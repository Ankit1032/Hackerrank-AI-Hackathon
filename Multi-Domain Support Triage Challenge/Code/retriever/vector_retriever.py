"""
Vector Retriever — semantic search over the support corpus.

Architecture:
- OpenAI text-embedding-3-small for embeddings (fast, cheap, high quality)
- FAISS for approximate nearest-neighbor search (in-process, no server needed)
- Persists index to disk so rebuild only happens on --rebuild-index

Domain-aware retrieval:
- When company is known, we boost/filter chunks from that company's corpus
- Fallback to all-corpus search when company is "None" or unrecognized
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from openai import OpenAI

from utils.models import RetrievedChunk
from utils.config import Config

logger = logging.getLogger("triage_agent.retriever")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available — falling back to numpy cosine search")


class VectorRetriever:
    """
    Semantic retriever with FAISS backend and domain-aware filtering.
    """

    INDEX_FILE = "faiss.index"
    META_FILE = "chunks_meta.pkl"
    EMBEDDINGS_FILE = "embeddings.npy"

    def __init__(self, config: Config, rebuild: bool = False):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None  # FAISS index

        index_dir = config.index_dir

        if not rebuild and self._index_exists(index_dir):
            logger.info("Loading existing vector index from disk...")
            self._load_index(index_dir)
        else:
            logger.info("Vector index not found or rebuild requested — will build after ingestion.")

    def _index_exists(self, index_dir: Path) -> bool:
        return (
            (index_dir / self.META_FILE).exists()
            and (index_dir / self.EMBEDDINGS_FILE).exists()
        )

    def _load_index(self, index_dir: Path):
        with open(index_dir / self.META_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        self.embeddings = np.load(index_dir / self.EMBEDDINGS_FILE)

        if FAISS_AVAILABLE and (index_dir / self.INDEX_FILE).exists():
            self.index = faiss.read_index(str(index_dir / self.INDEX_FILE))
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        else:
            logger.info(f"Loaded {len(self.chunks)} chunks (numpy fallback)")

    def _save_index(self, index_dir: Path):
        with open(index_dir / self.META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(index_dir / self.EMBEDDINGS_FILE, self.embeddings)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(index_dir / self.INDEX_FILE))
        logger.info("Index saved to disk.")

    def build_index(self, chunks: List[Dict]):
        """
        Embed all chunks and build the FAISS index.
        Called once during initialization.
        """
        self.chunks = chunks
        logger.info(f"Embedding {len(chunks)} chunks with {self.config.embedding_model}...")

        texts = [c["content"] for c in chunks]
        embeddings = self._embed_batch(texts)
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        self.embeddings = self.embeddings / norms

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vecs
            self.index.add(self.embeddings)
            logger.info(f"FAISS index built with {self.index.ntotal} vectors (dim={dim})")
        else:
            logger.info(f"Numpy index built with {len(self.embeddings)} vectors")

        self._save_index(self.config.index_dir)

    def _embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed texts in batches to respect API limits."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate to avoid token limits
            batch = [t[:6000] for t in batch]

            try:
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=batch,
                    dimensions=self.config.embedding_dimensions,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                if i % 500 == 0 and i > 0:
                    logger.info(f"  Embedded {i}/{len(texts)} chunks...")

            except Exception as e:
                logger.error(f"Embedding batch {i} failed: {e}")
                # Return zero vectors for failed batch
                all_embeddings.extend([[0.0] * self.config.embedding_dimensions] * len(batch))

        return all_embeddings

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=[query[:6000]],
            dimensions=self.config.embedding_dimensions,
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def retrieve(
        self,
        query: str,
        company: str = "None",
        top_k: int = 6,
        min_score: float = 0.25,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k most relevant chunks for a query.

        Domain-aware strategy:
        1. If company is known, retrieve top_k from that company's corpus
        2. Also retrieve top_k/2 from all companies (cross-domain)
        3. Deduplicate and rank by score
        """
        if self.embeddings is None or len(self.chunks) == 0:
            logger.warning("Index not built — returning empty results")
            return []

        query_vec = self._embed_query(query)

        # Get company-specific indices
        company_norm = company.strip().title()
        company_mask = np.array(
            [i for i, c in enumerate(self.chunks) if c["company"] == company_norm]
        )
        all_indices = np.arange(len(self.chunks))

        results: Dict[int, float] = {}

        # Search within company corpus (boosted)
        if len(company_mask) > 0 and company_norm != "None":
            company_results = self._search(query_vec, indices=company_mask, top_k=top_k)
            for idx, score in company_results:
                results[idx] = score * 1.2  # boost company-specific results

        # Search across all (catches cross-domain queries)
        global_results = self._search(query_vec, indices=all_indices, top_k=top_k)
        for idx, score in global_results:
            if idx not in results:
                results[idx] = score

        # Sort by score descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        retrieved = []
        seen_content = set()

        for idx, score in sorted_results[:top_k]:
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            # Deduplicate near-identical chunks
            content_key = chunk["content"][:100]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)

            retrieved.append(
                RetrievedChunk(
                    content=chunk["content"],
                    source=chunk["source"],
                    company=chunk["company"],
                    score=min(score, 1.0),  # cap at 1.0 due to boost
                    chunk_index=chunk["chunk_index"],
                )
            )

        logger.debug(f"Retrieved {len(retrieved)} chunks for query: {query[:60]}...")
        return retrieved

    def _search(
        self, query_vec: np.ndarray, indices: np.ndarray, top_k: int
    ) -> List[Tuple[int, float]]:
        """Search the index and return (global_index, score) pairs."""
        if len(indices) == 0:
            return []

        subset_embeddings = self.embeddings[indices]

        if FAISS_AVAILABLE and self.index is not None and len(indices) == len(self.chunks):
            # Use FAISS for full index search
            query_2d = query_vec.reshape(1, -1)
            k = min(top_k, self.index.ntotal)
            scores, faiss_indices = self.index.search(query_2d, k)
            return [(int(i), float(s)) for i, s in zip(faiss_indices[0], scores[0]) if i >= 0]
        else:
            # Numpy cosine similarity for subsets
            scores = subset_embeddings @ query_vec
            top_k_local = min(top_k, len(indices))
            top_local_indices = np.argpartition(scores, -top_k_local)[-top_k_local:]
            top_local_indices = top_local_indices[np.argsort(scores[top_local_indices])[::-1]]
            return [(int(indices[i]), float(scores[i])) for i in top_local_indices]
