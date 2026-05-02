"""
Corpus Loader — reads support documentation from the data/ directory.

Strategy:
- Walks all .md, .txt, .html files under data/hackerrank/, data/claude/, data/visa/
- Splits into overlapping chunks for retrieval
- Tags each chunk with its source company for domain-filtered retrieval
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("triage_agent.corpus_loader")


@dataclass
class RawDocument:
    content: str
    source: str       # relative file path
    company: str      # HackerRank | Claude | Visa


def _strip_html(text: str) -> str:
    """Very lightweight HTML tag removal."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    return text


def _clean_text(text: str) -> str:
    """Normalise whitespace."""
    text = _strip_html(text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_corpus(corpus_dirs: Dict[str, Path]) -> List[RawDocument]:
    """
    Load all documents from the corpus directories.
    
    Args:
        corpus_dirs: mapping of company_name -> directory_path
        
    Returns:
        List of RawDocument objects
    """
    documents = []
    extensions = {".md", ".txt", ".html", ".htm", ".rst"}

    for company, dir_path in corpus_dirs.items():
        if not dir_path.exists():
            logger.warning(f"Corpus directory not found: {dir_path}")
            continue

        file_count = 0
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() not in extensions:
                continue
            if file_path.stat().st_size == 0:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                content = _clean_text(content)
                if len(content) < 50:  # skip near-empty files
                    continue

                rel_path = str(file_path.relative_to(dir_path.parent.parent))
                documents.append(
                    RawDocument(
                        content=content,
                        source=rel_path,
                        company=company,
                    )
                )
                file_count += 1

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        logger.info(f"Loaded {file_count} documents for {company}")

    logger.info(f"Total corpus documents: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split documents into overlapping chunks suitable for embedding.
    
    Returns list of dicts with keys: content, source, company, chunk_index
    """
    chunks = []

    for doc in documents:
        text = doc.content
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to split at sentence boundary
            if end < len(text):
                split_pos = chunk_text.rfind(". ")
                if split_pos > chunk_size // 2:
                    chunk_text = chunk_text[: split_pos + 1]

            chunk_text = chunk_text.strip()
            if len(chunk_text) > 30:
                chunks.append(
                    {
                        "content": chunk_text,
                        "source": doc.source,
                        "company": doc.company,
                        "chunk_index": chunk_idx,
                    }
                )
                chunk_idx += 1

            # Advance with overlap
            advance = len(chunk_text) - chunk_overlap
            if advance <= 0:
                advance = max(1, len(chunk_text))
            start += advance

    logger.info(f"Total chunks after splitting: {len(chunks)}")
    return chunks
