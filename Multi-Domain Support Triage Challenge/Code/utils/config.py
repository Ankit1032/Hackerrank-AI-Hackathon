"""
Configuration management for the triage agent.
All secrets are read from environment variables — never hardcoded.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── API Keys (from env) ──────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

    # ── Model Selection ──────────────────────────────────────────────────────
    # GPT-4o for reasoning/generation (quality); text-embedding-3-small for retrieval (speed+cost)
    triage_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Paths ────────────────────────────────────────────────────────────────
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def corpus_dirs(self) -> dict:
        return {
            "HackerRank": self.data_dir / "hackerrank",
            "Claude": self.data_dir / "claude",
            "Visa": self.data_dir / "visa",
        }

    @property
    def index_dir(self) -> Path:
        idx = self.base_dir / "code" / ".index"
        idx.mkdir(parents=True, exist_ok=True)
        return idx

    # ── Retrieval ────────────────────────────────────────────────────────────
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # overlap between chunks
    top_k_retrieval: int = 6       # docs to retrieve per query
    similarity_threshold: float = 0.25  # min cosine similarity

    # ── Agent Behaviour ──────────────────────────────────────────────────────
    temperature: float = 0.0       # deterministic outputs
    max_tokens: int = 1024
    max_retries: int = 3

    # ── Escalation thresholds ────────────────────────────────────────────────
    # Confidence below this → escalate
    min_confidence_to_reply: float = 0.40

    def validate(self):
        if not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running: export OPENAI_API_KEY=sk-..."
            )
        return self
