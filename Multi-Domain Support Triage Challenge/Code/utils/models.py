"""
Shared data models for the triage agent.
Using dataclasses for clean, typed, serialisable structures.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class Status(str, Enum):
    REPLIED = "replied"
    ESCALATED = "escalated"


class RequestType(str, Enum):
    PRODUCT_ISSUE = "product_issue"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    INVALID = "invalid"


@dataclass
class SupportTicket:
    """Incoming support ticket."""
    issue: str
    subject: str
    company: str
    row_index: int = 0

    @property
    def company_normalized(self) -> str:
        c = (self.company or "").strip()
        if c.lower() in ("none", "n/a", "", "null"):
            return "None"
        return c

    @property
    def full_text(self) -> str:
        parts = []
        if self.subject and self.subject.strip():
            parts.append(f"Subject: {self.subject.strip()}")
        if self.issue and self.issue.strip():
            parts.append(f"Issue: {self.issue.strip()}")
        return "\n".join(parts)


@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from the corpus."""
    content: str
    source: str          # filename/path
    company: str         # HackerRank | Claude | Visa
    score: float         # cosine similarity
    chunk_index: int = 0


@dataclass
class TriageResult:
    """Output produced by the agent for one ticket."""
    status: str                  # replied | escalated
    product_area: str
    response: str
    justification: str
    request_type: str            # product_issue | feature_request | bug | invalid

    # Internal metadata (not written to output CSV)
    confidence: float = 0.0
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    escalation_reason: str = ""

    def to_dict(self, ticket: "SupportTicket") -> dict:
        return {
            "issue": ticket.issue,
            "subject": ticket.subject,
            "company": ticket.company,
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
        }
