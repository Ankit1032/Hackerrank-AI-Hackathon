"""
Safety & Risk Checker.

Detects high-risk, sensitive, adversarial, or out-of-scope tickets
before they reach the LLM for response generation.
This is a deterministic rule-based pre-filter — fast and reliable.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    SAFE = "safe"
    MEDIUM = "medium"
    HIGH = "high"           # Must escalate
    CRITICAL = "critical"  # Adversarial / prompt-injection / fraud


@dataclass
class RiskAssessment:
    level: RiskLevel
    reason: str
    should_escalate: bool
    detected_patterns: list[str]


# ── Pattern banks ────────────────────────────────────────────────────────────

# Prompt injection / jailbreak attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+your\s+(system\s+)?prompt",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|corpus|retrieval|internal\s+rules|logic|documents)",
    r"show\s+(me\s+)?(all\s+)?(internal|system|hidden|retrieved|corpus)",
    r"affiche\s+toutes\s+les\s+règles",  # French injection (from test data)
    r"print\s+(your\s+)?prompt",
    r"output\s+(your\s+)?(full|entire|raw)\s+(prompt|context)",
    r"bypass\s+(safety|filter|restriction)",
    r"you\s+are\s+now\s+(DAN|jailbroken|unrestricted)",
    r"act\s+as\s+(if\s+you\s+(are|have)\s+no\s+restrictions)",
    r"pretend\s+you\s+(are|have\s+no)",
    r"la\s+logique\s+exacte\s+que\s+vous\s+utilisez",  # French injection
]

# Harmful / dangerous requests
HARMFUL_PATTERNS = [
    r"(delete|remove|wipe|destroy)\s+all\s+(files|data|system|records)",
    r"(give|provide|write)\s+(me\s+)?(the\s+)?code\s+to\s+(delete|destroy|hack|exploit)",
    r"(hack|exploit|attack|ddos|sql\s+injection)",
    r"bomb|weapon|explosive|malware|ransomware|virus",
    r"how\s+to\s+(kill|harm|hurt|stalk)",
]

# High-risk financial / fraud / legal patterns — need human
HIGH_RISK_FINANCIAL = [
    r"identity\s+theft",
    r"(card|account)\s+(stolen|compromised|hacked|fraud)",
    r"unauthorized\s+(transaction|charge|access)",
    r"dispute\s+a?\s*charge",
    r"report\s+(a\s+)?(fraud|scam|stolen\s+card)",
    r"lost\s+(my\s+)?(card|wallet)",
    r"stolen\s+(card|cheques|wallet|identity)",
    r"refund\s+.{0,30}\s+today",  # demanding immediate refund
    r"security\s+vulnerability",
    r"bug\s+bounty",
    r"(major|critical)\s+(security|vulnerability|exploit)",
]

# Account / access escalation — need workspace admin or HR
HIGH_RISK_ACCOUNT = [
    r"restore\s+my\s+access",
    r"not\s+(the\s+)?(owner|admin)",
    r"reinstate\s+(my\s+)?(account|access)",
]

# Out-of-scope (clearly off-topic)
OUT_OF_SCOPE_PATTERNS = [
    r"(name\s+of\s+the\s+actor|who\s+played|movie\s+trivia)",
    r"(weather|recipe|cook|sports\s+score)",
    r"give\s+me\s+(the\s+)?code\s+to\s+delete\s+all\s+files",
]


def _match_patterns(text: str, patterns: list[str]) -> list[str]:
    """Return list of matched pattern descriptions."""
    text_lower = text.lower()
    matched = []
    for pat in patterns:
        if re.search(pat, text_lower, re.IGNORECASE | re.DOTALL):
            matched.append(pat)
    return matched


class SafetyChecker:
    """Deterministic pre-filter applied before any LLM call."""

    def assess(self, issue: str, subject: str = "", company: str = "") -> RiskAssessment:
        combined = f"{subject} {issue}".strip()

        # 1. Prompt injection / adversarial — CRITICAL
        injection_hits = _match_patterns(combined, INJECTION_PATTERNS)
        if injection_hits:
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                reason="Potential prompt-injection or system-disclosure attack detected.",
                should_escalate=True,
                detected_patterns=injection_hits,
            )

        # 2. Harmful / dangerous request
        harmful_hits = _match_patterns(combined, HARMFUL_PATTERNS)
        if harmful_hits:
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                reason="Harmful or dangerous request detected. Cannot be processed.",
                should_escalate=True,
                detected_patterns=harmful_hits,
            )

        # 3. High-risk financial / fraud
        fin_hits = _match_patterns(combined, HIGH_RISK_FINANCIAL)
        if fin_hits:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                reason="High-risk financial/security issue — requires human review.",
                should_escalate=True,
                detected_patterns=fin_hits,
            )

        # 4. Account escalation
        acc_hits = _match_patterns(combined, HIGH_RISK_ACCOUNT)
        if acc_hits:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                reason="Account access/permission issue requiring admin intervention.",
                should_escalate=True,
                detected_patterns=acc_hits,
            )

        # 5. Out of scope — reply with "out of scope" not escalate
        oos_hits = _match_patterns(combined, OUT_OF_SCOPE_PATTERNS)
        if oos_hits:
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                reason="Out of scope — not related to supported products.",
                should_escalate=False,  # reply saying out of scope
                detected_patterns=oos_hits,
            )

        return RiskAssessment(
            level=RiskLevel.SAFE,
            reason="No immediate risk flags detected.",
            should_escalate=False,
            detected_patterns=[],
        )
