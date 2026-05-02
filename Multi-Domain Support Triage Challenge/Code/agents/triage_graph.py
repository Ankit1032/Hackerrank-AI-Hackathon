"""
LangGraph Triage Agent — core orchestration graph.

Graph nodes:
  1. safety_check     → pre-filter for injections, harmful content
  2. domain_detect    → identify company/domain if "None"
  3. retrieve         → fetch relevant corpus chunks
  4. classify         → determine product_area and request_type
  5. decide           → reply vs escalate decision
  6. generate         → produce final user-facing response
  7. format_output    → assemble final TriageResult

Edges are conditional based on risk level and confidence.
"""

import logging
from typing import TypedDict, Optional, List, Annotated
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from utils.config import Config
from utils.models import SupportTicket, TriageResult, RetrievedChunk
from utils.safety import SafetyChecker, RiskLevel
from retriever.vector_retriever import VectorRetriever

logger = logging.getLogger("triage_agent.graph")


# ── Graph State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    ticket: dict                      # SupportTicket serialized

    # Safety
    risk_level: str
    risk_reason: str
    should_escalate_safety: bool

    # Retrieval
    retrieved_chunks: List[dict]
    retrieval_context: str

    # Classification
    product_area: str
    request_type: str
    domain_confidence: float

    # Decision
    should_escalate: bool
    escalation_reason: str
    confidence: float

    # Output
    response: str
    justification: str
    status: str
    final_result: dict


# ── Node Implementations ─────────────────────────────────────────────────────

PRODUCT_AREAS_BY_COMPANY = {
    "HackerRank": [
        "screen", "interview", "community", "certification", "work",
        "assessment", "candidate_management", "test_management",
        "billing", "account", "technical_issue", "integration",
    ],
    "Claude": [
        "privacy", "billing", "api", "conversation_management",
        "account", "data_usage", "technical_issue", "enterprise",
        "education", "safety", "web_crawl",
    ],
    "Visa": [
        "card_services", "fraud_prevention", "travel_support",
        "dispute_resolution", "general_support", "merchant_services",
        "cash_access", "security",
    ],
    "None": [
        "general_support", "out_of_scope", "conversation_management",
    ],
}


def _build_context(chunks: List[dict], max_chars: int = 4000) -> str:
    """Build a context string from retrieved chunks."""
    if not chunks:
        return "No relevant documentation found in the corpus."

    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        snippet = f"[Source: {chunk['source']} | Company: {chunk['company']}]\n{chunk['content']}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)

    return "\n\n---\n\n".join(parts)


class TriageGraph:
    """
    LangGraph-based triage orchestration.
    Builds and compiles the state machine graph.
    """

    def __init__(self, config: Config, retriever: VectorRetriever):
        self.config = config
        self.retriever = retriever
        self.safety_checker = SafetyChecker()

        # LLM — GPT-4o for high-quality reasoning, temp=0 for determinism
        self.llm = ChatOpenAI(
            model=config.triage_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key,
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("safety_check", self._node_safety_check)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("classify_and_decide", self._node_classify_and_decide)
        workflow.add_node("generate_response", self._node_generate_response)
        workflow.add_node("handle_escalation", self._node_handle_escalation)
        workflow.add_node("handle_out_of_scope", self._node_handle_out_of_scope)

        # Entry point
        workflow.set_entry_point("safety_check")

        # Safety check → conditional routing
        workflow.add_conditional_edges(
            "safety_check",
            self._route_after_safety,
            {
                "escalate": "handle_escalation",
                "out_of_scope": "handle_out_of_scope",
                "retrieve": "retrieve",
            },
        )

        # Retrieve → classify & decide
        workflow.add_edge("retrieve", "classify_and_decide")

        # Classify & decide → conditional routing
        workflow.add_conditional_edges(
            "classify_and_decide",
            self._route_after_classify,
            {
                "escalate": "handle_escalation",
                "generate": "generate_response",
            },
        )

        # Terminal nodes → END
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_escalation", END)
        workflow.add_edge("handle_out_of_scope", END)

        return workflow.compile()

    # ── Router Functions ─────────────────────────────────────────────────────

    def _route_after_safety(self, state: AgentState) -> str:
        if state["should_escalate_safety"]:
            if state["risk_level"] == RiskLevel.CRITICAL:
                return "escalate"
            return "escalate"
        # Check for out-of-scope (medium risk, no escalate)
        if state.get("risk_level") == RiskLevel.MEDIUM and not state["should_escalate_safety"]:
            return "out_of_scope"
        return "retrieve"

    def _route_after_classify(self, state: AgentState) -> str:
        return "escalate" if state["should_escalate"] else "generate"

    # ── Node: Safety Check ───────────────────────────────────────────────────

    def _node_safety_check(self, state: AgentState) -> dict:
        ticket = state["ticket"]
        assessment = self.safety_checker.assess(
            issue=ticket["issue"],
            subject=ticket["subject"],
            company=ticket["company"],
        )
        logger.debug(f"Safety: {assessment.level} — {assessment.reason}")
        return {
            "risk_level": assessment.level,
            "risk_reason": assessment.reason,
            "should_escalate_safety": assessment.should_escalate,
        }

    # ── Node: Retrieve ───────────────────────────────────────────────────────

    def _node_retrieve(self, state: AgentState) -> dict:
        ticket = state["ticket"]
        query = f"{ticket['subject']} {ticket['issue']}".strip()

        chunks = self.retriever.retrieve(
            query=query,
            company=ticket["company"],
            top_k=self.config.top_k_retrieval,
            min_score=self.config.similarity_threshold,
        )

        chunk_dicts = [
            {
                "content": c.content,
                "source": c.source,
                "company": c.company,
                "score": c.score,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        context = _build_context(chunk_dicts)
        return {
            "retrieved_chunks": chunk_dicts,
            "retrieval_context": context,
        }

    # ── Node: Classify & Decide ──────────────────────────────────────────────

    def _node_classify_and_decide(self, state: AgentState) -> dict:
        ticket = state["ticket"]
        context = state.get("retrieval_context", "")
        company = ticket["company"]

        valid_areas = PRODUCT_AREAS_BY_COMPANY.get(
            company, PRODUCT_AREAS_BY_COMPANY["None"]
        )
        all_areas = list(
            set(a for areas in PRODUCT_AREAS_BY_COMPANY.values() for a in areas)
        )

        system_prompt = f"""You are a support triage classifier for a multi-domain support system.
Your job is to analyze a support ticket and return a JSON classification.

Companies supported: HackerRank, Claude (Anthropic), Visa
Valid request_type values: product_issue, feature_request, bug, invalid
Preferred product_area values for {company}: {valid_areas}
All possible product_area values: {all_areas}

You MUST return ONLY valid JSON — no markdown, no explanation.
Return this exact JSON structure:
{{
  "product_area": "<area>",
  "request_type": "<type>",
  "should_escalate": <true|false>,
  "escalation_reason": "<reason or empty string>",
  "confidence": <0.0-1.0>,
  "analysis": "<1-2 sentence analysis>"
}}

Escalate when:
- Billing disputes, payment failures, or refund demands requiring account access
- Account permission changes requiring admin action
- Platform-wide outages (cannot verify or fix from docs)
- Security vulnerabilities or bug bounty reports
- Legal, regulatory, or compliance issues
- Ticket is totally irrelevant/malicious — reply with invalid, don't escalate

Do NOT escalate for:
- Standard how-to questions answerable from docs
- Feature requests
- Out-of-scope/invalid tickets (reply saying out of scope)
"""

        user_prompt = f"""Ticket to classify:
Company: {ticket['company']}
Subject: {ticket['subject']}
Issue: {ticket['issue']}

Retrieved corpus context:
{context[:2000]}
"""
        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            import json, re
            raw = response.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)

            return {
                "product_area": data.get("product_area", "general_support"),
                "request_type": data.get("request_type", "product_issue"),
                "should_escalate": bool(data.get("should_escalate", False)),
                "escalation_reason": data.get("escalation_reason", ""),
                "confidence": float(data.get("confidence", 0.5)),
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "product_area": "general_support",
                "request_type": "product_issue",
                "should_escalate": True,
                "escalation_reason": f"Classification error: {e}",
                "confidence": 0.0,
            }

    # ── Node: Generate Response ──────────────────────────────────────────────

    def _node_generate_response(self, state: AgentState) -> dict:
        ticket = state["ticket"]
        context = state.get("retrieval_context", "")
        product_area = state.get("product_area", "general_support")
        request_type = state.get("request_type", "product_issue")

        has_context = (
            context
            and context != "No relevant documentation found in the corpus."
            and len(state.get("retrieved_chunks", [])) > 0
        )

        system_prompt = """You are a helpful support agent for a multi-domain support system covering HackerRank, Claude (Anthropic), and Visa.

CRITICAL RULES:
1. ONLY use information from the provided corpus context to answer. Do NOT use your parametric knowledge.
2. If the corpus does not contain enough information to answer, say so honestly and suggest the user contact support directly.
3. Be specific, actionable, and concise.
4. Do not hallucinate policies, phone numbers, URLs, or steps not in the context.
5. Address all parts of the user's question.
6. If the ticket is invalid/irrelevant, politely say it's out of scope for supported products.

Response format: Plain text, step-by-step when needed. No markdown headers. Keep it under 300 words.
"""

        context_section = (
            f"Relevant documentation from corpus:\n{context}"
            if has_context
            else "No directly relevant documentation found in corpus."
        )

        user_prompt = f"""Support Ticket:
Company: {ticket['company']}
Subject: {ticket['subject']}
Issue: {ticket['issue']}
Product Area: {product_area}
Request Type: {request_type}

{context_section}

Generate a helpful, grounded response to this support ticket:"""

        try:
            response = self.llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            generated_response = response.content.strip()

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            generated_response = (
                "We're sorry, we were unable to process your request at this time. "
                "Please contact our support team directly for assistance."
            )

        # Generate justification
        justification = self._generate_justification(
            ticket=ticket,
            product_area=product_area,
            request_type=request_type,
            has_context=has_context,
            chunks=state.get("retrieved_chunks", []),
        )

        return {
            "response": generated_response,
            "justification": justification,
            "status": "replied",
            "final_result": {
                "status": "replied",
                "product_area": product_area,
                "response": generated_response,
                "justification": justification,
                "request_type": request_type,
            },
        }

    def _generate_justification(
        self,
        ticket: dict,
        product_area: str,
        request_type: str,
        has_context: bool,
        chunks: List[dict],
    ) -> str:
        sources = list(
            {c["source"].split("/")[-1] for c in chunks[:3]}
        )
        source_str = ", ".join(sources) if sources else "no corpus match"

        base = f"Classified as {request_type} under {product_area}."
        if has_context:
            base += f" Response grounded in corpus ({source_str})."
        else:
            base += " No strong corpus match; response based on general support guidance."
        return base

    # ── Node: Handle Escalation ──────────────────────────────────────────────

    def _node_handle_escalation(self, state: AgentState) -> dict:
        ticket = state["ticket"]
        reason = (
            state.get("risk_reason")
            or state.get("escalation_reason")
            or "Issue requires human review."
        )
        risk_level = state.get("risk_level", "high")
        product_area = state.get("product_area", "general_support")
        request_type = state.get("request_type", "product_issue")

        # Safety escalation for adversarial input — minimal response
        if risk_level in (RiskLevel.CRITICAL,):
            response = (
                "We're unable to process this request. "
                "If you have a genuine support issue, please contact us through official channels."
            )
            justification = f"Escalated: {reason}"
        else:
            response = (
                f"Thank you for reaching out. Your request has been escalated to our support team "
                f"as it requires human review. A specialist will get back to you shortly. "
                f"Please reference your ticket details when you hear from us."
            )
            justification = f"Escalated to human agent. Reason: {reason}"

        result = {
            "status": "escalated",
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type,
        }

        return {
            "response": response,
            "justification": justification,
            "status": "escalated",
            "final_result": result,
        }

    # ── Node: Handle Out-of-Scope ────────────────────────────────────────────

    def _node_handle_out_of_scope(self, state: AgentState) -> dict:
        response = (
            "I'm sorry, this request is outside the scope of our supported products. "
            "We handle support for HackerRank, Claude (Anthropic), and Visa. "
            "If your issue relates to one of these products, please provide more details."
        )
        justification = (
            "Ticket is irrelevant or out of scope for the supported products. "
            "Responded with an out-of-scope message rather than escalating."
        )
        result = {
            "status": "replied",
            "product_area": "out_of_scope",
            "response": response,
            "justification": justification,
            "request_type": "invalid",
        }
        return {
            "response": response,
            "justification": justification,
            "status": "replied",
            "product_area": "out_of_scope",
            "request_type": "invalid",
            "final_result": result,
        }

    # ── Main Invoke ──────────────────────────────────────────────────────────

    def run(self, ticket: SupportTicket) -> TriageResult:
        """Run the graph for a single ticket."""
        initial_state: AgentState = {
            "ticket": {
                "issue": ticket.issue,
                "subject": ticket.subject,
                "company": ticket.company_normalized,
            },
            "risk_level": "",
            "risk_reason": "",
            "should_escalate_safety": False,
            "retrieved_chunks": [],
            "retrieval_context": "",
            "product_area": "general_support",
            "request_type": "product_issue",
            "domain_confidence": 0.0,
            "should_escalate": False,
            "escalation_reason": "",
            "confidence": 0.0,
            "response": "",
            "justification": "",
            "status": "replied",
            "final_result": {},
        }

        final_state = self.graph.invoke(initial_state)
        result_dict = final_state.get("final_result", {})

        return TriageResult(
            status=result_dict.get("status", "escalated"),
            product_area=result_dict.get("product_area", "general_support"),
            response=result_dict.get("response", ""),
            justification=result_dict.get("justification", ""),
            request_type=result_dict.get("request_type", "product_issue"),
            confidence=final_state.get("confidence", 0.0),
            retrieved_chunks=[
                RetrievedChunk(
                    content=c["content"],
                    source=c["source"],
                    company=c["company"],
                    score=c["score"],
                    chunk_index=c["chunk_index"],
                )
                for c in final_state.get("retrieved_chunks", [])
            ],
            escalation_reason=final_state.get("escalation_reason", ""),
        )
