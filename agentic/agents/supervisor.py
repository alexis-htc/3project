"""Supervisor agent: central orchestrator that routes tickets to the right agent."""

import json
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from agentic.schemas import SupervisorDecision
from agentic.prompts import SUPERVISOR_ROUTING_PROMPT

logger = logging.getLogger("uda_hub.agents.supervisor")


def supervisor_route(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Decide which agent should handle the ticket next.

    Routing logic (with LLM fallback):
    - If not yet classified → classifier
    - If classified as critical → escalation
    - If classified as complex and already attempted resolution → escalation
    - Otherwise → resolver (which may invoke action tools internally)
    - If resolver completed with high confidence → end
    """
    llm = config["configurable"]["llm"]
    actions_taken = state.get("actions_taken", [])
    classification = state.get("classification")
    resolver_response = state.get("resolver_response")

    def _route(next_step: str, reason: str) -> Dict[str, Any]:
        logger.info(
            "Routing decision",
            extra={
                "event": "route",
                "node": "supervisor",
                "next_step": next_step,
                "reason": reason,
                "ticket_id": state.get("ticket_id"),
            },
        )
        return {"actions_taken": ["supervisor_route"], "next_step": next_step}

    # Rule-based fast path
    if classification is None and "classify_ticket" not in actions_taken:
        return _route("classifier", "ticket_not_yet_classified")

    if classification and classification.urgency == "critical" and "escalate_ticket" not in actions_taken:
        return _route("escalation", "critical_urgency")

    if resolver_response and not resolver_response.needs_escalation and resolver_response.confidence >= 0.4:
        return _route("end", f"resolved_with_confidence_{resolver_response.confidence}")

    if resolver_response and (resolver_response.needs_escalation or resolver_response.confidence < 0.4):
        if "escalate_ticket" not in actions_taken:
            return _route("escalation", f"low_confidence_{resolver_response.confidence}_or_needs_escalation")
        return _route("end", "already_escalated")

    if classification and "resolve_ticket" not in actions_taken:
        return _route("resolver", "classified_ready_to_resolve")

    # LLM-based routing for ambiguous cases
    logger.debug(
        "Using LLM for ambiguous routing",
        extra={"event": "route_llm_fallback", "node": "supervisor", "ticket_id": state.get("ticket_id")},
    )
    classification_str = json.dumps(classification.model_dump()) if classification else "Not classified"
    customer_history = json.dumps(state.get("customer_history") or [])

    prompt = SUPERVISOR_ROUTING_PROMPT.format(
        ticket_content=state.get("user_input", ""),
        classification=classification_str,
        customer_history=customer_history,
        current_state=json.dumps(actions_taken),
    )

    structured_llm = llm.with_structured_output(SupervisorDecision)
    decision = structured_llm.invoke(prompt)

    return _route(decision.next_agent, f"llm_decision: {decision.reasoning[:120]}")
