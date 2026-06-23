"""Supervisor agent: central orchestrator that routes tickets to the right agent."""

import json
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from agentic.schemas import SupervisorDecision
from agentic.prompts import SUPERVISOR_ROUTING_PROMPT


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

    # Rule-based fast path
    if classification is None and "classify_ticket" not in actions_taken:
        return {
            "actions_taken": ["supervisor_route"],
            "next_step": "classifier",
        }

    if classification and classification.urgency == "critical" and "escalate_ticket" not in actions_taken:
        return {
            "actions_taken": ["supervisor_route"],
            "next_step": "escalation",
        }

    if resolver_response and not resolver_response.needs_escalation and resolver_response.confidence >= 0.4:
        return {
            "actions_taken": ["supervisor_route"],
            "next_step": "end",
        }

    if resolver_response and (resolver_response.needs_escalation or resolver_response.confidence < 0.4):
        if "escalate_ticket" not in actions_taken:
            return {
                "actions_taken": ["supervisor_route"],
                "next_step": "escalation",
            }
        return {
            "actions_taken": ["supervisor_route"],
            "next_step": "end",
        }

    if classification and "resolve_ticket" not in actions_taken:
        return {
            "actions_taken": ["supervisor_route"],
            "next_step": "resolver",
        }

    # LLM-based routing for ambiguous cases
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

    return {
        "actions_taken": ["supervisor_route"],
        "next_step": decision.next_agent,
    }
