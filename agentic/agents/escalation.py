"""Escalation agent: handles tickets that cannot be auto-resolved."""

import json
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from agentic.schemas import EscalationResponse
from agentic.prompts import get_escalation_prompt

logger = logging.getLogger("uda_hub.agents.escalation")


def escalate_ticket(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Prepare an escalation package for a human agent.

    Summarises the ticket, explains why automation failed, and suggests
    next steps for the human support team.
    """
    llm = config["configurable"]["llm"]
    structured_llm = llm.with_structured_output(EscalationResponse)

    classification = state.get("classification")
    classification_str = json.dumps(classification.model_dump()) if classification else "Not classified"

    resolver_resp = state.get("resolver_response")
    resolver_str = json.dumps(resolver_resp.model_dump()) if resolver_resp else "No previous resolution attempt"

    customer_history = json.dumps(state.get("customer_history") or [])

    prompt_template = get_escalation_prompt()
    messages = prompt_template.invoke({
        "ticket_content": state.get("user_input", ""),
        "classification": classification_str,
        "resolver_response": resolver_str,
        "customer_history": customer_history,
        "chat_history": state.get("messages", []),
    }).to_messages()

    escalation_response = structured_llm.invoke(messages)

    logger.info(
        "Ticket escalated",
        extra={
            "event": "escalate",
            "node": "escalation",
            "ticket_id": state.get("ticket_id"),
            "assigned_to": escalation_response.assigned_to,
            "reason": escalation_response.reason,
        },
    )

    return {
        "escalation_response": escalation_response,
        "actions_taken": ["escalate_ticket"],
        "next_step": "end",
    }
