"""Classifier agent: categorizes incoming support tickets."""

import json
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

from agentic.schemas import TicketClassification
from agentic.prompts import TICKET_CLASSIFICATION_PROMPT


def classify_ticket(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Classify the incoming ticket by category, urgency, and complexity.

    Updates the state with:
      - classification (TicketClassification)
      - actions_taken  (appends 'classify_ticket')
      - next_step      (set to 'supervisor' for re-routing)
    """
    llm = config["configurable"]["llm"]
    structured_llm = llm.with_structured_output(TicketClassification)

    ticket_content = state.get("user_input", "")
    ticket_metadata = json.dumps(state.get("ticket_metadata") or {})
    customer_history = json.dumps(state.get("customer_history") or [])

    prompt = TICKET_CLASSIFICATION_PROMPT.format(
        ticket_content=ticket_content,
        ticket_metadata=ticket_metadata,
        customer_history=customer_history,
    )

    classification = structured_llm.invoke(prompt)

    return {
        "classification": classification,
        "actions_taken": ["classify_ticket"],
        "next_step": "supervisor",
    }
