"""
UDA-Hub LangGraph Workflow
--------------------------
Defines the multi-agent graph using a Supervisor pattern.

Graph structure:

  supervisor ──► classifier ──► supervisor
      │                             │
      ├──► resolver ────────────────┤
      │         │                   │
      │         └──► (tools)        │
      │                             │
      ├──► action ──────────────────┤
      │                             │
      ├──► escalation ──► update_memory ──► END
      │                                     ▲
      └──► update_memory ──────────────────┘

The supervisor is re-entered after classifier, resolver, and action so it
can decide the next step. When the supervisor decides "end", it routes
through update_memory to persist long-term memory, then terminates.
"""

from typing import Dict, Any, List, Optional, Annotated
import logging
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from agentic.agents.supervisor import supervisor_route
from agentic.agents.classifier import classify_ticket
from agentic.agents.resolver import resolve_ticket
from agentic.agents.escalation import escalate_ticket
from agentic.agents.action_agent import execute_action
from agentic.schemas import TicketClassification, ResolverResponse, EscalationResponse, MemorySummary
from agentic.prompts import MEMORY_SUMMARY_PROMPT

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate

logger = logging.getLogger("uda_hub.workflow")


# ---------------------------------------------------------------------------
# State definition (TypedDict with LangGraph reducers)
# ---------------------------------------------------------------------------

from typing import TypedDict


class WorkflowState(TypedDict):
    # Conversation messages (accumulated)
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: Optional[str]

    # Ticket info
    ticket_id: Optional[str]
    customer_id: Optional[str]
    ticket_metadata: Optional[Dict[str, Any]]

    # Classification
    classification: Optional[TicketClassification]

    # Routing
    next_step: str

    # Resolver
    resolver_response: Optional[ResolverResponse]

    # Escalation
    escalation_response: Optional[EscalationResponse]

    # Memory
    conversation_summary: str
    customer_history: Optional[List[Dict[str, Any]]]

    # Session
    session_id: Optional[str]

    # Tool tracking
    tools_used: List[str]

    # Tracing — uses operator.add reducer so each node appends its name
    actions_taken: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Update-memory node (runs before END to persist long-term memory)
# ---------------------------------------------------------------------------

def update_memory(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Summarise the interaction for long-term storage.

    Uses the LLM to produce a MemorySummary and writes it via the
    long_term_memory object stored in config (if provided).
    """
    llm = config["configurable"]["llm"]
    long_term_memory = config["configurable"].get("long_term_memory")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(MemorySummary)

    try:
        summary = structured_llm.invoke(prompt)
    except Exception:
        summary = MemorySummary(
            issue_summary=state.get("user_input", ""),
            resolution="Processed by UDA-Hub",
            customer_preferences=[],
        )

    logger.info(
        "Memory update",
        extra={
            "event": "update_memory",
            "node": "update_memory",
            "ticket_id": state.get("ticket_id"),
            "customer_id": state.get("customer_id") or "unknown",
        },
    )

    # Persist to long-term memory if available
    customer_id = state.get("customer_id") or "unknown"
    if long_term_memory is not None:
        category = ""
        classification = state.get("classification")
        if classification:
            category = classification.category

        long_term_memory.store_resolved_ticket(
            customer_id=customer_id,
            issue_summary=summary.issue_summary,
            resolution=summary.resolution,
            category=category,
            ticket_id=state.get("ticket_id", ""),
        )

        for pref in summary.customer_preferences:
            long_term_memory.store_customer_preference(
                customer_id=customer_id,
                key="preference",
                value=pref,
            )

    return {
        "conversation_summary": summary.issue_summary,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }


# ---------------------------------------------------------------------------
# Router function (reads next_step from state)
# ---------------------------------------------------------------------------

def route_from_supervisor(state: Dict[str, Any]) -> str:
    next_step = state.get("next_step", "end")
    mapping = {
        "classifier": "classifier",
        "resolver": "resolver",
        "escalation": "escalation",
        "action": "action",
        "end": "update_memory",
    }
    return mapping.get(next_step, "update_memory")


def route_after_classifier(state: Dict[str, Any]) -> str:
    return "supervisor"


def route_after_resolver(state: Dict[str, Any]) -> str:
    next_step = state.get("next_step", "supervisor")
    if next_step == "escalation":
        return "escalation"
    if next_step == "end":
        return "update_memory"
    return "supervisor"


def route_after_action(state: Dict[str, Any]) -> str:
    return "supervisor"


def route_after_escalation(state: Dict[str, Any]) -> str:
    return "update_memory"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_workflow():
    """
    Build and compile the UDA-Hub LangGraph workflow.

    Returns a compiled graph with InMemorySaver checkpointer for
    thread-level (short-term) memory persistence.
    """
    workflow = StateGraph(WorkflowState)

    # Add all agent nodes
    workflow.add_node("supervisor", supervisor_route)
    workflow.add_node("classifier", classify_ticket)
    workflow.add_node("resolver", resolve_ticket)
    workflow.add_node("escalation", escalate_ticket)
    workflow.add_node("action", execute_action)
    workflow.add_node("update_memory", update_memory)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routes to the next agent
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "classifier": "classifier",
            "resolver": "resolver",
            "escalation": "escalation",
            "action": "action",
            "update_memory": "update_memory",
        },
    )

    # After classifier → back to supervisor for routing decision
    workflow.add_edge("classifier", "supervisor")

    # After resolver → conditional (may escalate, end, or go back to supervisor)
    workflow.add_conditional_edges(
        "resolver",
        route_after_resolver,
        {
            "escalation": "escalation",
            "update_memory": "update_memory",
            "supervisor": "supervisor",
        },
    )

    # After action → back to supervisor
    workflow.add_edge("action", "supervisor")

    # After escalation → update memory
    workflow.add_edge("escalation", "update_memory")

    # After update_memory → END
    workflow.add_edge("update_memory", END)

    # Compile with InMemorySaver for short-term (thread-level) memory
    return workflow.compile(checkpointer=InMemorySaver())
