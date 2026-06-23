"""Pydantic schemas for UDA-Hub agent state, ticket classification, and responses."""

from typing import List, Optional, Dict, Any, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
import operator


# ---------------------------------------------------------------------------
# Structured output schemas used by LLM calls
# ---------------------------------------------------------------------------

class TicketClassification(BaseModel):
    """Result of classifying an incoming support ticket."""
    category: Literal["billing", "technical", "account", "subscription", "general"] = Field(
        description="High-level ticket category"
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="How urgent the ticket is"
    )
    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Estimated complexity to resolve"
    )
    requires_tool: bool = Field(
        default=False,
        description="Whether a support tool (account lookup, refund, etc.) is likely needed",
    )
    reasoning: str = Field(description="Brief explanation for the classification")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ResolverResponse(BaseModel):
    """Structured response from the resolver agent."""
    answer: str = Field(description="The proposed response to the customer")
    sources: List[str] = Field(default_factory=list, description="KB article IDs used")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_escalation: bool = Field(default=False)
    tools_invoked: List[str] = Field(default_factory=list)


class EscalationResponse(BaseModel):
    """Structured response when escalating a ticket."""
    summary: str = Field(description="Summary of the issue for the human agent")
    reason: str = Field(description="Why the ticket is being escalated")
    suggested_actions: List[str] = Field(default_factory=list)
    assigned_to: str = Field(default="senior_support", description="Team to assign to")


class SupervisorDecision(BaseModel):
    """Supervisor routing decision."""
    next_agent: Literal["classifier", "resolver", "escalation", "action", "end"] = Field(
        description="Which agent to route to next"
    )
    reasoning: str = Field(description="Why this routing was chosen")


class MemorySummary(BaseModel):
    """Summary produced after processing for long-term storage."""
    issue_summary: str = Field(description="Concise summary of the customer's issue")
    resolution: str = Field(description="How it was resolved or escalated")
    customer_preferences: List[str] = Field(
        default_factory=list, description="Inferred customer preferences"
    )


# ---------------------------------------------------------------------------
# LangGraph agent state
# ---------------------------------------------------------------------------

class AgentState(Dict):
    """
    The shared state flowing through the LangGraph workflow.

    Uses Annotated types for LangGraph reducers:
    - messages: accumulated via add_messages
    - actions_taken: accumulated via operator.add
    """
    # Conversation
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: Optional[str]

    # Ticket data
    ticket_id: Optional[str]
    customer_id: Optional[str]
    ticket_metadata: Optional[Dict[str, Any]]

    # Classification
    classification: Optional[TicketClassification]

    # Routing
    next_step: str

    # Resolver output
    resolver_response: Optional[ResolverResponse]

    # Escalation output
    escalation_response: Optional[EscalationResponse]

    # Memory
    conversation_summary: str
    customer_history: Optional[List[Dict[str, Any]]]

    # Session management
    session_id: Optional[str]

    # Tool tracking
    tools_used: List[str]

    # Tracing
    actions_taken: Annotated[List[str], operator.add]
