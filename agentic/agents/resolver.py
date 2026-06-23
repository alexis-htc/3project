"""Resolver agent: attempts to resolve tickets using RAG and tools."""

import json
import logging
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent

from agentic.schemas import ResolverResponse
from agentic.prompts import get_resolver_prompt

logger = logging.getLogger("uda_hub.agents.resolver")


def resolve_ticket(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Attempt to resolve the ticket using knowledge base search and support tools.

    The resolver:
    1. Builds a prompt with ticket context.
    2. Invokes a ReAct agent with all available tools.
    3. Parses the structured response (answer, confidence, sources).
    4. Routes to escalation if confidence is below threshold.
    """
    llm = config["configurable"]["llm"]
    tools = config["configurable"]["tools"]

    classification = state.get("classification")
    classification_str = json.dumps(classification.model_dump()) if classification else "Not classified"
    customer_history = json.dumps(state.get("customer_history") or [])

    prompt_template = get_resolver_prompt()
    messages = prompt_template.invoke({
        "ticket_content": state.get("user_input", ""),
        "classification": classification_str,
        "customer_history": customer_history,
        "chat_history": state.get("messages", []),
    }).to_messages()

    llm_with_tools = llm.bind_tools(tools)
    agent = create_react_agent(
        model=llm_with_tools,
        tools=tools,
        response_format=ResolverResponse,
    )

    logger.debug(
        "Resolver invoked",
        extra={"event": "resolve_start", "node": "resolver", "ticket_id": state.get("ticket_id")},
    )

    result = agent.invoke({"messages": messages})
    result_messages = result.get("messages", [])
    tools_used = [m.name for m in result_messages if isinstance(m, ToolMessage)]

    # Extract structured response from the last message
    last_msg = result_messages[-1] if result_messages else None
    resolver_response = None
    if last_msg and hasattr(last_msg, "content"):
        content = last_msg.content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                resolver_response = ResolverResponse(**parsed)
            except (json.JSONDecodeError, ValueError):
                resolver_response = ResolverResponse(
                    answer=content,
                    confidence=0.5,
                    sources=[],
                    needs_escalation=False,
                    tools_invoked=tools_used,
                )
        elif isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
            combined = "\n".join(text_parts) if text_parts else str(content)
            resolver_response = ResolverResponse(
                answer=combined,
                confidence=0.7,
                sources=[],
                needs_escalation=False,
                tools_invoked=tools_used,
            )

    if resolver_response is None:
        resolver_response = ResolverResponse(
            answer="I was unable to find a resolution. Escalating to a human agent.",
            confidence=0.0,
            needs_escalation=True,
            tools_invoked=tools_used,
        )

    # Determine next step based on confidence
    confidence_threshold = 0.4
    if resolver_response.needs_escalation or resolver_response.confidence < confidence_threshold:
        next_step = "escalation"
    else:
        next_step = "end"

    logger.info(
        "Resolution attempt complete",
        extra={
            "event": "resolve",
            "node": "resolver",
            "ticket_id": state.get("ticket_id"),
            "confidence": resolver_response.confidence,
            "needs_escalation": resolver_response.needs_escalation,
            "tools_invoked": tools_used,
            "next_step": next_step,
        },
    )

    return {
        "messages": result_messages,
        "resolver_response": resolver_response,
        "tools_used": tools_used,
        "actions_taken": ["resolve_ticket"],
        "next_step": next_step,
    }
