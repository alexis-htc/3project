"""Action agent: executes support operations via tools."""

from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent


ACTION_SYSTEM_PROMPT = """You are a support operations agent for CultPass.

Execute the requested support operation using the available tools:
- account_lookup: Look up customer account information
- subscription_management: Check or modify subscriptions
- refund_processing: Process refund requests

Follow these rules:
1. Use the appropriate tool for the requested operation.
2. Return the tool result clearly.
3. If the operation fails, explain why and suggest alternatives.
4. Never make up data; only return what the tools provide.
"""


def execute_action(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Execute a support operation using the available tools.

    Invokes a ReAct agent that picks the right tool based on the
    current ticket context and classification.
    """
    llm = config["configurable"]["llm"]
    tools = config["configurable"]["tools"]

    ticket_content = state.get("user_input", "")
    customer_id = state.get("customer_id", "unknown")

    messages = [
        HumanMessage(content=(
            f"Customer ID: {customer_id}\n"
            f"Request: {ticket_content}\n\n"
            "Execute the appropriate support operation using the available tools."
        ))
    ]

    llm_with_tools = llm.bind_tools(tools)
    agent = create_react_agent(
        model=llm_with_tools,
        tools=tools,
    )

    result = agent.invoke({"messages": messages})
    result_messages = result.get("messages", [])
    tools_used = [m.name for m in result_messages if isinstance(m, ToolMessage)]

    return {
        "messages": result_messages,
        "tools_used": tools_used,
        "actions_taken": ["execute_action"],
        "next_step": "supervisor",
    }
