"""
Utility functions for UDA-Hub.
Includes the interactive chat interface, helper functions,
and structured JSON logging configuration.
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Ensure solution/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line (JSONL) for structured, searchable logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        # Merge any extra structured fields attached to the record
        for key in ("event", "ticket_id", "customer_id", "category",
                     "urgency", "complexity", "confidence", "next_step",
                     "reason", "tools_used", "tools_invoked", "action",
                     "result", "assigned_to", "needs_escalation",
                     "query", "matches", "operation", "outcome",
                     "session_id", "node", "requires_tool", "reasoning"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure the ``uda_hub`` logger hierarchy to emit JSONL.

    * Writes structured JSON lines to *log_file* (default ``logs/uda_hub.jsonl``).
    * Also attaches a concise console handler at INFO level for human readability.
    * Returns the root ``uda_hub`` logger.
    """
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "uda_hub.jsonl")

    root_logger = logging.getLogger("uda_hub")
    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(level)

    # --- File handler (JSONL) ---
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JSONFormatter())
    root_logger.addHandler(fh)

    # --- Console handler (human-readable) ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root_logger.addHandler(ch)

    return root_logger

from agentic.workflow import create_workflow
from agentic.memory import LongTermMemory
from agentic.tools import (
    create_account_lookup_tool,
    create_subscription_management_tool,
    create_refund_processing_tool,
    create_knowledge_search_tool,
)

# ---------------------------------------------------------------------------
# Paths (relative to solution/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DB = os.path.join(BASE_DIR, "data", "external", "cultpass.db")
CORE_DB = os.path.join(BASE_DIR, "data", "core", "uda_hub.db")
MEMORY_DB = os.path.join(BASE_DIR, "data", "core", "memory.db")


def _init_vector_store(core_db: str):
    """
    Attempt to initialise a ChromaDB vector store with embeddings from
    the knowledge articles. Returns (embeddings, vector_store) or (None, None)
    if ChromaDB is not available.
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        import sqlite3

        embeddings = OpenAIEmbeddings()
        conn = sqlite3.connect(core_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content, category, tags FROM knowledge")
        articles = cursor.fetchall()
        conn.close()

        if not articles:
            return None, None

        from langchain_core.documents import Document

        docs = []
        for a in articles:
            docs.append(Document(
                page_content=a["content"],
                metadata={
                    "id": a["id"],
                    "title": a["title"],
                    "category": a["category"],
                    "tags": a["tags"],
                },
            ))

        persist_dir = os.path.join(BASE_DIR, "data", "core", "chroma_db")
        vector_store = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_dir,
            collection_name="cultpass_kb",
        )
        return embeddings, vector_store
    except Exception as e:
        print(f"[INFO] Vector store not initialised ({e}). Using keyword search fallback.")
        return None, None


def build_tools(external_db: str = EXTERNAL_DB, core_db: str = CORE_DB):
    """Build all support tools."""
    embeddings, vector_store = _init_vector_store(core_db)
    return [
        create_account_lookup_tool(external_db),
        create_subscription_management_tool(external_db),
        create_refund_processing_tool(external_db),
        create_knowledge_search_tool(core_db, embeddings=embeddings, vector_store=vector_store),
    ]


def create_llm(api_key: str, model_name: str = "gpt-4o", temperature: float = 0.1):
    """Create the ChatOpenAI LLM instance."""
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "model": model_name,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


class UDAHub:
    """
    High-level wrapper that ties together the LangGraph workflow,
    tools, LLM, and memory for interactive use.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        self.logger = setup_logging()
        self.llm = create_llm(api_key, model_name, temperature)
        self.tools = build_tools()
        self.workflow = create_workflow()
        self.long_term_memory = LongTermMemory(MEMORY_DB)
        self.session_id: Optional[str] = None
        self.customer_id: Optional[str] = None
        self.logger.info("UDAHub initialised", extra={"event": "hub_init", "model": model_name})

    def start_session(self, customer_id: str = "unknown") -> str:
        self.session_id = str(uuid.uuid4())
        self.customer_id = customer_id
        self.logger.info(
            "Session started",
            extra={"event": "session_start", "session_id": self.session_id, "customer_id": customer_id},
        )
        return self.session_id

    def process_ticket(self, ticket_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.session_id:
            self.start_session()

        ticket_id = str(uuid.uuid4())[:8]
        self.logger.info(
            "Ticket received",
            extra={
                "event": "ticket_received",
                "ticket_id": ticket_id,
                "customer_id": self.customer_id,
                "session_id": self.session_id,
            },
        )

        customer_history = self.long_term_memory.get_customer_history(self.customer_id or "unknown")

        config = {
            "configurable": {
                "thread_id": self.session_id,
                "llm": self.llm,
                "tools": self.tools,
                "long_term_memory": self.long_term_memory,
            }
        }

        initial_state = {
            "messages": [],
            "user_input": ticket_text,
            "ticket_id": ticket_id,
            "customer_id": self.customer_id,
            "ticket_metadata": metadata or {},
            "classification": None,
            "next_step": "supervisor",
            "resolver_response": None,
            "escalation_response": None,
            "conversation_summary": "",
            "customer_history": customer_history,
            "session_id": self.session_id,
            "tools_used": [],
            "actions_taken": [],
        }

        try:
            final_state = self.workflow.invoke(initial_state, config=config)

            self.logger.info(
                "Ticket processing complete",
                extra={
                    "event": "ticket_complete",
                    "ticket_id": ticket_id,
                    "actions_taken": final_state.get("actions_taken", []),
                    "tools_used": final_state.get("tools_used", []),
                },
            )

            # Store conversation in long-term memory
            self.long_term_memory.store_conversation_message(
                session_id=self.session_id,
                customer_id=self.customer_id or "unknown",
                role="customer",
                content=ticket_text,
            )

            # Extract response
            response_text = ""
            resolver_resp = final_state.get("resolver_response")
            escalation_resp = final_state.get("escalation_response")

            if resolver_resp:
                response_text = resolver_resp.answer
            elif escalation_resp:
                response_text = (
                    f"[ESCALATED] {escalation_resp.summary}\n"
                    f"Reason: {escalation_resp.reason}\n"
                    f"Assigned to: {escalation_resp.assigned_to}"
                )
            elif final_state.get("messages"):
                response_text = final_state["messages"][-1].content

            self.long_term_memory.store_conversation_message(
                session_id=self.session_id,
                customer_id=self.customer_id or "unknown",
                role="agent",
                content=response_text,
            )

            classification = final_state.get("classification")

            outcome = "escalated" if escalation_resp is not None else "resolved"
            self.logger.info(
                f"Ticket outcome: {outcome}",
                extra={
                    "event": "ticket_outcome",
                    "ticket_id": ticket_id,
                    "outcome": outcome,
                    "confidence": resolver_resp.confidence if resolver_resp else None,
                    "category": classification.category if classification else None,
                },
            )

            return {
                "success": True,
                "response": response_text,
                "classification": classification.model_dump() if classification else None,
                "tools_used": final_state.get("tools_used", []),
                "actions_taken": final_state.get("actions_taken", []),
                "summary": final_state.get("conversation_summary", ""),
                "escalated": escalation_resp is not None,
            }

        except Exception as e:
            self.logger.error(
                f"Ticket processing failed: {e}",
                extra={"event": "ticket_error", "ticket_id": ticket_id, "reason": str(e)},
            )
            return {
                "success": False,
                "error": str(e),
                "response": None,
            }


def chat_interface() -> None:
    """
    Simple interactive chat loop for UDA-Hub.
    Accepts customer support tickets and processes them through the agent workflow.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Create a .env file with your API key.")
        return

    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    temperature = float(os.getenv("TEMPERATURE", "0.1"))

    print("\n" + "=" * 60)
    print("  UDA-Hub - Universal Decision Agent")
    print("  CultPass Customer Support System")
    print("=" * 60 + "\n")

    hub = UDAHub(api_key=api_key, model_name=model_name, temperature=temperature)

    customer_id = input("Enter customer ID (or press Enter for 'guest'): ").strip() or "guest"
    session_id = hub.start_session(customer_id=customer_id)
    print(f"Session started: {session_id}")
    print(f"Customer: {customer_id}\n")

    # Show previous interactions if any
    history = hub.long_term_memory.get_customer_history(customer_id, limit=3)
    if history:
        print("--- Previous Interactions ---")
        for h in history:
            print(f"  [{h['created_at'][:10]}] {h['issue_summary'][:80]}")
        print("---\n")

    print("Commands: /quit, /history, /help")
    print("Enter your support request below.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("\nThank you for using CultPass support. Goodbye!")
                break
            elif user_input.lower() == "/history":
                history = hub.long_term_memory.get_customer_history(customer_id)
                if history:
                    print("\n--- Ticket History ---")
                    for h in history:
                        print(f"  [{h['created_at'][:10]}] {h['issue_summary'][:80]}")
                        print(f"    Resolution: {h['resolution'][:80]}")
                    print("---\n")
                else:
                    print("No previous history found.\n")
                continue
            elif user_input.lower() == "/help":
                print("\nUDA-Hub Support Assistant")
                print("  Type your support question or issue.")
                print("  /history  - View past interactions")
                print("  /quit     - Exit\n")
                continue

            print("\nProcessing your request...\n")
            result = hub.process_ticket(user_input)

            if result["success"]:
                print(f"Agent: {result['response']}\n")
                if result.get("classification"):
                    c = result["classification"]
                    print(f"  [Category: {c['category']} | Urgency: {c['urgency']} | Complexity: {c['complexity']}]")
                if result.get("tools_used"):
                    print(f"  [Tools used: {', '.join(result['tools_used'])}]")
                if result.get("escalated"):
                    print("  [Status: ESCALATED to human agent]")
                if result.get("actions_taken"):
                    print(f"  [Agent flow: {' → '.join(result['actions_taken'])}]")
                print()
            else:
                print(f"Error: {result.get('error', 'Unknown error')}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")
