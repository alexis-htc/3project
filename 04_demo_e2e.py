"""
04_demo_e2e.py
--------------
Demonstrates end-to-end ticket processing with structured logging.

Processes 5 sample tickets through the UDA-Hub workflow and writes
structured JSONL logs to ``logs/demo_output.jsonl``.

Usage:
    # With a live LLM (requires OPENAI_API_KEY in .env):
    python 04_demo_e2e.py

    # Offline simulated demo (no API key needed):
    python 04_demo_e2e.py --simulate

The resulting log file is machine-parseable (one JSON object per line)
and can be queried with ``jq``, ``grep``, or loaded into any log tool:

    # Show all routing decisions
    jq 'select(.event == "route")' logs/demo_output.jsonl

    # Count escalations vs resolutions
    jq 'select(.event == "ticket_outcome")' logs/demo_output.jsonl

    # Find all tool calls
    jq 'select(.event == "tool_call")' logs/demo_output.jsonl
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DEMO_LOG = os.path.join(LOG_DIR, "demo_output.jsonl")

# ---------------------------------------------------------------------------
# Sample tickets covering diverse categories and both outcomes
# ---------------------------------------------------------------------------
SAMPLE_TICKETS = [
    {
        "customer_id": "1",
        "text": (
            "I was charged $19.99 but I cancelled my subscription last week. "
            "Can I get a refund?"
        ),
        "label": "Billing refund request (expect: resolved with refund tool)",
    },
    {
        "customer_id": "3",
        "text": (
            "The CultPass app keeps crashing on my iPhone when I try to book "
            "an event. I've tried reinstalling but the problem persists."
        ),
        "label": "Technical crash report (expect: resolved via KB)",
    },
    {
        "customer_id": "2",
        "text": "I forgot my password and can't log into my CultPass account.",
        "label": "Account/password reset (expect: resolved via KB)",
    },
    {
        "customer_id": "5",
        "text": (
            "I've been trying for a week to resolve a payment issue. I've "
            "called three times and nobody can help. This is unacceptable. "
            "I want to speak to a manager immediately."
        ),
        "label": "Escalation scenario (expect: escalated due to urgency)",
    },
    {
        "customer_id": "1",
        "text": (
            "Can you look up my account? My customer ID is 1. I want to "
            "know my current subscription plan and recent bookings."
        ),
        "label": "Account lookup with tool usage (expect: resolved with tools)",
    },
]


# ---------------------------------------------------------------------------
# Simulated demo (produces realistic logs without an LLM)
# ---------------------------------------------------------------------------

def _simulate_demo() -> None:
    """Write a realistic simulated JSONL log for offline review."""
    from utils import setup_logging, JSONFormatter

    os.makedirs(LOG_DIR, exist_ok=True)
    # Clear previous demo log
    if os.path.exists(DEMO_LOG):
        os.remove(DEMO_LOG)

    fh = logging.FileHandler(DEMO_LOG, mode="w", encoding="utf-8")
    fh.setFormatter(JSONFormatter())
    sim_logger = logging.getLogger("uda_hub.sim")
    sim_logger.setLevel(logging.DEBUG)
    sim_logger.addHandler(fh)

    scenarios = [
        # Ticket 1: billing refund -- resolved
        [
            {"event": "ticket_received", "ticket_id": "t-sim-001", "customer_id": "1", "session_id": "sess-demo-001"},
            {"event": "route", "node": "supervisor", "next_step": "classifier", "reason": "ticket_not_yet_classified", "ticket_id": "t-sim-001"},
            {"event": "classify", "node": "classifier", "ticket_id": "t-sim-001", "category": "billing", "urgency": "high", "complexity": "moderate", "requires_tool": True, "confidence": 0.92, "reasoning": "Customer requesting refund for cancelled subscription charge"},
            {"event": "route", "node": "supervisor", "next_step": "resolver", "reason": "classified_ready_to_resolve", "ticket_id": "t-sim-001"},
            {"event": "tool_call", "operation": "knowledge_search", "query": "refund policy cancelled subscription", "outcome": "found", "matches": 2},
            {"event": "tool_call", "operation": "account_lookup", "query": "1", "outcome": "found", "customer_id": 1},
            {"event": "tool_call", "operation": "refund_processing", "outcome": "success", "customer_id": "1", "result": {"refund_id": "REF-00012", "amount": 19.99}},
            {"event": "resolve", "node": "resolver", "ticket_id": "t-sim-001", "confidence": 0.88, "needs_escalation": False, "tools_invoked": ["knowledge_search", "account_lookup", "refund_processing"], "next_step": "end"},
            {"event": "route", "node": "supervisor", "next_step": "end", "reason": "resolved_with_confidence_0.88", "ticket_id": "t-sim-001"},
            {"event": "update_memory", "node": "update_memory", "ticket_id": "t-sim-001", "customer_id": "1"},
            {"event": "ticket_outcome", "ticket_id": "t-sim-001", "outcome": "resolved", "confidence": 0.88, "category": "billing"},
        ],
        # Ticket 2: technical crash -- resolved via KB
        [
            {"event": "ticket_received", "ticket_id": "t-sim-002", "customer_id": "3", "session_id": "sess-demo-001"},
            {"event": "route", "node": "supervisor", "next_step": "classifier", "reason": "ticket_not_yet_classified", "ticket_id": "t-sim-002"},
            {"event": "classify", "node": "classifier", "ticket_id": "t-sim-002", "category": "technical", "urgency": "medium", "complexity": "moderate", "requires_tool": False, "confidence": 0.90, "reasoning": "App crash on iOS during event booking"},
            {"event": "route", "node": "supervisor", "next_step": "resolver", "reason": "classified_ready_to_resolve", "ticket_id": "t-sim-002"},
            {"event": "tool_call", "operation": "knowledge_search", "query": "app crash iPhone booking event", "outcome": "found", "matches": 3},
            {"event": "resolve", "node": "resolver", "ticket_id": "t-sim-002", "confidence": 0.82, "needs_escalation": False, "tools_invoked": ["knowledge_search"], "next_step": "end"},
            {"event": "route", "node": "supervisor", "next_step": "end", "reason": "resolved_with_confidence_0.82", "ticket_id": "t-sim-002"},
            {"event": "update_memory", "node": "update_memory", "ticket_id": "t-sim-002", "customer_id": "3"},
            {"event": "ticket_outcome", "ticket_id": "t-sim-002", "outcome": "resolved", "confidence": 0.82, "category": "technical"},
        ],
        # Ticket 3: account/password reset -- resolved
        [
            {"event": "ticket_received", "ticket_id": "t-sim-003", "customer_id": "2", "session_id": "sess-demo-001"},
            {"event": "route", "node": "supervisor", "next_step": "classifier", "reason": "ticket_not_yet_classified", "ticket_id": "t-sim-003"},
            {"event": "classify", "node": "classifier", "ticket_id": "t-sim-003", "category": "account", "urgency": "medium", "complexity": "simple", "requires_tool": False, "confidence": 0.95, "reasoning": "Password reset request"},
            {"event": "route", "node": "supervisor", "next_step": "resolver", "reason": "classified_ready_to_resolve", "ticket_id": "t-sim-003"},
            {"event": "tool_call", "operation": "knowledge_search", "query": "password reset login", "outcome": "found", "matches": 2},
            {"event": "resolve", "node": "resolver", "ticket_id": "t-sim-003", "confidence": 0.93, "needs_escalation": False, "tools_invoked": ["knowledge_search"], "next_step": "end"},
            {"event": "route", "node": "supervisor", "next_step": "end", "reason": "resolved_with_confidence_0.93", "ticket_id": "t-sim-003"},
            {"event": "update_memory", "node": "update_memory", "ticket_id": "t-sim-003", "customer_id": "2"},
            {"event": "ticket_outcome", "ticket_id": "t-sim-003", "outcome": "resolved", "confidence": 0.93, "category": "account"},
        ],
        # Ticket 4: escalation -- critical urgency
        [
            {"event": "ticket_received", "ticket_id": "t-sim-004", "customer_id": "5", "session_id": "sess-demo-001"},
            {"event": "route", "node": "supervisor", "next_step": "classifier", "reason": "ticket_not_yet_classified", "ticket_id": "t-sim-004"},
            {"event": "classify", "node": "classifier", "ticket_id": "t-sim-004", "category": "billing", "urgency": "critical", "complexity": "complex", "requires_tool": True, "confidence": 0.91, "reasoning": "Repeated unresolved payment issue, customer extremely frustrated, requesting manager"},
            {"event": "route", "node": "supervisor", "next_step": "escalation", "reason": "critical_urgency", "ticket_id": "t-sim-004"},
            {"event": "escalate", "node": "escalation", "ticket_id": "t-sim-004", "assigned_to": "senior_support", "reason": "Customer has contacted support 3 times over a week with no resolution on a payment issue. Urgency is critical and customer is requesting manager intervention."},
            {"event": "update_memory", "node": "update_memory", "ticket_id": "t-sim-004", "customer_id": "5"},
            {"event": "ticket_outcome", "ticket_id": "t-sim-004", "outcome": "escalated", "confidence": None, "category": "billing"},
        ],
        # Ticket 5: account lookup with tools -- resolved
        [
            {"event": "ticket_received", "ticket_id": "t-sim-005", "customer_id": "1", "session_id": "sess-demo-001"},
            {"event": "route", "node": "supervisor", "next_step": "classifier", "reason": "ticket_not_yet_classified", "ticket_id": "t-sim-005"},
            {"event": "classify", "node": "classifier", "ticket_id": "t-sim-005", "category": "account", "urgency": "low", "complexity": "simple", "requires_tool": True, "confidence": 0.94, "reasoning": "Customer requesting account and subscription information"},
            {"event": "route", "node": "supervisor", "next_step": "resolver", "reason": "classified_ready_to_resolve", "ticket_id": "t-sim-005"},
            {"event": "tool_call", "operation": "account_lookup", "query": "1", "outcome": "found", "customer_id": 1},
            {"event": "tool_call", "operation": "subscription_management", "action": "check", "outcome": "success", "customer_id": "1"},
            {"event": "resolve", "node": "resolver", "ticket_id": "t-sim-005", "confidence": 0.91, "needs_escalation": False, "tools_invoked": ["account_lookup", "subscription_management"], "next_step": "end"},
            {"event": "route", "node": "supervisor", "next_step": "end", "reason": "resolved_with_confidence_0.91", "ticket_id": "t-sim-005"},
            {"event": "update_memory", "node": "update_memory", "ticket_id": "t-sim-005", "customer_id": "1"},
            {"event": "ticket_outcome", "ticket_id": "t-sim-005", "outcome": "resolved", "confidence": 0.91, "category": "account"},
        ],
    ]

    for ticket, events in zip(SAMPLE_TICKETS, scenarios):
        print(f"\n{'='*60}")
        print(f"  TICKET: {ticket['label']}")
        print(f"{'='*60}")
        for evt in events:
            level = logging.ERROR if evt.get("outcome") == "error" else logging.INFO
            sim_logger.log(level, evt.get("event", ""), extra=evt)
            print(f"  [{evt['event']:20s}] {json.dumps({k: v for k, v in evt.items() if k != 'event'}, default=str)}")

    sim_logger.removeHandler(fh)
    fh.close()
    print(f"\nDemo log written to: {DEMO_LOG}")
    print(f"Total log entries: {sum(len(s) for s in scenarios)}")


# ---------------------------------------------------------------------------
# Live demo (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def _live_demo() -> None:
    """Process real tickets through the full LangGraph workflow."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found. Run with --simulate for offline demo.")
        sys.exit(1)

    from utils import UDAHub, setup_logging

    os.makedirs(LOG_DIR, exist_ok=True)
    # Clear previous demo log and point logger at demo file
    if os.path.exists(DEMO_LOG):
        os.remove(DEMO_LOG)
    setup_logging(log_file=DEMO_LOG)

    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    hub = UDAHub(api_key=api_key, model_name=model_name)

    for ticket in SAMPLE_TICKETS:
        print(f"\n{'='*60}")
        print(f"  TICKET: {ticket['label']}")
        print(f"  Customer ID: {ticket['customer_id']}")
        print(f"{'='*60}")

        hub.start_session(customer_id=ticket["customer_id"])
        result = hub.process_ticket(ticket["text"])

        if result["success"]:
            print(f"  Outcome   : {'ESCALATED' if result.get('escalated') else 'RESOLVED'}")
            print(f"  Category  : {result['classification']['category'] if result.get('classification') else 'N/A'}")
            print(f"  Tools used: {result.get('tools_used', [])}")
            print(f"  Flow      : {' -> '.join(result.get('actions_taken', []))}")
        else:
            print(f"  ERROR: {result.get('error')}")

    print(f"\nDemo log written to: {DEMO_LOG}")
    with open(DEMO_LOG) as f:
        lines = f.readlines()
    print(f"Total log entries: {len(lines)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    simulate = "--simulate" in sys.argv
    if simulate:
        _simulate_demo()
    else:
        _live_demo()
