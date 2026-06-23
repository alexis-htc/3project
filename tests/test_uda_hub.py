"""
Test suite for UDA-Hub - Universal Decision Agent.

Tests cover:
  1. Database setup and data integrity
  2. Knowledge base articles (count and diversity)
  3. Tool functionality (account lookup, subscription, refund, knowledge search)
  4. Agent schemas and validation
  5. Workflow graph structure
  6. Memory system (long-term persistence)
  7. End-to-end ticket processing (requires OPENAI_API_KEY)
"""

import os
import sys
import json
import sqlite3
import pytest

# Ensure the solution directory is on the path
SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOLUTION_DIR)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXTERNAL_DB = os.path.join(SOLUTION_DIR, "data", "external", "cultpass.db")
CORE_DB = os.path.join(SOLUTION_DIR, "data", "core", "uda_hub.db")
MEMORY_DB = os.path.join(SOLUTION_DIR, "data", "core", "memory_test.db")
ARTICLES_PATH = os.path.join(SOLUTION_DIR, "data", "core", "cultpass_articles.jsonl")


# ===========================================================================
# 1. Database Setup Tests
# ===========================================================================

class TestExternalDatabase:
    """Tests for the CultPass external database."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(EXTERNAL_DB):
            # Run setup
            exec(open(os.path.join(SOLUTION_DIR, "01_external_db_setup.py")).read())
        self.conn = sqlite3.connect(EXTERNAL_DB)
        self.conn.row_factory = sqlite3.Row
        yield
        self.conn.close()

    def test_customers_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        assert count >= 10, f"Expected at least 10 customers, got {count}"

    def test_subscriptions_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM subscriptions")
        count = cursor.fetchone()[0]
        assert count >= 10, f"Expected at least 10 subscriptions, got {count}"

    def test_payments_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM payments")
        count = cursor.fetchone()[0]
        assert count > 0, "Expected payments to be populated"

    def test_events_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        assert count >= 10, f"Expected at least 10 events, got {count}"

    def test_customer_has_required_fields(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM customers LIMIT 1")
        row = cursor.fetchone()
        assert row is not None
        keys = row.keys()
        for field in ["id", "name", "email", "phone", "plan", "status", "created_at"]:
            assert field in keys, f"Missing field '{field}' in customers table"

    def test_data_retrieval(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, email, plan FROM customers WHERE id = 1")
        row = cursor.fetchone()
        assert row is not None
        assert row["name"] == "Alice Johnson"
        assert row["plan"] == "premium"


class TestCoreDatabase:
    """Tests for the UDA-Hub core database."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(CORE_DB):
            exec(open(os.path.join(SOLUTION_DIR, "02_core_db_setup.py")).read())
        self.conn = sqlite3.connect(CORE_DB)
        self.conn.row_factory = sqlite3.Row
        yield
        self.conn.close()

    def test_required_tables_exist(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        required = {"accounts", "users", "tickets", "ticket_metadata", "ticket_messages", "knowledge"}
        missing = required - tables
        assert not missing, f"Missing tables: {missing}"

    def test_knowledge_base_count(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        count = cursor.fetchone()[0]
        assert count >= 14, f"Expected at least 14 knowledge articles, got {count}"

    def test_knowledge_categories_diversity(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM knowledge")
        categories = {row["category"] for row in cursor.fetchall()}
        expected = {"billing", "technical", "account", "subscription", "general"}
        overlap = categories & expected
        assert len(overlap) >= 4, f"Expected diverse categories, got: {categories}"

    def test_sample_tickets_exist(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tickets")
        count = cursor.fetchone()[0]
        assert count >= 5, f"Expected at least 5 sample tickets, got {count}"

    def test_cultpass_account_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM accounts WHERE name = 'CultPass'")
        account = cursor.fetchone()
        assert account is not None, "CultPass account should exist"


# ===========================================================================
# 2. Knowledge Base Articles Tests
# ===========================================================================

class TestKnowledgeArticles:
    """Tests for the cultpass_articles.jsonl file."""

    def test_articles_file_exists(self):
        assert os.path.exists(ARTICLES_PATH), f"Articles file not found: {ARTICLES_PATH}"

    def test_article_count(self):
        articles = []
        with open(ARTICLES_PATH, "r") as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))
        assert len(articles) >= 14, f"Expected at least 14 articles, got {len(articles)}"

    def test_article_fields(self):
        with open(ARTICLES_PATH, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                article = json.loads(line)
                for field in ["id", "title", "category", "tags", "content"]:
                    assert field in article, f"Article missing field '{field}': {article.get('id', 'unknown')}"

    def test_article_category_diversity(self):
        categories = set()
        with open(ARTICLES_PATH, "r") as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    categories.add(article["category"])
        assert len(categories) >= 4, f"Expected at least 4 categories, got {categories}"


# ===========================================================================
# 3. Tool Tests
# ===========================================================================

class TestAccountLookupTool:
    """Tests for the account lookup tool."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(EXTERNAL_DB):
            exec(open(os.path.join(SOLUTION_DIR, "01_external_db_setup.py")).read())

    def test_lookup_by_id(self):
        from agentic.tools.account_lookup import create_account_lookup_tool
        tool = create_account_lookup_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_identifier": "1"})
        data = json.loads(result)
        assert data["name"] == "Alice Johnson"
        assert data["plan"] == "premium"

    def test_lookup_by_email(self):
        from agentic.tools.account_lookup import create_account_lookup_tool
        tool = create_account_lookup_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_identifier": "bob.smith@email.com"})
        data = json.loads(result)
        assert data["name"] == "Bob Smith"

    def test_lookup_not_found(self):
        from agentic.tools.account_lookup import create_account_lookup_tool
        tool = create_account_lookup_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_identifier": "nonexistent@email.com"})
        assert "No customer found" in result


class TestSubscriptionManagementTool:
    """Tests for the subscription management tool."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(SOLUTION_DIR, "01_external_db_setup.py")],
            cwd=SOLUTION_DIR, check=True, capture_output=True,
        )

    def test_check_subscription(self):
        from agentic.tools.subscription_management import create_subscription_management_tool
        tool = create_subscription_management_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_id": "1", "action": "check"})
        data = json.loads(result)
        assert data["current_plan"] == "premium"
        assert data["status"] == "active"

    def test_upgrade_subscription(self):
        from agentic.tools.subscription_management import create_subscription_management_tool
        tool = create_subscription_management_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_id": "2", "action": "upgrade", "new_plan": "premium"})
        data = json.loads(result)
        assert data["success"] is True
        assert data["new_plan"] == "premium"

    def test_invalid_upgrade(self):
        from agentic.tools.subscription_management import create_subscription_management_tool
        tool = create_subscription_management_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_id": "1", "action": "upgrade", "new_plan": "basic"})
        assert "Cannot upgrade" in result


class TestRefundProcessingTool:
    """Tests for the refund processing tool."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(SOLUTION_DIR, "01_external_db_setup.py")],
            cwd=SOLUTION_DIR, check=True, capture_output=True,
        )

    def test_process_refund(self):
        from agentic.tools.refund_processing import create_refund_processing_tool
        tool = create_refund_processing_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_id": "1", "reason": "Accidental charge"})
        data = json.loads(result)
        assert data["success"] is True
        assert "refund_id" in data
        assert data["amount_refunded"] > 0

    def test_refund_nonexistent_customer(self):
        from agentic.tools.refund_processing import create_refund_processing_tool
        tool = create_refund_processing_tool(EXTERNAL_DB)
        result = tool.invoke({"customer_id": "999"})
        assert "No customer found" in result


class TestKnowledgeSearchTool:
    """Tests for the knowledge search tool."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not os.path.exists(CORE_DB):
            exec(open(os.path.join(SOLUTION_DIR, "02_core_db_setup.py")).read())

    def test_search_password_reset(self):
        from agentic.tools.knowledge_search import create_knowledge_search_tool
        tool = create_knowledge_search_tool(CORE_DB)
        result = tool.invoke({"query": "how to reset password"})
        assert "KB-001" in result or "password" in result.lower()

    def test_search_billing(self):
        from agentic.tools.knowledge_search import create_knowledge_search_tool
        tool = create_knowledge_search_tool(CORE_DB)
        result = tool.invoke({"query": "refund policy billing"})
        assert "refund" in result.lower()

    def test_search_no_results(self):
        from agentic.tools.knowledge_search import create_knowledge_search_tool
        tool = create_knowledge_search_tool(CORE_DB)
        result = tool.invoke({"query": "xyznonexistent1234"})
        assert "No relevant articles" in result or "escalation" in result.lower()


# ===========================================================================
# 4. Schema Validation Tests
# ===========================================================================

class TestSchemas:
    """Tests for Pydantic schema validation."""

    def test_ticket_classification_valid(self):
        from agentic.schemas import TicketClassification
        tc = TicketClassification(
            category="billing",
            urgency="high",
            complexity="moderate",
            requires_tool=True,
            reasoning="Customer is asking about a refund",
            confidence=0.9,
        )
        assert tc.category == "billing"
        assert tc.confidence == 0.9

    def test_ticket_classification_invalid_category(self):
        from agentic.schemas import TicketClassification
        with pytest.raises(Exception):
            TicketClassification(
                category="invalid_category",
                urgency="high",
                complexity="moderate",
                reasoning="test",
            )

    def test_ticket_classification_confidence_bounds(self):
        from agentic.schemas import TicketClassification
        with pytest.raises(Exception):
            TicketClassification(
                category="billing",
                urgency="high",
                complexity="moderate",
                reasoning="test",
                confidence=1.5,
            )

    def test_resolver_response(self):
        from agentic.schemas import ResolverResponse
        rr = ResolverResponse(
            answer="Here's how to reset your password...",
            sources=["KB-001"],
            confidence=0.85,
            needs_escalation=False,
        )
        assert rr.confidence == 0.85
        assert len(rr.sources) == 1

    def test_escalation_response(self):
        from agentic.schemas import EscalationResponse
        er = EscalationResponse(
            summary="Customer cannot login after multiple attempts",
            reason="Account may be locked, requires manual intervention",
            suggested_actions=["Check account lock status", "Verify identity"],
            assigned_to="senior_support",
        )
        assert "login" in er.summary.lower()

    def test_supervisor_decision(self):
        from agentic.schemas import SupervisorDecision
        sd = SupervisorDecision(next_agent="resolver", reasoning="Ticket classified, ready to resolve")
        assert sd.next_agent == "resolver"

    def test_supervisor_decision_invalid_agent(self):
        from agentic.schemas import SupervisorDecision
        with pytest.raises(Exception):
            SupervisorDecision(next_agent="nonexistent_agent", reasoning="test")


# ===========================================================================
# 5. Workflow Structure Tests
# ===========================================================================

class TestWorkflow:
    """Tests for the LangGraph workflow structure."""

    def test_workflow_compiles(self):
        from agentic.workflow import create_workflow
        workflow = create_workflow()
        assert workflow is not None

    def test_workflow_has_nodes(self):
        from agentic.workflow import create_workflow
        workflow = create_workflow()
        graph = workflow.get_graph()
        node_ids = set(graph.nodes.keys())
        expected = {"supervisor", "classifier", "resolver", "escalation", "action", "update_memory"}
        missing = expected - node_ids
        assert not missing, f"Missing nodes: {missing}"

    def test_workflow_entry_point(self):
        from agentic.workflow import create_workflow
        workflow = create_workflow()
        graph = workflow.get_graph()
        # The entry point should connect to supervisor
        first_node = graph.first_node()
        assert first_node is not None


# ===========================================================================
# 6. Memory System Tests
# ===========================================================================

class TestLongTermMemory:
    """Tests for the long-term memory system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from agentic.memory import LongTermMemory
        # Use a test-specific database
        if os.path.exists(MEMORY_DB):
            os.remove(MEMORY_DB)
        self.memory = LongTermMemory(MEMORY_DB)
        yield
        if os.path.exists(MEMORY_DB):
            os.remove(MEMORY_DB)

    def test_store_and_retrieve_preference(self):
        self.memory.store_customer_preference("cust-1", "communication", "email")
        prefs = self.memory.get_customer_preferences("cust-1")
        assert len(prefs) >= 1
        assert prefs[0]["key"] == "communication"
        assert prefs[0]["value"] == "email"

    def test_store_and_retrieve_resolved_ticket(self):
        self.memory.store_resolved_ticket(
            customer_id="cust-1",
            issue_summary="Password reset request",
            resolution="Guided through password reset flow",
            category="account",
            ticket_id="T-001",
        )
        history = self.memory.get_customer_history("cust-1")
        assert len(history) >= 1
        assert "password" in history[0]["issue_summary"].lower()

    def test_store_and_retrieve_conversation(self):
        self.memory.store_conversation_message("sess-1", "cust-1", "customer", "I need help with my account")
        self.memory.store_conversation_message("sess-1", "cust-1", "agent", "I'd be happy to help")
        history = self.memory.get_conversation_history("cust-1")
        assert len(history) >= 2

    def test_search_resolved_tickets(self):
        self.memory.store_resolved_ticket(
            customer_id="cust-2",
            issue_summary="Refund for duplicate charge",
            resolution="Refund of $19.99 processed",
            category="billing",
        )
        results = self.memory.search_resolved_tickets("refund charge")
        assert len(results) >= 1

    def test_empty_history(self):
        history = self.memory.get_customer_history("nonexistent")
        assert history == []


# ===========================================================================
# 7. End-to-End Tests (require OPENAI_API_KEY)
# ===========================================================================

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping end-to-end tests",
)
class TestEndToEnd:
    """End-to-end tests that require a live LLM connection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # Ensure databases exist
        if not os.path.exists(EXTERNAL_DB):
            exec(open(os.path.join(SOLUTION_DIR, "01_external_db_setup.py")).read())
        if not os.path.exists(CORE_DB):
            exec(open(os.path.join(SOLUTION_DIR, "02_core_db_setup.py")).read())

        from utils import UDAHub
        self.hub = UDAHub(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        )
        self.hub.start_session(customer_id="1")

    def test_billing_ticket_resolution(self):
        result = self.hub.process_ticket(
            "I was charged $19.99 but I cancelled my subscription last week. Can I get a refund?"
        )
        assert result["success"] is True
        assert result["response"] is not None
        assert result["classification"]["category"] in ("billing", "subscription")
        assert len(result["actions_taken"]) >= 3

    def test_technical_ticket_resolution(self):
        result = self.hub.process_ticket(
            "The CultPass app keeps crashing on my iPhone when I try to book an event."
        )
        assert result["success"] is True
        assert result["response"] is not None
        assert result["classification"]["category"] == "technical"

    def test_account_ticket_resolution(self):
        result = self.hub.process_ticket(
            "I forgot my password and can't log into my CultPass account."
        )
        assert result["success"] is True
        assert result["response"] is not None

    def test_escalation_scenario(self):
        result = self.hub.process_ticket(
            "I've been trying for a week to resolve a payment issue. "
            "I've called three times and nobody can help. This is unacceptable. "
            "I want to speak to a manager immediately."
        )
        assert result["success"] is True
        assert result["response"] is not None
        # This should likely be escalated given the urgency
        assert len(result["actions_taken"]) >= 2

    def test_tool_usage_in_resolution(self):
        result = self.hub.process_ticket(
            "Can you look up my account? My customer ID is 1. "
            "I want to know my current subscription plan."
        )
        assert result["success"] is True
        assert result["response"] is not None

    def test_conversation_memory_persists(self):
        # First interaction
        result1 = self.hub.process_ticket("What's the refund policy?")
        assert result1["success"] is True

        # Second interaction in same session should have context
        result2 = self.hub.process_ticket("And how long does the refund take to process?")
        assert result2["success"] is True

    def test_long_term_memory_storage(self):
        self.hub.process_ticket("I need help resetting my password.")
        # Check that long-term memory was updated
        history = self.hub.long_term_memory.get_customer_history("1")
        assert len(history) >= 1
