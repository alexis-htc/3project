# UDA-Hub: Universal Decision Agent

An intelligent multi-agent customer support system built with LangGraph for CultPass, a culture and entertainment subscription platform.

## Architecture

UDA-Hub uses a **Supervisor pattern** with 5 specialized agents:

```
Supervisor → Classifier → Resolver → (tools / escalation) → Memory → END
```

| Agent | Role |
|-------|------|
| **Supervisor** | Orchestrates routing between agents based on ticket state |
| **Classifier** | Categorizes tickets (billing, technical, account, subscription, general) with urgency and complexity |
| **Resolver** | Resolves tickets using RAG knowledge retrieval and support tools |
| **Escalation** | Summarizes and hands off tickets that cannot be auto-resolved |
| **Action** | Executes support operations (account lookup, refund, subscription change) |

### Tools

| Tool | Description |
|------|-------------|
| `knowledge_search` | RAG-based semantic search over 16 CultPass support articles |
| `account_lookup` | Queries customer account info from the CultPass database |
| `subscription_management` | Checks or modifies subscriptions (upgrade, downgrade, cancel) |
| `refund_processing` | Processes refund requests with eligibility validation |

### Memory

- **Short-term (session)**: LangGraph `InMemorySaver` checkpointer with `thread_id`. Maintains full message history and agent state within a session. Inspectable via `workflow.get_state(config)`.
- **Long-term (persistent)**: SQLite database storing resolved tickets, customer preferences, and conversation history across sessions. Used to personalize responses for returning customers.

### RAG Knowledge Retrieval

The knowledge base contains 16 articles covering billing, technical issues, account management, subscriptions, and general FAQs. When a ChromaDB vector store is available (with OpenAI embeddings), the system uses semantic similarity search. Otherwise, it falls back to keyword-based retrieval with TF scoring.

## Setup

### Requirements

```
Python 3.10+
```

### Installation

```bash
cd solution/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=your_key_here
#   MODEL_NAME=gpt-4o  (optional)
#   OPENAI_BASE_URL=... (optional, for custom endpoints)
```

### Database Setup

```bash
python 01_external_db_setup.py   # CultPass customer data
python 02_core_db_setup.py       # UDA-Hub core + knowledge base
```

### Run

```bash
python 03_agentic_app.py
```

## Project Structure

```
solution/
├── agentic/
│   ├── agents/
│   │   ├── supervisor.py        # Central orchestrator
│   │   ├── classifier.py        # Ticket classification
│   │   ├── resolver.py          # Knowledge-based resolution
│   │   ├── escalation.py        # Human handoff
│   │   └── action_agent.py      # Tool execution
│   ├── design/
│   │   └── architecture.md      # Architecture documentation
│   ├── tools/
│   │   ├── account_lookup.py    # Customer account queries
│   │   ├── subscription_management.py
│   │   ├── refund_processing.py
│   │   └── knowledge_search.py  # RAG retrieval
│   ├── memory.py                # Long-term memory (SQLite)
│   ├── prompts.py               # All prompt templates
│   ├── schemas.py               # Pydantic models + AgentState
│   └── workflow.py              # LangGraph graph definition
├── data/
│   ├── core/
│   │   └── cultpass_articles.jsonl  # 16 KB articles
│   └── external/
├── tests/
│   └── test_uda_hub.py          # Comprehensive test suite
├── 01_external_db_setup.py
├── 02_core_db_setup.py
├── 03_agentic_app.py            # Entry point
├── utils.py                     # UDAHub class + chat_interface()
├── requirements.txt
└── .env.example
```

## How State and Memory Works

### Structured Outputs

All LLM responses are enforced via Pydantic schemas using `llm.with_structured_output(Schema)`:

- `TicketClassification`: category, urgency, complexity, confidence
- `ResolverResponse`: answer, sources, confidence, needs_escalation
- `EscalationResponse`: summary, reason, suggested_actions
- `SupervisorDecision`: next_agent, reasoning
- `MemorySummary`: issue_summary, resolution, customer_preferences

### State Flow

The `WorkflowState` TypedDict flows through all nodes. Key fields use LangGraph reducers:
- `messages`: `Annotated[List[BaseMessage], add_messages]` — accumulated across nodes
- `actions_taken`: `Annotated[List[str], operator.add]` — traces the agent execution path

### Short-Term Memory

The workflow is compiled with `InMemorySaver()`. Each `process_ticket()` call passes a `thread_id` in the config, enabling LangGraph to persist and restore state across multiple invocations within the same session.

### Long-Term Memory

The `LongTermMemory` class (backed by SQLite) stores:
- **Resolved tickets**: issue summary + resolution for future reference
- **Customer preferences**: inferred from interactions (e.g., preferred channel)
- **Conversation history**: full message logs indexed by session and customer

At the start of each ticket, the system retrieves the customer's past interactions to provide context-aware responses.

## Testing

```bash
# Run all tests (structural + tool + schema + memory tests)
pytest tests/ -v

# Run end-to-end tests (requires OPENAI_API_KEY)
OPENAI_API_KEY=your_key pytest tests/ -v
```

## Example Conversations

### Billing Inquiry
```
You: I was charged $19.99 but I cancelled my subscription last week. Can I get a refund?
Agent: I understand your concern about the charge after cancellation. Based on our refund policy (KB-005),
       since you cancelled recently, you are eligible for a refund...
  [Category: billing | Urgency: medium | Complexity: moderate]
  [Tools used: knowledge_search, refund_processing]
  [Agent flow: supervisor_route → classify_ticket → supervisor_route → resolve_ticket → update_memory]
```

### Technical Issue
```
You: The CultPass app keeps crashing when I try to book an event on my iPhone.
Agent: I'm sorry to hear about the app crashes. Here are some troubleshooting steps (KB-004)...
  [Category: technical | Urgency: high | Complexity: moderate]
  [Tools used: knowledge_search]
```

### Escalation
```
You: I've called three times about a payment issue and nobody can help. I want to speak to a manager.
Agent: [ESCALATED] I sincerely apologize for the frustrating experience...
  [Category: billing | Urgency: critical | Complexity: complex]
  [Status: ESCALATED to human agent]
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | >=0.2.0 | LLM framework |
| langgraph | >=0.6.7 | Multi-agent graph orchestration |
| langchain-openai | >=0.1.0 | OpenAI LLM integration |
| langchain-community | >=0.2.0 | ChromaDB vector store |
| chromadb | >=0.4.0 | Vector database for RAG |
| pydantic | >=2.0.0 | Schema validation |
| python-dotenv | >=1.0.0 | Environment configuration |
| pytest | >=7.4.0 | Testing |
