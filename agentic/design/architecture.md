# UDA-Hub Architecture Design

## Overview

UDA-Hub (Universal Decision Agent Hub) is an intelligent multi-agent system that resolves customer support tickets for CultPass, a culture and entertainment subscription platform. It reads, reasons, routes, and resolves tickets using a **Supervisor pattern** built on LangGraph.

## Architecture Pattern: Supervisor

The system uses a **Supervisor** orchestration pattern where a central supervisor agent coordinates specialized worker agents. The supervisor inspects each incoming ticket, delegates to the appropriate agent, and synthesizes the final response.

```
                        ┌─────────────────────┐
                        │   Incoming Ticket    │
                        │  (text + metadata)   │
                        └─────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │   SUPERVISOR AGENT     │
                     │  (Orchestrator/Router) │
                     └────────────┬───────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
           ┌──────────────┐ ┌──────────┐ ┌──────────────┐
           │  CLASSIFIER  │ │ RESOLVER │ │  ESCALATION  │
           │    AGENT     │ │  AGENT   │ │    AGENT     │
           └──────┬───────┘ └────┬─────┘ └──────┬───────┘
                  │              │               │
                  │         ┌────┴────┐          │
                  │         ▼         ▼          │
                  │    ┌─────────┐ ┌──────┐      │
                  │    │   RAG   │ │ACTION│      │
                  │    │ Search  │ │AGENT │      │
                  │    └─────────┘ └──┬───┘      │
                  │                   │          │
                  │         ┌─────────┼──────┐   │
                  │         ▼         ▼      ▼   │
                  │    ┌────────┐ ┌───────┐ ┌────┴───┐
                  │    │Account │ │Refund │ │Subscr. │
                  │    │Lookup  │ │Process│ │Manage  │
                  │    └────────┘ └───────┘ └────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  MEMORY MANAGER  │
        │  (Short + Long)  │
        └──────────────────┘
```

## Agent Descriptions

### 1. Supervisor Agent
- **Role**: Central orchestrator; the entry point for every ticket.
- **Responsibilities**:
  - Receive incoming ticket with metadata (platform, urgency, history).
  - Delegate to Classifier for ticket categorization.
  - Route to Resolver or Escalation based on classification.
  - Synthesize final response and update memory.
- **Decision logic**: Uses LLM-based reasoning to decide routing.

### 2. Classifier Agent
- **Role**: Categorizes tickets by type, urgency, and complexity.
- **Responsibilities**:
  - Analyze ticket text and metadata.
  - Classify into categories: `billing`, `technical`, `account`, `subscription`, `general`.
  - Assign urgency: `low`, `medium`, `high`, `critical`.
  - Estimate complexity: `simple`, `moderate`, `complex`.
- **Output**: Structured classification with confidence scores.

### 3. Resolver Agent
- **Role**: Attempts to resolve tickets using the knowledge base and tools.
- **Responsibilities**:
  - Perform RAG retrieval against the knowledge base.
  - Generate responses grounded in retrieved articles.
  - Invoke support tools (account lookup, subscription management, refund) via the Action Agent when operational actions are needed.
  - Calculate a confidence score; if below threshold, delegate to Escalation.
- **RAG Pipeline**: Uses ChromaDB vector store with OpenAI embeddings for semantic search over CultPass support articles.

### 4. Escalation Agent
- **Role**: Handles tickets that cannot be resolved automatically.
- **Responsibilities**:
  - Summarize the ticket and all resolution attempts.
  - Assign to a human agent with context.
  - Record escalation reason and suggested actions.
  - Update long-term memory with escalation patterns.

### 5. Action Agent
- **Role**: Executes support operations via tools.
- **Responsibilities**:
  - Invoke account lookup, subscription management, and refund tools.
  - Return structured results to the Resolver.
  - Log all tool invocations.

## Tools

| Tool | Description | Database |
|------|-------------|----------|
| `account_lookup` | Query customer account info (name, email, plan, status) | CultPass external DB |
| `subscription_management` | Check/modify subscription plans, cancel, upgrade | CultPass external DB |
| `refund_processing` | Process refund requests with validation | CultPass external DB |
| `knowledge_search` | RAG-based semantic search over support articles | Core DB (ChromaDB) |

## Memory System

### Short-Term Memory (Session)
- **Implementation**: LangGraph `InMemorySaver` checkpointer with `thread_id`.
- **Scope**: Single conversation session.
- **Contents**: Message history, active ticket context, tools used, agent decisions.
- **Inspection**: Thread state can be inspected via `workflow.get_state(config)`.

### Long-Term Memory (Persistent)
- **Implementation**: SQLite database (`memory.db`) with semantic search.
- **Scope**: Across sessions for the same customer.
- **Contents**:
  - Resolved issues and their solutions.
  - Customer preferences (communication style, known issues).
  - Escalation history.
- **Retrieval**: Query by customer ID or semantic similarity.

## Data Flow

1. **Ticket arrives** with text content and metadata (platform, customer_id, urgency).
2. **Supervisor** receives ticket, sends to **Classifier**.
3. **Classifier** returns category, urgency, complexity.
4. **Supervisor** checks long-term memory for customer history.
5. **Supervisor** routes to **Resolver** (if resolvable) or **Escalation** (if critical/complex).
6. **Resolver** performs RAG search, optionally invokes **Action Agent** for operations.
7. **Resolver** returns response with confidence score.
8. If confidence < threshold → **Escalation Agent** takes over.
9. **Memory** is updated: short-term (session state) and long-term (resolution record).
10. Final response returned to customer.

## Database Schema

### External DB (CultPass - `cultpass.db`)
- `customers`: id, name, email, phone, created_at
- `subscriptions`: id, customer_id, plan, status, start_date, end_date, price
- `payments`: id, customer_id, amount, status, payment_date, description
- `events`: id, name, date, venue, category, price

### Core DB (UDA-Hub - `uda_hub.db`)
- `accounts`: id, name, domain, created_at
- `users`: id, account_id, name, email, role
- `tickets`: id, account_id, customer_id, subject, content, status, priority, channel, created_at
- `ticket_metadata`: id, ticket_id, key, value
- `ticket_messages`: id, ticket_id, sender_type, sender_id, content, created_at
- `knowledge`: id, account_id, title, content, category, tags, created_at

### Memory DB (`memory.db`)
- `customer_memory`: id, customer_id, memory_type, key, value, created_at
- `resolved_tickets`: id, ticket_id, customer_id, issue_summary, resolution, created_at

## Technology Stack

- **Orchestration**: LangGraph (StateGraph, conditional edges, checkpointer)
- **LLM**: OpenAI GPT-4o via LangChain
- **RAG**: ChromaDB + OpenAI Embeddings
- **Database**: SQLite (external, core, memory)
- **Schema Validation**: Pydantic v2
- **Testing**: pytest
