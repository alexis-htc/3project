"""
Memory system for UDA-Hub.

Short-term memory: LangGraph InMemorySaver (thread_id scoped).
Long-term memory: SQLite-based persistent storage for customer preferences,
resolved tickets, and conversation history across sessions.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class LongTermMemory:
    """Persistent memory backed by SQLite for cross-session recall."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS customer_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resolved_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT,
                    customer_id TEXT NOT NULL,
                    issue_summary TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    category TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    # ----- Customer Memory -----

    def store_customer_preference(
        self, customer_id: str, key: str, value: str
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO customer_memory (customer_id, memory_type, key, value, created_at) "
                "VALUES (?, 'preference', ?, ?, ?)",
                (customer_id, key, value, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_customer_preferences(self, customer_id: str) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key, value, created_at FROM customer_memory "
                "WHERE customer_id = ? AND memory_type = 'preference' ORDER BY created_at DESC",
                (customer_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_customer_history(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ticket_id, issue_summary, resolution, category, created_at "
                "FROM resolved_tickets WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
                (customer_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # ----- Resolved Tickets -----

    def store_resolved_ticket(
        self,
        customer_id: str,
        issue_summary: str,
        resolution: str,
        category: str = "",
        ticket_id: str = "",
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO resolved_tickets (ticket_id, customer_id, issue_summary, resolution, category, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ticket_id, customer_id, issue_summary, resolution, category, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    # ----- Conversation History -----

    def store_conversation_message(
        self, session_id: str, customer_id: str, role: str, content: str
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO conversation_history (session_id, customer_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, customer_id, role, content, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_conversation_history(
        self, customer_id: str, limit: int = 20
    ) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id, role, content, created_at FROM conversation_history "
                "WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
                (customer_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search_resolved_tickets(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword search over resolved tickets for semantic recall."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            keywords = [f"%{t}%" for t in query.lower().split() if len(t) > 2]
            if not keywords:
                return []
            conditions = " OR ".join(
                ["LOWER(issue_summary) LIKE ? OR LOWER(resolution) LIKE ?"] * len(keywords)
            )
            params = []
            for kw in keywords:
                params.extend([kw, kw])

            cursor.execute(
                f"SELECT ticket_id, customer_id, issue_summary, resolution, category, created_at "
                f"FROM resolved_tickets WHERE {conditions} ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
