"""
02_core_db_setup.py
-------------------
Sets up the UDA-Hub core database with application tables (Account, User,
Ticket, TicketMetadata, TicketMessage, Knowledge) and loads the CultPass
knowledge base articles from cultpass_articles.jsonl.

Run: python 02_core_db_setup.py
"""

import sqlite3
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DB_DIR = os.path.join(BASE_DIR, "data", "core")
CORE_DB_PATH = os.path.join(CORE_DB_DIR, "uda_hub.db")
ARTICLES_PATH = os.path.join(CORE_DB_DIR, "cultpass_articles.jsonl")
MEMORY_DB_DIR = os.path.join(BASE_DIR, "data", "core")
MEMORY_DB_PATH = os.path.join(MEMORY_DB_DIR, "memory.db")


def create_core_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            domain TEXT,
            api_key TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'agent',
            created_at TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            customer_id INTEGER,
            subject TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            priority TEXT DEFAULT 'medium',
            category TEXT,
            channel TEXT DEFAULT 'chat',
            created_at TEXT NOT NULL,
            updated_at TEXT,
            resolved_at TEXT,
            FOREIGN KEY (account_id) REFERENCES accounts(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            sender_type TEXT NOT NULL,
            sender_id TEXT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (ticket_id) REFERENCES tickets(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            account_id INTEGER,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id)
        )
    """)

    conn.commit()


def create_memory_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
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

    cursor.execute("""
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


def seed_account(conn: sqlite3.Connection) -> int:
    now = datetime.now().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO accounts (name, domain, created_at) VALUES (?, ?, ?)",
        ("CultPass", "cultpass.com", now),
    )
    conn.commit()
    return cursor.lastrowid


def seed_users(conn: sqlite3.Connection, account_id: int) -> None:
    now = datetime.now().isoformat()
    users = [
        (account_id, "Sarah Admin", "sarah@cultpass.com", "admin", now),
        (account_id, "Mike Support", "mike@cultpass.com", "agent", now),
        (account_id, "Lisa Support", "lisa@cultpass.com", "agent", now),
        (account_id, "Tom Manager", "tom@cultpass.com", "manager", now),
    ]
    conn.executemany(
        "INSERT INTO users (account_id, name, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
        users,
    )
    conn.commit()


def seed_sample_tickets(conn: sqlite3.Connection, account_id: int) -> None:
    now = datetime.now().isoformat()
    tickets = [
        (account_id, 1, "Cannot login to my account", "I keep getting 'Invalid credentials' when trying to log in. I'm sure my password is correct.", "open", "high", "technical", "chat", now),
        (account_id, 2, "Request for refund", "I was charged $19.99 but I cancelled my subscription last week. Please refund.", "open", "medium", "billing", "email", now),
        (account_id, 3, "Upgrade to VIP plan", "I'd like to upgrade from Premium to VIP. How does the prorated billing work?", "open", "low", "subscription", "chat", now),
        (account_id, 5, "App crashes on event booking", "Every time I try to book an event, the app crashes immediately. Using iPhone 14, iOS 17.", "open", "high", "technical", "chat", now),
        (account_id, 8, "Payment keeps failing", "My credit card payment fails every month. I've updated my card but it still doesn't work.", "open", "critical", "billing", "email", now),
    ]
    for t in tickets:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tickets (account_id, customer_id, subject, content, status, priority, category, channel, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            t,
        )
        tid = cursor.lastrowid
        conn.execute(
            "INSERT INTO ticket_messages (ticket_id, sender_type, sender_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (tid, "customer", str(t[1]), t[3], now),
        )
    conn.commit()


def load_knowledge_articles(conn: sqlite3.Connection, account_id: int) -> int:
    if not os.path.exists(ARTICLES_PATH):
        print(f"Warning: Articles file not found at {ARTICLES_PATH}")
        return 0

    count = 0
    now = datetime.now().isoformat()
    with open(ARTICLES_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            conn.execute(
                "INSERT OR REPLACE INTO knowledge (id, account_id, title, content, category, tags, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    article["id"],
                    account_id,
                    article["title"],
                    article["content"],
                    article.get("category", "general"),
                    json.dumps(article.get("tags", [])),
                    now,
                ),
            )
            count += 1
    conn.commit()
    return count


def verify_setup(conn: sqlite3.Connection, label: str) -> None:
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n--- {label} Verification ---")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} records")
    print(f"--- {label} Complete ---\n")


def main() -> None:
    os.makedirs(CORE_DB_DIR, exist_ok=True)
    os.makedirs(MEMORY_DB_DIR, exist_ok=True)

    # Core DB
    if os.path.exists(CORE_DB_PATH):
        os.remove(CORE_DB_PATH)
        print(f"Removed existing core database: {CORE_DB_PATH}")

    conn = sqlite3.connect(CORE_DB_PATH)
    try:
        create_core_tables(conn)
        account_id = seed_account(conn)
        seed_users(conn, account_id)
        seed_sample_tickets(conn, account_id)
        article_count = load_knowledge_articles(conn, account_id)
        print(f"Loaded {article_count} knowledge base articles")
        verify_setup(conn, "UDA-Hub Core Database")
    finally:
        conn.close()

    # Memory DB
    if os.path.exists(MEMORY_DB_PATH):
        os.remove(MEMORY_DB_PATH)
        print(f"Removed existing memory database: {MEMORY_DB_PATH}")

    mem_conn = sqlite3.connect(MEMORY_DB_PATH)
    try:
        create_memory_tables(mem_conn)
        verify_setup(mem_conn, "Memory Database")
    finally:
        mem_conn.close()

    print(f"Core database created at: {CORE_DB_PATH}")
    print(f"Memory database created at: {MEMORY_DB_PATH}")


if __name__ == "__main__":
    main()
