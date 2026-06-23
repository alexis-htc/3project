"""
01_external_db_setup.py
-----------------------
Sets up the CultPass external database with sample customer, subscription,
payment, and event data. This simulates the data received from CultPass,
the first customer to purchase UDA-Hub.

Run: python 01_external_db_setup.py
"""

import sqlite3
import os
from datetime import datetime, timedelta
import random

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "external")
DB_PATH = os.path.join(DB_DIR, "cultpass.db")


def create_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            plan TEXT DEFAULT 'basic',
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            plan TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            start_date TEXT NOT NULL,
            end_date TEXT,
            price REAL NOT NULL,
            billing_cycle TEXT DEFAULT 'monthly',
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed',
            payment_date TEXT NOT NULL,
            description TEXT,
            payment_method TEXT DEFAULT 'credit_card',
            refundable INTEGER DEFAULT 1,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            venue TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL DEFAULT 0.0,
            capacity INTEGER DEFAULT 100,
            available_spots INTEGER DEFAULT 100
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            event_id INTEGER NOT NULL,
            status TEXT DEFAULT 'confirmed',
            booked_at TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (event_id) REFERENCES events(id)
        )
    """)

    conn.commit()


def seed_customers(conn: sqlite3.Connection) -> None:
    customers = [
        ("Alice Johnson", "alice.johnson@email.com", "+1-555-0101", "premium", "active", "2024-01-15"),
        ("Bob Smith", "bob.smith@email.com", "+1-555-0102", "basic", "active", "2024-02-20"),
        ("Carol Williams", "carol.w@email.com", "+1-555-0103", "vip", "active", "2024-01-10"),
        ("David Brown", "david.b@email.com", "+1-555-0104", "premium", "active", "2024-03-05"),
        ("Emma Davis", "emma.d@email.com", "+1-555-0105", "basic", "inactive", "2024-02-01"),
        ("Frank Miller", "frank.m@email.com", "+1-555-0106", "vip", "active", "2023-11-20"),
        ("Grace Lee", "grace.lee@email.com", "+1-555-0107", "premium", "active", "2024-04-01"),
        ("Henry Wilson", "henry.w@email.com", "+1-555-0108", "basic", "suspended", "2024-01-25"),
        ("Iris Chen", "iris.chen@email.com", "+1-555-0109", "premium", "active", "2024-03-15"),
        ("Jack Taylor", "jack.t@email.com", "+1-555-0110", "basic", "active", "2024-05-01"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO customers (name, email, phone, plan, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        customers,
    )
    conn.commit()


def seed_subscriptions(conn: sqlite3.Connection) -> None:
    plan_prices = {"basic": 9.99, "premium": 19.99, "vip": 39.99}
    cursor = conn.cursor()
    cursor.execute("SELECT id, plan, created_at FROM customers")
    for cid, plan, created in cursor.fetchall():
        conn.execute(
            "INSERT INTO subscriptions (customer_id, plan, status, start_date, price, billing_cycle) "
            "VALUES (?, ?, 'active', ?, ?, 'monthly')",
            (cid, plan, created, plan_prices.get(plan, 9.99)),
        )
    conn.commit()


def seed_payments(conn: sqlite3.Connection) -> None:
    plan_prices = {"basic": 9.99, "premium": 19.99, "vip": 39.99}
    cursor = conn.cursor()
    cursor.execute("SELECT id, plan, created_at FROM customers")
    for cid, plan, created in cursor.fetchall():
        base = datetime.strptime(created, "%Y-%m-%d")
        price = plan_prices.get(plan, 9.99)
        for month_offset in range(6):
            pdate = base + timedelta(days=30 * month_offset)
            if pdate > datetime.now():
                break
            status = "completed" if random.random() > 0.05 else "failed"
            conn.execute(
                "INSERT INTO payments (customer_id, amount, status, payment_date, description, payment_method) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cid, price, status, pdate.strftime("%Y-%m-%d"),
                 f"Monthly subscription - {plan} plan", "credit_card"),
            )
    conn.commit()


def seed_events(conn: sqlite3.Connection) -> None:
    events = [
        ("Jazz Night at the Blue Note", "2025-07-15", "Blue Note Jazz Club", "music", 25.00, 200, 150),
        ("Modern Art Exhibition", "2025-07-20", "Metropolitan Museum", "museum", 0.00, 500, 320),
        ("Shakespeare in the Park", "2025-08-01", "Central Park", "theater", 15.00, 300, 200),
        ("Food & Wine Festival", "2025-08-10", "Convention Center", "festival", 45.00, 1000, 650),
        ("Pottery Workshop", "2025-07-25", "Craft Studio Downtown", "workshop", 35.00, 20, 8),
        ("Symphony Orchestra Gala", "2025-09-01", "Carnegie Hall", "music", 75.00, 500, 400),
        ("Street Photography Walk", "2025-07-18", "Brooklyn Bridge", "workshop", 10.00, 30, 22),
        ("Comedy Night Special", "2025-07-22", "Laugh Factory", "comedy", 20.00, 150, 90),
        ("Indie Film Festival", "2025-08-15", "Tribeca Cinema", "festival", 30.00, 250, 180),
        ("Sculpture Garden Tour", "2025-07-30", "Noguchi Museum", "museum", 0.00, 40, 25),
    ]
    conn.executemany(
        "INSERT INTO events (name, date, venue, category, price, capacity, available_spots) VALUES (?, ?, ?, ?, ?, ?, ?)",
        events,
    )
    conn.commit()


def seed_bookings(conn: sqlite3.Connection) -> None:
    bookings = [
        (1, 1, "confirmed", "2025-06-20"),
        (1, 3, "confirmed", "2025-06-22"),
        (3, 6, "confirmed", "2025-06-15"),
        (3, 2, "confirmed", "2025-06-18"),
        (4, 5, "confirmed", "2025-06-25"),
        (6, 1, "confirmed", "2025-06-10"),
        (6, 8, "cancelled", "2025-06-12"),
        (7, 9, "confirmed", "2025-06-28"),
        (9, 4, "confirmed", "2025-07-01"),
        (10, 7, "confirmed", "2025-07-05"),
    ]
    conn.executemany(
        "INSERT INTO bookings (customer_id, event_id, status, booked_at) VALUES (?, ?, ?, ?)",
        bookings,
    )
    conn.commit()


def verify_setup(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    tables = ["customers", "subscriptions", "payments", "events", "bookings"]
    print("\n--- CultPass External Database Verification ---")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} records")
    print("--- Setup Complete ---\n")


def main() -> None:
    os.makedirs(DB_DIR, exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        create_tables(conn)
        seed_customers(conn)
        seed_subscriptions(conn)
        seed_payments(conn)
        seed_events(conn)
        seed_bookings(conn)
        verify_setup(conn)
    finally:
        conn.close()

    print(f"CultPass external database created at: {DB_PATH}")


if __name__ == "__main__":
    main()
