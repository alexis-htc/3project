"""Account lookup tool: queries the CultPass external database for customer info."""

import sqlite3
import os
import json
from langchain.tools import tool


def create_account_lookup_tool(db_path: str):
    """Create a tool that looks up customer account information."""

    @tool
    def account_lookup(customer_identifier: str) -> str:
        """
        Look up a CultPass customer account by email address or customer ID.

        Args:
            customer_identifier: The customer's email address or numeric customer ID.

        Returns:
            Customer account details including name, email, plan, status, and recent bookings.
        """
        if not os.path.exists(db_path):
            return f"Error: Database not found at {db_path}. Run 01_external_db_setup.py first."

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()

            # Try to find by email or ID
            if customer_identifier.isdigit():
                cursor.execute(
                    "SELECT * FROM customers WHERE id = ?",
                    (int(customer_identifier),),
                )
            else:
                cursor.execute(
                    "SELECT * FROM customers WHERE email = ? OR name LIKE ?",
                    (customer_identifier, f"%{customer_identifier}%"),
                )

            customer = cursor.fetchone()
            if not customer:
                return f"No customer found matching '{customer_identifier}'."

            cid = customer["id"]

            # Get subscription info
            cursor.execute(
                "SELECT plan, status, start_date, price, billing_cycle FROM subscriptions "
                "WHERE customer_id = ? ORDER BY id DESC LIMIT 1",
                (cid,),
            )
            subscription = cursor.fetchone()

            # Get recent payments
            cursor.execute(
                "SELECT amount, status, payment_date, description FROM payments "
                "WHERE customer_id = ? ORDER BY payment_date DESC LIMIT 3",
                (cid,),
            )
            payments = cursor.fetchall()

            # Get bookings
            cursor.execute(
                "SELECT b.status, b.booked_at, e.name as event_name, e.date as event_date "
                "FROM bookings b JOIN events e ON b.event_id = e.id "
                "WHERE b.customer_id = ? ORDER BY b.booked_at DESC LIMIT 5",
                (cid,),
            )
            bookings = cursor.fetchall()

            result = {
                "customer_id": customer["id"],
                "name": customer["name"],
                "email": customer["email"],
                "phone": customer["phone"],
                "plan": customer["plan"],
                "status": customer["status"],
                "member_since": customer["created_at"],
            }

            if subscription:
                result["subscription"] = {
                    "plan": subscription["plan"],
                    "status": subscription["status"],
                    "start_date": subscription["start_date"],
                    "price": subscription["price"],
                    "billing_cycle": subscription["billing_cycle"],
                }

            if payments:
                result["recent_payments"] = [
                    {
                        "amount": p["amount"],
                        "status": p["status"],
                        "date": p["payment_date"],
                        "description": p["description"],
                    }
                    for p in payments
                ]

            if bookings:
                result["recent_bookings"] = [
                    {
                        "event": b["event_name"],
                        "event_date": b["event_date"],
                        "status": b["status"],
                        "booked_at": b["booked_at"],
                    }
                    for b in bookings
                ]

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error looking up account: {str(e)}"
        finally:
            conn.close()

    return account_lookup
