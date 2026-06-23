"""Subscription management tool: check or modify CultPass subscriptions."""

import sqlite3
import os
import json
from datetime import datetime
from langchain.tools import tool


def create_subscription_management_tool(db_path: str):
    """Create a tool that manages customer subscriptions."""

    @tool
    def subscription_management(
        customer_id: str,
        action: str = "check",
        new_plan: str = "",
    ) -> str:
        """
        Manage a CultPass customer subscription. Supports checking status,
        upgrading, downgrading, or cancelling.

        Args:
            customer_id: The numeric customer ID.
            action: One of 'check', 'upgrade', 'downgrade', or 'cancel'.
            new_plan: The target plan for upgrade/downgrade (basic, premium, or vip).

        Returns:
            Result of the subscription operation with details.
        """
        if not os.path.exists(db_path):
            return f"Error: Database not found at {db_path}. Run 01_external_db_setup.py first."

        plan_prices = {"basic": 9.99, "premium": 19.99, "vip": 39.99}
        plan_order = ["basic", "premium", "vip"]

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()

            # Get current subscription
            cursor.execute(
                "SELECT s.*, c.name, c.email FROM subscriptions s "
                "JOIN customers c ON s.customer_id = c.id "
                "WHERE s.customer_id = ? ORDER BY s.id DESC LIMIT 1",
                (int(customer_id),),
            )
            sub = cursor.fetchone()
            if not sub:
                return f"No subscription found for customer {customer_id}."

            current_plan = sub["plan"]

            if action == "check":
                return json.dumps({
                    "customer_id": customer_id,
                    "customer_name": sub["name"],
                    "current_plan": current_plan,
                    "status": sub["status"],
                    "price": plan_prices.get(current_plan, 0),
                    "start_date": sub["start_date"],
                    "billing_cycle": sub["billing_cycle"],
                }, indent=2)

            if action in ("upgrade", "downgrade"):
                new_plan_lower = new_plan.lower().strip()
                if new_plan_lower not in plan_prices:
                    return f"Invalid plan '{new_plan}'. Choose from: basic, premium, vip."

                current_idx = plan_order.index(current_plan) if current_plan in plan_order else 0
                new_idx = plan_order.index(new_plan_lower)

                if action == "upgrade" and new_idx <= current_idx:
                    return f"Cannot upgrade from {current_plan} to {new_plan_lower}. The target plan must be higher."

                if action == "downgrade" and new_idx >= current_idx:
                    return f"Cannot downgrade from {current_plan} to {new_plan_lower}. The target plan must be lower."

                now = datetime.now().isoformat()
                cursor.execute(
                    "UPDATE subscriptions SET plan = ?, price = ? WHERE id = ?",
                    (new_plan_lower, plan_prices[new_plan_lower], sub["id"]),
                )
                cursor.execute(
                    "UPDATE customers SET plan = ? WHERE id = ?",
                    (new_plan_lower, int(customer_id)),
                )
                conn.commit()

                return json.dumps({
                    "success": True,
                    "action": action,
                    "previous_plan": current_plan,
                    "new_plan": new_plan_lower,
                    "new_price": plan_prices[new_plan_lower],
                    "effective": "immediately" if action == "upgrade" else "next billing cycle",
                    "timestamp": now,
                }, indent=2)

            if action == "cancel":
                now = datetime.now().isoformat()
                cursor.execute(
                    "UPDATE subscriptions SET status = 'cancelled', end_date = ? WHERE id = ?",
                    (now, sub["id"]),
                )
                cursor.execute(
                    "UPDATE customers SET status = 'inactive' WHERE id = ?",
                    (int(customer_id),),
                )
                conn.commit()

                return json.dumps({
                    "success": True,
                    "action": "cancel",
                    "plan": current_plan,
                    "access_until": "end of current billing period",
                    "cancelled_at": now,
                    "refund_eligible": True,
                    "note": "Access continues until end of billing period. Reactivation possible anytime.",
                }, indent=2)

            return f"Unknown action '{action}'. Use: check, upgrade, downgrade, or cancel."

        except Exception as e:
            return f"Error managing subscription: {str(e)}"
        finally:
            conn.close()

    return subscription_management
