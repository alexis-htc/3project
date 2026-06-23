"""Refund processing tool: handles refund requests against CultPass payments."""

import sqlite3
import os
import json
import logging
from datetime import datetime
from langchain.tools import tool

logger = logging.getLogger("uda_hub.tools.refund")


def create_refund_processing_tool(db_path: str):
    """Create a tool that processes customer refund requests."""

    @tool
    def refund_processing(
        customer_id: str,
        reason: str = "",
        amount: str = "",
    ) -> str:
        """
        Process a refund request for a CultPass customer. Checks eligibility
        based on payment history and refund policy, then processes if eligible.

        Args:
            customer_id: The numeric customer ID.
            reason: The reason for the refund request.
            amount: Optional specific amount to refund. If empty, refunds the most recent eligible payment.

        Returns:
            Refund processing result with status and details.
        """
        if not os.path.exists(db_path):
            return f"Error: Database not found at {db_path}. Run 01_external_db_setup.py first."

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()

            # Verify customer exists
            cursor.execute("SELECT * FROM customers WHERE id = ?", (int(customer_id),))
            customer = cursor.fetchone()
            if not customer:
                return f"No customer found with ID {customer_id}."

            # Get refundable payments
            cursor.execute(
                "SELECT * FROM payments WHERE customer_id = ? AND status = 'completed' "
                "AND refundable = 1 ORDER BY payment_date DESC",
                (int(customer_id),),
            )
            payments = cursor.fetchall()

            if not payments:
                return json.dumps({
                    "success": False,
                    "reason": "No eligible payments found for refund.",
                    "customer_id": customer_id,
                }, indent=2)

            # Select the payment to refund
            target_payment = None
            if amount:
                try:
                    refund_amount = float(amount)
                    for p in payments:
                        if abs(p["amount"] - refund_amount) < 0.01:
                            target_payment = p
                            break
                except ValueError:
                    return f"Invalid amount '{amount}'. Provide a numeric value."

            if target_payment is None:
                target_payment = payments[0]

            # Process the refund
            now = datetime.now().isoformat()
            refund_amount = target_payment["amount"]

            cursor.execute(
                "UPDATE payments SET status = 'refunded', refundable = 0 WHERE id = ?",
                (target_payment["id"],),
            )
            conn.commit()

            result = {
                "success": True,
                "refund_id": f"REF-{target_payment['id']:05d}",
                "customer_id": customer_id,
                "customer_name": customer["name"],
                "amount_refunded": refund_amount,
                "original_payment_date": target_payment["payment_date"],
                "original_description": target_payment["description"],
                "reason": reason or "Customer request",
                "refund_method": "original payment method",
                "processing_time": "5-10 business days",
                "processed_at": now,
            }
            logger.info(
                "Refund processed",
                extra={
                    "event": "tool_call",
                    "operation": "refund_processing",
                    "outcome": "success",
                    "customer_id": customer_id,
                    "result": {"refund_id": result["refund_id"], "amount": refund_amount},
                },
            )
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(
                f"Refund error: {e}",
                extra={"event": "tool_call", "operation": "refund_processing", "outcome": "error"},
            )
            return f"Error processing refund: {str(e)}"
        finally:
            conn.close()

    return refund_processing
