"""robots/live_order_tracker.py — tracks live order IDs and persists state."""

import json
import os

from position.position import Position
from stocks.base_stock import StockInterface


class LiveOrderTracker:
    """Tracks live Binance order IDs and loan state, with JSON crash recovery.

    Args:
        position: Position facade for recording fills and serialization.
        stock: Exchange interface for cancelling/querying orders.
        persist_path: File path for atomic JSON state persistence.
    """

    def __init__(self, position: Position, stock: StockInterface, persist_path: str) -> None:
        self.position = position
        self.stock = stock
        self.persist_path = persist_path
        self.buy_id: str = ""
        self.sell_id: str = ""
        self.loan_id: str = ""
        self.loan_amount: float = 0.0

    # ------------------------------------------------------------------
    # Order ID setters
    # ------------------------------------------------------------------

    def set_buy_order(self, order_id: str) -> None:
        """Set the active buy order ID and persist state."""
        self.buy_id = order_id
        self.save()

    def set_sell_order(self, order_id: str) -> None:
        """Set the active sell order ID and persist state."""
        self.sell_id = order_id
        self.save()

    def set_loan(self, loan_id: str, amount: float) -> None:
        """Record an active margin loan and persist state."""
        self.loan_id = loan_id
        self.loan_amount = amount
        self.save()

    def get_loan(self) -> tuple[str, float]:
        """Return (loan_id, loan_amount)."""
        return (self.loan_id, self.loan_amount)

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    def cancel_buy(self) -> None:
        """Cancel the active buy order on the exchange, clear buy_id, and save."""
        self.stock.cancel_order(self.buy_id)
        self.buy_id = ""
        self.save()

    def cancel_sell(self) -> None:
        """Cancel the active sell order on the exchange, clear sell_id, and save."""
        self.stock.cancel_order(self.sell_id)
        self.sell_id = ""
        self.save()

    def check_fill(self, order_id: str) -> tuple[str, dict]:
        """Query fill status for an order.

        Returns:
            (status, fill_dict) — passed through unchanged from stock.order_info().
        """
        return self.stock.order_info(order_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically serialize tracker + position state to persist_path.

        Writes to a .tmp file first, then renames to avoid partial writes.
        """
        data: dict = {
            "buy_id": self.buy_id,
            "sell_id": self.sell_id,
            "loan_id": self.loan_id,
            "loan_amount": self.loan_amount,
        }
        # Merge position dict (may be {} if no position is open)
        data.update(self.position.to_dict())

        tmp_path = self.persist_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, self.persist_path)

    def load(self) -> bool:
        """Load state from persist_path if it exists.

        Restores buy_id, sell_id, loan_id, loan_amount, and calls
        position.from_dict() to reconstruct the position.

        Returns:
            True if file existed and was loaded; False if absent.
        """
        if not os.path.exists(self.persist_path):
            return False

        with open(self.persist_path, "r") as f:
            data = json.load(f)

        self.buy_id = data.get("buy_id", "")
        self.sell_id = data.get("sell_id", "")
        self.loan_id = data.get("loan_id", "")
        self.loan_amount = data.get("loan_amount", 0.0)
        self.position.from_dict(data)
        return True

    def clear(self) -> None:
        """Reset all tracked IDs and delete the persist file if it exists."""
        self.buy_id = ""
        self.sell_id = ""
        self.loan_id = ""
        self.loan_amount = 0.0
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)
