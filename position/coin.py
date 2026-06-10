"""Coin — lightweight currency tracker for position management."""


class Coin:
    """Tracks available, used, returned, borrowed, and pending amounts for a single currency.

    Attributes:
        name: Currency name (e.g., "usdt", "link")
        size: Available amount currently held
        used: Amount deployed/committed in current action
        returned: Amount recovered from fills
        loan: Amount borrowed on margin
        want_to_use: Intended allocation for position opening
        action_amount: Amount pending in current order (set before placing order)
    """

    def __init__(self, name: str):
        """Initialize Coin with currency name.

        Args:
            name: Currency name (e.g., "usdt", "btc", "eth")
        """
        self.name: str = name
        self.size: float = 0.0
        self.used: float = 0.0
        self.returned: float = 0.0
        self.loan: float = 0.0
        self.want_to_use: float = 0.0
        self.action_amount: float = 0.0

    def log(self) -> None:
        """Print current state of all attributes with label."""
        print(f"Coin({self.name}):")
        print(f"  size={self.size}")
        print(f"  used={self.used}")
        print(f"  returned={self.returned}")
        print(f"  loan={self.loan}")
        print(f"  want_to_use={self.want_to_use}")
        print(f"  action_amount={self.action_amount}")

    def to_dict(self) -> dict:
        """Serialize all attributes to JSON-compatible dict.

        Returns:
            Dictionary with all attributes (name and all float amounts)
        """
        return {
            "name": self.name,
            "size": self.size,
            "used": self.used,
            "returned": self.returned,
            "loan": self.loan,
            "want_to_use": self.want_to_use,
            "action_amount": self.action_amount,
        }

    def from_dict(self, data: dict) -> None:
        """Restore all attributes from dict (in-place update).

        Args:
            data: Dictionary with "name" and all amount fields
        """
        self.name = data["name"]
        self.size = data["size"]
        self.used = data["used"]
        self.returned = data["returned"]
        self.loan = data["loan"]
        self.want_to_use = data["want_to_use"]
        self.action_amount = data["action_amount"]

    def reset(self) -> None:
        """Set all amounts to 0.0 (keeps name).

        Called during position.finalize() to prepare coin for reuse.
        """
        self.size = 0.0
        self.used = 0.0
        self.returned = 0.0
        self.loan = 0.0
        self.want_to_use = 0.0
        self.action_amount = 0.0
