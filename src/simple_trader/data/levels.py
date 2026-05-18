from typing import Any, Dict, List, Optional


class Levels:
    """
    Manages support and resistance price levels.

    Levels are loaded once per session from a flat file and queried
    by the Strategy layer to set stop-loss and take-profit prices.
    """

    def __init__(self) -> None:
        self._levels: List[Dict[str, Any]] = []

    def add(self, price: float, label: str = "") -> None:
        self._levels.append({"price": price, "label": label})

    def load(self, levels: List[Dict[str, Any]]) -> None:
        self._levels = [dict(d) for d in levels]

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._levels)

    def clear(self) -> None:
        self._levels = []

    def nearest(self, price: float) -> Optional[Dict[str, Any]]:
        if not self._levels:
            return None
        return min(self._levels, key=lambda lvl: abs(lvl["price"] - price))
