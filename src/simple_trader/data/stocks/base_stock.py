from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseStock(ABC):

    @property
    @abstractmethod
    def fee(self) -> float:
        """Taker fee as a decimal fraction (e.g. 0.001 = 0.1%)."""

    @abstractmethod
    def get_candles_history(
        self,
        count: int,
        symbol: str,
        time_point: int,
        tf_minutes: int = 1,
    ) -> List[Any]:
        """Fetch OHLCV candles from the exchange."""

    @abstractmethod
    def trade(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place an order on the exchange."""

    @abstractmethod
    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Query the fill status of an existing order."""
