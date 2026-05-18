import pytest
from typing import Any, Dict, List, Optional
from simple_trader.data.stocks.base_stock import BaseStock


def test_base_stock_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseStock()  # type: ignore[abstract]


def test_subclass_missing_trade_raises_type_error():
    class Partial(BaseStock):
        @property
        def fee(self) -> float:
            return 0.001

        def get_candles_history(self, count: int, symbol: str, time_point: int, tf_minutes: int = 1) -> List[Any]:
            return []

        def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
            return {}

    with pytest.raises(TypeError):
        Partial()


def test_fully_implemented_subclass_instantiates():
    class FakeStock(BaseStock):
        @property
        def fee(self) -> float:
            return 0.001

        def get_candles_history(self, count: int, symbol: str, time_point: int, tf_minutes: int = 1) -> List[Any]:
            return []

        def trade(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
            return {"orderId": "1"}

        def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
            return {"status": "FILLED", "executedQty": "1.0"}

    stock = FakeStock()
    assert stock.fee == 0.001


def test_get_candles_history_returns_list():
    class FakeStock(BaseStock):
        @property
        def fee(self) -> float:
            return 0.001

        def get_candles_history(self, count: int, symbol: str, time_point: int, tf_minutes: int = 1) -> List[Any]:
            return [["t", "o", "h", "l", "c", "v"]]

        def trade(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
            return {"orderId": "1"}

        def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
            return {"status": "FILLED", "executedQty": "1.0"}

    stock = FakeStock()
    result = stock.get_candles_history(1, "LINKUSDT", 0)
    assert isinstance(result, list)
    assert len(result) == 1
