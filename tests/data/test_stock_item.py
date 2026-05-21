import pytest
import simple_trader.data.stocks.stock_item as stock_item
from simple_trader.data.stocks.base_stock import BaseStock


class FakeStock(BaseStock):
    @property
    def fee(self) -> float:
        return 0.002

    def get_candles_history(self, count: int, symbol: str, time_point: int, tf_minutes: int = 1) -> list:
        return []

    def trade(self, symbol: str, side: str, order_type: str, quantity: float, price: float | None = None) -> dict:
        return {"orderId": "99"}

    def get_order_status(self, symbol: str, order_id: str) -> dict:
        return {"status": "FILLED", "executedQty": "1.0"}


@pytest.fixture(autouse=True)
def reset():
    """Ensure singleton is cleared before and after each test."""
    stock_item.set(None)
    yield
    stock_item.set(None)


def test_get_raises_before_set():
    with pytest.raises(RuntimeError, match="not initialized"):
        stock_item.get()


def test_get_returns_instance_after_set():
    fake = FakeStock()
    stock_item.set(fake)
    assert stock_item.get() is fake


def test_set_overwrites_previous():
    fake1 = FakeStock()
    fake2 = FakeStock()
    stock_item.set(fake1)
    stock_item.set(fake2)
    assert stock_item.get() is fake2


def test_set_none_clears_instance():
    stock_item.set(FakeStock())
    stock_item.set(None)
    with pytest.raises(RuntimeError):
        stock_item.get()
