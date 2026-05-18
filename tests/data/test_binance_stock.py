import pytest
from unittest.mock import MagicMock, patch
from simple_trader.data.stocks.binance_stock import BinanceStock


@pytest.fixture
def mock_client():
    with patch("simple_trader.data.stocks.binance_stock.Client") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        yield client


def test_fee_returns_constructor_value(mock_client):
    stock = BinanceStock("key", "secret", taker_fee=0.001)
    assert stock.fee == 0.001


def test_get_candles_history_calls_get_klines(mock_client):
    mock_client.get_klines.return_value = [["ts", "o", "h", "l", "c", "v"]]
    stock = BinanceStock("key", "secret")
    result = stock.get_candles_history(count=100, symbol="LINKUSDT", time_point=0)
    mock_client.get_klines.assert_called_once()
    assert result == [["ts", "o", "h", "l", "c", "v"]]


def test_get_candles_history_uses_tf_minutes_map(mock_client):
    mock_client.get_klines.return_value = []
    stock = BinanceStock("key", "secret")
    stock.get_candles_history(count=10, symbol="LINKUSDT", time_point=0, tf_minutes=60)
    call_kwargs = mock_client.get_klines.call_args[1]
    assert call_kwargs["interval"] == "1h"


def test_trade_calls_create_order(mock_client):
    mock_client.create_order.return_value = {"orderId": "42"}
    stock = BinanceStock("key", "secret")
    result = stock.trade("LINKUSDT", "BUY", "LIMIT", 10.0, price=15.5)
    assert result["orderId"] == "42"
    mock_client.create_order.assert_called_once()
    call_kwargs = mock_client.create_order.call_args[1]
    assert call_kwargs.get("timeInForce") == "GTC"


def test_get_order_status_calls_get_order(mock_client):
    mock_client.get_order.return_value = {"status": "FILLED", "executedQty": "10.0"}
    stock = BinanceStock("key", "secret")
    result = stock.get_order_status("LINKUSDT", "42")
    assert result["status"] == "FILLED"
    mock_client.get_order.assert_called_once_with(symbol="LINKUSDT", orderId="42")


def test_trade_market_order_no_time_in_force(mock_client):
    mock_client.create_order.return_value = {"orderId": "55"}
    stock = BinanceStock("key", "secret")
    stock.trade("LINKUSDT", "BUY", "MARKET", 5.0)
    call_kwargs = mock_client.create_order.call_args[1]
    assert "timeInForce" not in call_kwargs
    assert "price" not in call_kwargs


def test_get_candles_history_raises_for_unknown_tf_minutes(mock_client):
    stock = BinanceStock("key", "secret")
    with pytest.raises(ValueError, match="Unsupported tf_minutes"):
        stock.get_candles_history(count=10, symbol="LINKUSDT", time_point=0, tf_minutes=30)
