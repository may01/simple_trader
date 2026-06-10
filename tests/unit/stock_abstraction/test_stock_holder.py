import pytest
from constants import STATUS_SUCCESS


def test_singleton_same_object():
    from stocks_holder import stock_holder as s1
    from stocks_holder import stock_holder as s2
    assert s1 is s2


def test_before_init_returns_empty():
    import importlib, stocks_holder
    importlib.reload(stocks_holder)
    result = stocks_holder.stock_holder.item.get_candles_history([1], "link")
    assert result == {}


def test_mock_init():
    from stocks_holder import do_stock_init, stock_holder as stock
    do_stock_init("mock")
    assert stock.item.was_init == True
    assert stock.item.fee == 0.001


def test_mock_trade():
    from stocks_holder import do_stock_init, stock_holder as stock
    from constants import TRADE_BUY
    do_stock_init("mock")
    status, info = stock.item.trade(TRADE_BUY, 20.0, 5.0)
    assert status == STATUS_SUCCESS


def test_mock_order_info_filled():
    from stocks_holder import do_stock_init, stock_holder as stock
    do_stock_init("mock")
    status, info = stock.item.order_info("mock-001")
    assert status == STATUS_SUCCESS
    assert info["status"] == "FILLED"


def test_mock_binance_missing_fixture(monkeypatch, tmp_path):
    import stocks.mock_stock
    monkeypatch.setattr(stocks.mock_stock, "FIXTURE_PATH", str(tmp_path / "nonexistent.pkl"))
    from stocks_holder import do_stock_init
    with pytest.raises(FileNotFoundError):
        do_stock_init("mock_binance")


def test_mock_binance_returns_candles():
    from stocks_holder import do_stock_init, stock_holder as stock
    do_stock_init("mock_binance")
    result = stock.item.get_candles_history([1, 5, 15], "link")
    assert set(result.keys()) == {1, 5, 15}
    assert len(result[1]) > 0
