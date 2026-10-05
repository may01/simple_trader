# tests/unit/stock_abstraction/test_readonly_stocks.py
# Pair naming for the paper-mode mock, and the keyless real-candles stock
# (STOCK_TYPE=binance_candles) — no real network calls.

from unittest.mock import MagicMock, patch

import pytest

from constants import STATUS_FAIL, TRADE_BUY


# ---------------------------------------------------------------------------
# Stock_MockBinance.get_pair_name
# ---------------------------------------------------------------------------

def test_mock_binance_reports_its_pair():
    # Regression: without an override this fell through to StockInterface's
    # default and returned "", so every indicator main/ published in paper
    # mode reached trade_executor under pair "".
    from stocks.mock_stock import Stock_MockBinance
    assert Stock_MockBinance().get_pair_name() == "LINKUSDT"


# ---------------------------------------------------------------------------
# Stock_BinanceCandles
# ---------------------------------------------------------------------------

@pytest.fixture
def candles_env(monkeypatch):
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("EXCHANGE_FEE", "0.001")
    # Present in the environment on purpose: the stock must not pick them up.
    monkeypatch.setenv("BINANCE_API_KEY", "real_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "real_secret")


@pytest.fixture
def stock(candles_env):
    with patch("stocks.binance_candles_stock.Client") as client_cls:
        client_cls.return_value = MagicMock()
        from stocks.binance_candles_stock import Stock_BinanceCandles
        s = Stock_BinanceCandles()
        s._client_cls = client_cls
        yield s


def test_builds_a_keyless_client_even_when_keys_are_in_the_env(stock):
    stock._client_cls.assert_called_once_with()
    assert stock.key == "" and stock.secret == ""


def test_pair_and_fee_come_from_env(stock):
    assert stock.get_pair_name() == "LINKUSDT"
    assert stock.fee == 0.001
    assert stock.was_init is True


def test_works_without_any_api_keys(monkeypatch):
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("EXCHANGE_FEE", "0.001")
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    with patch("stocks.binance_candles_stock.Client"):
        from stocks.binance_candles_stock import Stock_BinanceCandles
        assert Stock_BinanceCandles().get_pair_name() == "LINKUSDT"


def test_candles_come_from_the_public_klines_endpoint(stock):
    stock.client.get_historical_klines.return_value = []
    stock.get_candles_history([1], "link")
    stock.client.get_historical_klines.assert_called()
    assert stock.client.get_historical_klines.call_args[0][0] == "LINKUSDT"


@pytest.mark.parametrize("call", [
    lambda s: s.trade(TRADE_BUY, 20.0, 5.0),
    lambda s: s.trade(TRADE_BUY, 20.0, 5.0, force=True),
    lambda s: s.order_info("1"),
    lambda s: s.cancel_order("1"),
    lambda s: s.funds("usdt"),
    lambda s: s.get_aviable_loan("usdt"),
    lambda s: s.borrow("usdt", 1.0),
    lambda s: s.repay("usdt", 1.0),
])
def test_every_account_or_order_call_is_refused_without_touching_the_client(stock, call):
    status, _ = call(stock)
    assert status == STATUS_FAIL
    stock.client.create_margin_order.assert_not_called()
    stock.client.create_margin_loan.assert_not_called()
    stock.client.repay_margin_loan.assert_not_called()
    stock.client.get_margin_account.assert_not_called()


def test_stock_holder_knows_binance_candles(candles_env):
    with patch("stocks.binance_candles_stock.Client"):
        from stocks_holder import do_stock_init, stock_holder
        do_stock_init("binance_candles")
        assert stock_holder.item.stock_name == "binance_candles"
        assert stock_holder.item.get_pair_name() == "LINKUSDT"
