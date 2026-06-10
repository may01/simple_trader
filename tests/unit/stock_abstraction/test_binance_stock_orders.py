# tests/unit/stock_abstraction/test_binance_stock_orders.py
# Unit tests for Stock_Binance order management methods.

import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd

from stocks.binance_stock import Stock_Binance
from constants import STATUS_SUCCESS, STATUS_FAIL, TRADE_BUY, TRADE_SELL


def _symbol_info():
    return {"filters": [
        {"filterType": "LOT_SIZE", "minQty": "0.01", "maxQty": "9000", "stepSize": "0.01"},
        {"filterType": "MIN_NOTIONAL", "minNotional": "10.0"},
    ]}


@pytest.fixture
def stock(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("EXCHANGE_FEE", "0.001")
    with patch("stocks.binance_stock.Client") as mc:
        mc.return_value.get_symbol_info.return_value = _symbol_info()
        s = Stock_Binance()
        s.client = mc.return_value
        yield s


# ------------------------------------------------------------------
# trade()
# ------------------------------------------------------------------

def test_trade_maps_buy_type(stock):
    stock.is_invalid_amount = MagicMock(return_value=False)
    stock.client.create_margin_order.return_value = {"orderId": "123"}
    status, info = stock.trade(TRADE_BUY, 20.0, 10.0)
    call_kwargs = stock.client.create_margin_order.call_args[1]
    assert call_kwargs.get("side") == "BUY"


def test_trade_maps_sell_type(stock):
    stock.is_invalid_amount = MagicMock(return_value=False)
    stock.client.create_margin_order.return_value = {"orderId": "456"}
    status, info = stock.trade(TRADE_SELL, 20.0, 10.0)
    call_kwargs = stock.client.create_margin_order.call_args[1]
    assert call_kwargs.get("side") == "SELL"


def test_trade_returns_success_with_order_id(stock):
    stock.is_invalid_amount = MagicMock(return_value=False)
    stock.client.create_margin_order.return_value = {"orderId": 789}
    status, info = stock.trade(TRADE_BUY, 20.0, 10.0)
    assert status == STATUS_SUCCESS
    assert info["order_id"] == "789"


def test_trade_invalid_amount_returns_fail(stock):
    stock.client.get_symbol_info.return_value = _symbol_info()
    status, info = stock.trade(TRADE_BUY, 20.0, 0.0001)
    assert status == STATUS_FAIL
    assert info == {}


def test_trade_exception_returns_fail(stock):
    stock.is_invalid_amount = MagicMock(return_value=False)
    stock.client.create_margin_order.side_effect = Exception("api error")
    status, info = stock.trade(TRADE_BUY, 20.0, 10.0)
    assert status == STATUS_FAIL
    assert info == {}


def test_trade_increments_weight(stock):
    stock.is_invalid_amount = MagicMock(return_value=False)
    stock.client.create_margin_order.return_value = {"orderId": "1"}
    stock.weight = 0
    stock.trade(TRADE_BUY, 20.0, 10.0)
    assert stock.weight == 6


# ------------------------------------------------------------------
# order_info()
# ------------------------------------------------------------------

def test_order_info_returns_status(stock):
    stock.client.get_margin_order.return_value = {
        "status": "FILLED", "origQty": "10.0", "executedQty": "10.0", "price": "20.0"
    }
    status, info = stock.order_info("123")
    assert status == STATUS_SUCCESS
    assert info["status"] == "FILLED"


def test_order_info_calculates_left_amount(stock):
    stock.client.get_margin_order.return_value = {
        "status": "PARTIALLY_FILLED",
        "origQty": "10.0",
        "executedQty": "4.0",
        "price": "20.0",
    }
    status, info = stock.order_info("123")
    assert status == STATUS_SUCCESS
    assert info["start_amount"] == 10.0
    assert info["left_amount"] == pytest.approx(6.0)
    assert info["rate"] == 20.0


def test_order_info_exception_returns_fail(stock):
    stock.client.get_margin_order.side_effect = Exception("network")
    status, info = stock.order_info("123")
    assert status == STATUS_FAIL
    assert info == {}


def test_order_info_increments_weight(stock):
    stock.client.get_margin_order.return_value = {
        "status": "NEW", "origQty": "10.0", "executedQty": "0.0", "price": "20.0"
    }
    stock.weight = 0
    stock.order_info("123")
    assert stock.weight == 10


# ------------------------------------------------------------------
# cancel_order()
# ------------------------------------------------------------------

def test_cancel_order_success(stock):
    stock.client.cancel_margin_order.return_value = {"status": "CANCELED"}
    stock.client.get_margin_order.return_value = {
        "status": "CANCELED", "origQty": "10.0", "executedQty": "0.0", "price": "20.0"
    }
    status, info = stock.cancel_order("123")
    assert status == STATUS_SUCCESS
    assert info["status"] == "CANCELED"


def test_cancel_polls_pending_cancel(stock):
    stock.client.cancel_margin_order.return_value = {"status": "PENDING_CANCEL"}
    stock.order_info = MagicMock(side_effect=[
        (STATUS_SUCCESS, {"status": "PENDING_CANCEL", "start_amount": 10.0, "left_amount": 5.0, "rate": 20.0}),
        (STATUS_SUCCESS, {"status": "CANCELED", "start_amount": 10.0, "left_amount": 0.0, "rate": 20.0}),
    ])
    with patch("time.sleep"):
        status, info = stock.cancel_order("123")
    assert status == STATUS_SUCCESS
    assert info["status"] == "CANCELED"


def test_cancel_order_exception_returns_fail(stock):
    stock.client.cancel_margin_order.side_effect = Exception("api error")
    status, info = stock.cancel_order("123")
    assert status == STATUS_FAIL
    assert info == {}


# ------------------------------------------------------------------
# depth()
# ------------------------------------------------------------------

def test_depth_returns_two_dataframes(stock):
    stock.client.get_order_book.return_value = {
        "asks": [["20.0", "5.0"], ["20.1", "3.0"]],
        "bids": [["19.9", "4.0"], ["19.8", "2.0"]],
    }
    asks, bids = stock.depth(100)
    assert isinstance(asks, pd.DataFrame)
    assert isinstance(bids, pd.DataFrame)
    assert "price" in asks.columns
    assert "cumulative" in bids.columns


def test_depth_cumulative_is_cumsum(stock):
    stock.client.get_order_book.return_value = {
        "asks": [["20.0", "5.0"], ["20.1", "3.0"]],
        "bids": [["19.9", "4.0"], ["19.8", "2.0"]],
    }
    asks, bids = stock.depth(100)
    # asks are sorted ascending by price: 20.0 then 20.1
    assert asks.iloc[0]["cumulative"] == pytest.approx(5.0)
    assert asks.iloc[1]["cumulative"] == pytest.approx(8.0)


def test_depth_weight_small(stock):
    stock.client.get_order_book.return_value = {"asks": [], "bids": []}
    stock.weight = 0
    stock.depth(50)
    assert stock.weight == 5


def test_depth_weight_medium(stock):
    stock.client.get_order_book.return_value = {"asks": [], "bids": []}
    stock.weight = 0
    stock.depth(100)
    assert stock.weight == 25


def test_depth_weight_large(stock):
    stock.client.get_order_book.return_value = {"asks": [], "bids": []}
    stock.weight = 0
    stock.depth(500)
    assert stock.weight == 50


def test_depth_weight_xlarge(stock):
    stock.client.get_order_book.return_value = {"asks": [], "bids": []}
    stock.weight = 0
    stock.depth(1000)
    assert stock.weight == 250


# ------------------------------------------------------------------
# info()
# ------------------------------------------------------------------

def test_info_returns_symbol_info(stock):
    stock.client.get_symbol_info.return_value = _symbol_info()
    result = stock.info()
    assert "filters" in result


def test_info_exception_returns_empty_dict(stock):
    stock.client.get_symbol_info.side_effect = Exception("error")
    result = stock.info()
    assert result == {}


# ------------------------------------------------------------------
# is_invalid_amount()
# ------------------------------------------------------------------

def test_is_invalid_amount_small_qty(stock):
    stock.client.get_symbol_info.return_value = _symbol_info()
    result = stock.is_invalid_amount(0.0001, 20.0)
    assert result is True


def test_is_invalid_amount_small_notional(stock):
    # amount=0.5 * 20 = 10 < 2*10 = 20, so notional invalid
    stock.client.get_symbol_info.return_value = _symbol_info()
    result = stock.is_invalid_amount(0.5, 20.0)
    assert result is True


def test_is_invalid_amount_valid(stock):
    stock.client.get_symbol_info.return_value = _symbol_info()
    result = stock.is_invalid_amount(10.0, 20.0)
    assert result is False


def test_is_invalid_amount_info_fails(stock):
    stock.client.get_symbol_info.return_value = {}
    result = stock.is_invalid_amount(10.0, 20.0)
    assert result is True


# ------------------------------------------------------------------
# funds()
# ------------------------------------------------------------------

def test_funds_returns_balance(stock):
    stock.client.get_margin_account.return_value = {
        "userAssets": [
            {"asset": "LINK", "free": "5.5", "locked": "0.0"},
            {"asset": "USDT", "free": "100.0", "locked": "0.0"},
        ]
    }
    status, amount = stock.funds("LINK")
    assert status == STATUS_SUCCESS
    assert amount == pytest.approx(5.5)


def test_funds_missing_coin_returns_zero(stock):
    stock.client.get_margin_account.return_value = {"userAssets": []}
    status, amount = stock.funds("BTC")
    assert status == STATUS_SUCCESS
    assert amount == 0.0


def test_funds_exception_returns_fail(stock):
    stock.client.get_margin_account.side_effect = Exception("net")
    status, amount = stock.funds("LINK")
    assert status == STATUS_FAIL
    assert amount == 0.0


def test_funds_locked_asset_type(stock):
    stock.client.get_margin_account.return_value = {
        "userAssets": [{"asset": "LINK", "free": "5.0", "locked": "2.0"}]
    }
    status, amount = stock.funds("LINK", asset_type="locked")
    assert status == STATUS_SUCCESS
    assert amount == pytest.approx(2.0)


# ------------------------------------------------------------------
# get_aviable_loan()
# ------------------------------------------------------------------

def test_get_aviable_loan_returns_amount(stock):
    stock.client.get_max_margin_loan.return_value = {"amount": "50.0"}
    status, amount = stock.get_aviable_loan("link")
    assert status == STATUS_SUCCESS
    assert amount == pytest.approx(50.0)
    stock.client.get_max_margin_loan.assert_called_once_with(asset="LINK")


def test_get_aviable_loan_exception_returns_fail(stock):
    stock.client.get_max_margin_loan.side_effect = Exception("api")
    status, amount = stock.get_aviable_loan("LINK")
    assert status == STATUS_FAIL
    assert amount == 0.0


# ------------------------------------------------------------------
# borrow()
# ------------------------------------------------------------------

def test_borrow_returns_success(stock):
    stock.client.create_margin_loan.return_value = {}
    status, amount = stock.borrow("link", 10.0)
    assert status == STATUS_SUCCESS
    assert amount == 10.0
    stock.client.create_margin_loan.assert_called_once_with(asset="LINK", amount="10.0")


def test_borrow_exception_returns_fail(stock):
    stock.client.create_margin_loan.side_effect = Exception("api")
    status, amount = stock.borrow("LINK", 10.0)
    assert status == STATUS_FAIL
    assert amount == 0.0


# ------------------------------------------------------------------
# repay()
# ------------------------------------------------------------------

def test_repay_returns_success(stock):
    stock.client.repay_margin_loan.return_value = {}
    status, amount = stock.repay("link", 5.0)
    assert status == STATUS_SUCCESS
    assert amount == 5.0
    stock.client.repay_margin_loan.assert_called_once_with(asset="LINK", amount="5.0")


def test_repay_exception_returns_fail(stock):
    stock.client.repay_margin_loan.side_effect = Exception("api")
    status, amount = stock.repay("LINK", 5.0)
    assert status == STATUS_FAIL
    assert amount == 0.0


# ------------------------------------------------------------------
# set_operation_sleep()
# ------------------------------------------------------------------

def test_set_operation_sleep_no_sleep_below_limit(stock):
    stock.weight = 100
    stock.overflow_weight = 5000
    with patch("time.sleep") as mock_sleep:
        stock.set_operation_sleep()
    mock_sleep.assert_not_called()


def test_set_operation_sleep_sleeps_above_limit(stock):
    stock.weight = 6000
    stock.overflow_weight = 5000
    stock.overflow_weight_time = 60
    with patch("time.sleep") as mock_sleep:
        stock.set_operation_sleep()
    mock_sleep.assert_called_once_with(60)
    assert stock.weight == 0
