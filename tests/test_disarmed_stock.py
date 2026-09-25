"""Tests for stocks/disarmed_stock.py — main/ must place no orders.

The property under test is not "returns a failure". It is "the wrapped
stock was never called": a refusal that still reached the exchange would
pass a returns-a-failure test and double-trade in production.
"""

import logging

import pytest

from constants import STATUS_FAIL, STATUS_SUCCESS, TRADE_BUY, TRADE_SELL
from stocks.base_stock import StockInterface
from stocks.disarmed_stock import DisarmedStock, is_disarmed_result


class SpyStock(StockInterface):
    """Records every call, so a test can assert one never happened."""

    def __init__(self):
        super().__init__(key="k", secret="s", coin="LINK", coin_base="USDT")
        self.calls: list[tuple] = []
        self.fee = 0.001

    # --- write path (must never be reached through DisarmedStock) ---
    def trade(self, trade_type, price, amount, force=False):
        self.calls.append(("trade", trade_type, price, amount, force))
        return (STATUS_SUCCESS, {"order_id": "real-1"})

    def borrow(self, coin, amount):
        self.calls.append(("borrow", coin, amount))
        return (STATUS_SUCCESS, amount)

    def repay(self, coin, amount):
        self.calls.append(("repay", coin, amount))
        return (STATUS_SUCCESS, amount)

    def cancel_order(self, order_id):
        self.calls.append(("cancel_order", order_id))
        return (STATUS_SUCCESS, {})

    # --- read path (must pass through unchanged) ---
    def order_info(self, order_id):
        self.calls.append(("order_info", order_id))
        return (STATUS_SUCCESS, {"status": "FILLED"})

    def funds(self, coin, asset_type="free"):
        self.calls.append(("funds", coin, asset_type))
        return (STATUS_SUCCESS, 123.45)

    def depth(self, quantity):
        self.calls.append(("depth", quantity))
        return (STATUS_SUCCESS, {"bids": [], "asks": []})

    def get_pair_name(self):
        self.calls.append(("get_pair_name",))
        return "LINKUSDT"

    def get_aviable_loan(self, coin):
        self.calls.append(("get_aviable_loan", coin))
        return (STATUS_SUCCESS, 10.0)


@pytest.fixture
def spy_and_disarmed():
    spy = SpyStock()
    return spy, DisarmedStock(spy)


# ---------------------------------------------------------------------
# Refused: nothing reaches the wrapped stock
# ---------------------------------------------------------------------

def test_trade_never_reaches_the_wrapped_stock(spy_and_disarmed):
    spy, disarmed = spy_and_disarmed
    status, payload = disarmed.trade(TRADE_BUY, 15.0, 2.0)
    assert status == STATUS_FAIL
    assert is_disarmed_result(payload)
    assert spy.calls == [], "a refused order must not reach the exchange"


def test_force_does_not_bypass_the_refusal(spy_and_disarmed):
    """`force=True` is the flag most likely to be reached for in an
    emergency. It must not arm a disarmed stock."""
    spy, disarmed = spy_and_disarmed
    status, payload = disarmed.trade(TRADE_SELL, 15.0, 2.0, force=True)
    assert status == STATUS_FAIL
    assert is_disarmed_result(payload)
    assert spy.calls == []


def test_borrow_repay_and_cancel_never_reach_the_wrapped_stock(spy_and_disarmed):
    spy, disarmed = spy_and_disarmed
    assert disarmed.borrow("LINK", 5.0)[0] == STATUS_FAIL
    assert disarmed.repay("LINK", 5.0)[0] == STATUS_FAIL
    assert disarmed.cancel_order("abc")[0] == STATUS_FAIL
    assert spy.calls == []


def test_a_refusal_is_distinguishable_from_a_genuine_failure(spy_and_disarmed):
    """Both return STATUS_FAIL. Only one deserves an ERROR in the log, so
    the caller has to be able to tell them apart."""
    _spy, disarmed = spy_and_disarmed
    _status, refused = disarmed.trade(TRADE_BUY, 15.0, 2.0)
    assert is_disarmed_result(refused)
    # A real exchange failure carries no such marker.
    assert not is_disarmed_result({})
    assert not is_disarmed_result({"error": "insufficient balance"})
    assert not is_disarmed_result(None)


def test_refusals_log_at_debug_not_error(spy_and_disarmed, caplog):
    _spy, disarmed = spy_and_disarmed
    with caplog.at_level(logging.DEBUG):
        disarmed.trade(TRADE_BUY, 15.0, 2.0)
        disarmed.borrow("LINK", 1.0)
        disarmed.cancel_order("abc")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], \
        "a refusal is a normal outcome under the default config, not a warning"
    assert any(r.levelno == logging.DEBUG for r in caplog.records)


# ---------------------------------------------------------------------
# Reads pass through
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,args,expected",
    [
        ("order_info", ("abc",), (STATUS_SUCCESS, {"status": "FILLED"})),
        ("funds", ("USDT",), (STATUS_SUCCESS, 123.45)),
        ("depth", (10,), (STATUS_SUCCESS, {"bids": [], "asks": []})),
        ("get_pair_name", (), "LINKUSDT"),
        ("get_aviable_loan", ("LINK",), (STATUS_SUCCESS, 10.0)),
    ],
)
def test_reads_delegate_and_return_the_inner_value(spy_and_disarmed, name, args, expected):
    """The regression this guards: DisarmedStock used to subclass
    StockInterface, whose concrete no-ops shadowed __getattr__, so every
    read returned (STATUS_FAIL, {}) and printed "INIT STOCK" instead of
    reaching the real stock. Disarming the write path must not blind the
    read path."""
    spy, disarmed = spy_and_disarmed
    assert getattr(disarmed, name)(*args) == expected
    assert spy.calls and spy.calls[0][0] == name, "the call must reach the wrapped stock"


def test_attributes_delegate_too(spy_and_disarmed):
    spy, disarmed = spy_and_disarmed
    assert disarmed.fee == spy.fee
    assert disarmed.coin == "LINK"
    assert disarmed.coin_base == "USDT"


def test_every_write_method_on_the_interface_is_explicitly_refused():
    """Guards against a write method being added to StockInterface and
    silently delegating through `__getattr__`.

    If this fails, the new method needs an explicit refusal above — or, if
    it is genuinely a read, adding to the allowlist here with a note.
    """
    write_methods = {"trade", "borrow", "repay", "cancel_order"}
    for name in write_methods:
        assert name in DisarmedStock.__dict__, f"{name} must be refused explicitly"
    assert not issubclass(DisarmedStock, StockInterface), (
        "DisarmedStock must not inherit StockInterface: the base defines a "
        "concrete no-op for every method, which would shadow __getattr__ and "
        "blind every read"
    )

    known_reads = {
        "get_candles_history", "get_candles_range", "order_info", "depth", "info",
        "is_invalid_amount", "funds", "get_aviable_loan", "get_pair_name",
        "set_operation_sleep", "_resample_to_tf",
    }
    declared = {
        name for name, attr in vars(StockInterface).items()
        if callable(attr) and not name.startswith("__")
    }
    unclassified = declared - write_methods - known_reads
    assert not unclassified, (
        f"StockInterface gained {sorted(unclassified)}; classify each as a read "
        "(add to known_reads) or a write (add an explicit refusal to DisarmedStock)"
    )
