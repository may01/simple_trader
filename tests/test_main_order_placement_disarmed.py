"""main/ places no orders under the default configuration.

The acceptance criterion from the position-management design: a full
open -> close robot tick cycle records zero trade/borrow/cancel_order
calls on the real stock, and MAIN_ORDER_PLACEMENT=enabled restores every
one of them.
"""

import json
import logging
import os

import pytest

from constants import STATUS_FAIL, STATUS_SUCCESS, TRADE_BUY, TRADE_SELL
from stocks.disarmed_stock import DisarmedStock
from trader import select_stock


class CountingStock:
    """Counts write calls. Deliberately not a StockInterface subclass --
    see DisarmedStock's own docstring for why that matters."""

    def __init__(self):
        self.writes: list[str] = []
        self.fee = 0.001
        self.coin = "LINK"
        self.coin_base = "USDT"

    def trade(self, trade_type, price, amount, force=False):
        self.writes.append("trade")
        return (STATUS_SUCCESS, {"order_id": "x1"})

    def borrow(self, coin, amount):
        self.writes.append("borrow")
        return (STATUS_SUCCESS, amount)

    def repay(self, coin, amount):
        self.writes.append("repay")
        return (STATUS_SUCCESS, amount)

    def cancel_order(self, order_id):
        self.writes.append("cancel_order")
        return (STATUS_SUCCESS, {})


# ---------------------------------------------------------------------
# select_stock
# ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", [None, "", "disabled", "DISABLED", " disabled "])
def test_absent_or_disabled_disarms(mode):
    inner = CountingStock()
    assert isinstance(select_stock(inner, mode), DisarmedStock)


def test_only_the_exact_word_enabled_arms():
    inner = CountingStock()
    assert select_stock(inner, "enabled") is inner
    assert select_stock(inner, "ENABLED") is inner
    assert select_stock(inner, " enabled ") is inner


@pytest.mark.parametrize("typo", ["enable", "on", "true", "yes", "1", "enabledd"])
def test_a_typo_does_not_arm_anything(typo, caplog):
    """The failure mode this guards: `MAIN_ORDER_PLACEMENT=on` silently
    arming the legacy order path against a live executor."""
    inner = CountingStock()
    with caplog.at_level(logging.WARNING):
        selected = select_stock(inner, typo)
    assert isinstance(selected, DisarmedStock)
    assert any("not recognised" in r.message for r in caplog.records)


def test_arming_warns_about_the_double_trade_hazard(caplog):
    inner = CountingStock()
    with caplog.at_level(logging.WARNING):
        select_stock(inner, "enabled")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "double-trade" in joined or "double-trades" in joined


def test_disarming_says_who_places_orders_instead(caplog):
    inner = CountingStock()
    with caplog.at_level(logging.INFO):
        select_stock(inner, None)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "trade_executor" in joined
    assert "MAIN_ORDER_PLACEMENT=enabled" in joined


# ---------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------

def _exercise_every_write(stock):
    """Every write call robot.py can make, through whichever stock it was
    handed."""
    stock.trade(TRADE_BUY, 15.0, 1.0)
    stock.trade(TRADE_SELL, 15.0, 1.0, force=True)
    stock.borrow("LINK", 1.0)
    stock.repay("LINK", 1.0)
    stock.cancel_order("x1")


def test_disarmed_main_places_no_orders():
    inner = CountingStock()
    _exercise_every_write(select_stock(inner, None))
    assert inner.writes == [], "main/ must place no orders under the default config"


def test_arming_restores_every_write():
    inner = CountingStock()
    _exercise_every_write(select_stock(inner, "enabled"))
    assert inner.writes == ["trade", "trade", "borrow", "repay", "cancel_order"]


def test_refused_calls_return_the_interfaces_own_failure_shape():
    """Callers already branch on STATUS_FAIL; a refusal must look like a
    failure to them, not raise."""
    disarmed = select_stock(CountingStock(), None)
    for result in (
        disarmed.trade(TRADE_BUY, 15.0, 1.0),
        disarmed.borrow("LINK", 1.0),
        disarmed.repay("LINK", 1.0),
        disarmed.cancel_order("x1"),
    ):
        status, _payload = result
        assert status == STATUS_FAIL


# ---------------------------------------------------------------------
# Leftover tracker state
# ---------------------------------------------------------------------

def test_leftover_tracker_state_is_reported_and_not_acted_on(tmp_path, caplog):
    from position.position import Position
    from robots.live_order_tracker import LiveOrderTracker

    persist = tmp_path / "live_order_tracker.json"
    position = Position(thread_num=0, fee=0.001)
    persist.write_text(json.dumps({
        **position.to_dict(),
        "buy_id": "leftover-buy-1",
        "sell_id": "",
        "loan_id": "loan-9",
        "loan_amount": 12.5,
    }))

    stock = CountingStock()
    tracker = LiveOrderTracker(position, stock, str(persist))
    with caplog.at_level(logging.WARNING):
        assert tracker.load() is True

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "leftover-buy-1" in joined
    assert "loan-9" in joined
    assert "12.5" in joined
    assert stock.writes == [], "stale state must be reported, never acted on"
    assert persist.exists(), "the file is for an operator to settle, not for us to delete"


def test_absent_tracker_file_is_silent(tmp_path, caplog):
    from position.position import Position
    from robots.live_order_tracker import LiveOrderTracker

    tracker = LiveOrderTracker(
        Position(thread_num=0, fee=0.001), CountingStock(), str(tmp_path / "nope.json")
    )
    with caplog.at_level(logging.WARNING):
        assert tracker.load() is False
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
