"""Tests for LiveOrderTracker — TDD: write RED tests first, then implement."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call
from position.position import Position
from stocks.base_stock import StockInterface
from constants import STATUS_SUCCESS, STATUS_FAIL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tracker(tmp_path, stock=None):
    """Return a fresh LiveOrderTracker with a temp persist_path."""
    from robots.live_order_tracker import LiveOrderTracker

    if stock is None:
        stock = MagicMock(spec=StockInterface)

    pos = Position(fee=0.001)
    persist_path = str(tmp_path / "tracker_state.json")
    return LiveOrderTracker(position=pos, stock=stock, persist_path=persist_path), pos, stock, persist_path


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_ids_are_empty_strings(self, tmp_path):
        tracker, pos, stock, _ = make_tracker(tmp_path)
        assert tracker.buy_id == ""
        assert tracker.sell_id == ""
        assert tracker.loan_id == ""

    def test_default_loan_amount_is_zero(self, tmp_path):
        tracker, pos, stock, _ = make_tracker(tmp_path)
        assert tracker.loan_amount == 0.0

    def test_attributes_assigned(self, tmp_path):
        tracker, pos, stock, persist_path = make_tracker(tmp_path)
        assert tracker.position is pos
        assert tracker.stock is stock
        assert tracker.persist_path == persist_path

    def test_ids_are_strings_not_none(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        assert tracker.buy_id is not None
        assert tracker.sell_id is not None
        assert tracker.loan_id is not None


# ---------------------------------------------------------------------------
# set_buy_order / set_sell_order
# ---------------------------------------------------------------------------

class TestSetOrders:
    def test_set_buy_order_updates_buy_id(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_buy_order("order-123")
        assert tracker.buy_id == "order-123"

    def test_set_buy_order_calls_save(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_buy_order("order-123")
        assert os.path.exists(persist_path)

    def test_set_sell_order_updates_sell_id(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_sell_order("sell-456")
        assert tracker.sell_id == "sell-456"

    def test_set_sell_order_calls_save(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_sell_order("sell-456")
        assert os.path.exists(persist_path)


# ---------------------------------------------------------------------------
# set_loan / get_loan
# ---------------------------------------------------------------------------

class TestLoan:
    def test_set_loan_sets_loan_id_and_amount(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_loan("loan-789", 500.0)
        assert tracker.loan_id == "loan-789"
        assert tracker.loan_amount == 500.0

    def test_set_loan_calls_save(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_loan("loan-789", 500.0)
        assert os.path.exists(persist_path)

    def test_get_loan_returns_tuple(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_loan("loan-abc", 250.5)
        result = tracker.get_loan()
        assert result == ("loan-abc", 250.5)

    def test_get_loan_default_state(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        assert tracker.get_loan() == ("", 0.0)


# ---------------------------------------------------------------------------
# cancel_buy / cancel_sell
# ---------------------------------------------------------------------------

class TestCancel:
    def test_cancel_buy_calls_stock_cancel_order(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        stock.cancel_order.return_value = (STATUS_SUCCESS, {})
        tracker, _, _, _ = make_tracker(tmp_path, stock=stock)
        tracker.set_buy_order("order-buy-1")
        tracker.cancel_buy()
        stock.cancel_order.assert_called_once_with("order-buy-1")

    def test_cancel_buy_clears_buy_id(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        stock.cancel_order.return_value = (STATUS_SUCCESS, {})
        tracker, _, _, _ = make_tracker(tmp_path, stock=stock)
        tracker.set_buy_order("order-buy-1")
        tracker.cancel_buy()
        assert tracker.buy_id == ""

    def test_cancel_buy_saves_after_clear(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        stock.cancel_order.return_value = (STATUS_SUCCESS, {})
        tracker, _, _, persist_path = make_tracker(tmp_path, stock=stock)
        tracker.set_buy_order("order-buy-1")
        tracker.cancel_buy()
        # Verify the saved file shows empty buy_id
        with open(persist_path) as f:
            data = json.load(f)
        assert data["buy_id"] == ""

    def test_cancel_sell_calls_stock_cancel_order(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        stock.cancel_order.return_value = (STATUS_SUCCESS, {})
        tracker, _, _, _ = make_tracker(tmp_path, stock=stock)
        tracker.set_sell_order("order-sell-1")
        tracker.cancel_sell()
        stock.cancel_order.assert_called_once_with("order-sell-1")

    def test_cancel_sell_clears_sell_id(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        stock.cancel_order.return_value = (STATUS_SUCCESS, {})
        tracker, _, _, _ = make_tracker(tmp_path, stock=stock)
        tracker.set_sell_order("order-sell-1")
        tracker.cancel_sell()
        assert tracker.sell_id == ""


# ---------------------------------------------------------------------------
# check_fill
# ---------------------------------------------------------------------------

class TestCheckFill:
    def test_check_fill_delegates_to_stock_order_info(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        fill_dict = {"status": "FILLED", "qty": 10.0}
        stock.order_info.return_value = (STATUS_SUCCESS, fill_dict)
        tracker, _, _, _ = make_tracker(tmp_path, stock=stock)
        result = tracker.check_fill("order-xyz")
        stock.order_info.assert_called_once_with("order-xyz")
        assert result == (STATUS_SUCCESS, fill_dict)

    def test_check_fill_returns_status_and_dict(self, tmp_path):
        stock = MagicMock(spec=StockInterface)
        stock.order_info.return_value = (STATUS_FAIL, {})
        tracker, _, _, _ = make_tracker(tmp_path, stock=stock)
        status, info = tracker.check_fill("bad-order")
        assert status == STATUS_FAIL
        assert info == {}


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_buy_order("b-1")
        assert os.path.exists(persist_path)

    def test_save_writes_correct_fields(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_buy_order("b-1")
        tracker.set_sell_order("s-2")
        tracker.set_loan("l-3", 999.0)
        with open(persist_path) as f:
            data = json.load(f)
        assert data["buy_id"] == "b-1"
        assert data["sell_id"] == "s-2"
        assert data["loan_id"] == "l-3"
        assert data["loan_amount"] == 999.0

    def test_save_is_atomic_no_tmp_file_left(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_buy_order("b-1")
        tmp_path_str = persist_path + ".tmp"
        assert not os.path.exists(tmp_path_str)

    def test_load_returns_false_when_file_absent(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        result = tracker.load()
        assert result is False

    def test_load_returns_true_when_file_exists(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_buy_order("b-1")
        result = tracker.load()
        assert result is True

    def test_load_restores_buy_id(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_buy_order("b-loaded")
        tracker.buy_id = ""  # clobber in-memory state
        tracker.load()
        assert tracker.buy_id == "b-loaded"

    def test_load_restores_sell_id(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_sell_order("s-loaded")
        tracker.sell_id = ""
        tracker.load()
        assert tracker.sell_id == "s-loaded"

    def test_load_restores_loan_fields(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_loan("l-loaded", 123.45)
        tracker.loan_id = ""
        tracker.loan_amount = 0.0
        tracker.load()
        assert tracker.loan_id == "l-loaded"
        assert tracker.loan_amount == 123.45

    def test_load_calls_position_from_dict(self, tmp_path):
        """load() must call position.from_dict() with the saved data."""
        from robots.live_order_tracker import LiveOrderTracker

        stock = MagicMock(spec=StockInterface)
        pos = MagicMock(spec=Position)
        pos.to_dict.return_value = {"position_type": "POSITION_TYPE_LONG", "state": "open"}
        persist_path = str(tmp_path / "tracker_state.json")
        tracker = LiveOrderTracker(position=pos, stock=stock, persist_path=persist_path)
        tracker.set_buy_order("b-1")

        # Reset mock call history, then load
        pos.from_dict.reset_mock()
        tracker.load()
        pos.from_dict.assert_called_once()

    def test_save_includes_position_dict(self, tmp_path):
        """save() should serialize position.to_dict() into the JSON."""
        from robots.live_order_tracker import LiveOrderTracker

        stock = MagicMock(spec=StockInterface)
        pos = MagicMock(spec=Position)
        pos.to_dict.return_value = {"position_type": "POSITION_TYPE_LONG"}
        persist_path = str(tmp_path / "tracker_state.json")
        tracker = LiveOrderTracker(position=pos, stock=stock, persist_path=persist_path)
        tracker.set_buy_order("b-1")

        with open(persist_path) as f:
            data = json.load(f)
        assert "position_type" in data


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_buy_id(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_buy_order("b-1")
        tracker.clear()
        assert tracker.buy_id == ""

    def test_clear_resets_sell_id(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_sell_order("s-1")
        tracker.clear()
        assert tracker.sell_id == ""

    def test_clear_resets_loan_fields(self, tmp_path):
        tracker, _, _, _ = make_tracker(tmp_path)
        tracker.set_loan("l-1", 200.0)
        tracker.clear()
        assert tracker.loan_id == ""
        assert tracker.loan_amount == 0.0

    def test_clear_deletes_persist_file(self, tmp_path):
        tracker, _, _, persist_path = make_tracker(tmp_path)
        tracker.set_buy_order("b-1")
        assert os.path.exists(persist_path)
        tracker.clear()
        assert not os.path.exists(persist_path)

    def test_clear_safe_when_no_file(self, tmp_path):
        """clear() should not raise if persist file doesn't exist yet."""
        tracker, _, _, persist_path = make_tracker(tmp_path)
        assert not os.path.exists(persist_path)
        tracker.clear()  # should not raise
        assert tracker.buy_id == ""
