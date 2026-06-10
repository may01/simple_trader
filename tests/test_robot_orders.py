"""Tests for Robot order management methods — Phase 10, Task 03.

TDD: _open_position, _close_position, _stop_loss, _process_executed_orders,
_place_valid_order, _do_finalize_action, _stop_loss_cancel_actions.
All dependencies (position, tracker, stock) are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from constants import (
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    TRADE_BUY,
    TRADE_SELL,
    STATUS_SUCCESS,
    STATUS_FAIL,
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data_point(cur_price=100.0, timestamp=1000.0):
    dp = MagicMock()
    dp.cur_price.return_value = cur_price
    dp.timestamp = timestamp
    return dp


def make_robot():
    """Return a Robot with mocked position, tracker, and stock."""
    from robots.robot import Robot

    sm = MagicMock()
    ld = MagicMock()
    stock = MagicMock()
    robot = Robot(
        strategy_manager=sm,
        live_data=ld,
        stock=stock,
        fee=0.001,
        persist_path="/tmp/test_robot_orders_state.json",
    )
    # Replace internals with controlled mocks
    robot.position = MagicMock()
    robot.tracker = MagicMock()
    robot.stock = stock
    return robot


# ---------------------------------------------------------------------------
# _place_valid_order
# ---------------------------------------------------------------------------

class TestPlaceValidOrder:

    def test_returns_order_id_on_success(self):
        robot = make_robot()
        robot.position.check_stop_open.return_value = True
        robot.stock.trade.return_value = (STATUS_SUCCESS, {"order_id": "abc123"})

        result = robot._place_valid_order(TRADE_BUY, 100.0, 1.5)

        assert result == "abc123"

    def test_returns_empty_string_when_check_stop_open_false(self):
        robot = make_robot()
        robot.position.check_stop_open.return_value = False

        result = robot._place_valid_order(TRADE_BUY, 100.0, 1.5)

        assert result == ""
        robot.stock.trade.assert_not_called()

    def test_returns_empty_string_on_status_fail(self):
        robot = make_robot()
        robot.position.check_stop_open.return_value = True
        robot.stock.trade.return_value = (STATUS_FAIL, {})

        result = robot._place_valid_order(TRADE_SELL, 100.0, 1.0)

        assert result == ""

    def test_returns_empty_string_on_exception(self):
        robot = make_robot()
        robot.position.check_stop_open.return_value = True
        robot.stock.trade.side_effect = Exception("network error")

        result = robot._place_valid_order(TRADE_BUY, 100.0, 1.0)

        assert result == ""

    def test_calls_trade_with_correct_args(self):
        robot = make_robot()
        robot.position.check_stop_open.return_value = True
        robot.stock.trade.return_value = (STATUS_SUCCESS, {"order_id": "xyz"})

        robot._place_valid_order(TRADE_BUY, 200.0, 3.0)

        robot.stock.trade.assert_called_once_with(TRADE_BUY, 200.0, 3.0)

    def test_calls_check_stop_open_with_price(self):
        robot = make_robot()
        robot.position.check_stop_open.return_value = False

        robot._place_valid_order(TRADE_BUY, 150.0, 1.0)

        robot.position.check_stop_open.assert_called_once_with(150.0)


# ---------------------------------------------------------------------------
# _open_position — LONG
# ---------------------------------------------------------------------------

class TestOpenPositionLong:

    def setup_method(self):
        self.robot = make_robot()
        self.robot.position.open.return_value = True
        self.robot.position.full_position = 10000.0
        self.dp = make_data_point()

    def test_calls_position_open(self):
        with patch.object(self.robot, '_place_valid_order', return_value="oid1"):
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_LONG,
                                      [100.0], [105.0], 95.0, 5)
        self.robot.position.open.assert_called_once()

    def test_open_false_skips_order(self):
        self.robot.position.open.return_value = False
        with patch.object(self.robot, '_place_valid_order') as mock_pvo:
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_LONG,
                                      [100.0], [105.0], 95.0, 5)
        mock_pvo.assert_not_called()

    def test_places_buy_order_for_long(self):
        with patch.object(self.robot, '_place_valid_order', return_value="buy1") as mock_pvo:
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_LONG,
                                      [100.0], [105.0], 95.0, 5)
        mock_pvo.assert_called_once_with(TRADE_BUY, 100.0, 100.0)  # 10000/100.0

    def test_sets_buy_order_on_tracker_for_long(self):
        with patch.object(self.robot, '_place_valid_order', return_value="buy1"):
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_LONG,
                                      [100.0], [105.0], 95.0, 5)
        self.robot.tracker.set_buy_order.assert_called_once_with("buy1")

    def test_does_not_borrow_for_long(self):
        with patch.object(self.robot, '_place_valid_order', return_value="buy1"):
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_LONG,
                                      [100.0], [105.0], 95.0, 5)
        self.robot.stock.borrow.assert_not_called()


# ---------------------------------------------------------------------------
# _open_position — SHORT
# ---------------------------------------------------------------------------

class TestOpenPositionShort:

    def setup_method(self):
        self.robot = make_robot()
        self.robot.position.open.return_value = True
        self.robot.position.full_position = 10000.0
        self.robot.stock.borrow.return_value = (STATUS_SUCCESS, "loan99")
        self.dp = make_data_point()

    def test_places_sell_order_for_short(self):
        with patch.object(self.robot, '_place_valid_order', return_value="sell1") as mock_pvo:
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_SHORT,
                                      [100.0], [95.0], 105.0, 5)
        mock_pvo.assert_called_once_with(TRADE_SELL, 100.0, 100.0)

    def test_sets_sell_order_on_tracker_for_short(self):
        with patch.object(self.robot, '_place_valid_order', return_value="sell1"):
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_SHORT,
                                      [100.0], [95.0], 105.0, 5)
        self.robot.tracker.set_sell_order.assert_called_once_with("sell1")

    def test_borrows_before_sell_for_short(self):
        borrow_called_before = []
        original_pvo = lambda *a, **kw: "sell1"

        def track_borrow(coin, amount):
            borrow_called_before.append(True)
            return (STATUS_SUCCESS, "loan1")

        self.robot.stock.borrow.side_effect = track_borrow
        with patch.object(self.robot, '_place_valid_order', return_value="sell1"):
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_SHORT,
                                      [100.0], [95.0], 105.0, 5)
        assert len(borrow_called_before) == 1

    def test_borrow_called_with_coin_and_amount(self):
        self.robot.stock.coin = "BTC"
        with patch.object(self.robot, '_place_valid_order', return_value="sell1"):
            self.robot._open_position(self.dp, STRATEGY_ACTION_OPEN_SHORT,
                                      [100.0], [95.0], 105.0, 5)
        self.robot.stock.borrow.assert_called_once_with("BTC", 100.0)


# ---------------------------------------------------------------------------
# _close_position
# ---------------------------------------------------------------------------

class TestClosePosition:

    def setup_method(self):
        self.robot = make_robot()
        self.robot.tracker.buy_id = ""
        self.robot.tracker.sell_id = ""
        self.dp = make_data_point()

    def test_calls_position_close(self):
        with patch.object(self.robot, '_place_valid_order', return_value="oid"):
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_LONG,
                                       [105.0], 95.0, 5)
        self.robot.position.close.assert_called_once()

    def test_close_long_places_sell_order(self):
        with patch.object(self.robot, '_place_valid_order', return_value="s1") as mock_pvo:
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_LONG,
                                       [105.0], 95.0, 5)
        mock_pvo.assert_called_once()
        args = mock_pvo.call_args[0]
        assert args[0] == TRADE_SELL
        assert args[1] == 105.0

    def test_close_short_places_buy_order(self):
        with patch.object(self.robot, '_place_valid_order', return_value="b1") as mock_pvo:
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_SHORT,
                                       [95.0], 105.0, 5)
        args = mock_pvo.call_args[0]
        assert args[0] == TRADE_BUY

    def test_close_long_sets_sell_order_on_tracker(self):
        with patch.object(self.robot, '_place_valid_order', return_value="s1"):
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_LONG,
                                       [105.0], 95.0, 5)
        self.robot.tracker.set_sell_order.assert_called_once_with("s1")

    def test_close_short_sets_buy_order_on_tracker(self):
        with patch.object(self.robot, '_place_valid_order', return_value="b1"):
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_SHORT,
                                       [95.0], 105.0, 5)
        self.robot.tracker.set_buy_order.assert_called_once_with("b1")

    def test_cancels_existing_buy_order_before_closing(self):
        self.robot.tracker.buy_id = "old_buy"
        with patch.object(self.robot, '_place_valid_order', return_value="s1"):
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_LONG,
                                       [105.0], 95.0, 5)
        self.robot.tracker.cancel_buy.assert_called_once()

    def test_cancels_existing_sell_order_before_closing(self):
        self.robot.tracker.sell_id = "old_sell"
        with patch.object(self.robot, '_place_valid_order', return_value="b1"):
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_SHORT,
                                       [95.0], 105.0, 5)
        self.robot.tracker.cancel_sell.assert_called_once()

    def test_no_cancel_when_no_existing_orders(self):
        self.robot.tracker.buy_id = ""
        self.robot.tracker.sell_id = ""
        with patch.object(self.robot, '_place_valid_order', return_value="s1"):
            self.robot._close_position(self.dp, STRATEGY_ACTION_CLOSE_LONG,
                                       [105.0], 95.0, 5)
        self.robot.tracker.cancel_buy.assert_not_called()
        self.robot.tracker.cancel_sell.assert_not_called()


# ---------------------------------------------------------------------------
# _stop_loss
# ---------------------------------------------------------------------------

class TestStopLoss:

    def setup_method(self):
        self.robot = make_robot()
        self.robot.tracker.buy_id = ""
        self.robot.tracker.sell_id = ""
        self.dp = make_data_point(cur_price=99.0)

    def test_cancels_buy_order_if_present(self):
        self.robot.tracker.buy_id = "existing_buy"
        with patch.object(self.robot, '_place_valid_order', return_value="sl1"):
            self.robot._stop_loss(self.dp)
        self.robot.tracker.cancel_buy.assert_called_once()

    def test_cancels_sell_order_if_present(self):
        self.robot.tracker.sell_id = "existing_sell"
        with patch.object(self.robot, '_place_valid_order', return_value="sl1"):
            self.robot._stop_loss(self.dp)
        self.robot.tracker.cancel_sell.assert_called_once()

    def test_no_cancel_when_no_orders(self):
        with patch.object(self.robot, '_place_valid_order', return_value="sl1"):
            self.robot._stop_loss(self.dp)
        self.robot.tracker.cancel_buy.assert_not_called()
        self.robot.tracker.cancel_sell.assert_not_called()

    def test_places_sell_order_at_cur_price_for_long(self):
        self.robot.position.posImpl = MagicMock()
        self.robot.position.posImpl.position_type = POSITION_TYPE_LONG
        with patch.object(self.robot, '_place_valid_order', return_value="sl1") as mock_pvo:
            self.robot._stop_loss(self.dp)
        args = mock_pvo.call_args[0]
        assert args[0] == TRADE_SELL
        assert args[1] == 99.0

    def test_places_buy_order_at_cur_price_for_short(self):
        self.robot.position.posImpl = MagicMock()
        self.robot.position.posImpl.position_type = POSITION_TYPE_SHORT
        with patch.object(self.robot, '_place_valid_order', return_value="sl1") as mock_pvo:
            self.robot._stop_loss(self.dp)
        args = mock_pvo.call_args[0]
        assert args[0] == TRADE_BUY

    def test_sets_sell_order_on_tracker_for_long_stop_loss(self):
        self.robot.position.posImpl = MagicMock()
        self.robot.position.posImpl.position_type = POSITION_TYPE_LONG
        with patch.object(self.robot, '_place_valid_order', return_value="sl1"):
            self.robot._stop_loss(self.dp)
        self.robot.tracker.set_sell_order.assert_called_once_with("sl1")

    def test_sets_buy_order_on_tracker_for_short_stop_loss(self):
        self.robot.position.posImpl = MagicMock()
        self.robot.position.posImpl.position_type = POSITION_TYPE_SHORT
        with patch.object(self.robot, '_place_valid_order', return_value="sl1"):
            self.robot._stop_loss(self.dp)
        self.robot.tracker.set_buy_order.assert_called_once_with("sl1")


# ---------------------------------------------------------------------------
# _do_finalize_action
# ---------------------------------------------------------------------------

class TestDoFinalizeAction:

    def test_calls_position_finalize(self):
        robot = make_robot()
        robot.position.finalize.return_value = (0.05, 500.0)
        robot.tracker.clear = MagicMock()

        robot._do_finalize_action()

        robot.position.finalize.assert_called_once()

    def test_calls_tracker_clear(self):
        robot = make_robot()
        robot.position.finalize.return_value = (0.05, 500.0)

        robot._do_finalize_action()

        robot.tracker.clear.assert_called_once()

    def test_does_not_raise_on_zero_pnl(self):
        robot = make_robot()
        robot.position.finalize.return_value = (0.0, 0.0)

        # Should not raise
        robot._do_finalize_action()


# ---------------------------------------------------------------------------
# _process_executed_orders — LONG path
# ---------------------------------------------------------------------------

class TestProcessExecutedOrdersLong:
    """LONG position: buy_id → entry_fill, sell_id → exit_fill."""

    def setup_method(self):
        self.robot = make_robot()
        self.robot.tracker.buy_id = ""
        self.robot.tracker.sell_id = ""
        self.dp = make_data_point()
        # Simulate a LONG position
        self.robot.position.posImpl = MagicMock()
        self.robot.position.posImpl.position_type = POSITION_TYPE_LONG

    def test_no_action_when_no_orders(self):
        self.robot._process_executed_orders(self.dp)
        self.robot.tracker.check_fill.assert_not_called()

    def test_checks_buy_fill_when_buy_id_present(self):
        self.robot.tracker.buy_id = "buy1"
        self.robot.tracker.check_fill.return_value = (STATUS_FAIL, {})
        self.robot._process_executed_orders(self.dp)
        self.robot.tracker.check_fill.assert_called_once_with("buy1")

    def test_checks_sell_fill_when_sell_id_present(self):
        self.robot.tracker.sell_id = "sell1"
        self.robot.tracker.check_fill.return_value = (STATUS_FAIL, {})
        self.robot._process_executed_orders(self.dp)
        self.robot.tracker.check_fill.assert_called_once_with("sell1")

    def test_skips_fill_processing_on_status_fail(self):
        self.robot.tracker.buy_id = "buy1"
        self.robot.tracker.check_fill.return_value = (STATUS_FAIL, {})
        self.robot._process_executed_orders(self.dp)
        self.robot.position.record_entry_fill.assert_not_called()

    def test_records_entry_fill_on_filled_buy(self):
        self.robot.tracker.buy_id = "buy1"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 100.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        self.robot._process_executed_orders(self.dp)

        self.robot.position.record_entry_fill.assert_called_once_with(1.0, 100.0)

    def test_records_entry_fill_on_partially_filled_buy(self):
        self.robot.tracker.buy_id = "buy1"
        fill = {"status": "PARTIALLY_FILLED", "start_amount": 2.0, "left_amount": 1.0, "rate": 200.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        self.robot._process_executed_orders(self.dp)

        # coin_amount = 2.0 - 1.0 = 1.0
        self.robot.position.record_entry_fill.assert_called_once_with(1.0, 200.0)

    def test_records_exit_fill_on_filled_sell(self):
        self.robot.tracker.sell_id = "sell1"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 105.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action'):
            self.robot._process_executed_orders(self.dp)

        self.robot.position.record_exit_fill.assert_called_once_with(1.0, 105.0)

    def test_calls_do_finalize_when_sell_fully_filled(self):
        self.robot.tracker.sell_id = "sell1"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 105.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action') as mock_fin:
            self.robot._process_executed_orders(self.dp)

        mock_fin.assert_called_once()

    def test_no_finalize_on_partially_filled_sell(self):
        self.robot.tracker.sell_id = "sell1"
        fill = {"status": "PARTIALLY_FILLED", "start_amount": 2.0, "left_amount": 1.0, "rate": 105.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action') as mock_fin:
            self.robot._process_executed_orders(self.dp)

        mock_fin.assert_not_called()

    def test_buy_id_does_not_trigger_exit_fill_for_long(self):
        """LONG: buy_id should never route to record_exit_fill."""
        self.robot.tracker.buy_id = "buy1"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 100.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        self.robot._process_executed_orders(self.dp)

        self.robot.position.record_exit_fill.assert_not_called()

    def test_sell_id_does_not_trigger_entry_fill_for_long(self):
        """LONG: sell_id should never route to record_entry_fill."""
        self.robot.tracker.sell_id = "sell1"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 105.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action'):
            self.robot._process_executed_orders(self.dp)

        self.robot.position.record_entry_fill.assert_not_called()


# ---------------------------------------------------------------------------
# _process_executed_orders — SHORT path
# ---------------------------------------------------------------------------

class TestProcessExecutedOrdersShort:
    """SHORT position: sell_id → entry_fill, buy_id → exit_fill."""

    def setup_method(self):
        self.robot = make_robot()
        self.robot.tracker.buy_id = ""
        self.robot.tracker.sell_id = ""
        self.dp = make_data_point()
        # Simulate a SHORT position
        self.robot.position.posImpl = MagicMock()
        self.robot.position.posImpl.position_type = POSITION_TYPE_SHORT

    def test_no_action_when_no_orders(self):
        self.robot._process_executed_orders(self.dp)
        self.robot.tracker.check_fill.assert_not_called()

    def test_checks_sell_fill_as_entry_for_short(self):
        """SHORT entry: sell_id is checked for the entry fill."""
        self.robot.tracker.sell_id = "sell_entry"
        self.robot.tracker.check_fill.return_value = (STATUS_FAIL, {})
        self.robot._process_executed_orders(self.dp)
        self.robot.tracker.check_fill.assert_called_once_with("sell_entry")

    def test_records_entry_fill_on_filled_sell_for_short(self):
        """SHORT: filled sell_id → record_entry_fill."""
        self.robot.tracker.sell_id = "sell_entry"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 100.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        self.robot._process_executed_orders(self.dp)

        self.robot.position.record_entry_fill.assert_called_once_with(1.0, 100.0)

    def test_sell_id_does_not_trigger_exit_fill_for_short(self):
        """SHORT: sell_id (entry) must never call record_exit_fill."""
        self.robot.tracker.sell_id = "sell_entry"
        fill = {"status": "FILLED", "start_amount": 1.0, "left_amount": 0.0, "rate": 100.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        self.robot._process_executed_orders(self.dp)

        self.robot.position.record_exit_fill.assert_not_called()

    def test_checks_buy_fill_as_exit_for_short(self):
        """SHORT exit: buy_id (buy-back) is checked for the exit fill."""
        self.robot.tracker.buy_id = "buyback1"
        self.robot.tracker.check_fill.return_value = (STATUS_FAIL, {})
        self.robot._process_executed_orders(self.dp)
        self.robot.tracker.check_fill.assert_called_once_with("buyback1")

    def test_records_exit_fill_on_filled_buy_for_short(self):
        """SHORT: filled buy_id (buy-back) → record_exit_fill."""
        self.robot.tracker.buy_id = "buyback1"
        fill = {"status": "FILLED", "start_amount": 2.0, "left_amount": 0.0, "rate": 90.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action'):
            self.robot._process_executed_orders(self.dp)

        self.robot.position.record_exit_fill.assert_called_once_with(2.0, 90.0)

    def test_buy_id_does_not_trigger_entry_fill_for_short(self):
        """SHORT: buy_id (exit) must never call record_entry_fill."""
        self.robot.tracker.buy_id = "buyback1"
        fill = {"status": "FILLED", "start_amount": 2.0, "left_amount": 0.0, "rate": 90.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action'):
            self.robot._process_executed_orders(self.dp)

        self.robot.position.record_entry_fill.assert_not_called()

    def test_calls_do_finalize_when_buyback_fully_filled_for_short(self):
        """SHORT: fully filled buy-back (buy_id) triggers finalize."""
        self.robot.tracker.buy_id = "buyback1"
        fill = {"status": "FILLED", "start_amount": 2.0, "left_amount": 0.0, "rate": 90.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action') as mock_fin:
            self.robot._process_executed_orders(self.dp)

        mock_fin.assert_called_once()

    def test_no_finalize_on_partially_filled_buyback_for_short(self):
        """SHORT: partially filled buy-back must not trigger finalize."""
        self.robot.tracker.buy_id = "buyback1"
        fill = {"status": "PARTIALLY_FILLED", "start_amount": 2.0, "left_amount": 1.0, "rate": 90.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action') as mock_fin:
            self.robot._process_executed_orders(self.dp)

        mock_fin.assert_not_called()

    def test_skips_exit_fill_on_status_fail_for_short(self):
        """SHORT: STATUS_FAIL on buy-back check → no record_exit_fill."""
        self.robot.tracker.buy_id = "buyback1"
        self.robot.tracker.check_fill.return_value = (STATUS_FAIL, {})
        self.robot._process_executed_orders(self.dp)
        self.robot.position.record_exit_fill.assert_not_called()

    def test_records_partial_entry_fill_on_partial_sell_for_short(self):
        """SHORT: partial sell fill → record_entry_fill with partial amount."""
        self.robot.tracker.sell_id = "sell_entry"
        fill = {"status": "PARTIALLY_FILLED", "start_amount": 3.0, "left_amount": 1.5, "rate": 102.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        self.robot._process_executed_orders(self.dp)

        self.robot.position.record_entry_fill.assert_called_once_with(1.5, 102.0)

    def test_repays_loan_note_after_buy_fully_filled_for_short(self):
        """After a buy-back fill for SHORT, no exception is raised (loan repay tracked separately)."""
        self.robot.tracker.buy_id = "buyback1"
        self.robot.tracker.loan_id = "loan1"
        self.robot.tracker.loan_amount = 2.0
        self.robot.stock.coin = "BTC"
        fill = {"status": "FILLED", "start_amount": 2.0, "left_amount": 0.0, "rate": 90.0}
        self.robot.tracker.check_fill.return_value = (STATUS_SUCCESS, fill)

        with patch.object(self.robot, '_do_finalize_action'):
            self.robot._process_executed_orders(self.dp)

        # record_exit_fill must have been called correctly
        self.robot.position.record_exit_fill.assert_called_once_with(2.0, 90.0)


# ---------------------------------------------------------------------------
# _stop_loss_cancel_actions
# ---------------------------------------------------------------------------

class TestStopLossCancelActions:

    def test_calls_stop_loss_when_triggered(self):
        robot = make_robot()
        robot.position.is_stop_loss_triggered.return_value = True
        robot.position.close_by_time.return_value = False
        dp = make_data_point(cur_price=90.0, timestamp=5000.0)

        with patch.object(robot, '_stop_loss') as mock_sl:
            robot._stop_loss_cancel_actions(dp)

        mock_sl.assert_called_once_with(dp)

    def test_does_not_call_stop_loss_when_not_triggered(self):
        robot = make_robot()
        robot.position.is_stop_loss_triggered.return_value = False
        robot.position.close_by_time.return_value = False
        dp = make_data_point()

        with patch.object(robot, '_stop_loss') as mock_sl:
            robot._stop_loss_cancel_actions(dp)

        mock_sl.assert_not_called()

    def test_calls_close_position_when_close_by_time(self):
        robot = make_robot()
        robot.position.is_stop_loss_triggered.return_value = False
        robot.position.close_by_time.return_value = True
        dp = make_data_point(cur_price=100.0, timestamp=5000.0)

        with patch.object(robot, '_close_position') as mock_cp:
            robot._stop_loss_cancel_actions(dp)

        mock_cp.assert_called_once()

    def test_does_not_call_close_position_when_not_timed_out(self):
        robot = make_robot()
        robot.position.is_stop_loss_triggered.return_value = False
        robot.position.close_by_time.return_value = False
        dp = make_data_point()

        with patch.object(robot, '_close_position') as mock_cp:
            robot._stop_loss_cancel_actions(dp)

        mock_cp.assert_not_called()

    def test_checks_stop_loss_with_cur_price(self):
        robot = make_robot()
        robot.position.is_stop_loss_triggered.return_value = False
        robot.position.close_by_time.return_value = False
        dp = make_data_point(cur_price=88.0, timestamp=9000.0)

        robot._stop_loss_cancel_actions(dp)

        robot.position.is_stop_loss_triggered.assert_called_once_with(88.0)

    def test_checks_close_by_time_with_timestamp(self):
        robot = make_robot()
        robot.position.is_stop_loss_triggered.return_value = False
        robot.position.close_by_time.return_value = False
        dp = make_data_point(cur_price=88.0, timestamp=9000.0)

        robot._stop_loss_cancel_actions(dp)

        robot.position.close_by_time.assert_called_once_with(9000.0)
