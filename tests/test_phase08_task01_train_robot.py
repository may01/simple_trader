"""Tests for TrainRobot — Phase 08, Task 01.

TDD: tests written before implementation.
All dependencies (StrategyManager, Position) are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from constants import (
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
    STRATEGY_ACTION_DO_STOP_LOSS,
    POSITION_STATE_WAIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data_point(cur_price=100.0, timestamp=1000.0):
    """Return a MagicMock DataPoint with sensible defaults."""
    dp = MagicMock()
    dp.cur_price.return_value = cur_price
    dp.timestamp = timestamp
    return dp


def make_strategy_manager(action=STRATEGY_ACTION_NOTHING,
                           open_prices=None, close_prices=None,
                           stop_price=0.0, tf=0):
    """Return a mocked StrategyManager whose check() returns the given tuple."""
    sm = MagicMock()
    sm.check.return_value = (
        action,
        open_prices if open_prices is not None else [],
        close_prices if close_prices is not None else [],
        stop_price,
        tf,
    )
    return sm


def make_position(state=POSITION_STATE_WAIT, full_position=10000.0,
                  open_returns=True,
                  stop_loss_triggered=False, close_by_time=False,
                  target_reached=False,
                  finalize_returns=(0.02, 200.0)):
    """Return a mocked Position with sensible defaults."""
    pos = MagicMock()
    pos.get_state.return_value = state
    pos.full_position = full_position
    pos.open.return_value = open_returns
    pos.is_stop_loss_triggered.return_value = stop_loss_triggered
    pos.is_target_reached.return_value = target_reached
    pos.close_by_time.return_value = close_by_time
    pos.finalize.return_value = finalize_returns
    return pos


# ---------------------------------------------------------------------------
# Import under test (after helpers so we can patch Position in the module)
# ---------------------------------------------------------------------------

from robots.train_robot import TrainRobot


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTrainRobotConstruction:

    def test_attributes_set_on_init(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)

        assert robot.strategy_manager is sm
        assert robot.fee == 0.001
        assert robot.trade_count == 0
        assert robot.revenue_history == []
        assert robot._current_coin_amount == 0.0

    def test_position_created_with_fee(self):
        """Position should be created with the fee passed to TrainRobot."""
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.005)
        assert robot.position.fee == pytest.approx(0.005)

    def test_position_is_position_instance(self):
        from position.position import Position
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        assert isinstance(robot.position, Position)


# ---------------------------------------------------------------------------
# get_results
# ---------------------------------------------------------------------------

class TestGetResults:

    def test_get_results_empty(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        result = robot.get_results()
        assert result['total_trades'] == 0
        assert result['revenue_history'] == []
        assert result['total_revenue_abs'] == 0.0
        assert result['avg_revenue_pct'] == 0.0

    def test_get_results_after_trades(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.revenue_history = [(0.02, 200.0), (0.01, 100.0)]
        robot.trade_count = 2
        result = robot.get_results()
        assert result['total_trades'] == 2
        assert result['revenue_history'] == [(0.02, 200.0), (0.01, 100.0)]
        assert result['total_revenue_abs'] == pytest.approx(300.0)
        assert result['avg_revenue_pct'] == pytest.approx(0.015)

    def test_get_results_single_trade(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.revenue_history = [(0.03, 300.0)]
        robot.trade_count = 1
        result = robot.get_results()
        assert result['total_revenue_abs'] == pytest.approx(300.0)
        assert result['avg_revenue_pct'] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# _finalize
# ---------------------------------------------------------------------------

class TestFinalize:

    def test_finalize_calls_position_finalize(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 5.0
        robot._finalize()
        robot.position.finalize.assert_called_once()

    def test_finalize_appends_revenue_history(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 5.0
        robot._finalize()
        assert robot.revenue_history == [(0.02, 200.0)]

    def test_finalize_increments_trade_count(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 5.0
        robot._finalize()
        assert robot.trade_count == 1

    def test_finalize_resets_coin_amount(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 5.0
        robot._finalize()
        assert robot._current_coin_amount == 0.0

    def test_finalize_multiple_trades_accumulate(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.01, 100.0))
        robot._current_coin_amount = 1.0
        robot._finalize()
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 1.0
        robot._finalize()
        assert robot.trade_count == 2
        assert len(robot.revenue_history) == 2


# ---------------------------------------------------------------------------
# buy
# ---------------------------------------------------------------------------

class TestBuy:

    def test_buy_calls_position_open(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(full_position=10000.0, open_returns=True)
        dp = make_data_point()

        robot.buy(dp, STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15)

        robot.position.open.assert_called_once_with(
            STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15, None
        )

    def test_buy_sets_current_coin_amount_on_success(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(full_position=10000.0, open_returns=True)
        dp = make_data_point()

        robot.buy(dp, STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15)

        # 10000.0 / 50.0 = 200.0
        assert robot._current_coin_amount == pytest.approx(200.0)

    def test_buy_calls_record_entry_fill_on_success(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(full_position=10000.0, open_returns=True)
        dp = make_data_point()

        robot.buy(dp, STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15)

        robot.position.record_entry_fill.assert_called_once_with(200.0, 50.0)

    def test_buy_does_not_set_coin_amount_on_failure(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(full_position=10000.0, open_returns=False)
        dp = make_data_point()

        robot.buy(dp, STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15)

        assert robot._current_coin_amount == 0.0

    def test_buy_does_not_call_record_entry_fill_on_failure(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(full_position=10000.0, open_returns=False)
        dp = make_data_point()

        robot.buy(dp, STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15)

        robot.position.record_entry_fill.assert_not_called()


# ---------------------------------------------------------------------------
# sell
# ---------------------------------------------------------------------------

class TestSell:

    def test_sell_calls_position_close(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 100.0
        dp = make_data_point()

        robot.sell(dp, STRATEGY_ACTION_CLOSE_LONG, [55.0], 45.0, 15)

        robot.position.close.assert_called_once_with(
            STRATEGY_ACTION_CLOSE_LONG, [55.0], 45.0, 15, None
        )

    def test_sell_calls_record_exit_fill(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 100.0
        dp = make_data_point()

        robot.sell(dp, STRATEGY_ACTION_CLOSE_LONG, [55.0], 45.0, 15)

        robot.position.record_exit_fill.assert_called_once_with(100.0, 55.0)

    def test_sell_calls_finalize(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(finalize_returns=(0.02, 200.0))
        robot._current_coin_amount = 100.0
        dp = make_data_point()

        robot.sell(dp, STRATEGY_ACTION_CLOSE_LONG, [55.0], 45.0, 15)

        robot.position.finalize.assert_called_once()
        assert robot.trade_count == 1
        assert robot.revenue_history == [(0.02, 200.0)]


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------

class TestWait:

    def test_wait_no_triggers_does_nothing(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(stop_loss_triggered=False, close_by_time=False)
        dp = make_data_point(cur_price=100.0, timestamp=5000.0)

        robot.wait(dp)

        robot.position.close.assert_not_called()
        robot.position.finalize.assert_not_called()

    def test_wait_stop_loss_triggers_sell(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(
            stop_loss_triggered=True, close_by_time=False,
            finalize_returns=(0.01, 100.0)
        )
        robot._current_coin_amount = 50.0
        dp = make_data_point(cur_price=80.0, timestamp=5000.0)

        robot.wait(dp)

        robot.position.close.assert_called_once_with(
            STRATEGY_ACTION_DO_STOP_LOSS, [80.0], 0.0, 0, None
        )
        robot.position.record_exit_fill.assert_called_once_with(50.0, 80.0)

    def test_wait_close_by_time_triggers_sell(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(
            stop_loss_triggered=False, close_by_time=True,
            finalize_returns=(0.01, 100.0)
        )
        robot._current_coin_amount = 50.0
        dp = make_data_point(cur_price=105.0, timestamp=9999.0)

        robot.wait(dp)

        robot.position.close.assert_called_once_with(
            STRATEGY_ACTION_DO_STOP_LOSS, [105.0], 0.0, 0, None
        )

    def test_wait_checks_stop_loss_with_cur_price(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(stop_loss_triggered=False, close_by_time=False)
        dp = make_data_point(cur_price=90.0, timestamp=1234.0)

        robot.wait(dp)

        robot.position.is_stop_loss_triggered.assert_called_once_with(90.0)

    def test_wait_checks_close_by_time_with_timestamp(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(stop_loss_triggered=False, close_by_time=False)
        dp = make_data_point(cur_price=90.0, timestamp=7777.0)

        robot.wait(dp)

        robot.position.close_by_time.assert_called_once_with(7777.0)


# ---------------------------------------------------------------------------
# _do dispatch
# ---------------------------------------------------------------------------

class TestDoDispatch:

    def _make_robot_with_position(self, sm):
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position(full_position=10000.0, open_returns=True,
                                       finalize_returns=(0.01, 100.0))
        return robot

    def test_do_calls_strategy_manager_check(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = self._make_robot_with_position(sm)
        dp = make_data_point(timestamp=1000.0)
        robot._do(dp)
        sm.check.assert_called_once()

    def test_do_passes_position_state_and_time(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = self._make_robot_with_position(sm)
        robot.position.get_state.return_value = POSITION_STATE_WAIT
        dp = make_data_point(timestamp=1234.0)
        robot._do(dp)
        sm.check.assert_called_once_with(dp, POSITION_STATE_WAIT, 1234.0, action_msg=None)

    def test_do_dispatches_open_long_to_buy(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_OPEN_LONG,
                                   open_prices=[50.0], close_prices=[60.0],
                                   stop_price=45.0, tf=15)
        robot = self._make_robot_with_position(sm)
        dp = make_data_point()
        robot._do(dp)
        robot.position.open.assert_called_once()

    def test_do_dispatches_open_short_to_buy(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_OPEN_SHORT,
                                   open_prices=[50.0], close_prices=[40.0],
                                   stop_price=55.0, tf=15)
        robot = self._make_robot_with_position(sm)
        dp = make_data_point()
        robot._do(dp)
        robot.position.open.assert_called_once()

    def test_do_dispatches_close_long_to_sell(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_LONG,
                                   close_prices=[55.0], stop_price=45.0, tf=15)
        robot = self._make_robot_with_position(sm)
        robot._current_coin_amount = 100.0
        dp = make_data_point()
        robot._do(dp)
        robot.position.close.assert_called_once()

    def test_do_dispatches_close_short_to_sell(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_SHORT,
                                   close_prices=[45.0], stop_price=55.0, tf=15)
        robot = self._make_robot_with_position(sm)
        robot._current_coin_amount = 100.0
        dp = make_data_point()
        robot._do(dp)
        robot.position.close.assert_called_once()

    def test_do_dispatches_close_long_part_to_sell(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_LONG_PART,
                                   close_prices=[55.0], stop_price=45.0, tf=15)
        robot = self._make_robot_with_position(sm)
        robot._current_coin_amount = 100.0
        dp = make_data_point()
        robot._do(dp)
        robot.position.close.assert_called_once()

    def test_do_dispatches_close_short_part_to_sell(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_SHORT_PART,
                                   close_prices=[45.0], stop_price=55.0, tf=15)
        robot = self._make_robot_with_position(sm)
        robot._current_coin_amount = 100.0
        dp = make_data_point()
        robot._do(dp)
        robot.position.close.assert_called_once()

    def test_do_dispatches_do_stop_loss_to_sell(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_DO_STOP_LOSS,
                                   close_prices=[48.0], stop_price=0.0, tf=0)
        robot = self._make_robot_with_position(sm)
        robot._current_coin_amount = 100.0
        dp = make_data_point()
        robot._do(dp)
        robot.position.close.assert_called_once()

    def test_do_dispatches_nothing_to_wait(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = self._make_robot_with_position(sm)
        robot.position.is_stop_loss_triggered.return_value = False
        robot.position.is_target_reached.return_value = False
        robot.position.close_by_time.return_value = False
        dp = make_data_point()
        robot._do(dp)
        # wait checks stop loss and close_by_time; no close should have been called
        robot.position.close.assert_not_called()
        robot.position.is_stop_loss_triggered.assert_called_once()


# ---------------------------------------------------------------------------
# step (exception handling)
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_calls_do(self):
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position()
        dp = make_data_point()

        with patch.object(robot, '_do') as mock_do:
            robot.step(dp)
            mock_do.assert_called_once_with(dp)

    def test_step_catches_exception_and_does_not_reraise(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        dp = make_data_point()

        with patch.object(robot, '_do', side_effect=ValueError("boom")):
            # Should not raise
            robot.step(dp)

    def test_step_logs_exception(self):
        sm = make_strategy_manager()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        dp = make_data_point()

        with patch.object(robot, '_do', side_effect=RuntimeError("test error")):
            with patch('robots.train_robot.log_error') as mock_log:
                robot.step(dp)
                mock_log.assert_called_once()

    def test_step_does_not_store_data_point(self):
        """data_point must never be stored as an instance attribute."""
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = TrainRobot(strategy_manager=sm, fee=0.001)
        robot.position = make_position()
        dp = make_data_point()
        robot.step(dp)
        # data_point should not be on robot
        assert not hasattr(robot, 'data_point')
        assert not hasattr(robot, 'dp')


# ---------------------------------------------------------------------------
# Full flow integration (mocked): open → sell → results
# ---------------------------------------------------------------------------

class TestFullFlowMocked:

    def test_full_trade_cycle(self):
        """Simulate one complete open→close trade using mocked position."""
        sm = MagicMock()
        robot = TrainRobot(strategy_manager=sm, fee=0.001)

        pos = make_position(full_position=10000.0, open_returns=True,
                            finalize_returns=(0.02, 200.0))
        robot.position = pos

        dp_open = make_data_point(cur_price=50.0, timestamp=1000.0)
        dp_close = make_data_point(cur_price=55.0, timestamp=2000.0)

        # Simulate buy
        robot.buy(dp_open, STRATEGY_ACTION_OPEN_LONG, [50.0], [60.0], 45.0, 15)
        assert robot._current_coin_amount == pytest.approx(200.0)
        assert robot.trade_count == 0

        # Simulate sell
        robot.sell(dp_close, STRATEGY_ACTION_CLOSE_LONG, [55.0], 45.0, 15)
        assert robot._current_coin_amount == 0.0
        assert robot.trade_count == 1
        assert robot.revenue_history == [(0.02, 200.0)]

        results = robot.get_results()
        assert results['total_trades'] == 1
        assert results['total_revenue_abs'] == pytest.approx(200.0)
        assert results['avg_revenue_pct'] == pytest.approx(0.02)
