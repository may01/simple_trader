"""Tests for Robot — Phase 10, Task 02.

TDD: polling loop scaffold, do() dispatch, stop().
All dependencies are mocked.
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

def make_data_point(timestamp=1000.0):
    """Return a MagicMock DataPoint."""
    dp = MagicMock()
    dp.timestamp = timestamp
    return dp


def make_live_data(data_point=None):
    """Return a MagicMock LiveData object with a data_point property."""
    ld = MagicMock()
    ld.data_point = data_point if data_point is not None else make_data_point()
    return ld


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


def make_robot(strategy_manager=None, live_data=None, stock=None,
               fee=0.001, persist_path="/tmp/test_robot_state.json"):
    """Return a Robot with mocked dependencies."""
    from robots.robot import Robot

    if strategy_manager is None:
        strategy_manager = make_strategy_manager()
    if live_data is None:
        live_data = make_live_data()
    if stock is None:
        stock = MagicMock()

    return Robot(
        strategy_manager=strategy_manager,
        live_data=live_data,
        stock=stock,
        fee=fee,
        persist_path=persist_path,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRobotConstruction:

    def test_attributes_set_on_init(self):
        from robots.robot import Robot
        sm = make_strategy_manager()
        ld = make_live_data()
        stock = MagicMock()
        robot = Robot(strategy_manager=sm, live_data=ld, stock=stock,
                      fee=0.002, persist_path="/tmp/r.json")

        assert robot.strategy_manager is sm
        assert robot.live_data is ld
        assert robot.stock is stock
        assert robot.fee == 0.002
        assert robot.running is False

    def test_position_created_internally(self):
        from position.position import Position
        robot = make_robot(fee=0.003)
        assert isinstance(robot.position, Position)
        assert robot.position.fee == 0.003

    def test_tracker_created_internally(self):
        from robots.live_order_tracker import LiveOrderTracker
        stock = MagicMock()
        robot = make_robot(stock=stock, persist_path="/tmp/tracker.json")
        assert isinstance(robot.tracker, LiveOrderTracker)
        assert robot.tracker.position is robot.position
        assert robot.tracker.stock is stock
        assert robot.tracker.persist_path == "/tmp/tracker.json"

    def test_running_starts_false(self):
        robot = make_robot()
        assert robot.running is False


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_sets_running_false(self):
        robot = make_robot()
        robot.running = True
        robot.stop()
        assert robot.running is False

    def test_stop_when_already_false(self):
        robot = make_robot()
        robot.stop()
        assert robot.running is False


# ---------------------------------------------------------------------------
# do() — dispatch
# ---------------------------------------------------------------------------

class TestDo:

    def test_do_calls_strategy_manager_check(self):
        dp = make_data_point(timestamp=5000.0)
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        robot.do()
        sm.check.assert_called_once()

    def test_do_passes_data_point_to_check(self):
        dp = make_data_point(timestamp=5000.0)
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        robot.position = MagicMock()
        robot.position.get_state.return_value = POSITION_STATE_WAIT
        robot.do()
        args, kwargs = sm.check.call_args
        assert args[0] is dp

    def test_do_passes_position_state_to_check(self):
        dp = make_data_point(timestamp=5000.0)
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        robot.position = MagicMock()
        robot.position.get_state.return_value = POSITION_STATE_WAIT
        robot.do()
        args, kwargs = sm.check.call_args
        assert args[1] == POSITION_STATE_WAIT

    def test_do_passes_timestamp_to_check(self):
        dp = make_data_point(timestamp=9999.0)
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        robot.do()
        args, kwargs = sm.check.call_args
        assert args[2] == 9999.0

    def test_do_dispatches_open_long_to_open_position(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_OPEN_LONG,
                                   open_prices=[100.0], close_prices=[105.0],
                                   stop_price=95.0, tf=5)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_open_position') as mock_open:
            robot.do()
            mock_open.assert_called_once()

    def test_do_dispatches_open_short_to_open_position(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_OPEN_SHORT,
                                   open_prices=[100.0], close_prices=[95.0],
                                   stop_price=105.0, tf=5)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_open_position') as mock_open:
            robot.do()
            mock_open.assert_called_once()

    def test_do_dispatches_close_long_to_close_position(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_LONG)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_close_position') as mock_close:
            robot.do()
            mock_close.assert_called_once()

    def test_do_dispatches_close_short_to_close_position(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_SHORT)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_close_position') as mock_close:
            robot.do()
            mock_close.assert_called_once()

    def test_do_dispatches_close_long_part_to_close_position(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_LONG_PART)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_close_position') as mock_close:
            robot.do()
            mock_close.assert_called_once()

    def test_do_dispatches_close_short_part_to_close_position(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_CLOSE_SHORT_PART)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_close_position') as mock_close:
            robot.do()
            mock_close.assert_called_once()

    def test_do_dispatches_do_stop_loss_to_stop_loss(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_DO_STOP_LOSS)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_stop_loss') as mock_sl:
            robot.do()
            mock_sl.assert_called_once()

    def test_do_dispatches_nothing_to_wait(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, 'wait') as mock_wait:
            robot.do()
            mock_wait.assert_called_once_with(dp)

    def test_do_nothing_does_not_call_open_or_close(self):
        dp = make_data_point()
        ld = make_live_data(data_point=dp)
        sm = make_strategy_manager(action=STRATEGY_ACTION_NOTHING)
        robot = make_robot(strategy_manager=sm, live_data=ld)
        with patch.object(robot, '_open_position') as mock_open, \
             patch.object(robot, '_close_position') as mock_close, \
             patch.object(robot, '_stop_loss') as mock_sl, \
             patch.object(robot, 'wait'):
            robot.do()
            mock_open.assert_not_called()
            mock_close.assert_not_called()
            mock_sl.assert_not_called()


# ---------------------------------------------------------------------------
# wait()
# ---------------------------------------------------------------------------

class TestWait:

    def test_wait_calls_process_executed_orders(self):
        dp = make_data_point()
        robot = make_robot()
        with patch.object(robot, '_process_executed_orders') as mock_peo, \
             patch.object(robot, '_stop_loss_cancel_actions'):
            robot.wait(dp)
            mock_peo.assert_called_once_with(dp)

    def test_wait_calls_stop_loss_cancel_actions(self):
        dp = make_data_point()
        robot = make_robot()
        with patch.object(robot, '_process_executed_orders'), \
             patch.object(robot, '_stop_loss_cancel_actions') as mock_slca:
            robot.wait(dp)
            mock_slca.assert_called_once_with(dp)

    def test_wait_calls_both_in_order(self):
        dp = make_data_point()
        robot = make_robot()
        call_order = []
        with patch.object(robot, '_process_executed_orders',
                          side_effect=lambda _: call_order.append('peo')), \
             patch.object(robot, '_stop_loss_cancel_actions',
                          side_effect=lambda _: call_order.append('slca')):
            robot.wait(dp)
        assert call_order == ['peo', 'slca']


# ---------------------------------------------------------------------------
# run_instantly() — loop mechanics
# ---------------------------------------------------------------------------

class TestRunInstantly:

    def test_run_instantly_calls_tracker_load(self):
        robot = make_robot()
        with patch.object(robot.tracker, 'load') as mock_load, \
             patch.object(robot, 'do', side_effect=lambda: setattr(robot, 'running', False)), \
             patch('time.sleep'):
            robot.run_instantly()
            mock_load.assert_called_once()

    def test_run_instantly_sets_running_true_then_calls_do(self):
        robot = make_robot()
        call_log = []

        def fake_do():
            call_log.append(('running_during_do', robot.running))
            robot.running = False  # stop after first iteration

        with patch.object(robot.tracker, 'load'), \
             patch.object(robot, 'do', side_effect=fake_do), \
             patch('time.sleep'):
            robot.run_instantly()

        assert ('running_during_do', True) in call_log

    def test_run_instantly_calls_do_in_loop(self):
        robot = make_robot()
        iteration = [0]

        def fake_do():
            iteration[0] += 1
            if iteration[0] >= 3:
                robot.running = False

        with patch.object(robot.tracker, 'load'), \
             patch.object(robot, 'do', side_effect=fake_do), \
             patch('time.sleep'):
            robot.run_instantly()

        assert iteration[0] == 3

    def test_run_instantly_sleeps_1_second(self):
        robot = make_robot()
        sleep_calls = []

        def fake_do():
            robot.running = False

        with patch.object(robot.tracker, 'load'), \
             patch.object(robot, 'do', side_effect=fake_do), \
             patch('time.sleep', side_effect=lambda s: sleep_calls.append(s)):
            robot.run_instantly()

        assert len(sleep_calls) >= 1
        assert all(s == 1 for s in sleep_calls)

    def test_run_instantly_handles_keyboard_interrupt(self):
        robot = make_robot()

        def fake_do():
            raise KeyboardInterrupt

        with patch.object(robot.tracker, 'load'), \
             patch.object(robot, 'do', side_effect=fake_do), \
             patch('time.sleep'):
            # Should not raise
            robot.run_instantly()

        assert robot.running is False

    def test_run_instantly_sets_running_false_after_keyboard_interrupt(self):
        robot = make_robot()

        def fake_do():
            raise KeyboardInterrupt

        with patch.object(robot.tracker, 'load'), \
             patch.object(robot, 'do', side_effect=fake_do), \
             patch('time.sleep'):
            robot.run_instantly()

        assert robot.running is False


# ---------------------------------------------------------------------------
# Stub methods exist and are callable
# ---------------------------------------------------------------------------

class TestStubMethods:

    def test_open_position_exists_and_is_passthrough(self):
        robot = make_robot()
        # Should not raise
        robot._open_position(MagicMock(), STRATEGY_ACTION_OPEN_LONG, [], [], 0.0, 0)

    def test_close_position_exists_and_is_passthrough(self):
        robot = make_robot()
        robot._close_position(MagicMock(), STRATEGY_ACTION_CLOSE_LONG, [], 0.0, 0)

    def test_stop_loss_exists_and_is_passthrough(self):
        robot = make_robot()
        robot._stop_loss(MagicMock(), [], 0.0, 0)

    def test_process_executed_orders_exists_and_is_passthrough(self):
        robot = make_robot()
        robot._process_executed_orders(MagicMock())

    def test_stop_loss_cancel_actions_exists_and_is_passthrough(self):
        robot = make_robot()
        robot._stop_loss_cancel_actions(MagicMock())
