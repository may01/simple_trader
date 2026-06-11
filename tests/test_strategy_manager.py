"""Tests for StrategyManager (Phase 07, Task 03).

TDD: these tests are written BEFORE the implementation.
"""

import pytest
import pandas as pd

from data import LiveDataPoint
from strategies.strategy import Strategy
from signals_lib.signal_manager import SignalManager, SignalChain
from signals_lib.base_signal import BaseSignal
from constants import (
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
    STRATEGY_ACTION_MOVE_STOP_LOSS_LONG,
    STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT,
    STRATEGY_ACTION_DO_STOP_LOSS,
    POSITION_STATE_WAIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dp(close_val: float = 100.0, sar_002_02: float = 95.0, atr: float = 2.0) -> LiveDataPoint:
    """Create a minimal LiveDataPoint with tf=1 data."""
    df = pd.DataFrame({
        "1_close": [close_val],
        "1_sar_002_02": [sar_002_02],
        "1_atr_14": [atr],
        "1_is_closed": [True],
    })
    df.index = pd.to_datetime(["2024-01-01 00:01:00"], utc=True)
    return LiveDataPoint({1: df})


class AlwaysFiresSignal(BaseSignal):
    """Signal that always fires."""

    def check(self, data_point, levels, action) -> bool:
        return True

    def get_marker_pos(self, data_point) -> float:
        return 1.0


class StrategyReturning:
    """Factory to create a strategy that always returns a specific action."""

    def __call__(self, action: str, open_prices=None, close_prices=None,
                 stop_price=0.0, tf=1, fee=0.001):
        if open_prices is None:
            open_prices = [100.0]
        if close_prices is None:
            close_prices = [100.8]

        _action = action
        _open_prices = open_prices
        _close_prices = close_prices
        _stop_price = stop_price
        _tf = tf

        class _Impl(Strategy):
            def register_signals(self):
                pass

            def check_conditions(self, data_point, position_state, action_msg):
                return True

            def check(self, data_point, position_state, cur_time, action_msg):
                return (_action, _open_prices, _close_prices, _stop_price, _tf)

        return _Impl(fee)


class StrategyReturningNothing:
    """Factory to create a strategy that always returns NOTHING."""

    def __call__(self, fee=0.001):
        class _Impl(Strategy):
            def register_signals(self):
                pass

            def check_conditions(self, data_point, position_state, action_msg):
                return True

            def check(self, data_point, position_state, cur_time, action_msg):
                return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)

        return _Impl(fee)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestStrategyManagerConstruction:
    def test_can_instantiate_with_fee(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        assert sm.fee == 0.001

    def test_fee_stored(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.0025)
        assert sm.fee == 0.0025

    def test_strategies_list_initially_empty(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        assert sm.strategies == []

    def test_strategies_is_list(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        assert isinstance(sm.strategies, list)


# ---------------------------------------------------------------------------
# register() tests
# ---------------------------------------------------------------------------

class TestStrategyManagerRegister:
    def test_register_appends_strategy(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        strategy = StrategyReturningNothing()()
        sm.register(strategy)
        assert len(sm.strategies) == 1
        assert sm.strategies[0] is strategy

    def test_register_multiple_strategies(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        s1 = StrategyReturningNothing()()
        s2 = StrategyReturningNothing()()
        sm.register(s1)
        sm.register(s2)
        assert len(sm.strategies) == 2
        assert sm.strategies[0] is s1
        assert sm.strategies[1] is s2

    def test_register_preserves_order(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        strategies = [StrategyReturningNothing()() for _ in range(5)]
        for s in strategies:
            sm.register(s)
        assert sm.strategies == strategies


# ---------------------------------------------------------------------------
# check() return type and structure
# ---------------------------------------------------------------------------

class TestStrategyManagerCheckReturnStructure:
    def test_check_returns_tuple(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        result = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert isinstance(result, tuple)

    def test_check_returns_5_tuple(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        result = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert len(result) == 5

    def test_check_tuple_elements_types(self):
        """(action: str, open_prices: list, close_prices: list, stop_price: float, tf: int)"""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        action, open_prices, close_prices, stop_price, tf = sm.check(
            dp, POSITION_STATE_WAIT, 1000.0, None
        )
        assert isinstance(action, str)
        assert isinstance(open_prices, list)
        assert isinstance(close_prices, list)
        assert isinstance(stop_price, float)
        assert isinstance(tf, int)


# ---------------------------------------------------------------------------
# check() — empty strategies list
# ---------------------------------------------------------------------------

class TestStrategyManagerCheckEmpty:
    def test_returns_nothing_when_no_strategies(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING

    def test_returns_empty_lists_when_no_strategies(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        dp = make_dp()
        _, open_prices, close_prices, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert open_prices == []
        assert close_prices == []

    def test_returns_zero_prices_when_no_strategies(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        dp = make_dp()
        _, _, _, stop_price, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert stop_price == 0.0
        assert tf == 0


# ---------------------------------------------------------------------------
# check() — all strategies return NOTHING
# ---------------------------------------------------------------------------

class TestStrategyManagerCheckAllNothing:
    def test_returns_nothing_when_all_strategies_nothing(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturningNothing()())
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING


# ---------------------------------------------------------------------------
# check() — single action from single strategy
# ---------------------------------------------------------------------------

class TestStrategyManagerCheckSingleAction:
    def test_returns_single_open_long(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(
            STRATEGY_ACTION_OPEN_LONG,
            open_prices=[100.0],
            close_prices=[100.8],
            stop_price=95.0,
            tf=1
        ))
        dp = make_dp()
        action, open_prices, close_prices, stop_price, tf = sm.check(
            dp, POSITION_STATE_WAIT, 1000.0, None
        )
        assert action == STRATEGY_ACTION_OPEN_LONG
        assert open_prices == [100.0]
        assert close_prices == [100.8]
        assert stop_price == 95.0
        assert tf == 1

    def test_returns_single_close_long(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(
            STRATEGY_ACTION_CLOSE_LONG,
            open_prices=[],
            close_prices=[102.0],
            stop_price=0.0,
            tf=5
        ))
        dp = make_dp()
        action, _, _, _, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_CLOSE_LONG
        assert tf == 5


# ---------------------------------------------------------------------------
# Conflict resolution: DO_STOP_LOSS priority
# ---------------------------------------------------------------------------

class TestStrategyManagerConflictResolutionDoStopLoss:
    def test_do_stop_loss_takes_priority_over_open(self):
        """DO_STOP_LOSS has highest priority."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_DO_STOP_LOSS, stop_price=90.0, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_LONG, stop_price=95.0, tf=1))
        dp = make_dp()
        action, _, _, stop_price, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_DO_STOP_LOSS
        assert stop_price == 90.0

    def test_do_stop_loss_takes_priority_over_close(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_LONG, stop_price=92.0, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_DO_STOP_LOSS, stop_price=90.0, tf=1))
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_DO_STOP_LOSS

    def test_do_stop_loss_takes_priority_over_move_stop_loss(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, stop_price=92.0, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_DO_STOP_LOSS, stop_price=90.0, tf=1))
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_DO_STOP_LOSS


# ---------------------------------------------------------------------------
# Conflict resolution: CLOSE priority over MOVE_STOP_LOSS and OPEN
# ---------------------------------------------------------------------------

class TestStrategyManagerConflictResolutionClose:
    def test_close_long_takes_priority_over_move_stop_loss(self):
        """CLOSE takes priority over MOVE_STOP_LOSS."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_LONG, tf=5))
        dp = make_dp()
        action, _, _, _, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_CLOSE_LONG
        assert tf == 5

    def test_close_short_takes_priority_over_move_stop_loss(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_SHORT, tf=5))
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_CLOSE_SHORT

    def test_close_long_part_takes_priority_over_move_stop_loss(self):
        """Partial close also counts as CLOSE action."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_LONG_PART, tf=5))
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_CLOSE_LONG_PART

    def test_close_takes_priority_over_open(self):
        """CLOSE takes priority over OPEN."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_LONG, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_LONG, tf=5))
        dp = make_dp()
        action, _, _, _, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_CLOSE_LONG
        assert tf == 5

    def test_multiple_close_signals_returns_first(self):
        """Multiple CLOSE signals → return first one."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_LONG, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_CLOSE_LONG, tf=5))
        dp = make_dp()
        action, _, _, _, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_CLOSE_LONG
        assert tf == 1  # First strategy


# ---------------------------------------------------------------------------
# Conflict resolution: MOVE_STOP_LOSS priority over OPEN
# ---------------------------------------------------------------------------

class TestStrategyManagerConflictResolutionMoveStopLoss:
    def test_move_stop_loss_long_takes_priority_over_open(self):
        """MOVE_STOP_LOSS takes priority over OPEN."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_LONG, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, tf=5))
        dp = make_dp()
        action, _, _, _, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_MOVE_STOP_LOSS_LONG
        assert tf == 5

    def test_move_stop_loss_short_takes_priority_over_open(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_SHORT, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT, tf=5))
        dp = make_dp()
        action, _, _, _, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT
        assert tf == 5


# ---------------------------------------------------------------------------
# Conflict resolution: Multiple OPEN signals → NOTHING
# ---------------------------------------------------------------------------

class TestStrategyManagerConflictResolutionMultipleOpen:
    def test_multiple_open_long_returns_nothing(self):
        """Multiple different OPEN signals → return NOTHING (conflict)."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_LONG, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_LONG, tf=5))
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING

    def test_multiple_open_short_returns_nothing(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_SHORT, tf=1))
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_SHORT, tf=5))
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING

    def test_single_open_long_allowed(self):
        """Single OPEN_LONG is allowed."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_LONG, tf=1))
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_OPEN_LONG

    def test_single_open_short_allowed(self):
        """Single OPEN_SHORT is allowed."""
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_OPEN_SHORT, tf=1))
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        action, _, _, _, _ = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_OPEN_SHORT


# ---------------------------------------------------------------------------
# Conflict resolution: Only MOVE_STOP_LOSS
# ---------------------------------------------------------------------------

class TestStrategyManagerConflictResolutionOnlyMoveStopLoss:
    def test_only_move_stop_loss_long_returned(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, tf=1, stop_price=95.0))
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        action, _, _, stop_price, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_MOVE_STOP_LOSS_LONG
        assert tf == 1
        assert stop_price == 95.0

    def test_only_move_stop_loss_short_returned(self):
        from strategies.strategy_manager import StrategyManager
        sm = StrategyManager(fee=0.001)
        sm.register(StrategyReturning()(STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT, tf=5, stop_price=105.0))
        sm.register(StrategyReturningNothing()())
        dp = make_dp()
        action, _, _, stop_price, tf = sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT
        assert tf == 5
        assert stop_price == 105.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestStrategyManagerIntegration:
    def test_calls_each_strategy_check(self):
        """check() calls check() on each registered strategy."""
        from strategies.strategy_manager import StrategyManager
        call_log = []

        class LoggingStrategy(Strategy):
            def __init__(self, fee, log):
                self.log = log
                super().__init__(fee)

            def register_signals(self):
                pass

            def check_conditions(self, dp, ps, am):
                return True

            def check(self, data_point, position_state, cur_time, action_msg):
                self.log.append("check_called")
                return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)

        sm = StrategyManager(fee=0.001)
        s1 = LoggingStrategy(fee=0.001, log=call_log)
        s2 = LoggingStrategy(fee=0.001, log=call_log)
        sm.register(s1)
        sm.register(s2)
        dp = make_dp()
        sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert call_log == ["check_called", "check_called"]

    def test_strategies_evaluated_in_order(self):
        """Strategies are checked in registration order."""
        from strategies.strategy_manager import StrategyManager
        order = []

        class OrderTrackingStrategy(Strategy):
            def __init__(self, fee, strategy_id, order_list):
                self.strategy_id = strategy_id
                self.order_list = order_list
                super().__init__(fee)

            def register_signals(self):
                pass

            def check_conditions(self, dp, ps, am):
                return True

            def check(self, data_point, position_state, cur_time, action_msg):
                self.order_list.append(self.strategy_id)
                return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)

        sm = StrategyManager(fee=0.001)
        for i in range(3):
            s = OrderTrackingStrategy(fee=0.001, strategy_id=i, order_list=order)
            sm.register(s)
        dp = make_dp()
        sm.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert order == [0, 1, 2]

    def test_verification_snippet(self):
        """Verify StrategyManager works as in task spec."""
        from strategies.strategy_manager import StrategyManager
        from strategies.example_strategy_long import ExampleStrategyLong
        from constants import POSITION_STATE_WAIT

        sm = StrategyManager(fee=0.001)
        sm.register(ExampleStrategyLong(fee=0.001))
        assert len(sm.strategies) == 1
