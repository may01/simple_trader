"""Tests for Strategy abstract base class (Phase 07, Task 01).

TDD: these tests are written BEFORE the implementation.
"""

import pytest
import pandas as pd

from data import LiveDataPoint
from signals_lib.signal_manager import SignalManager, SignalChain
from signals_lib.base_signal import BaseSignal
from constants import (
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_MOVE_STOP_LOSS_LONG,
    STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT,
    POSITION_STATE_WAIT,
    LEVEL_TYPE_AUTO_RESISTANCE,
    LEVEL_TYPE_SHORT_RESISTANCE,
)


# ---------------------------------------------------------------------------
# Helpers / Test doubles
# ---------------------------------------------------------------------------

def make_dp(close_val: float = 100.0, sar: float = 95.0, atr: float = 2.0) -> LiveDataPoint:
    """Create a minimal LiveDataPoint with tf=1 data."""
    df = pd.DataFrame({
        "1_close": [close_val],
        "1_sar": [sar],
        "1_atr_14": [atr],
        "1_is_closed": [True],
    })
    df.index = pd.to_datetime(["2024-01-01 00:01:00"], utc=True)
    return LiveDataPoint({1: df})


def make_dp_multi_tf(close_1: float = 100.0, sar_5: float = 95.0, atr_5: float = 2.0) -> LiveDataPoint:
    """Create a LiveDataPoint with tf=1 and tf=5 data."""
    df_1 = pd.DataFrame({
        "1_close": [close_1],
        "1_sar": [sar_5],
        "1_atr_14": [atr_5],
        "1_is_closed": [True],
    })
    df_5 = pd.DataFrame({
        "5_close": [close_1],
        "5_sar": [sar_5],
        "5_atr_14": [atr_5],
        "5_is_closed": [True],
    })
    df_1.index = pd.to_datetime(["2024-01-01 00:01:00"], utc=True)
    df_5.index = pd.to_datetime(["2024-01-01 00:01:00"], utc=True)
    return LiveDataPoint({1: df_1, 5: df_5})


class AlwaysFiresSignal(BaseSignal):
    """Signal that always fires."""

    def check(self, data_point, levels, action) -> bool:
        return True

    def get_marker_pos(self, data_point) -> float:
        return 1.0


class NeverFiresSignal(BaseSignal):
    """Signal that never fires."""

    def check(self, data_point, levels, action) -> bool:
        return False

    def get_marker_pos(self, data_point) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Concrete strategy implementations for testing
# ---------------------------------------------------------------------------

class ConcreteStrategy:
    """Minimal concrete strategy that returns True from check_conditions."""

    def _make_impl(self, fee):
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self):
                pass  # no chains

            def check_conditions(self, data_point, position_state, action_msg):
                return True

        return _Impl(fee)

    def __call__(self, fee=0.001):
        return self._make_impl(fee)


class StrategyWithChain:
    """Strategy that fires a single-signal chain on every tick."""

    def __call__(self, action, tf=1, fee=0.001):
        from strategies.strategy import Strategy
        _action = action
        _tf = tf

        class _Impl(Strategy):
            def register_signals(self):
                chain = SignalChain("test_chain", _action, tf=_tf, notify=False)
                chain.add(AlwaysFiresSignal())
                self.signals.add_chain(chain)

            def check_conditions(self, data_point, position_state, action_msg):
                return True

        return _Impl(fee)


class StrategyWithConditionGate:
    """Strategy whose check_conditions returns a configurable value."""

    def __call__(self, condition_result: bool, action=STRATEGY_ACTION_OPEN_LONG, tf=1, fee=0.001):
        from strategies.strategy import Strategy
        _result = condition_result
        _action = action
        _tf = tf

        class _Impl(Strategy):
            def register_signals(self):
                chain = SignalChain("gated_chain", _action, tf=_tf, notify=False)
                chain.add(AlwaysFiresSignal())
                self.signals.add_chain(chain)

            def check_conditions(self, data_point, position_state, action_msg):
                return _result

        return _Impl(fee)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestStrategyConstruction:
    def test_cannot_instantiate_without_fee(self):
        """Strategy.__init__ requires fee argument — no default."""
        from strategies.strategy import Strategy
        with pytest.raises(TypeError):
            class BadImpl(Strategy):
                def register_signals(self): pass
                def check_conditions(self, dp, ps, am): return True
            BadImpl()  # missing fee

    def test_fee_attribute_stored(self):
        strategy = ConcreteStrategy()(fee=0.001)
        assert strategy.fee == 0.001

    def test_fee_attribute_zero(self):
        strategy = ConcreteStrategy()(fee=0.0)
        assert strategy.fee == 0.0

    def test_signals_attribute_is_signal_manager(self):
        strategy = ConcreteStrategy()()
        assert isinstance(strategy.signals, SignalManager)

    def test_register_signals_called_during_init(self):
        """register_signals is called in __init__ after self.signals is set."""
        from strategies.strategy import Strategy
        call_log = []

        class _Impl(Strategy):
            def register_signals(self):
                call_log.append("registered")

            def check_conditions(self, dp, ps, am):
                return True

        _Impl(fee=0.001)
        assert call_log == ["registered"]

    def test_abstract_methods_required(self):
        """Cannot instantiate Strategy without implementing abstract methods."""
        from strategies.strategy import Strategy
        with pytest.raises(TypeError):
            Strategy(fee=0.001)


# ---------------------------------------------------------------------------
# check() return type and structure
# ---------------------------------------------------------------------------

class TestCheckReturnStructure:
    def test_check_returns_tuple(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        result = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert isinstance(result, tuple)

    def test_check_returns_5_tuple(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        result = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert len(result) == 5

    def test_check_tuple_elements_types(self):
        """(action: str, open_prices: list, close_prices: list, stop_price: float, tf: int)"""
        strategy = ConcreteStrategy()()
        dp = make_dp()
        action, open_prices, close_prices, stop_price, tf = strategy.check(
            dp, POSITION_STATE_WAIT, 1000.0, None
        )
        assert isinstance(action, str)
        assert isinstance(open_prices, list)
        assert isinstance(close_prices, list)
        assert isinstance(stop_price, float)
        assert isinstance(tf, int)


# ---------------------------------------------------------------------------
# check() — nothing action when no signals fire
# ---------------------------------------------------------------------------

class TestCheckNothingAction:
    def test_returns_nothing_when_no_chains(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        action, _, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING

    def test_nothing_tf_is_zero(self):
        """When action is NOTHING, tf must be 0."""
        strategy = ConcreteStrategy()()
        dp = make_dp()
        _, _, _, _, tf = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert tf == 0

    def test_nothing_open_prices_is_list(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        _, open_prices, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert isinstance(open_prices, list)

    def test_returns_nothing_when_check_conditions_false(self):
        strategy = StrategyWithConditionGate()(condition_result=False)
        dp = make_dp()
        action, _, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING


# ---------------------------------------------------------------------------
# check() — action when chain fires
# ---------------------------------------------------------------------------

class TestCheckWithAction:
    def test_returns_open_long_when_chain_fires(self):
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=1)
        dp = make_dp()
        action, _, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_OPEN_LONG

    def test_returns_open_short_when_chain_fires(self):
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_SHORT, tf=1)
        dp = make_dp()
        action, _, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_OPEN_SHORT

    def test_tf_matches_fired_chain_tf(self):
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=5)
        dp = make_dp_multi_tf()
        _, _, _, _, tf = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert tf == 5

    def test_check_conditions_false_skips_signals(self):
        """When check_conditions returns False, signals are not evaluated."""
        strategy = StrategyWithConditionGate()(
            condition_result=False,
            action=STRATEGY_ACTION_OPEN_LONG,
            tf=1,
        )
        dp = make_dp()
        action, _, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert action == STRATEGY_ACTION_NOTHING


# ---------------------------------------------------------------------------
# Default open prices
# ---------------------------------------------------------------------------

class TestDefaultOpenPrices:
    def test_open_price_is_current_1min_close(self):
        """Default open_prices = [current 1-min close]."""
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=1)
        dp = make_dp(close_val=42.0)
        _, open_prices, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert open_prices == [42.0]

    def test_open_prices_is_list_of_one(self):
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=1)
        dp = make_dp(close_val=100.0)
        _, open_prices, _, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert len(open_prices) == 1


# ---------------------------------------------------------------------------
# Default close prices
# ---------------------------------------------------------------------------

class TestDefaultClosePrices:
    def test_close_price_long_is_plus_0_8_percent(self):
        """Default close_prices (long) = [open_price * 1.008]."""
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=1)
        dp = make_dp(close_val=100.0)
        _, open_prices, close_prices, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        expected = open_prices[0] * 1.008
        assert close_prices == [pytest.approx(expected)]

    def test_close_price_short_is_minus_0_8_percent(self):
        """Default close_prices (short) = [open_price * 0.992]."""
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_SHORT, tf=1)
        dp = make_dp(close_val=100.0)
        _, open_prices, close_prices, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        expected = open_prices[0] * 0.992
        assert close_prices == [pytest.approx(expected)]

    def test_close_prices_is_list_of_one(self):
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=1)
        dp = make_dp(close_val=100.0)
        _, _, close_prices, _, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert len(close_prices) == 1


# ---------------------------------------------------------------------------
# Default stop-loss price
# ---------------------------------------------------------------------------

class TestDefaultStopLossPrice:
    def test_stop_loss_long_is_sar_minus_0_3_atr(self):
        """Default stop_loss (long) = sar - 0.3 * atr_14."""
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=1)
        dp = make_dp(close_val=100.0, sar=95.0, atr=2.0)
        _, _, _, stop_price, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        expected = 95.0 - 0.3 * 2.0  # = 94.4
        assert stop_price == pytest.approx(expected)

    def test_stop_loss_short_is_sar_plus_0_3_atr(self):
        """Default stop_loss (short) = sar + 0.3 * atr_14."""
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_SHORT, tf=1)
        dp = make_dp(close_val=100.0, sar=105.0, atr=2.0)
        _, _, _, stop_price, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        expected = 105.0 + 0.3 * 2.0  # = 105.6
        assert stop_price == pytest.approx(expected)

    def test_stop_loss_uses_fired_chain_tf(self):
        """stop_loss uses the tf of the fired chain, not hardcoded 1."""
        strategy = StrategyWithChain()(STRATEGY_ACTION_OPEN_LONG, tf=5)
        dp = make_dp_multi_tf(close_1=100.0, sar_5=90.0, atr_5=3.0)
        _, _, _, stop_price, _ = strategy.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        expected = 90.0 - 0.3 * 3.0  # = 89.1
        assert stop_price == pytest.approx(expected)


# ---------------------------------------------------------------------------
# select_final_action — single action
# ---------------------------------------------------------------------------

class TestSelectFinalActionSingle:
    def test_single_action_used_directly(self):
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        actions = [[STRATEGY_ACTION_OPEN_LONG, 15, 100.0, {}]]
        action, tf = s.select_final_action(actions, POSITION_STATE_WAIT)
        assert action == STRATEGY_ACTION_OPEN_LONG
        assert tf == 15

    def test_single_close_long_used_directly(self):
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        actions = [[STRATEGY_ACTION_CLOSE_LONG, 5, 100.0, {}]]
        action, tf = s.select_final_action(actions, POSITION_STATE_WAIT)
        assert action == STRATEGY_ACTION_CLOSE_LONG
        assert tf == 5


# ---------------------------------------------------------------------------
# select_final_action — empty actions list
# ---------------------------------------------------------------------------

class TestSelectFinalActionEmpty:
    def test_empty_list_returns_nothing(self):
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        action, tf = s.select_final_action([], POSITION_STATE_WAIT)
        assert action == STRATEGY_ACTION_NOTHING
        assert tf == 0


# ---------------------------------------------------------------------------
# select_final_action — multiple actions, POSITION_STATE_WAIT
# ---------------------------------------------------------------------------

class TestSelectFinalActionMultipleWait:
    def test_prefers_open_long_over_close_long_when_waiting(self):
        """When waiting, prefer OPEN over CLOSE."""
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        actions = [
            [STRATEGY_ACTION_CLOSE_LONG, 5, 100.0, {}],
            [STRATEGY_ACTION_OPEN_LONG, 15, 100.0, {}],
        ]
        action, _ = s.select_final_action(actions, POSITION_STATE_WAIT)
        assert action == STRATEGY_ACTION_OPEN_LONG

    def test_prefers_open_short_over_move_stop_loss_when_waiting(self):
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        actions = [
            [STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, 5, 100.0, {}],
            [STRATEGY_ACTION_OPEN_SHORT, 15, 100.0, {}],
        ]
        action, _ = s.select_final_action(actions, POSITION_STATE_WAIT)
        assert action == STRATEGY_ACTION_OPEN_SHORT


# ---------------------------------------------------------------------------
# select_final_action — multiple actions, position open
# ---------------------------------------------------------------------------

class TestSelectFinalActionMultipleOpen:
    def test_prefers_close_over_move_stop_loss_when_position_open(self):
        """When position open, prefer CLOSE over MOVE_STOP_LOSS."""
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        # Use any non-WAIT state
        from constants import POSITION_STATE_WAIT_BUY
        actions = [
            [STRATEGY_ACTION_MOVE_STOP_LOSS_LONG, 5, 100.0, {}],
            [STRATEGY_ACTION_CLOSE_LONG, 15, 100.0, {}],
        ]
        action, _ = s.select_final_action(actions, POSITION_STATE_WAIT_BUY)
        assert action == STRATEGY_ACTION_CLOSE_LONG

    def test_prefers_close_short_over_move_stop_loss_short(self):
        from strategies.strategy import Strategy
        class _Impl(Strategy):
            def register_signals(self): pass
            def check_conditions(self, dp, ps, am): return True

        s = _Impl(fee=0.001)
        from constants import POSITION_STATE_WAIT_SELL
        actions = [
            [STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT, 5, 100.0, {}],
            [STRATEGY_ACTION_CLOSE_SHORT, 15, 100.0, {}],
        ]
        action, _ = s.select_final_action(actions, POSITION_STATE_WAIT_SELL)
        assert action == STRATEGY_ACTION_CLOSE_SHORT


# ---------------------------------------------------------------------------
# get_level_values — levels dict structure
# ---------------------------------------------------------------------------

class TestGetLevelValues:
    def test_returns_dict(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        levels = strategy.get_level_values(dp)
        assert isinstance(levels, dict)

    def test_contains_auto_resistance_key(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        levels = strategy.get_level_values(dp)
        assert LEVEL_TYPE_AUTO_RESISTANCE in levels

    def test_contains_short_resistance_key(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        levels = strategy.get_level_values(dp)
        assert LEVEL_TYPE_SHORT_RESISTANCE in levels

    def test_auto_resistance_value_is_list(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        levels = strategy.get_level_values(dp)
        assert isinstance(levels[LEVEL_TYPE_AUTO_RESISTANCE], list)

    def test_short_resistance_value_is_list(self):
        strategy = ConcreteStrategy()()
        dp = make_dp()
        levels = strategy.get_level_values(dp)
        assert isinstance(levels[LEVEL_TYPE_SHORT_RESISTANCE], list)


# ---------------------------------------------------------------------------
# Price method overrides
# ---------------------------------------------------------------------------

class TestPriceMethodOverrides:
    def test_can_override_open_prices(self):
        """Subclass can override get_open_prices() to return custom values."""
        from strategies.strategy import Strategy

        class CustomOpenStrategy(Strategy):
            def register_signals(self):
                chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=1, notify=False)
                chain.add(AlwaysFiresSignal())
                self.signals.add_chain(chain)

            def check_conditions(self, dp, ps, am):
                return True

            def get_open_prices(self, data_point, tf):
                return [999.0, 998.0]  # two entry targets

        s = CustomOpenStrategy(fee=0.001)
        dp = make_dp(close_val=100.0)
        _, open_prices, _, _, _ = s.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert open_prices == [999.0, 998.0]

    def test_can_override_close_prices(self):
        """Subclass can override get_close_prices() to return custom values."""
        from strategies.strategy import Strategy

        class CustomCloseStrategy(Strategy):
            def register_signals(self):
                chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=1, notify=False)
                chain.add(AlwaysFiresSignal())
                self.signals.add_chain(chain)

            def check_conditions(self, dp, ps, am):
                return True

            def get_close_prices(self, data_point, tf, action):
                return [777.0]

        s = CustomCloseStrategy(fee=0.001)
        dp = make_dp(close_val=100.0)
        _, _, close_prices, _, _ = s.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert close_prices == [777.0]

    def test_can_override_stop_loss_price(self):
        """Subclass can override get_stop_loss_price() to return custom value."""
        from strategies.strategy import Strategy

        class CustomStopStrategy(Strategy):
            def register_signals(self):
                chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=1, notify=False)
                chain.add(AlwaysFiresSignal())
                self.signals.add_chain(chain)

            def check_conditions(self, dp, ps, am):
                return True

            def get_stop_loss_price(self, data_point, tf, action):
                return 55.0

        s = CustomStopStrategy(fee=0.001)
        dp = make_dp(close_val=100.0)
        _, _, _, stop_price, _ = s.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert stop_price == 55.0


# ---------------------------------------------------------------------------
# check() passes levels to signals.check()
# ---------------------------------------------------------------------------

class TestCheckPassesLevels:
    def test_levels_passed_to_signal_check(self):
        """get_level_values() result is forwarded to signals.check()."""
        from strategies.strategy import Strategy

        received_levels = {}

        class LevelCapturingSignal(BaseSignal):
            def check(self, data_point, levels, action) -> bool:
                received_levels.update(levels)
                return True

            def get_marker_pos(self, data_point) -> float:
                return 1.0

        class LevelCapturingStrategy(Strategy):
            def register_signals(self):
                chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=1, notify=False)
                chain.add(LevelCapturingSignal())
                self.signals.add_chain(chain)

            def check_conditions(self, dp, ps, am):
                return True

        s = LevelCapturingStrategy(fee=0.001)
        dp = make_dp()
        s.check(dp, POSITION_STATE_WAIT, 1000.0, None)
        assert LEVEL_TYPE_AUTO_RESISTANCE in received_levels
