"""Tests for ExampleStrategyLong and ExampleStrategyShort (Phase 07, Task 02).

TDD: these tests are written BEFORE the implementation.
"""

import pytest
import pandas as pd

from data import LiveDataPoint
from constants import (
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_SHORT,
    POSITION_STATE_WAIT,
)


def make_dp_with_cci(cci_15_val: float = 0.0, cci_15_prev: float = -150.0) -> LiveDataPoint:
    """Create a LiveDataPoint with CCI_14 on tf=15 and minimal tf=1 data.

    Args:
        cci_15_val: Current CCI value at shift=0
        cci_15_prev: Previous CCI value at shift=1

    Returns:
        LiveDataPoint with tf=15 data containing CCI_14 values and tf=1 fallback
    """
    df_15 = pd.DataFrame({
        "15_close": [100.0, 100.0],
        "15_cci_14": [cci_15_val, cci_15_prev],
        "15_sar_002_02": [95.0, 95.0],
        "15_atr_14": [2.0, 2.0],
        "15_is_closed": [True, True],
    })
    df_15.index = pd.to_datetime(
        ["2024-01-01 00:15:00", "2024-01-01 00:00:00"],
        utc=True
    )
    # Also add minimal tf=1 data for base class methods that need it
    df_1 = pd.DataFrame({
        "1_close": [100.0],
        "1_sar_002_02": [95.0],
        "1_atr_14": [2.0],
        "1_is_closed": [True],
    })
    df_1.index = pd.to_datetime(["2024-01-01 00:15:00"], utc=True)
    return LiveDataPoint({15: df_15, 1: df_1})


class TestExampleStrategyLongConstruction:
    """Test ExampleStrategyLong basic construction."""

    def test_can_instantiate_with_fee(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        assert strategy.fee == 0.001

    def test_has_signals_manager(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        from signals_lib.signal_manager import SignalManager
        strategy = ExampleStrategyLong(fee=0.001)
        assert isinstance(strategy.signals, SignalManager)

    def test_registers_exactly_two_chains(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        assert len(strategy.signals.chains) == 2

    def test_chains_are_long_entry_and_long_exit(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        chain_names = {chain.name for chain in strategy.signals.chains}
        assert "long_entry" in chain_names
        assert "long_exit" in chain_names

    def test_long_entry_chain_has_open_long_action(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        entry_chain = next(c for c in strategy.signals.chains if c.name == "long_entry")
        assert entry_chain.result_action == STRATEGY_ACTION_OPEN_LONG

    def test_long_exit_chain_has_close_long_action(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        exit_chain = next(c for c in strategy.signals.chains if c.name == "long_exit")
        assert exit_chain.result_action == STRATEGY_ACTION_CLOSE_LONG

    def test_both_chains_use_tf_15(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        for chain in strategy.signals.chains:
            assert chain.tf == 15

    def test_each_chain_has_exactly_one_signal(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        for chain in strategy.signals.chains:
            assert len(chain.signals) == 1


class TestExampleStrategyLongSignals:
    """Test signal types in ExampleStrategyLong chains."""

    def test_long_entry_uses_cross_up_val_signal(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        from signals_lib.common import Cross_Up_Val_Signal
        strategy = ExampleStrategyLong(fee=0.001)
        entry_chain = next(c for c in strategy.signals.chains if c.name == "long_entry")
        signal = entry_chain.signals[0]
        assert isinstance(signal, Cross_Up_Val_Signal)

    def test_long_entry_signal_is_cci_14_threshold_minus_100(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        entry_chain = next(c for c in strategy.signals.chains if c.name == "long_entry")
        signal = entry_chain.signals[0]
        assert signal.indi_1 == "cci_14"
        assert signal.val == -100

    def test_long_exit_uses_cross_down_val_signal(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        from signals_lib.common import Cross_Down_Val_Signal
        strategy = ExampleStrategyLong(fee=0.001)
        exit_chain = next(c for c in strategy.signals.chains if c.name == "long_exit")
        signal = exit_chain.signals[0]
        assert isinstance(signal, Cross_Down_Val_Signal)

    def test_long_exit_signal_is_cci_14_threshold_100(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        exit_chain = next(c for c in strategy.signals.chains if c.name == "long_exit")
        signal = exit_chain.signals[0]
        assert signal.indi_1 == "cci_14"
        assert signal.val == 100


class TestExampleStrategyLongCheckConditions:
    """Test check_conditions() always returns True."""

    def test_check_conditions_returns_true_in_wait_state(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        result = strategy.check_conditions(None, POSITION_STATE_WAIT, None)
        assert result is True

    def test_check_conditions_returns_true_with_data_point(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        dp = make_dp_with_cci()
        result = strategy.check_conditions(dp, POSITION_STATE_WAIT, None)
        assert result is True


class TestExampleStrategyLongPriceMethods:
    """Test that price methods delegate to base class."""

    def test_get_open_prices_delegates_to_base(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        dp = make_dp_with_cci()
        # Base implementation returns [current 1-min close]
        # We need tf=1 data for this, so we'll just verify it's a list
        prices = strategy.get_open_prices(dp, 15)
        assert isinstance(prices, list)

    def test_get_close_prices_delegates_to_base(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        dp = make_dp_with_cci()
        prices = strategy.get_close_prices(dp, 15, STRATEGY_ACTION_OPEN_LONG)
        assert isinstance(prices, list)

    def test_get_stop_loss_price_delegates_to_base(self):
        from strategies.example_strategy_long import ExampleStrategyLong
        strategy = ExampleStrategyLong(fee=0.001)
        dp = make_dp_with_cci()
        price = strategy.get_stop_loss_price(dp, 15, STRATEGY_ACTION_OPEN_LONG)
        assert isinstance(price, float)


class TestExampleStrategyShortConstruction:
    """Test ExampleStrategyShort basic construction."""

    def test_can_instantiate_with_fee(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        assert strategy.fee == 0.001

    def test_has_signals_manager(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        from signals_lib.signal_manager import SignalManager
        strategy = ExampleStrategyShort(fee=0.001)
        assert isinstance(strategy.signals, SignalManager)

    def test_registers_exactly_two_chains(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        assert len(strategy.signals.chains) == 2

    def test_chains_are_short_entry_and_short_exit(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        chain_names = {chain.name for chain in strategy.signals.chains}
        assert "short_entry" in chain_names
        assert "short_exit" in chain_names

    def test_short_entry_chain_has_open_short_action(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        entry_chain = next(c for c in strategy.signals.chains if c.name == "short_entry")
        assert entry_chain.result_action == STRATEGY_ACTION_OPEN_SHORT

    def test_short_exit_chain_has_close_short_action(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        exit_chain = next(c for c in strategy.signals.chains if c.name == "short_exit")
        assert exit_chain.result_action == STRATEGY_ACTION_CLOSE_SHORT

    def test_both_chains_use_tf_15(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        for chain in strategy.signals.chains:
            assert chain.tf == 15

    def test_each_chain_has_exactly_one_signal(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        for chain in strategy.signals.chains:
            assert len(chain.signals) == 1


class TestExampleStrategyShortSignals:
    """Test signal types in ExampleStrategyShort chains."""

    def test_short_entry_uses_cross_down_val_signal(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        from signals_lib.common import Cross_Down_Val_Signal
        strategy = ExampleStrategyShort(fee=0.001)
        entry_chain = next(c for c in strategy.signals.chains if c.name == "short_entry")
        signal = entry_chain.signals[0]
        assert isinstance(signal, Cross_Down_Val_Signal)

    def test_short_entry_signal_is_cci_14_threshold_100(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        entry_chain = next(c for c in strategy.signals.chains if c.name == "short_entry")
        signal = entry_chain.signals[0]
        assert signal.indi_1 == "cci_14"
        assert signal.val == 100

    def test_short_exit_uses_cross_up_val_signal(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        from signals_lib.common import Cross_Up_Val_Signal
        strategy = ExampleStrategyShort(fee=0.001)
        exit_chain = next(c for c in strategy.signals.chains if c.name == "short_exit")
        signal = exit_chain.signals[0]
        assert isinstance(signal, Cross_Up_Val_Signal)

    def test_short_exit_signal_is_cci_14_threshold_minus_100(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        exit_chain = next(c for c in strategy.signals.chains if c.name == "short_exit")
        signal = exit_chain.signals[0]
        assert signal.indi_1 == "cci_14"
        assert signal.val == -100


class TestExampleStrategyShortCheckConditions:
    """Test check_conditions() always returns True."""

    def test_check_conditions_returns_true_in_wait_state(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        result = strategy.check_conditions(None, POSITION_STATE_WAIT, None)
        assert result is True

    def test_check_conditions_returns_true_with_data_point(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        dp = make_dp_with_cci()
        result = strategy.check_conditions(dp, POSITION_STATE_WAIT, None)
        assert result is True


class TestExampleStrategyShortPriceMethods:
    """Test that price methods delegate to base class."""

    def test_get_open_prices_delegates_to_base(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        dp = make_dp_with_cci()
        prices = strategy.get_open_prices(dp, 15)
        assert isinstance(prices, list)

    def test_get_close_prices_delegates_to_base(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        dp = make_dp_with_cci()
        prices = strategy.get_close_prices(dp, 15, STRATEGY_ACTION_OPEN_SHORT)
        assert isinstance(prices, list)

    def test_get_stop_loss_price_delegates_to_base(self):
        from strategies.example_strategy_short import ExampleStrategyShort
        strategy = ExampleStrategyShort(fee=0.001)
        dp = make_dp_with_cci()
        price = strategy.get_stop_loss_price(dp, 15, STRATEGY_ACTION_OPEN_SHORT)
        assert isinstance(price, float)


class TestExampleStrategiesIntegration:
    """Integration tests for both strategies."""

    def test_verification_snippet_long(self):
        """Verify ExampleStrategyLong works as expected."""
        from strategies.example_strategy_long import ExampleStrategyLong
        s_long = ExampleStrategyLong(fee=0.001)
        assert s_long.check_conditions(None, POSITION_STATE_WAIT, None) == True
        assert len(s_long.signals.chains) == 2

    def test_verification_snippet_short(self):
        """Verify ExampleStrategyShort works as expected."""
        from strategies.example_strategy_short import ExampleStrategyShort
        s_short = ExampleStrategyShort(fee=0.001)
        assert s_short.check_conditions(None, POSITION_STATE_WAIT, None) == True
        assert len(s_short.signals.chains) == 2
