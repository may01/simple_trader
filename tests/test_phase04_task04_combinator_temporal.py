"""Tests for combinator and temporal signals (Phase 04, Task 04)."""

import math
import pytest
from data import DataPoint
from signals_lib.base_signal import BaseSignal
from signals_lib.operations import (
    And_Signal,
    Or_Signal,
    Not_Signal,
    History_Signal,
    Continuous_Signal,
    True_Signal,
    False_Signal,
    BoolValue_Signal,
    PersistentCounter_Signal,
    DataValue_Signal,
)


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------


class MockDataPoint(DataPoint):
    """Mock DataPoint for testing signals."""

    def __init__(self, data: dict = None):
        self._data = data or {}
        self._ts = None

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        key = (col, tf, shift)
        if key in self._data:
            return self._data[key]
        return float("nan")

    def get_df(self, tf: int):
        return None

    @property
    def timestamp(self):
        return self._ts


class FixedSignal(BaseSignal):
    """Signal that always returns a fixed value."""

    def __init__(self, result: bool):
        super().__init__()
        self._result = result

    def check(self, data_point, levels, action) -> bool:
        return self._result


class TrackingSignal(BaseSignal):
    """Signal that records whether it was called."""

    def __init__(self, result: bool = True):
        super().__init__()
        self._result = result
        self.called = False
        self.last_shift = None

    def check(self, data_point, levels, action) -> bool:
        self.called = True
        self.last_shift = self.shift
        return self._result


# ---------------------------------------------------------------------------
# And_Signal tests
# ---------------------------------------------------------------------------


class TestAndSignal:
    """Test And_Signal: check() evaluates ALL nested signals, returns all(...)."""

    def test_all_true(self):
        """And_Signal returns True when all nested signals are True."""
        sig = And_Signal([FixedSignal(True), FixedSignal(True)])
        assert sig.check(None, {}, None) is True

    def test_any_false(self):
        """And_Signal returns False when any nested signal is False."""
        sig = And_Signal([FixedSignal(True), FixedSignal(False)])
        assert sig.check(None, {}, None) is False

    def test_all_false(self):
        """And_Signal returns False when all nested signals are False."""
        sig = And_Signal([FixedSignal(False), FixedSignal(False)])
        assert sig.check(None, {}, None) is False

    def test_no_short_circuit_evaluates_all(self):
        """And_Signal MUST evaluate all signals even when first is False (no short-circuit)."""
        tracker = TrackingSignal(result=True)
        false_sig = False_Signal()
        and_sig = And_Signal([false_sig, tracker])
        result = and_sig.check(None, {}, None)
        assert result is False
        assert tracker.called is True  # MUST be True — all evaluated

    def test_no_short_circuit_second_is_true(self):
        """And_Signal evaluates all even when first True and second False."""
        tracker = TrackingSignal(result=False)
        true_sig = True_Signal()
        and_sig = And_Signal([true_sig, tracker])
        result = and_sig.check(None, {}, None)
        assert result is False
        assert tracker.called is True

    def test_single_signal_true(self):
        """And_Signal works with a single signal returning True."""
        assert And_Signal([FixedSignal(True)]).check(None, {}, None) is True

    def test_single_signal_false(self):
        """And_Signal works with a single signal returning False."""
        assert And_Signal([FixedSignal(False)]).check(None, {}, None) is False

    def test_set_shift_propagates_to_all(self):
        """And_Signal.set_shift() propagates to all nested signals."""
        t1 = TrackingSignal()
        t2 = TrackingSignal()
        sig = And_Signal([t1, t2])
        sig.set_shift(3)
        assert t1.shift == 3
        assert t2.shift == 3

    def test_set_shift_three_signals(self):
        """And_Signal.set_shift() propagates to all three nested signals."""
        signals = [TrackingSignal() for _ in range(3)]
        sig = And_Signal(signals)
        sig.set_shift(7)
        for s in signals:
            assert s.shift == 7


# ---------------------------------------------------------------------------
# Or_Signal tests
# ---------------------------------------------------------------------------


class TestOrSignal:
    """Test Or_Signal: check() returns any(...)."""

    def test_all_true(self):
        """Or_Signal returns True when all are True."""
        assert Or_Signal([FixedSignal(True), FixedSignal(True)]).check(None, {}, None) is True

    def test_first_true(self):
        """Or_Signal returns True when first is True."""
        assert Or_Signal([FixedSignal(True), FixedSignal(False)]).check(None, {}, None) is True

    def test_second_true(self):
        """Or_Signal returns True when second is True."""
        assert Or_Signal([FixedSignal(False), FixedSignal(True)]).check(None, {}, None) is True

    def test_all_false(self):
        """Or_Signal returns False when all are False."""
        assert Or_Signal([FixedSignal(False), FixedSignal(False)]).check(None, {}, None) is False

    def test_set_shift_propagates_to_all(self):
        """Or_Signal.set_shift() propagates to all nested signals."""
        t1 = TrackingSignal()
        t2 = TrackingSignal()
        sig = Or_Signal([t1, t2])
        sig.set_shift(5)
        assert t1.shift == 5
        assert t2.shift == 5


# ---------------------------------------------------------------------------
# Not_Signal tests
# ---------------------------------------------------------------------------


class TestNotSignal:
    """Test Not_Signal: check() returns not signal.check(...)."""

    def test_negates_true(self):
        """Not_Signal returns False when nested signal is True."""
        assert Not_Signal(FixedSignal(True)).check(None, {}, None) is False

    def test_negates_false(self):
        """Not_Signal returns True when nested signal is False."""
        assert Not_Signal(FixedSignal(False)).check(None, {}, None) is True

    def test_set_shift_propagates(self):
        """Not_Signal.set_shift() propagates to nested signal."""
        inner = TrackingSignal()
        sig = Not_Signal(inner)
        sig.set_shift(4)
        assert inner.shift == 4


# ---------------------------------------------------------------------------
# True_Signal / False_Signal tests
# ---------------------------------------------------------------------------


class TestTrueFalseSignals:
    """Test True_Signal and False_Signal utility signals."""

    def test_true_signal_always_true(self):
        assert True_Signal().check(None, {}, None) is True

    def test_false_signal_always_false(self):
        assert False_Signal().check(None, {}, None) is False

    def test_true_signal_multiple_calls(self):
        sig = True_Signal()
        for _ in range(5):
            assert sig.check(None, {}, None) is True

    def test_false_signal_multiple_calls(self):
        sig = False_Signal()
        for _ in range(5):
            assert sig.check(None, {}, None) is False


# ---------------------------------------------------------------------------
# BoolValue_Signal tests
# ---------------------------------------------------------------------------


class TestBoolValueSignal:
    """Test BoolValue_Signal: returns True if data_point.get(value, tf, shift) != 0."""

    def test_nonzero_value_true(self):
        """BoolValue_Signal returns True for nonzero value."""
        pt = MockDataPoint({("trend_up_50", 5, 0): 1.0})
        sig = BoolValue_Signal("trend_up_50", 5)
        assert sig.check(pt, {}, None) is True

    def test_zero_value_false(self):
        """BoolValue_Signal returns False for zero value."""
        pt = MockDataPoint({("trend_up_50", 5, 0): 0.0})
        sig = BoolValue_Signal("trend_up_50", 5)
        assert sig.check(pt, {}, None) is False

    def test_nan_value_false(self):
        """BoolValue_Signal returns False for NaN value."""
        pt = MockDataPoint({})  # missing key → NaN
        sig = BoolValue_Signal("trend_up_50", 5)
        assert sig.check(pt, {}, None) is False

    def test_negative_value_true(self):
        """BoolValue_Signal returns True for any nonzero value including negative."""
        pt = MockDataPoint({("trend_up_50", 5, 0): -1.0})
        sig = BoolValue_Signal("trend_up_50", 5)
        assert sig.check(pt, {}, None) is True

    def test_respects_shift(self):
        """BoolValue_Signal uses self.shift when reading data."""
        pt = MockDataPoint({("flag", 5, 2): 1.0, ("flag", 5, 0): 0.0})
        sig = BoolValue_Signal("flag", 5)
        sig.set_shift(2)
        assert sig.check(pt, {}, None) is True

    def test_shift_zero_uses_shift_zero(self):
        """BoolValue_Signal with shift=0 reads shift=0."""
        pt = MockDataPoint({("flag", 5, 0): 0.0, ("flag", 5, 2): 1.0})
        sig = BoolValue_Signal("flag", 5)
        # default shift is 0
        assert sig.check(pt, {}, None) is False


# ---------------------------------------------------------------------------
# History_Signal tests
# ---------------------------------------------------------------------------


class TestHistorySignal:
    """Test History_Signal: evaluates signal at historical offset."""

    def test_evaluates_at_history_offset(self):
        """History_Signal uses history + shift as the child's effective shift."""
        inner = TrackingSignal(result=True)
        sig = History_Signal(inner, history=3)
        sig.check(None, {}, None)
        # child should have been set to history (3) + shift (0) = 3
        assert inner.last_shift == 3

    def test_combined_shift(self):
        """History_Signal uses history + self.shift for child shift."""
        inner = TrackingSignal(result=True)
        sig = History_Signal(inner, history=3)
        sig.set_shift(2)
        sig.check(None, {}, None)
        assert inner.last_shift == 5  # 3 + 2

    def test_returns_inner_result_true(self):
        """History_Signal returns True when inner signal is True."""
        inner = FixedSignal(True)
        sig = History_Signal(inner, history=2)
        assert sig.check(None, {}, None) is True

    def test_returns_inner_result_false(self):
        """History_Signal returns False when inner signal is False."""
        inner = FixedSignal(False)
        sig = History_Signal(inner, history=2)
        assert sig.check(None, {}, None) is False

    def test_set_shift_updates_self_shift(self):
        """History_Signal.set_shift() updates self.shift."""
        inner = TrackingSignal()
        sig = History_Signal(inner, history=1)
        sig.set_shift(4)
        assert sig.shift == 4

    def test_history_zero(self):
        """History_Signal with history=0 uses just self.shift as child shift."""
        inner = TrackingSignal(result=True)
        sig = History_Signal(inner, history=0)
        sig.check(None, {}, None)
        assert inner.last_shift == 0

    def test_reads_historical_data(self):
        """History_Signal reads historical data via inner signal's shift."""
        # Value at shift=0: 40 (below 50), value at shift=2: 60 (above 50)
        from signals_lib.common import Greater_Val_Signal

        pt = MockDataPoint({("rsi_14", 5, 0): 40.0, ("rsi_14", 5, 2): 60.0})
        inner = Greater_Val_Signal(5, "rsi_14", 50)
        sig = History_Signal(inner, history=2)
        # should read rsi_14 at shift=2 (60 > 50) → True
        assert sig.check(pt, {}, None) is True


# ---------------------------------------------------------------------------
# Continuous_Signal tests
# ---------------------------------------------------------------------------


class TestContinuousSignal:
    """Test Continuous_Signal: fires if signal is True >= multiply times across history candles."""

    def test_all_true_fires(self):
        """Continuous_Signal returns True when all checks in history are True."""
        inner = FixedSignal(True)
        sig = Continuous_Signal(inner, history=3, multiply=3)
        assert sig.check(None, {}, None) is True

    def test_all_false_does_not_fire(self):
        """Continuous_Signal returns False when no checks are True."""
        inner = FixedSignal(False)
        sig = Continuous_Signal(inner, history=3, multiply=1)
        assert sig.check(None, {}, None) is False

    def test_count_meets_multiply(self):
        """Continuous_Signal returns True when count >= multiply."""
        # signal is True at shifts 0,1 (FixedSignal always True), history=3, multiply=2
        inner = FixedSignal(True)
        sig = Continuous_Signal(inner, history=3, multiply=2)
        assert sig.check(None, {}, None) is True

    def test_count_below_multiply(self):
        """Continuous_Signal returns False when count < multiply."""
        inner = FixedSignal(False)
        sig = Continuous_Signal(inner, history=3, multiply=2)
        assert sig.check(None, {}, None) is False

    def test_default_multiply_is_1(self):
        """Continuous_Signal default multiply=1 fires if signal is True at least once."""
        inner = FixedSignal(True)
        sig = Continuous_Signal(inner, history=3)
        assert sig.check(None, {}, None) is True

    def test_restores_original_shift(self):
        """Continuous_Signal restores the inner signal's original shift after evaluation."""
        inner = TrackingSignal(result=True)
        inner.set_shift(5)
        sig = Continuous_Signal(inner, history=3)
        sig.check(None, {}, None)
        # shift should be restored to the original value
        assert inner.shift == 5

    def test_evaluates_across_history_candles(self):
        """Continuous_Signal evaluates signal at each i in range(history)."""
        pt = MockDataPoint({
            ("rsi_14", 5, 0): 60.0,  # shift=0: 60>50 → True
            ("rsi_14", 5, 1): 40.0,  # shift=1: 40<50 → False
            ("rsi_14", 5, 2): 70.0,  # shift=2: 70>50 → True
        })
        from signals_lib.common import Greater_Val_Signal

        inner = Greater_Val_Signal(5, "rsi_14", 50)
        sig = Continuous_Signal(inner, history=3, multiply=2)
        # 2 out of 3 are True, multiply=2 → True
        assert sig.check(pt, {}, None) is True

    def test_evaluates_across_history_candles_not_enough(self):
        """Continuous_Signal returns False when count < multiply over history."""
        pt = MockDataPoint({
            ("rsi_14", 5, 0): 40.0,  # False
            ("rsi_14", 5, 1): 40.0,  # False
            ("rsi_14", 5, 2): 70.0,  # True
        })
        from signals_lib.common import Greater_Val_Signal

        inner = Greater_Val_Signal(5, "rsi_14", 50)
        sig = Continuous_Signal(inner, history=3, multiply=2)
        # only 1 True, multiply=2 → False
        assert sig.check(pt, {}, None) is False


# ---------------------------------------------------------------------------
# PersistentCounter_Signal tests
# ---------------------------------------------------------------------------


class TestPersistentCounterSignal:
    """Test PersistentCounter_Signal: fires every fire_each calls."""

    def test_first_call_fires_when_fire_each_1(self):
        """PersistentCounter_Signal fires on first call when fire_each=1."""
        sig = PersistentCounter_Signal(fire_each=1)
        assert sig.check(None, {}, None) is True

    def test_fires_every_n_calls(self):
        """PersistentCounter_Signal fires every fire_each calls."""
        sig = PersistentCounter_Signal(fire_each=3)
        results = [sig.check(None, {}, None) for _ in range(6)]
        # calls 1,2,3,4,5,6; fires at 3,6 (count % 3 == 0)
        assert results == [False, False, True, False, False, True]

    def test_first_call_is_call_number_1(self):
        """Counter starts at 1 — first call is call #1."""
        sig = PersistentCounter_Signal(fire_each=2)
        # Call 1: 1 % 2 == 1 → False
        # Call 2: 2 % 2 == 0 → True
        assert sig.check(None, {}, None) is False
        assert sig.check(None, {}, None) is True

    def test_fire_each_1_always_fires(self):
        """PersistentCounter_Signal with fire_each=1 fires on every call."""
        sig = PersistentCounter_Signal(fire_each=1)
        for _ in range(5):
            assert sig.check(None, {}, None) is True

    def test_independent_instances(self):
        """Two PersistentCounter_Signal instances have independent counters."""
        sig1 = PersistentCounter_Signal(fire_each=2)
        sig2 = PersistentCounter_Signal(fire_each=2)
        sig1.check(None, {}, None)  # call 1
        # sig2 is still at 0 calls, should behave independently
        assert sig2.check(None, {}, None) is False  # call 1 for sig2


# ---------------------------------------------------------------------------
# DataValue_Signal tests
# ---------------------------------------------------------------------------


class TestDataValueSignal:
    """Test DataValue_Signal: armed/fired flag pattern."""

    def test_check_returns_value_set_false_when_not_armed(self):
        """DataValue_Signal.check() returns False when not armed."""
        sig = DataValue_Signal("my_val", None, False)
        assert sig.check(None, {}, None) is False

    def test_check_returns_value_set_true_when_armed(self):
        """DataValue_Signal.check() returns True when armed."""
        sig = DataValue_Signal("my_val", None, True)
        assert sig.check(None, {}, None) is True

    def test_set_value_arms_signal(self):
        """DataValue_Signal.set_value() sets value and arms the signal."""
        sig = DataValue_Signal("my_val", None, False)
        sig.set_value(42.0)
        assert sig.check(None, {}, None) is True

    def test_get_data_empty_when_not_armed(self):
        """DataValue_Signal.get_data() returns empty dict when not armed."""
        sig = DataValue_Signal("price", None, False)
        assert sig.get_data() == {}

    def test_get_data_returns_value_when_armed(self):
        """DataValue_Signal.get_data() returns data_name→value when armed."""
        sig = DataValue_Signal("price", None, False)
        sig.set_value(100.5)
        assert sig.get_data() == {"price": 100.5}

    def test_get_data_initial_value_when_armed_at_init(self):
        """DataValue_Signal with initial value_set=True exposes value via get_data."""
        sig = DataValue_Signal("level", 55.0, True)
        assert sig.get_data() == {"level": 55.0}


# ---------------------------------------------------------------------------
# Integration: combinator with data signals
# ---------------------------------------------------------------------------


class TestCombinatorIntegration:
    """Integration tests combining multiple signal types."""

    def test_and_with_bool_value_signals(self):
        """And_Signal with BoolValue_Signal reads data correctly."""
        pt = MockDataPoint({
            ("trend_up_50", 5, 0): 1.0,
            ("trend_dn", 5, 0): 0.0,
        })
        sig = And_Signal([BoolValue_Signal("trend_up_50", 5), BoolValue_Signal("trend_dn", 5)])
        assert sig.check(pt, {}, None) is False

    def test_or_with_bool_value_signals(self):
        """Or_Signal returns True when at least one BoolValue_Signal is True."""
        pt = MockDataPoint({
            ("trend_up_50", 5, 0): 0.0,
            ("trend_dn", 5, 0): 1.0,
        })
        sig = Or_Signal([BoolValue_Signal("trend_up_50", 5), BoolValue_Signal("trend_dn", 5)])
        assert sig.check(pt, {}, None) is True

    def test_not_wrapping_and(self):
        """Not_Signal wrapping And_Signal inverts the combined result."""
        sig = Not_Signal(And_Signal([True_Signal(), True_Signal()]))
        assert sig.check(None, {}, None) is False

    def test_nested_and_or(self):
        """Nested And_Signal inside Or_Signal works correctly."""
        # Or([And([T, T]), F]) → Or([T, F]) → T
        sig = Or_Signal([
            And_Signal([True_Signal(), True_Signal()]),
            False_Signal(),
        ])
        assert sig.check(None, {}, None) is True

    def test_set_shift_propagates_through_nested_combinators(self):
        """set_shift() propagates through nested combinator hierarchy."""
        leaf1 = TrackingSignal()
        leaf2 = TrackingSignal()
        # Not(And([leaf1, leaf2]))
        sig = Not_Signal(And_Signal([leaf1, leaf2]))
        sig.set_shift(6)
        assert leaf1.shift == 6
        assert leaf2.shift == 6
