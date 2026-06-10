"""Tests for atomic comparison signals (Phase 04, Task 02)."""

import math

import pytest
from data import DataPoint
from signals_lib.base_signal import BaseSignal
from signals_lib.common import (
    Less_Signal,
    Greater_Signal,
    Less_Val_Signal,
    Greater_Val_Signal,
    Rising_Signal,
    Falling_Signal,
    Diff_Greater_Signal,
    Diff_Less_Signal,
    Diff_LessIndi_Signal,
    Diff_GreaterIndi_Signal,
)


# Mock DataPoint for testing
class MockDataPoint(DataPoint):
    """Mock DataPoint for testing signals."""

    def __init__(self, data: dict = None):
        """Initialize with optional data dict keyed by (col, tf, shift)."""
        self._data = data or {}
        self._ts = None

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        """Return mocked data or NaN if not found."""
        key = (col, tf, shift)
        if key in self._data:
            return self._data[key]
        return float("nan")

    def get_df(self, tf: int):
        """Return None (not needed for these tests)."""
        return None

    @property
    def timestamp(self):
        """Return mocked timestamp."""
        return self._ts


class TestLessSignal:
    """Test Less_Signal: indi_1 < indi_2."""

    def test_less_signal_true(self):
        """Less_Signal returns True when indi_1 < indi_2."""
        sig = Less_Signal(5, "rsi_14", "close")
        data = {
            ("rsi_14", 5, 0): 30.0,
            ("close", 5, 0): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_less_signal_false(self):
        """Less_Signal returns False when indi_1 >= indi_2."""
        sig = Less_Signal(5, "rsi_14", "close")
        data = {
            ("rsi_14", 5, 0): 60.0,
            ("close", 5, 0): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_less_signal_equal(self):
        """Less_Signal returns False when indi_1 == indi_2."""
        sig = Less_Signal(5, "rsi_14", "close")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("close", 5, 0): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_less_signal_with_shift(self):
        """Less_Signal applies self.shift to both get() calls."""
        sig = Less_Signal(5, "rsi_14", "close")
        sig.set_shift(2)
        data = {
            ("rsi_14", 5, 2): 30.0,
            ("close", 5, 2): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_less_signal_nan_returns_false(self):
        """Less_Signal returns False when either value is NaN."""
        sig = Less_Signal(5, "rsi_14", "close")
        # Missing close value (will be NaN)
        data = {("rsi_14", 5, 0): 30.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

        # Missing rsi_14 value (will be NaN)
        data2 = {("close", 5, 0): 50.0}
        pt2 = MockDataPoint(data2)
        assert sig.check(pt2, {}, None) is False


class TestGreaterSignal:
    """Test Greater_Signal: indi_1 > indi_2."""

    def test_greater_signal_true(self):
        """Greater_Signal returns True when indi_1 > indi_2."""
        sig = Greater_Signal(5, "rsi_14", "close")
        data = {
            ("rsi_14", 5, 0): 60.0,
            ("close", 5, 0): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_greater_signal_false(self):
        """Greater_Signal returns False when indi_1 <= indi_2."""
        sig = Greater_Signal(5, "rsi_14", "close")
        data = {
            ("rsi_14", 5, 0): 30.0,
            ("close", 5, 0): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_greater_signal_with_shift(self):
        """Greater_Signal applies self.shift to both get() calls."""
        sig = Greater_Signal(5, "rsi_14", "close")
        sig.set_shift(1)
        data = {
            ("rsi_14", 5, 1): 60.0,
            ("close", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_greater_signal_nan_returns_false(self):
        """Greater_Signal returns False when either value is NaN."""
        sig = Greater_Signal(5, "rsi_14", "close")
        data = {("rsi_14", 5, 0): 60.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestLessValSignal:
    """Test Less_Val_Signal: indi_1 < val."""

    def test_less_val_signal_true(self):
        """Less_Val_Signal returns True when indi_1 < val."""
        sig = Less_Val_Signal(5, "rsi_14", 40.0)
        data = {("rsi_14", 5, 0): 30.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_less_val_signal_false(self):
        """Less_Val_Signal returns False when indi_1 >= val."""
        sig = Less_Val_Signal(5, "rsi_14", 40.0)
        data = {("rsi_14", 5, 0): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_less_val_signal_with_shift(self):
        """Less_Val_Signal applies self.shift to get() call."""
        sig = Less_Val_Signal(5, "rsi_14", 40.0)
        sig.set_shift(2)
        data = {("rsi_14", 5, 2): 30.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_less_val_signal_nan_returns_false(self):
        """Less_Val_Signal returns False when value is NaN."""
        sig = Less_Val_Signal(5, "rsi_14", 40.0)
        data = {}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestGreaterValSignal:
    """Test Greater_Val_Signal: indi_1 > val."""

    def test_greater_val_signal_true(self):
        """Greater_Val_Signal returns True when indi_1 > val."""
        sig = Greater_Val_Signal(5, "rsi_14", 40.0)
        data = {("rsi_14", 5, 0): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_greater_val_signal_false(self):
        """Greater_Val_Signal returns False when indi_1 <= val."""
        sig = Greater_Val_Signal(5, "rsi_14", 40.0)
        data = {("rsi_14", 5, 0): 30.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_greater_val_signal_with_shift(self):
        """Greater_Val_Signal applies self.shift to get() call."""
        sig = Greater_Val_Signal(5, "rsi_14", 40.0)
        sig.set_shift(1)
        data = {("rsi_14", 5, 1): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_greater_val_signal_nan_returns_false(self):
        """Greater_Val_Signal returns False when value is NaN."""
        sig = Greater_Val_Signal(5, "rsi_14", 40.0)
        data = {}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestRisingSignal:
    """Test Rising_Signal: indi_1[shift] > indi_1[shift+1]."""

    def test_rising_signal_true(self):
        """Rising_Signal returns True when current > previous."""
        sig = Rising_Signal(5, "rsi_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_14", 5, 1): 40.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_rising_signal_false(self):
        """Rising_Signal returns False when current <= previous."""
        sig = Rising_Signal(5, "rsi_14")
        data = {
            ("rsi_14", 5, 0): 40.0,
            ("rsi_14", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_rising_signal_with_shift(self):
        """Rising_Signal applies self.shift and self.shift+1."""
        sig = Rising_Signal(5, "rsi_14")
        sig.set_shift(2)
        # When shift=2, should compare get(..., shift=2) > get(..., shift=3)
        data = {
            ("rsi_14", 5, 2): 50.0,
            ("rsi_14", 5, 3): 40.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_rising_signal_nan_returns_false(self):
        """Rising_Signal returns False when either value is NaN."""
        sig = Rising_Signal(5, "rsi_14")
        # Missing shift+1 value
        data = {("rsi_14", 5, 0): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestFallingSignal:
    """Test Falling_Signal: indi_1[shift] < indi_1[shift+1]."""

    def test_falling_signal_true(self):
        """Falling_Signal returns True when current < previous."""
        sig = Falling_Signal(5, "rsi_14")
        data = {
            ("rsi_14", 5, 0): 40.0,
            ("rsi_14", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_falling_signal_false(self):
        """Falling_Signal returns False when current >= previous."""
        sig = Falling_Signal(5, "rsi_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_14", 5, 1): 40.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_falling_signal_with_shift(self):
        """Falling_Signal applies self.shift and self.shift+1."""
        sig = Falling_Signal(5, "rsi_14")
        sig.set_shift(1)
        data = {
            ("rsi_14", 5, 1): 40.0,
            ("rsi_14", 5, 2): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_falling_signal_nan_returns_false(self):
        """Falling_Signal returns False when either value is NaN."""
        sig = Falling_Signal(5, "rsi_14")
        data = {("rsi_14", 5, 0): 40.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestDiffGreaterSignal:
    """Test Diff_Greater_Signal: (indi_1 - indi_2) > val."""

    def test_diff_greater_signal_true(self):
        """Diff_Greater_Signal returns True when diff > val."""
        sig = Diff_Greater_Signal(5, "rsi_14", "rsi_7", 10.0)
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 35.0,
        }
        pt = MockDataPoint(data)
        # 50 - 35 = 15 > 10, should be True
        assert sig.check(pt, {}, None) is True

    def test_diff_greater_signal_false(self):
        """Diff_Greater_Signal returns False when diff <= val."""
        sig = Diff_Greater_Signal(5, "rsi_14", "rsi_7", 10.0)
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 45.0,
        }
        pt = MockDataPoint(data)
        # 50 - 45 = 5 not > 10, should be False
        assert sig.check(pt, {}, None) is False

    def test_diff_greater_signal_with_shift(self):
        """Diff_Greater_Signal applies self.shift to all get() calls."""
        sig = Diff_Greater_Signal(5, "rsi_14", "rsi_7", 10.0)
        sig.set_shift(2)
        data = {
            ("rsi_14", 5, 2): 50.0,
            ("rsi_7", 5, 2): 35.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_diff_greater_signal_nan_returns_false(self):
        """Diff_Greater_Signal returns False when either value is NaN."""
        sig = Diff_Greater_Signal(5, "rsi_14", "rsi_7", 10.0)
        data = {("rsi_14", 5, 0): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestDiffLessSignal:
    """Test Diff_Less_Signal: (indi_1 - indi_2) < val."""

    def test_diff_less_signal_true(self):
        """Diff_Less_Signal returns True when diff < val."""
        sig = Diff_Less_Signal(5, "rsi_14", "rsi_7", 10.0)
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 45.0,
        }
        pt = MockDataPoint(data)
        # 50 - 45 = 5 < 10, should be True
        assert sig.check(pt, {}, None) is True

    def test_diff_less_signal_false(self):
        """Diff_Less_Signal returns False when diff >= val."""
        sig = Diff_Less_Signal(5, "rsi_14", "rsi_7", 10.0)
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 35.0,
        }
        pt = MockDataPoint(data)
        # 50 - 35 = 15 not < 10, should be False
        assert sig.check(pt, {}, None) is False

    def test_diff_less_signal_with_shift(self):
        """Diff_Less_Signal applies self.shift to all get() calls."""
        sig = Diff_Less_Signal(5, "rsi_14", "rsi_7", 10.0)
        sig.set_shift(1)
        data = {
            ("rsi_14", 5, 1): 50.0,
            ("rsi_7", 5, 1): 45.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_diff_less_signal_nan_returns_false(self):
        """Diff_Less_Signal returns False when either value is NaN."""
        sig = Diff_Less_Signal(5, "rsi_14", "rsi_7", 10.0)
        data = {("rsi_7", 5, 0): 45.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestDiffLessIndiSignal:
    """Test Diff_LessIndi_Signal: (indi_1 - indi_2) > 0 AND (indi_1 - indi_2) < indi_dist."""

    def test_diff_less_indi_signal_true(self):
        """Diff_LessIndi_Signal returns True when diff > 0 AND diff < threshold."""
        sig = Diff_LessIndi_Signal(5, "rsi_14", "rsi_7", "atr_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 40.0,  # diff = 10
            ("atr_14", 5, 0): 15.0,  # threshold = 15
        }
        pt = MockDataPoint(data)
        # 50 - 40 = 10 > 0 AND 10 < 15, should be True
        assert sig.check(pt, {}, None) is True

    def test_diff_less_indi_signal_false_diff_negative(self):
        """Diff_LessIndi_Signal returns False when diff <= 0."""
        sig = Diff_LessIndi_Signal(5, "rsi_14", "rsi_7", "atr_14")
        data = {
            ("rsi_14", 5, 0): 30.0,
            ("rsi_7", 5, 0): 40.0,  # diff = -10
            ("atr_14", 5, 0): 15.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_diff_less_indi_signal_false_diff_too_large(self):
        """Diff_LessIndi_Signal returns False when diff >= threshold."""
        sig = Diff_LessIndi_Signal(5, "rsi_14", "rsi_7", "atr_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 30.0,  # diff = 20
            ("atr_14", 5, 0): 15.0,  # threshold = 15
        }
        pt = MockDataPoint(data)
        # 50 - 30 = 20 > 0 BUT 20 is not < 15, should be False
        assert sig.check(pt, {}, None) is False

    def test_diff_less_indi_signal_with_shift(self):
        """Diff_LessIndi_Signal applies self.shift to all get() calls."""
        sig = Diff_LessIndi_Signal(5, "rsi_14", "rsi_7", "atr_14")
        sig.set_shift(2)
        data = {
            ("rsi_14", 5, 2): 50.0,
            ("rsi_7", 5, 2): 40.0,
            ("atr_14", 5, 2): 15.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_diff_less_indi_signal_nan_returns_false(self):
        """Diff_LessIndi_Signal returns False when any value is NaN."""
        sig = Diff_LessIndi_Signal(5, "rsi_14", "rsi_7", "atr_14")
        data = {("rsi_14", 5, 0): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestDiffGreaterIndiSignal:
    """Test Diff_GreaterIndi_Signal: (indi_1 - indi_2) > 0 AND (indi_1 - indi_2) > indi_dist."""

    def test_diff_greater_indi_signal_true(self):
        """Diff_GreaterIndi_Signal returns True when diff > 0 AND diff > threshold."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 5, "rsi_7", 5, "atr_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 30.0,  # diff = 20
            ("atr_14", 5, 0): 15.0,  # threshold = 15
        }
        pt = MockDataPoint(data)
        # 50 - 30 = 20 > 0 AND 20 > 15, should be True
        assert sig.check(pt, {}, None) is True

    def test_diff_greater_indi_signal_false_diff_negative(self):
        """Diff_GreaterIndi_Signal returns False when diff <= 0."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 5, "rsi_7", 5, "atr_14")
        data = {
            ("rsi_14", 5, 0): 30.0,
            ("rsi_7", 5, 0): 50.0,  # diff = -20
            ("atr_14", 5, 0): 15.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_diff_greater_indi_signal_false_diff_too_small(self):
        """Diff_GreaterIndi_Signal returns False when diff <= threshold."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 5, "rsi_7", 5, "atr_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 5, 0): 40.0,  # diff = 10
            ("atr_14", 5, 0): 15.0,  # threshold = 15
        }
        pt = MockDataPoint(data)
        # 50 - 40 = 10 > 0 BUT 10 is not > 15, should be False
        assert sig.check(pt, {}, None) is False

    def test_diff_greater_indi_signal_with_shift(self):
        """Diff_GreaterIndi_Signal applies self.shift to all three get() calls."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 5, "rsi_7", 5, "atr_14")
        sig.set_shift(2)
        data = {
            ("rsi_14", 5, 2): 50.0,
            ("rsi_7", 5, 2): 30.0,
            ("atr_14", 5, 2): 15.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_diff_greater_indi_signal_cross_timeframe(self):
        """Diff_GreaterIndi_Signal can use different timeframes."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 15, "rsi_7", 1, "atr_14")
        data = {
            ("rsi_14", 5, 0): 50.0,
            ("rsi_7", 15, 0): 30.0,  # diff = 20
            ("atr_14", 1, 0): 15.0,  # threshold = 15
        }
        pt = MockDataPoint(data)
        # 50 - 30 = 20 > 0 AND 20 > 15, should be True
        assert sig.check(pt, {}, None) is True

    def test_diff_greater_indi_signal_stores_timeframes(self):
        """Diff_GreaterIndi_Signal stores separate tf values."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 15, "rsi_7", 1, "atr_14")
        # Verify the signal stores the correct timeframes
        assert sig.tf_1 == 5
        assert sig.tf_2 == 15
        assert sig.tf_dist == 1

    def test_diff_greater_indi_signal_nan_returns_false(self):
        """Diff_GreaterIndi_Signal returns False when any value is NaN."""
        sig = Diff_GreaterIndi_Signal(5, "rsi_14", 5, "rsi_7", 5, "atr_14")
        data = {("rsi_14", 5, 0): 50.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False
