"""Tests for crossover and level-based signals (Phase 04, Task 03)."""

import math
import pytest
from data import DataPoint
from signals_lib.common import (
    Cross_Up_Signal,
    Cross_Down_Signal,
    Cross_Up_Val_Signal,
    Cross_Down_Val_Signal,
    Near_Level_Signal,
    Near_Price_Level_Signal,
    Over_Level_Signal,
    Under_Level_Signal,
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


class TestCrossUpSignal:
    """Test Cross_Up_Signal: current indi_1 > indi_2 AND previous indi_1 < indi_2."""

    def test_cross_up_signal_true(self):
        """Cross_Up_Signal returns True when crossing above."""
        sig = Cross_Up_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 51.0,  # current > indi_2
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 48.0,  # previous < indi_2
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_up_signal_false_no_cross(self):
        """Cross_Up_Signal returns False when both above."""
        sig = Cross_Up_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 51.0,
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 52.0,  # also above
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_up_signal_false_both_below(self):
        """Cross_Up_Signal returns False when both below."""
        sig = Cross_Up_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 48.0,
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 47.0,
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_up_signal_with_shift(self):
        """Cross_Up_Signal applies self.shift to both current and previous."""
        sig = Cross_Up_Signal(5, "rsi_14", "rsi_7")
        sig.set_shift(1)
        data = {
            ("rsi_14", 5, 1): 51.0,
            ("rsi_7", 5, 1): 50.0,
            ("rsi_14", 5, 2): 48.0,
            ("rsi_7", 5, 2): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_up_signal_nan_current_returns_false(self):
        """Cross_Up_Signal returns False when current is NaN."""
        sig = Cross_Up_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 48.0,
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_up_signal_nan_previous_returns_false(self):
        """Cross_Up_Signal returns False when previous is NaN."""
        sig = Cross_Up_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 51.0,
            ("rsi_7", 5, 0): 50.0,
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestCrossDownSignal:
    """Test Cross_Down_Signal: current indi_1 < indi_2 AND previous indi_1 > indi_2."""

    def test_cross_down_signal_true(self):
        """Cross_Down_Signal returns True when crossing below."""
        sig = Cross_Down_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 48.0,  # current < indi_2
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 51.0,  # previous > indi_2
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_down_signal_false_no_cross(self):
        """Cross_Down_Signal returns False when both below."""
        sig = Cross_Down_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 48.0,
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 47.0,  # also below
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_down_signal_false_both_above(self):
        """Cross_Down_Signal returns False when both above."""
        sig = Cross_Down_Signal(5, "rsi_14", "rsi_7")
        data = {
            ("rsi_14", 5, 0): 51.0,
            ("rsi_7", 5, 0): 50.0,
            ("rsi_14", 5, 1): 52.0,
            ("rsi_7", 5, 1): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_down_signal_with_shift(self):
        """Cross_Down_Signal applies self.shift to both current and previous."""
        sig = Cross_Down_Signal(5, "rsi_14", "rsi_7")
        sig.set_shift(2)
        data = {
            ("rsi_14", 5, 2): 48.0,
            ("rsi_7", 5, 2): 50.0,
            ("rsi_14", 5, 3): 51.0,
            ("rsi_7", 5, 3): 50.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_down_signal_nan_returns_false(self):
        """Cross_Down_Signal returns False when any value is NaN."""
        sig = Cross_Down_Signal(5, "rsi_14", "rsi_7")
        data = {("rsi_14", 5, 0): 48.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestCrossUpValSignal:
    """Test Cross_Up_Val_Signal: current indi_1 > val AND previous indi_1 < val."""

    def test_cross_up_val_signal_true(self):
        """Cross_Up_Val_Signal returns True when crossing above threshold."""
        sig = Cross_Up_Val_Signal(5, "rsi_14", 50.0)
        data = {
            ("rsi_14", 5, 0): 51.0,  # current > 50
            ("rsi_14", 5, 1): 48.0,  # previous < 50
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_up_val_signal_false_no_cross(self):
        """Cross_Up_Val_Signal returns False when both above."""
        sig = Cross_Up_Val_Signal(5, "rsi_14", 50.0)
        data = {
            ("rsi_14", 5, 0): 53.0,
            ("rsi_14", 5, 1): 51.0,  # also > 50
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_up_val_signal_false_both_below(self):
        """Cross_Up_Val_Signal returns False when both below."""
        sig = Cross_Up_Val_Signal(5, "rsi_14", 50.0)
        data = {
            ("rsi_14", 5, 0): 48.0,
            ("rsi_14", 5, 1): 47.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_up_val_signal_with_shift(self):
        """Cross_Up_Val_Signal applies self.shift."""
        sig = Cross_Up_Val_Signal(15, "rsi_14", 50.0)
        sig.set_shift(1)
        data = {
            ("rsi_14", 15, 1): 51.0,
            ("rsi_14", 15, 2): 48.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_up_val_signal_nan_returns_false(self):
        """Cross_Up_Val_Signal returns False when value is NaN."""
        sig = Cross_Up_Val_Signal(5, "rsi_14", 50.0)
        data = {("rsi_14", 5, 0): 51.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestCrossDownValSignal:
    """Test Cross_Down_Val_Signal: current indi_1 < val AND previous indi_1 > val."""

    def test_cross_down_val_signal_true(self):
        """Cross_Down_Val_Signal returns True when crossing below threshold."""
        sig = Cross_Down_Val_Signal(5, "rsi_14", 50.0)
        data = {
            ("rsi_14", 5, 0): 48.0,  # current < 50
            ("rsi_14", 5, 1): 51.0,  # previous > 50
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_down_val_signal_false_no_cross(self):
        """Cross_Down_Val_Signal returns False when both below."""
        sig = Cross_Down_Val_Signal(5, "rsi_14", 50.0)
        data = {
            ("rsi_14", 5, 0): 48.0,
            ("rsi_14", 5, 1): 47.0,  # also < 50
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_down_val_signal_false_both_above(self):
        """Cross_Down_Val_Signal returns False when both above."""
        sig = Cross_Down_Val_Signal(5, "rsi_14", 50.0)
        data = {
            ("rsi_14", 5, 0): 53.0,
            ("rsi_14", 5, 1): 51.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_cross_down_val_signal_with_shift(self):
        """Cross_Down_Val_Signal applies self.shift."""
        sig = Cross_Down_Val_Signal(15, "rsi_14", 50.0)
        sig.set_shift(2)
        data = {
            ("rsi_14", 15, 2): 48.0,
            ("rsi_14", 15, 3): 51.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_cross_down_val_signal_nan_returns_false(self):
        """Cross_Down_Val_Signal returns False when value is NaN."""
        sig = Cross_Down_Val_Signal(5, "rsi_14", 50.0)
        data = {("rsi_14", 5, 1): 51.0}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestNearLevelSignal:
    """Test Near_Level_Signal: abs(price - level_val) <= buffer for any level."""

    def test_near_level_signal_true_exact_match(self):
        """Near_Level_Signal returns True when price matches a level exactly."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 100.0}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_near_level_signal_true_within_buffer(self):
        """Near_Level_Signal returns True when price within buffer of level."""
        sig = Near_Level_Signal(5, "close", 1, 1.0)
        data = {("close", 5, 0): 100.5}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_near_level_signal_true_multiple_levels(self):
        """Near_Level_Signal returns True for any level match."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 110.2}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_near_level_signal_false_outside_buffer(self):
        """Near_Level_Signal returns False when price outside all buffers."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 102.0}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_near_level_signal_false_empty_levels(self):
        """Near_Level_Signal returns False when no levels for type."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 100.0}
        levels = {1: [], 2: [100.0]}  # type 1 is empty
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_near_level_signal_false_missing_level_type(self):
        """Near_Level_Signal returns False when level_type missing."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 100.0}
        levels = {2: [100.0]}  # no type 1
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_near_level_signal_with_shift(self):
        """Near_Level_Signal applies self.shift to get() call."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        sig.set_shift(2)
        data = {("close", 5, 2): 100.0}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_near_level_signal_nan_returns_false(self):
        """Near_Level_Signal returns False when price is NaN."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_near_level_signal_get_data_exports_level(self):
        """Near_Level_Signal.get_data() exports matched level_val."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 100.2}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        sig.check(pt, levels, None)
        result = sig.get_data()
        assert "level_val" in result
        assert result["level_val"] == 100.0

    def test_near_level_signal_get_data_last_match_wins(self):
        """Near_Level_Signal.get_data() returns last matched level."""
        sig = Near_Level_Signal(5, "close", 1, 0.5)
        data = {("close", 5, 0): 100.1}
        levels = {1: [100.0, 100.2, 110.0]}
        pt = MockDataPoint(data)
        sig.check(pt, levels, None)
        result = sig.get_data()
        # Both 100.0 and 100.2 are within buffer; last one should win
        assert result["level_val"] == 100.2


class TestNearPriceLevelSignal:
    """Test Near_Price_Level_Signal: dynamic buffer = buffer × buffer_indi."""

    def test_near_price_level_signal_true(self):
        """Near_Price_Level_Signal returns True with dynamic buffer."""
        sig = Near_Price_Level_Signal(5, "close", 1, 0.1, "atr_14")
        data = {
            ("close", 5, 0): 100.5,
            ("atr_14", 5, 0): 5.0,  # effective_buffer = 0.1 * 5 = 0.5
        }
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        # |100.5 - 100| = 0.5 <= 0.5, should be True
        assert sig.check(pt, levels, None) is True

    def test_near_price_level_signal_false_outside_buffer(self):
        """Near_Price_Level_Signal returns False outside dynamic buffer."""
        sig = Near_Price_Level_Signal(5, "close", 1, 0.1, "atr_14")
        data = {
            ("close", 5, 0): 101.0,
            ("atr_14", 5, 0): 5.0,  # effective_buffer = 0.1 * 5 = 0.5
        }
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        # |101 - 100| = 1 > 0.5, should be False
        assert sig.check(pt, levels, None) is False

    def test_near_price_level_signal_with_shift(self):
        """Near_Price_Level_Signal.set_shift() propagates to inner signal."""
        sig = Near_Price_Level_Signal(5, "close", 1, 0.1, "atr_14")
        sig.set_shift(1)
        data = {
            ("close", 5, 1): 100.3,
            ("atr_14", 5, 1): 3.0,  # effective_buffer = 0.1 * 3 = 0.3
        }
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        # |100.3 - 100| = 0.3 <= 0.3, should be True
        assert sig.check(pt, levels, None) is True

    def test_near_price_level_signal_nan_buffer_indi(self):
        """Near_Price_Level_Signal returns False if buffer_indi is NaN."""
        sig = Near_Price_Level_Signal(5, "close", 1, 0.1, "atr_14")
        data = {("close", 5, 0): 100.5}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        # buffer_indi is NaN, should return False
        assert sig.check(pt, levels, None) is False


class TestOverLevelSignal:
    """Test Over_Level_Signal: indi_1 > level_val for any level."""

    def test_over_level_signal_true(self):
        """Over_Level_Signal returns True when above a level."""
        sig = Over_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 105.0}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_over_level_signal_false_below_all(self):
        """Over_Level_Signal returns False when below all levels."""
        sig = Over_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 95.0}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_over_level_signal_false_at_level(self):
        """Over_Level_Signal returns False when equal to level (not strictly >)."""
        sig = Over_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 100.0}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_over_level_signal_empty_levels(self):
        """Over_Level_Signal returns False when no levels for type."""
        sig = Over_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 105.0}
        levels = {1: []}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_over_level_signal_with_shift(self):
        """Over_Level_Signal applies self.shift."""
        sig = Over_Level_Signal(5, "close", 1)
        sig.set_shift(2)
        data = {("close", 5, 2): 105.0}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_over_level_signal_nan_returns_false(self):
        """Over_Level_Signal returns False when price is NaN."""
        sig = Over_Level_Signal(5, "close", 1)
        data = {}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False


class TestUnderLevelSignal:
    """Test Under_Level_Signal: indi_1 < level_val for any level."""

    def test_under_level_signal_true(self):
        """Under_Level_Signal returns True when below a level."""
        sig = Under_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 95.0}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_under_level_signal_false_above_all(self):
        """Under_Level_Signal returns False when above all levels."""
        sig = Under_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 115.0}
        levels = {1: [100.0, 110.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_under_level_signal_false_at_level(self):
        """Under_Level_Signal returns False when equal to level (not strictly <)."""
        sig = Under_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 100.0}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_under_level_signal_empty_levels(self):
        """Under_Level_Signal returns False when no levels for type."""
        sig = Under_Level_Signal(5, "close", 1)
        data = {("close", 5, 0): 95.0}
        levels = {1: []}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False

    def test_under_level_signal_with_shift(self):
        """Under_Level_Signal applies self.shift."""
        sig = Under_Level_Signal(5, "close", 1)
        sig.set_shift(1)
        data = {("close", 5, 1): 95.0}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is True

    def test_under_level_signal_nan_returns_false(self):
        """Under_Level_Signal returns False when price is NaN."""
        sig = Under_Level_Signal(5, "close", 1)
        data = {}
        levels = {1: [100.0]}
        pt = MockDataPoint(data)
        assert sig.check(pt, levels, None) is False
