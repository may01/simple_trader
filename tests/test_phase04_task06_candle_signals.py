"""Tests for candle pattern signals (Phase 04, Task 06)."""

import math
import pandas as pd

import pytest
from data import DataPoint, LiveDataPoint
from signals_lib.candle import (
    BigCandle_Signal,
    LongTale_Signal,
    ShortTale_Signal,
    CandleUP_Signal,
    CandleDOWN_Signal,
)
from constants import CANDLE_HIGH, CANDLE_LOW


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


class TestBigCandleSignal:
    """Test BigCandle_Signal: natr_14 > coef * natr_ma."""

    def test_big_candle_true(self):
        """BigCandle_Signal returns True when natr_14 > coef * natr_ma."""
        sig = BigCandle_Signal(15, 2.0)
        data = {
            ("natr_14", 15, 0): 2.5,  # 2.5 > 2.0 * 1.0 = True
            ("natr_ma", 15, 0): 1.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_big_candle_false(self):
        """BigCandle_Signal returns False when natr_14 <= coef * natr_ma."""
        sig = BigCandle_Signal(15, 2.0)
        data = {
            ("natr_14", 15, 0): 1.5,  # 1.5 <= 2.0 * 1.0 = False
            ("natr_ma", 15, 0): 1.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_big_candle_equal(self):
        """BigCandle_Signal returns False when natr_14 == coef * natr_ma."""
        sig = BigCandle_Signal(15, 2.0)
        data = {
            ("natr_14", 15, 0): 2.0,  # 2.0 = 2.0 * 1.0 = False (not greater)
            ("natr_ma", 15, 0): 1.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_big_candle_with_shift(self):
        """BigCandle_Signal applies self.shift to both get() calls."""
        sig = BigCandle_Signal(15, 2.0)
        sig.set_shift(1)
        data = {
            ("natr_14", 15, 1): 3.0,
            ("natr_ma", 15, 1): 1.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_big_candle_nan_natr_14(self):
        """BigCandle_Signal returns False when natr_14 is NaN."""
        sig = BigCandle_Signal(15, 2.0)
        data = {
            ("natr_ma", 15, 0): 1.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_big_candle_nan_natr_ma(self):
        """BigCandle_Signal returns False when natr_ma is NaN."""
        sig = BigCandle_Signal(15, 2.0)
        data = {
            ("natr_14", 15, 0): 2.5,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_big_candle_both_nan(self):
        """BigCandle_Signal returns False when both are NaN."""
        sig = BigCandle_Signal(15, 2.0)
        data = {}
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestLongTaleSignal:
    """Test LongTale_Signal: lower or upper wick > 0.5 * candle_size."""

    def test_long_tale_lower_wick_true(self):
        """LongTale_Signal (CANDLE_LOW) returns True for long lower wick."""
        sig = LongTale_Signal(15, CANDLE_LOW)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 21.0,
            ("high", 15, 0): 22.0,
            ("low", 15, 0): 18.0,
        }
        # candle_size = 22.0 - 18.0 = 4.0
        # min(20.0, 21.0) = 20.0
        # 20.0 - 18.0 = 2.0
        # 2.0 > 0.5 * 4.0 = 2.0 ? No, exactly equal => False
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_long_tale_lower_wick_true_greater(self):
        """LongTale_Signal (CANDLE_LOW) returns True for long lower wick (strictly greater)."""
        sig = LongTale_Signal(15, CANDLE_LOW)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 21.0,
            ("high", 15, 0): 22.0,
            ("low", 15, 0): 17.5,
        }
        # candle_size = 22.0 - 17.5 = 4.5
        # min(20.0, 21.0) = 20.0
        # 20.0 - 17.5 = 2.5
        # 2.5 > 0.5 * 4.5 = 2.25 ? Yes => True
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_long_tale_upper_wick_true(self):
        """LongTale_Signal (CANDLE_HIGH) returns True for long upper wick."""
        sig = LongTale_Signal(15, CANDLE_HIGH)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 19.5,
            ("high", 15, 0): 23.0,
            ("low", 15, 0): 18.0,
        }
        # candle_size = 23.0 - 18.0 = 5.0
        # max(20.0, 19.5) = 20.0
        # 23.0 - 20.0 = 3.0
        # 3.0 > 0.5 * 5.0 = 2.5 ? Yes => True
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_long_tale_doji_candle_size_zero(self):
        """LongTale_Signal returns False when candle_size == 0 (doji)."""
        sig = LongTale_Signal(15, CANDLE_LOW)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 20.0,
            ("high", 15, 0): 20.0,
            ("low", 15, 0): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_long_tale_with_shift(self):
        """LongTale_Signal applies self.shift to all get() calls."""
        sig = LongTale_Signal(15, CANDLE_LOW)
        sig.set_shift(2)
        data = {
            ("open", 15, 2): 20.0,
            ("close", 15, 2): 21.0,
            ("high", 15, 2): 22.0,
            ("low", 15, 2): 17.5,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_long_tale_nan_returns_false(self):
        """LongTale_Signal returns False when any OHLC value is NaN."""
        sig = LongTale_Signal(15, CANDLE_LOW)
        # Missing close value
        data = {
            ("open", 15, 0): 20.0,
            ("high", 15, 0): 22.0,
            ("low", 15, 0): 18.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestShortTaleSignal:
    """Test ShortTale_Signal: lower or upper wick < 0.2 * candle_size."""

    def test_short_tale_lower_wick_true(self):
        """ShortTale_Signal (CANDLE_LOW) returns True for short lower wick."""
        sig = ShortTale_Signal(15, CANDLE_LOW)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 21.0,
            ("high", 15, 0): 22.0,
            ("low", 15, 0): 19.5,
        }
        # candle_size = 22.0 - 19.5 = 2.5
        # min(20.0, 21.0) = 20.0
        # 20.0 - 19.5 = 0.5
        # 0.5 < 0.2 * 2.5 = 0.5 ? No, exactly equal => False
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_short_tale_lower_wick_true_less(self):
        """ShortTale_Signal (CANDLE_LOW) returns True for short lower wick (strictly less)."""
        sig = ShortTale_Signal(15, CANDLE_LOW)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 21.0,
            ("high", 15, 0): 22.0,
            ("low", 15, 0): 19.6,
        }
        # candle_size = 22.0 - 19.6 = 2.4
        # min(20.0, 21.0) = 20.0
        # 20.0 - 19.6 = 0.4
        # 0.4 < 0.2 * 2.4 = 0.48 ? Yes => True
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_short_tale_upper_wick_true(self):
        """ShortTale_Signal (CANDLE_HIGH) returns True for short upper wick."""
        sig = ShortTale_Signal(15, CANDLE_HIGH)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 19.5,
            ("high", 15, 0): 20.3,
            ("low", 15, 0): 18.0,
        }
        # candle_size = 20.3 - 18.0 = 2.3
        # max(20.0, 19.5) = 20.0
        # 20.3 - 20.0 = 0.3
        # 0.3 < 0.2 * 2.3 = 0.46 ? Yes => True
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_short_tale_doji_candle_size_zero(self):
        """ShortTale_Signal returns False when candle_size == 0 (doji)."""
        sig = ShortTale_Signal(15, CANDLE_LOW)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 20.0,
            ("high", 15, 0): 20.0,
            ("low", 15, 0): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_short_tale_with_shift(self):
        """ShortTale_Signal applies self.shift to all get() calls."""
        sig = ShortTale_Signal(15, CANDLE_HIGH)
        sig.set_shift(1)
        data = {
            ("open", 15, 1): 20.0,
            ("close", 15, 1): 19.5,
            ("high", 15, 1): 20.3,
            ("low", 15, 1): 18.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True


class TestCandleUPSignal:
    """Test CandleUP_Signal: open < close (bullish)."""

    def test_candle_up_true(self):
        """CandleUP_Signal returns True when open < close."""
        sig = CandleUP_Signal(15)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 21.5,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_candle_up_false(self):
        """CandleUP_Signal returns False when open >= close."""
        sig = CandleUP_Signal(15)
        data = {
            ("open", 15, 0): 21.5,
            ("close", 15, 0): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_candle_up_equal(self):
        """CandleUP_Signal returns False when open == close."""
        sig = CandleUP_Signal(15)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_candle_up_with_shift(self):
        """CandleUP_Signal applies self.shift to both get() calls."""
        sig = CandleUP_Signal(15)
        sig.set_shift(2)
        data = {
            ("open", 15, 2): 20.0,
            ("close", 15, 2): 21.5,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_candle_up_nan_returns_false(self):
        """CandleUP_Signal returns False when any value is NaN."""
        sig = CandleUP_Signal(15)
        data = {("open", 15, 0): 20.0}  # Missing close
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False


class TestCandleDOWNSignal:
    """Test CandleDOWN_Signal: open > close (bearish)."""

    def test_candle_down_true(self):
        """CandleDOWN_Signal returns True when open > close."""
        sig = CandleDOWN_Signal(15)
        data = {
            ("open", 15, 0): 21.5,
            ("close", 15, 0): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_candle_down_false(self):
        """CandleDOWN_Signal returns False when open <= close."""
        sig = CandleDOWN_Signal(15)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 21.5,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_candle_down_equal(self):
        """CandleDOWN_Signal returns False when open == close."""
        sig = CandleDOWN_Signal(15)
        data = {
            ("open", 15, 0): 20.0,
            ("close", 15, 0): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False

    def test_candle_down_with_shift(self):
        """CandleDOWN_Signal applies self.shift to both get() calls."""
        sig = CandleDOWN_Signal(15)
        sig.set_shift(2)
        data = {
            ("open", 15, 2): 21.5,
            ("close", 15, 2): 20.0,
        }
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is True

    def test_candle_down_nan_returns_false(self):
        """CandleDOWN_Signal returns False when any value is NaN."""
        sig = CandleDOWN_Signal(15)
        data = {("open", 15, 0): 20.0}  # Missing close
        pt = MockDataPoint(data)
        assert sig.check(pt, {}, None) is False
