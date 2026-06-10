"""Tests for divergence and bounce complex signals (Phase 04, Task 05)."""

import math
import pytest
from data import DataPoint
from signals_lib.base_signal import BaseSignal


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


# ---------------------------------------------------------------------------
# Diver_Bull_signal tests
# ---------------------------------------------------------------------------


class TestDiverBullSignal:
    """Tests for Diver_Bull_signal: bullish divergence detection."""

    def _make_dp(self, indicator_vals: list, close_vals: list, tf: int = 5):
        """Build a MockDataPoint from lists (index 0 = current, 1 = 1 ago, ...).

        Args:
            indicator_vals: indicator values at shift 0, 1, 2, ... respectively.
            close_vals: close price values at shift 0, 1, 2, ... respectively.
            tf: timeframe to use.
        """
        data = {}
        for i, v in enumerate(indicator_vals):
            data[("rsi", tf, i)] = v
        for i, v in enumerate(close_vals):
            data[("close", tf, i)] = v
        return MockDataPoint(data)

    def test_basic_bullish_divergence_fires(self):
        """Diver_Bull_signal fires when indicator < level and divergence pattern exists."""
        from signals_lib.complex import Diver_Bull_signal

        # current RSI = 25 (< level=30)
        # prior valley at shift=3: RSI=20 (< level=30, confirmed: RSI[2]>RSI[3]<RSI[4])
        # current RSI (25) > prior RSI (20) → higher low in indicator ✓
        # current close (90) < prior close at valley (100) → lower low in price ✓
        # finished=True: check confirmation at prior valley
        indicator_vals = [25, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is True

    def test_no_fire_when_indicator_above_level(self):
        """Diver_Bull_signal does not fire when current indicator >= level."""
        from signals_lib.complex import Diver_Bull_signal

        # current RSI=35 >= level=30 → should return False
        indicator_vals = [35, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_cancel_level_triggers_reset(self):
        """Diver_Bull_signal returns True from reset() when indicator > cancel_level."""
        from signals_lib.complex import Diver_Bull_signal

        # indicator currently above cancel_level → reset() should abort
        indicator_vals = [55, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.reset() is True

    def test_reset_returns_false_when_not_cancelled(self):
        """Diver_Bull_signal reset() returns False when no cancel triggered."""
        from signals_lib.complex import Diver_Bull_signal

        # normal situation, no cancel
        indicator_vals = [25, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.reset() is False

    def test_no_fire_when_prior_valley_above_level(self):
        """Diver_Bull_signal does not fire when all prior indicator values >= level."""
        from signals_lib.complex import Diver_Bull_signal

        # prior values all above level=30 → no valley found
        indicator_vals = [25, 35, 38, 40, 42, 45, 38, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_no_fire_when_indicator_not_higher_low(self):
        """Diver_Bull_signal does not fire when current indicator < prior valley (not higher low)."""
        from signals_lib.complex import Diver_Bull_signal

        # current RSI=18 < prior valley RSI=20 → not a higher low
        indicator_vals = [18, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_no_fire_when_price_not_lower_low(self):
        """Diver_Bull_signal does not fire when current price > prior price (not lower low)."""
        from signals_lib.complex import Diver_Bull_signal

        # current close=105 > prior close=100 → not a lower low in price
        indicator_vals = [25, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [105, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_finished_false_does_not_require_confirmation(self):
        """Diver_Bull_signal with finished=False does not require 3-candle valley confirmation.

        The data is constructed so that all sub-level valleys are unconfirmed (monotonically
        decreasing indicator within the scan window), ensuring finished=True returns False while
        finished=False returns True (finding the first unconfirmed valley with divergence).
        """
        from signals_lib.complex import Diver_Bull_signal

        # Monotonically decreasing indicator in scan window → all valleys unconfirmed.
        # indicator_vals[7] has NaN for its 'next' confirmation value (offset 8 is out of range).
        # Only unconfirmed valleys exist. finished=True → False. finished=False → True.
        indicator_vals = [25, 28, 29, 20, 17, 14, 12, 10]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        # finished=True should NOT fire (no confirmed valley)
        sig_finished = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig_finished.check(dp, {}, None) is False
        # finished=False should fire (finds first unconfirmed valley with divergence at offset=3)
        sig_unfinished = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=False)
        assert sig_unfinished.check(dp, {}, None) is True

    def test_get_data_returns_prior_valley_info(self):
        """Diver_Bull_signal.get_data() returns prior valley indicator and price values."""
        from signals_lib.complex import Diver_Bull_signal

        indicator_vals = [25, 28, 26, 20, 24, 35, 40, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        data = sig.get_data()
        assert "prior_indicator" in data
        assert "prior_price" in data
        assert data["prior_indicator"] == 20.0
        assert data["prior_price"] == 100.0

    def test_get_data_empty_when_no_divergence(self):
        """Diver_Bull_signal.get_data() returns empty dict when no divergence found."""
        from signals_lib.complex import Diver_Bull_signal

        # No prior valley below level
        indicator_vals = [25, 35, 38, 40, 42, 45, 38, 50]
        close_vals =     [90, 95, 97, 100, 98, 102, 110, 115]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.get_data() == {}

    def test_respects_shift(self):
        """Diver_Bull_signal with set_shift() uses shifted data lookups."""
        from signals_lib.complex import Diver_Bull_signal

        # Build data shifted by 1: signal at position 1 in history
        # At shift=1: indicator=25, at shift=4: indicator=20 (valley, confirmed)
        # close at shift=1: 90, close at shift=4: 100
        data = {}
        tf = 5
        # shift=1 is current when set_shift(1)
        # actual values at shifts 1..8 match the divergence pattern
        data[("rsi", tf, 1)] = 25.0
        data[("rsi", tf, 2)] = 28.0
        data[("rsi", tf, 3)] = 26.0
        data[("rsi", tf, 4)] = 20.0
        data[("rsi", tf, 5)] = 24.0
        data[("rsi", tf, 6)] = 35.0
        data[("rsi", tf, 7)] = 40.0
        data[("rsi", tf, 8)] = 50.0
        data[("close", tf, 1)] = 90.0
        data[("close", tf, 4)] = 100.0
        dp = MockDataPoint(data)
        sig = Diver_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.set_shift(1)
        assert sig.check(dp, {}, None) is True


# ---------------------------------------------------------------------------
# Diver_Bear_signal tests
# ---------------------------------------------------------------------------


class TestDiverBearSignal:
    """Tests for Diver_Bear_signal: bearish divergence detection."""

    def _make_dp(self, indicator_vals: list, close_vals: list, tf: int = 5):
        """Build a MockDataPoint for bearish divergence tests."""
        data = {}
        for i, v in enumerate(indicator_vals):
            data[("rsi", tf, i)] = v
        for i, v in enumerate(close_vals):
            data[("close", tf, i)] = v
        return MockDataPoint(data)

    def test_basic_bearish_divergence_fires(self):
        """Diver_Bear_signal fires when indicator > level and bearish divergence pattern exists."""
        from signals_lib.complex import Diver_Bear_signal

        # current RSI=75 (> level=70)
        # prior peak at shift=3: RSI=80 (> level=70, confirmed: RSI[2]<RSI[3]>RSI[4])
        # current RSI (75) < prior peak (80) → lower high in indicator ✓
        # current close (110) > prior close at peak (100) → higher high in price ✓
        indicator_vals = [75, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is True

    def test_no_fire_when_indicator_below_level(self):
        """Diver_Bear_signal does not fire when current indicator <= level."""
        from signals_lib.complex import Diver_Bear_signal

        indicator_vals = [65, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_cancel_level_triggers_reset(self):
        """Diver_Bear_signal returns True from reset() when indicator < cancel_level."""
        from signals_lib.complex import Diver_Bear_signal

        # indicator currently below cancel_level=50 → reset() should return True
        indicator_vals = [45, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.reset() is True

    def test_reset_returns_false_when_not_cancelled(self):
        """Diver_Bear_signal reset() returns False when no cancel triggered."""
        from signals_lib.complex import Diver_Bear_signal

        indicator_vals = [75, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.reset() is False

    def test_no_fire_when_prior_peak_below_level(self):
        """Diver_Bear_signal does not fire when all prior values are below level."""
        from signals_lib.complex import Diver_Bear_signal

        # prior peaks all below level=70
        indicator_vals = [75, 65, 64, 60, 62, 65, 63, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_no_fire_when_indicator_not_lower_high(self):
        """Diver_Bear_signal does not fire when current indicator > prior peak (not lower high)."""
        from signals_lib.complex import Diver_Bear_signal

        # current RSI=85 > prior peak=80 → not a lower high
        indicator_vals = [85, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_no_fire_when_price_not_higher_high(self):
        """Diver_Bear_signal does not fire when current price < prior price (not higher high)."""
        from signals_lib.complex import Diver_Bear_signal

        # current close=95 < prior close=100 → not a higher high
        indicator_vals = [75, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [95, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_get_data_returns_prior_peak_info(self):
        """Diver_Bear_signal.get_data() returns prior peak indicator and price values."""
        from signals_lib.complex import Diver_Bear_signal

        indicator_vals = [75, 72, 74, 80, 76, 65, 60, 50]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        data = sig.get_data()
        assert "prior_indicator" in data
        assert "prior_price" in data
        assert data["prior_indicator"] == 80.0
        assert data["prior_price"] == 100.0

    def test_finished_false_no_confirmation_required(self):
        """Diver_Bear_signal with finished=False does not require confirmation candles.

        The data is constructed so that all above-level peaks are unconfirmed (monotonically
        increasing indicator within the scan window), ensuring finished=True returns False while
        finished=False returns True.
        """
        from signals_lib.complex import Diver_Bear_signal

        # Monotonically increasing indicator in scan window → all peaks unconfirmed.
        # indicator[7] has NaN 'next' → confirmation fails for that too.
        # Only unconfirmed peaks exist. finished=True → False. finished=False → True.
        indicator_vals = [75, 72, 73, 74, 76, 78, 79, 80]
        close_vals =     [110, 105, 103, 100, 102, 98, 90, 85]
        dp = self._make_dp(indicator_vals, close_vals)
        sig_finished = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig_finished.check(dp, {}, None) is False
        sig_unfinished = Diver_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=False)
        assert sig_unfinished.check(dp, {}, None) is True


# ---------------------------------------------------------------------------
# Diver_Hidden_Bull_signal tests
# ---------------------------------------------------------------------------


class TestDiverHiddenBullSignal:
    """Tests for Diver_Hidden_Bull_signal: hidden bullish divergence detection."""

    def _make_dp(self, indicator_vals: list, low_vals: list, tf: int = 5):
        """Build a MockDataPoint for hidden bull divergence tests."""
        data = {}
        for i, v in enumerate(indicator_vals):
            data[("rsi", tf, i)] = v
        for i, v in enumerate(low_vals):
            data[("low", tf, i)] = v
        return MockDataPoint(data)

    def test_basic_hidden_bull_divergence_fires(self):
        """Diver_Hidden_Bull_signal fires when hidden bull pattern exists.

        Confirmed valley at offset=4: RSI[3]=25 > RSI[4]=22 < RSI[5]=35.
        prior_rsi=22 > cur_rsi=20 (lower low in indicator) and
        prior_low=88 < cur_low=95 (higher low in price).
        """
        from signals_lib.complex import Diver_Hidden_Bull_signal

        # Confirmed valley at offset=4: RSI[3]=25 > RSI[4]=22 < RSI[5]=35
        indicator_vals = [20, 28, 27, 25, 22, 35, 40, 50]
        low_vals =       [95, 92, 93, 90, 88, 100, 110, 115]
        dp = self._make_dp(indicator_vals, low_vals)
        sig = Diver_Hidden_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is True

    def test_no_fire_when_indicator_above_level(self):
        """Diver_Hidden_Bull_signal does not fire when current indicator >= level."""
        from signals_lib.complex import Diver_Hidden_Bull_signal

        indicator_vals = [35, 22, 24, 25, 28, 35, 40, 50]
        low_vals =       [95, 92, 93, 90, 88, 100, 110, 115]
        dp = self._make_dp(indicator_vals, low_vals)
        sig = Diver_Hidden_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_cancel_level_triggers_reset(self):
        """Diver_Hidden_Bull_signal cancel: if any historical value > cancel_level, reset() returns True."""
        from signals_lib.complex import Diver_Hidden_Bull_signal

        # A historical value rises above cancel_level=50 during scan
        indicator_vals = [20, 22, 24, 25, 55, 35, 40, 50]
        low_vals =       [95, 92, 93, 90, 88, 100, 110, 115]
        dp = self._make_dp(indicator_vals, low_vals)
        sig = Diver_Hidden_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.reset() is True

    def test_no_fire_when_price_not_higher_low(self):
        """Diver_Hidden_Bull_signal does not fire when current price <= prior price."""
        from signals_lib.complex import Diver_Hidden_Bull_signal

        # Use same indicator data with confirmed valley at offset=4 (RSI=22),
        # but current low=85 < prior low at valley=88 → not a higher low in price
        indicator_vals = [20, 28, 27, 25, 22, 35, 40, 50]
        low_vals =       [85, 92, 93, 90, 88, 100, 110, 115]  # cur=85 < hist[4]=88
        dp = self._make_dp(indicator_vals, low_vals)
        sig = Diver_Hidden_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_uses_low_not_close(self):
        """Diver_Hidden_Bull_signal uses 'low' for price comparison, not 'close'."""
        from signals_lib.complex import Diver_Hidden_Bull_signal

        # Confirmed valley at offset=4: RSI[3]=25 > RSI[4]=22 < RSI[5]=35
        indicator_vals = [20, 28, 27, 25, 22, 35, 40, 50]
        low_vals =       [95, 92, 93, 90, 88, 100, 110, 115]  # cur low > hist low → fires
        data = {}
        tf = 5
        for i, v in enumerate(indicator_vals):
            data[("rsi", tf, i)] = v
        for i, v in enumerate(low_vals):
            data[("low", tf, i)] = v
        # close values would fail (current close < prior close at valley)
        close_data = [80, 92, 93, 90, 95, 100, 110, 115]  # cur close=80 < hist close=95
        for i, v in enumerate(close_data):
            data[("close", tf, i)] = v
        dp = MockDataPoint(data)
        sig = Diver_Hidden_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        # Should still fire because 'low' passes (uses low, not close)
        assert sig.check(dp, {}, None) is True

    def test_get_data_returns_info(self):
        """Diver_Hidden_Bull_signal.get_data() returns prior valley info."""
        from signals_lib.complex import Diver_Hidden_Bull_signal

        indicator_vals = [20, 28, 27, 25, 22, 35, 40, 50]
        low_vals =       [95, 92, 93, 90, 88, 100, 110, 115]
        dp = self._make_dp(indicator_vals, low_vals)
        sig = Diver_Hidden_Bull_signal(tf=5, level=30.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        data = sig.get_data()
        assert "prior_indicator" in data
        assert "prior_price" in data


# ---------------------------------------------------------------------------
# Diver_Hidden_Bear_signal tests
# ---------------------------------------------------------------------------


class TestDiverHiddenBearSignal:
    """Tests for Diver_Hidden_Bear_signal: hidden bearish divergence detection."""

    def _make_dp(self, indicator_vals: list, high_vals: list, tf: int = 5):
        """Build a MockDataPoint for hidden bear divergence tests."""
        data = {}
        for i, v in enumerate(indicator_vals):
            data[("rsi", tf, i)] = v
        for i, v in enumerate(high_vals):
            data[("high", tf, i)] = v
        return MockDataPoint(data)

    def test_basic_hidden_bear_divergence_fires(self):
        """Diver_Hidden_Bear_signal fires when hidden bear pattern exists."""
        from signals_lib.complex import Diver_Hidden_Bear_signal

        # Hidden bear: current indicator higher high, current price lower high
        # current RSI=82 (> level=70)
        # prior peak at shift=3: RSI=78 (> level=70, confirmed)
        # prior RSI (78) < current RSI (82) → higher high in indicator ✓
        # prior high (110) > current high (105) → lower high in price ✓
        indicator_vals = [82, 79, 77, 78, 75, 65, 60, 50]
        high_vals =      [105, 108, 109, 110, 108, 98, 90, 85]
        dp = self._make_dp(indicator_vals, high_vals)
        sig = Diver_Hidden_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is True

    def test_no_fire_when_indicator_below_level(self):
        """Diver_Hidden_Bear_signal does not fire when current indicator <= level."""
        from signals_lib.complex import Diver_Hidden_Bear_signal

        indicator_vals = [65, 79, 77, 78, 75, 65, 60, 50]
        high_vals =      [105, 108, 109, 110, 108, 98, 90, 85]
        dp = self._make_dp(indicator_vals, high_vals)
        sig = Diver_Hidden_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_cancel_level_triggers_reset(self):
        """Diver_Hidden_Bear_signal cancel: if any historical value < cancel_level, reset() True."""
        from signals_lib.complex import Diver_Hidden_Bear_signal

        # A historical value drops below cancel_level=50
        indicator_vals = [82, 79, 77, 78, 45, 65, 60, 50]
        high_vals =      [105, 108, 109, 110, 108, 98, 90, 85]
        dp = self._make_dp(indicator_vals, high_vals)
        sig = Diver_Hidden_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        assert sig.reset() is True

    def test_no_fire_when_price_not_lower_high(self):
        """Diver_Hidden_Bear_signal does not fire when current price >= prior price."""
        from signals_lib.complex import Diver_Hidden_Bear_signal

        # current high=115 > prior high=110 → not a lower high in price
        indicator_vals = [82, 79, 77, 78, 75, 65, 60, 50]
        high_vals =      [115, 108, 109, 110, 108, 98, 90, 85]
        dp = self._make_dp(indicator_vals, high_vals)
        sig = Diver_Hidden_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is False

    def test_uses_high_not_close(self):
        """Diver_Hidden_Bear_signal uses 'high' for price comparison, not 'close'."""
        from signals_lib.complex import Diver_Hidden_Bear_signal

        indicator_vals = [82, 79, 77, 78, 75, 65, 60, 50]
        high_vals =      [105, 108, 109, 110, 108, 98, 90, 85]
        data = {}
        tf = 5
        for i, v in enumerate(indicator_vals):
            data[("rsi", tf, i)] = v
        for i, v in enumerate(high_vals):
            data[("high", tf, i)] = v
        # close would fail (higher close)
        for i, v in enumerate([120, 108, 109, 110, 108, 98, 90, 85]):
            data[("close", tf, i)] = v
        dp = MockDataPoint(data)
        sig = Diver_Hidden_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        assert sig.check(dp, {}, None) is True

    def test_get_data_returns_info(self):
        """Diver_Hidden_Bear_signal.get_data() returns prior peak info."""
        from signals_lib.complex import Diver_Hidden_Bear_signal

        indicator_vals = [82, 79, 77, 78, 75, 65, 60, 50]
        high_vals =      [105, 108, 109, 110, 108, 98, 90, 85]
        dp = self._make_dp(indicator_vals, high_vals)
        sig = Diver_Hidden_Bear_signal(tf=5, level=70.0, cancel_level=50.0, indicator="rsi", finished=True)
        sig.check(dp, {}, None)
        data = sig.get_data()
        assert "prior_indicator" in data
        assert "prior_price" in data


# ---------------------------------------------------------------------------
# Bounce_Long_Signal tests
# ---------------------------------------------------------------------------


class TestBounceLongSignal:
    """Tests for Bounce_Long_Signal: indi bounces back toward indi_ma from above."""

    def test_imports(self):
        """Bounce_Long_Signal can be imported."""
        from signals_lib.complex import Bounce_Long_Signal
        assert Bounce_Long_Signal is not None

    def test_delegates_to_inner_and_signal(self):
        """Bounce_Long_Signal has an inner And_Signal."""
        from signals_lib.complex import Bounce_Long_Signal
        from signals_lib.operations import And_Signal

        sig = Bounce_Long_Signal(tf=5, indi="macd", indi_ma="macd_signal")
        assert hasattr(sig, "_inner")
        assert isinstance(sig._inner, And_Signal)

    def test_check_delegates_to_inner(self):
        """Bounce_Long_Signal.check() delegates to inner And_Signal."""
        from signals_lib.complex import Bounce_Long_Signal

        # indi=5, indi_ma=3 → diff=2 (at shift=0 and 1): gap>1 ✓
        # at current (shift=0): indi=4, indi_ma=3.5 → diff=0.5 < 1 ✓ (gap closed)
        data = {}
        tf = 5
        # shift=0 (current): indi=4, indi_ma=3.5  → diff_indi_ma = 0.5 < 1 (Or branch fires)
        data[("indi", tf, 0)] = 4.0
        data[("indi_ma", tf, 0)] = 3.5
        # shift=1 (1 ago): indi=5, indi_ma=2 → diff=3 > 1 (History branch 1 fires)
        data[("indi", tf, 1)] = 5.0
        data[("indi_ma", tf, 1)] = 2.0
        # shift=2 (2 ago): indi=6, indi_ma=2 → diff=4 > 1 (History branch 2 fires)
        data[("indi", tf, 2)] = 6.0
        data[("indi_ma", tf, 2)] = 2.0

        dp = MockDataPoint(data)
        sig = Bounce_Long_Signal(tf=5, indi="indi", indi_ma="indi_ma")
        result = sig.check(dp, {}, None)
        assert result is True

    def test_no_fire_when_gap_not_previously_large(self):
        """Bounce_Long_Signal does not fire when historical gap was small."""
        from signals_lib.complex import Bounce_Long_Signal

        data = {}
        tf = 5
        # current gap small (ok), but historical gaps too small
        data[("indi", tf, 0)] = 4.0
        data[("indi_ma", tf, 0)] = 3.5  # diff=0.5 < 1 (ok)
        data[("indi", tf, 1)] = 4.5
        data[("indi_ma", tf, 1)] = 4.0  # diff=0.5 NOT > 1 (history fails)
        data[("indi", tf, 2)] = 4.5
        data[("indi_ma", tf, 2)] = 4.0  # diff=0.5 NOT > 1

        dp = MockDataPoint(data)
        sig = Bounce_Long_Signal(tf=5, indi="indi", indi_ma="indi_ma")
        result = sig.check(dp, {}, None)
        assert result is False

    def test_set_shift_propagates_to_inner(self):
        """Bounce_Long_Signal.set_shift() propagates to inner And_Signal."""
        from signals_lib.complex import Bounce_Long_Signal
        from signals_lib.operations import And_Signal

        sig = Bounce_Long_Signal(tf=5, indi="indi", indi_ma="indi_ma")
        sig.set_shift(3)
        # The inner And_Signal should have shift=3 propagated
        assert sig._inner.shift == 3


# ---------------------------------------------------------------------------
# Bounce_Short_Signal tests
# ---------------------------------------------------------------------------


class TestBounceShortSignal:
    """Tests for Bounce_Short_Signal: indi_ma bounces back toward indi from above."""

    def test_imports(self):
        """Bounce_Short_Signal can be imported."""
        from signals_lib.complex import Bounce_Short_Signal
        assert Bounce_Short_Signal is not None

    def test_delegates_to_inner_and_signal(self):
        """Bounce_Short_Signal has an inner And_Signal."""
        from signals_lib.complex import Bounce_Short_Signal
        from signals_lib.operations import And_Signal

        sig = Bounce_Short_Signal(tf=5, indi="macd", indi_ma="macd_signal")
        assert hasattr(sig, "_inner")
        assert isinstance(sig._inner, And_Signal)

    def test_check_delegates_to_inner(self):
        """Bounce_Short_Signal.check() delegates to inner And_Signal."""
        from signals_lib.complex import Bounce_Short_Signal

        # indi_ma was well above indi, now gap closed
        data = {}
        tf = 5
        # current: indi=3.5, indi_ma=4 → diff_indi_ma_indi = 0.5 < 1 (Or branch fires)
        data[("indi", tf, 0)] = 3.5
        data[("indi_ma", tf, 0)] = 4.0
        # 1 ago: indi_ma=6, indi=2 → diff_indi_ma_indi = 4 > 1 (History 1 fires)
        data[("indi", tf, 1)] = 2.0
        data[("indi_ma", tf, 1)] = 6.0
        # 2 ago: indi_ma=7, indi=2 → diff_indi_ma_indi = 5 > 1 (History 2 fires)
        data[("indi", tf, 2)] = 2.0
        data[("indi_ma", tf, 2)] = 7.0

        dp = MockDataPoint(data)
        sig = Bounce_Short_Signal(tf=5, indi="indi", indi_ma="indi_ma")
        result = sig.check(dp, {}, None)
        assert result is True

    def test_no_fire_when_gap_not_previously_large(self):
        """Bounce_Short_Signal does not fire when historical gap was small."""
        from signals_lib.complex import Bounce_Short_Signal

        data = {}
        tf = 5
        data[("indi", tf, 0)] = 3.5
        data[("indi_ma", tf, 0)] = 4.0  # gap=0.5 < 1 (ok)
        data[("indi", tf, 1)] = 3.5
        data[("indi_ma", tf, 1)] = 4.0  # diff=0.5 NOT > 1
        data[("indi", tf, 2)] = 3.5
        data[("indi_ma", tf, 2)] = 4.0  # diff=0.5 NOT > 1

        dp = MockDataPoint(data)
        sig = Bounce_Short_Signal(tf=5, indi="indi", indi_ma="indi_ma")
        result = sig.check(dp, {}, None)
        assert result is False

    def test_set_shift_propagates_to_inner(self):
        """Bounce_Short_Signal.set_shift() propagates to inner And_Signal."""
        from signals_lib.complex import Bounce_Short_Signal

        sig = Bounce_Short_Signal(tf=5, indi="indi", indi_ma="indi_ma")
        sig.set_shift(5)
        assert sig._inner.shift == 5
