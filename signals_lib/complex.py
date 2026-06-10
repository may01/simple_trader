"""Complex divergence and bounce signals for Phase 04, Task 05.

Divergence signals:
  - Diver_Bull_signal: bullish divergence (price lower low, indicator higher low)
  - Diver_Bear_signal: bearish divergence (price higher high, indicator lower high)
  - Diver_Hidden_Bull_signal: hidden bullish divergence (price higher low, indicator lower low)
  - Diver_Hidden_Bear_signal: hidden bearish divergence (price lower high, indicator higher high)

Bounce signals (composed from existing signals):
  - Bounce_Long_Signal: indi bounced back toward indi_ma from above
  - Bounce_Short_Signal: indi_ma bounced back toward indi from above
"""

import math

from data import DataPoint
from signals_lib.base_signal import BaseSignal
from signals_lib.common import Diff_Greater_Signal, Diff_Less_Signal
from signals_lib.operations import And_Signal, Or_Signal, History_Signal


# How many historical candles to scan for prior peak/valley
_PEAK_HISTORY = 7


class Diver_Bull_signal(BaseSignal):
    """Signal: bullish divergence — price makes lower low, indicator makes higher low below level.

    Fires when:
    1. Current indicator < level (indicator oversold)
    2. A prior valley (local minimum) below level exists within the last peak_history candles
    3. Current indicator > prior valley indicator (higher low in indicator)
    4. Current price < prior price at valley (lower low in price)

    If finished=True, the prior valley must be confirmed by a 3-candle pattern:
    indicator[i-1] > indicator[i] < indicator[i+1].

    reset() returns True when indicator > cancel_level (sequence aborted).
    """

    def __init__(
        self,
        tf: int,
        level: float,
        cancel_level: float,
        indicator: str,
        finished: bool = True,
    ) -> None:
        """Initialize Diver_Bull_signal.

        Args:
            tf: Timeframe (in minutes).
            level: Indicator threshold — only fire when indicator < level.
            cancel_level: If indicator > cancel_level, reset() returns True.
            indicator: Indicator name (without tf prefix).
            finished: If True, require prior valley to be confirmed by 3-candle pattern.
        """
        super().__init__()
        self.tf = tf
        self.level = level
        self.cancel_level = cancel_level
        self.indicator = indicator
        self.finished = finished
        self._cancelled = False
        self._prior_indicator: float | None = None
        self._prior_price: float | None = None

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check for bullish divergence.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if bullish divergence detected, False otherwise.
        """
        self._cancelled = False
        self._prior_indicator = None
        self._prior_price = None

        cur_indicator = data_point.get(self.indicator, self.tf, self.shift)
        if math.isnan(cur_indicator):
            return False

        # Cancel condition: indicator is above cancel_level
        if cur_indicator > self.cancel_level:
            self._cancelled = True
            return False

        # Condition 1: current indicator must be below level (oversold)
        if cur_indicator >= self.level:
            return False

        cur_price = data_point.get("close", self.tf, self.shift)
        if math.isnan(cur_price):
            return False

        # Scan for prior valley: local minimum in indicator below level
        for i in range(1, _PEAK_HISTORY + 1):
            offset = self.shift + i
            hist_indicator = data_point.get(self.indicator, self.tf, offset)
            if math.isnan(hist_indicator):
                continue

            # Valley must be below level
            if hist_indicator >= self.level:
                continue

            # If finished=True, require 3-candle confirmation: [i-1] > [i] < [i+1]
            if self.finished:
                prev_indicator = data_point.get(self.indicator, self.tf, offset - 1)
                next_indicator = data_point.get(self.indicator, self.tf, offset + 1)
                if math.isnan(prev_indicator) or math.isnan(next_indicator):
                    continue
                if not (prev_indicator > hist_indicator and next_indicator > hist_indicator):
                    continue

            hist_price = data_point.get("close", self.tf, offset)
            if math.isnan(hist_price):
                continue

            # Bullish divergence conditions:
            # indicator: higher low (current > prior valley)
            # price: lower low (current < prior price at valley)
            if cur_indicator > hist_indicator and cur_price < hist_price:
                self._prior_indicator = hist_indicator
                self._prior_price = hist_price
                return True

        return False

    def reset(self) -> bool:
        """Return True if cancel_level was breached (sequence aborted).

        Returns:
            bool: True if cancelled, False otherwise.
        """
        return self._cancelled

    def get_data(self) -> dict:
        """Return prior valley indicator and price values when divergence found.

        Returns:
            dict: {'prior_indicator': float, 'prior_price': float} or {}.
        """
        if self._prior_indicator is not None and self._prior_price is not None:
            return {
                "prior_indicator": self._prior_indicator,
                "prior_price": self._prior_price,
            }
        return {}


class Diver_Bear_signal(BaseSignal):
    """Signal: bearish divergence — price makes higher high, indicator makes lower high above level.

    Fires when:
    1. Current indicator > level (indicator overbought)
    2. A prior peak (local maximum) above level exists within the last peak_history candles
    3. Current indicator < prior peak indicator (lower high in indicator)
    4. Current price > prior price at peak (higher high in price)

    If finished=True, the prior peak must be confirmed by a 3-candle pattern:
    indicator[i-1] < indicator[i] > indicator[i+1].

    reset() returns True when indicator < cancel_level (sequence aborted).
    """

    def __init__(
        self,
        tf: int,
        level: float,
        cancel_level: float,
        indicator: str,
        finished: bool = True,
    ) -> None:
        """Initialize Diver_Bear_signal.

        Args:
            tf: Timeframe (in minutes).
            level: Indicator threshold — only fire when indicator > level.
            cancel_level: If indicator < cancel_level, reset() returns True.
            indicator: Indicator name (without tf prefix).
            finished: If True, require prior peak to be confirmed by 3-candle pattern.
        """
        super().__init__()
        self.tf = tf
        self.level = level
        self.cancel_level = cancel_level
        self.indicator = indicator
        self.finished = finished
        self._cancelled = False
        self._prior_indicator: float | None = None
        self._prior_price: float | None = None

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check for bearish divergence.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if bearish divergence detected, False otherwise.
        """
        self._cancelled = False
        self._prior_indicator = None
        self._prior_price = None

        cur_indicator = data_point.get(self.indicator, self.tf, self.shift)
        if math.isnan(cur_indicator):
            return False

        # Cancel condition: indicator is below cancel_level
        if cur_indicator < self.cancel_level:
            self._cancelled = True
            return False

        # Condition 1: current indicator must be above level (overbought)
        if cur_indicator <= self.level:
            return False

        cur_price = data_point.get("close", self.tf, self.shift)
        if math.isnan(cur_price):
            return False

        # Scan for prior peak: local maximum in indicator above level
        for i in range(1, _PEAK_HISTORY + 1):
            offset = self.shift + i
            hist_indicator = data_point.get(self.indicator, self.tf, offset)
            if math.isnan(hist_indicator):
                continue

            # Peak must be above level
            if hist_indicator <= self.level:
                continue

            # If finished=True, require 3-candle confirmation: [i-1] < [i] > [i+1]
            if self.finished:
                prev_indicator = data_point.get(self.indicator, self.tf, offset - 1)
                next_indicator = data_point.get(self.indicator, self.tf, offset + 1)
                if math.isnan(prev_indicator) or math.isnan(next_indicator):
                    continue
                if not (prev_indicator < hist_indicator and next_indicator < hist_indicator):
                    continue

            hist_price = data_point.get("close", self.tf, offset)
            if math.isnan(hist_price):
                continue

            # Bearish divergence conditions:
            # indicator: lower high (current < prior peak)
            # price: higher high (current > prior price at peak)
            if cur_indicator < hist_indicator and cur_price > hist_price:
                self._prior_indicator = hist_indicator
                self._prior_price = hist_price
                return True

        return False

    def reset(self) -> bool:
        """Return True if cancel_level was breached (sequence aborted).

        Returns:
            bool: True if cancelled, False otherwise.
        """
        return self._cancelled

    def get_data(self) -> dict:
        """Return prior peak indicator and price values when divergence found.

        Returns:
            dict: {'prior_indicator': float, 'prior_price': float} or {}.
        """
        if self._prior_indicator is not None and self._prior_price is not None:
            return {
                "prior_indicator": self._prior_indicator,
                "prior_price": self._prior_price,
            }
        return {}


class Diver_Hidden_Bull_signal(BaseSignal):
    """Signal: hidden bullish divergence — price makes higher low, indicator makes lower low.

    Uses 'low' price (not 'close') for price comparison.

    Fires when:
    1. Current indicator < level (oversold region)
    2. A prior valley below level exists within peak_history candles
    3. Prior indicator > current indicator (lower low in indicator)
    4. Prior price < current price (higher low in price via 'low' field)

    If finished=True, valley must be confirmed by 3-candle pattern.

    reset() returns True if any historical indicator value rises above cancel_level during scan.
    """

    def __init__(
        self,
        tf: int,
        level: float,
        cancel_level: float,
        indicator: str,
        finished: bool = True,
    ) -> None:
        """Initialize Diver_Hidden_Bull_signal.

        Args:
            tf: Timeframe (in minutes).
            level: Indicator threshold — only fire when indicator < level.
            cancel_level: If any historical indicator > cancel_level during scan, reset() True.
            indicator: Indicator name (without tf prefix).
            finished: If True, require prior valley to be confirmed by 3-candle pattern.
        """
        super().__init__()
        self.tf = tf
        self.level = level
        self.cancel_level = cancel_level
        self.indicator = indicator
        self.finished = finished
        self._cancelled = False
        self._prior_indicator: float | None = None
        self._prior_price: float | None = None

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check for hidden bullish divergence.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if hidden bullish divergence detected, False otherwise.
        """
        self._cancelled = False
        self._prior_indicator = None
        self._prior_price = None

        cur_indicator = data_point.get(self.indicator, self.tf, self.shift)
        if math.isnan(cur_indicator):
            return False

        # Condition 1: current indicator must be below level
        if cur_indicator >= self.level:
            return False

        cur_price = data_point.get("low", self.tf, self.shift)
        if math.isnan(cur_price):
            return False

        # Pre-scan: check for cancel over full range before pattern matching.
        # Cancel: if any historical indicator rises above cancel_level during scan.
        for i in range(1, _PEAK_HISTORY + 1):
            offset = self.shift + i
            hist_indicator = data_point.get(self.indicator, self.tf, offset)
            if not math.isnan(hist_indicator) and hist_indicator > self.cancel_level:
                self._cancelled = True
                break

        # Scan for prior valley: local minimum in indicator below level
        for i in range(1, _PEAK_HISTORY + 1):
            offset = self.shift + i
            hist_indicator = data_point.get(self.indicator, self.tf, offset)
            if math.isnan(hist_indicator):
                continue

            # Skip values at or above level (valley must be below level)
            if hist_indicator >= self.level:
                continue

            # If finished=True, require 3-candle confirmation
            if self.finished:
                prev_indicator = data_point.get(self.indicator, self.tf, offset - 1)
                next_indicator = data_point.get(self.indicator, self.tf, offset + 1)
                if math.isnan(prev_indicator) or math.isnan(next_indicator):
                    continue
                if not (prev_indicator > hist_indicator and next_indicator > hist_indicator):
                    continue

            hist_price = data_point.get("low", self.tf, offset)
            if math.isnan(hist_price):
                continue

            # Hidden bullish divergence:
            # indicator: lower low (prior > current, i.e., current made lower low)
            # price: higher low (prior < current, i.e., current made higher low)
            if hist_indicator > cur_indicator and hist_price < cur_price:
                self._prior_indicator = hist_indicator
                self._prior_price = hist_price
                return True

        return False

    def reset(self) -> bool:
        """Return True if any historical indicator exceeded cancel_level during scan.

        Returns:
            bool: True if cancelled, False otherwise.
        """
        return self._cancelled

    def get_data(self) -> dict:
        """Return prior valley indicator and price values when divergence found.

        Returns:
            dict: {'prior_indicator': float, 'prior_price': float} or {}.
        """
        if self._prior_indicator is not None and self._prior_price is not None:
            return {
                "prior_indicator": self._prior_indicator,
                "prior_price": self._prior_price,
            }
        return {}


class Diver_Hidden_Bear_signal(BaseSignal):
    """Signal: hidden bearish divergence — price makes lower high, indicator makes higher high.

    Uses 'high' price (not 'close') for price comparison.

    Fires when:
    1. Current indicator > level (overbought region)
    2. A prior peak above level exists within peak_history candles
    3. Prior indicator < current indicator (higher high in indicator)
    4. Prior price > current price (lower high in price via 'high' field)

    If finished=True, peak must be confirmed by 3-candle pattern.

    reset() returns True if any historical indicator value drops below cancel_level during scan.
    """

    def __init__(
        self,
        tf: int,
        level: float,
        cancel_level: float,
        indicator: str,
        finished: bool = True,
    ) -> None:
        """Initialize Diver_Hidden_Bear_signal.

        Args:
            tf: Timeframe (in minutes).
            level: Indicator threshold — only fire when indicator > level.
            cancel_level: If any historical indicator < cancel_level during scan, reset() True.
            indicator: Indicator name (without tf prefix).
            finished: If True, require prior peak to be confirmed by 3-candle pattern.
        """
        super().__init__()
        self.tf = tf
        self.level = level
        self.cancel_level = cancel_level
        self.indicator = indicator
        self.finished = finished
        self._cancelled = False
        self._prior_indicator: float | None = None
        self._prior_price: float | None = None

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check for hidden bearish divergence.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if hidden bearish divergence detected, False otherwise.
        """
        self._cancelled = False
        self._prior_indicator = None
        self._prior_price = None

        cur_indicator = data_point.get(self.indicator, self.tf, self.shift)
        if math.isnan(cur_indicator):
            return False

        # Condition 1: current indicator must be above level
        if cur_indicator <= self.level:
            return False

        cur_price = data_point.get("high", self.tf, self.shift)
        if math.isnan(cur_price):
            return False

        # Pre-scan: check for cancel over full range before pattern matching.
        # Cancel: if any historical indicator drops below cancel_level during scan.
        for i in range(1, _PEAK_HISTORY + 1):
            offset = self.shift + i
            hist_indicator = data_point.get(self.indicator, self.tf, offset)
            if not math.isnan(hist_indicator) and hist_indicator < self.cancel_level:
                self._cancelled = True
                break

        # Scan for prior peak: local maximum in indicator above level
        for i in range(1, _PEAK_HISTORY + 1):
            offset = self.shift + i
            hist_indicator = data_point.get(self.indicator, self.tf, offset)
            if math.isnan(hist_indicator):
                continue

            # Skip values at or below level (peak must be above level)
            if hist_indicator <= self.level:
                continue

            # If finished=True, require 3-candle confirmation
            if self.finished:
                prev_indicator = data_point.get(self.indicator, self.tf, offset - 1)
                next_indicator = data_point.get(self.indicator, self.tf, offset + 1)
                if math.isnan(prev_indicator) or math.isnan(next_indicator):
                    continue
                if not (prev_indicator < hist_indicator and next_indicator < hist_indicator):
                    continue

            hist_price = data_point.get("high", self.tf, offset)
            if math.isnan(hist_price):
                continue

            # Hidden bearish divergence:
            # indicator: higher high (prior < current, i.e., current made higher high)
            # price: lower high (prior > current, i.e., current made lower high)
            if hist_indicator < cur_indicator and hist_price > cur_price:
                self._prior_indicator = hist_indicator
                self._prior_price = hist_price
                return True

        return False

    def reset(self) -> bool:
        """Return True if any historical indicator dropped below cancel_level during scan.

        Returns:
            bool: True if cancelled, False otherwise.
        """
        return self._cancelled

    def get_data(self) -> dict:
        """Return prior peak indicator and price values when divergence found.

        Returns:
            dict: {'prior_indicator': float, 'prior_price': float} or {}.
        """
        if self._prior_indicator is not None and self._prior_price is not None:
            return {
                "prior_indicator": self._prior_indicator,
                "prior_price": self._prior_price,
            }
        return {}


class Bounce_Long_Signal(BaseSignal):
    """Signal: indi previously well above indi_ma (gap > 1 for last 2 candles), gap now closed.

    Composed from existing signals:
        And_Signal([
            Or_Signal([
                Diff_Less_Signal(tf, indi_ma, indi, 1),  # indi_ma slightly above indi
                Diff_Less_Signal(tf, indi, indi_ma, 1),  # indi slightly above indi_ma
            ]),
            History_Signal(Diff_Greater_Signal(tf, indi, indi_ma, 1), 1),
            History_Signal(Diff_Greater_Signal(tf, indi, indi_ma, 1), 2),
        ])

    Delegates entirely to inner And_Signal. set_shift() propagates to inner signal.
    """

    def __init__(self, tf: int, indi: str, indi_ma: str) -> None:
        """Initialize Bounce_Long_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi: Indicator name (without tf prefix).
            indi_ma: Indicator moving average name (without tf prefix).
        """
        super().__init__()
        self._inner = And_Signal([
            Or_Signal([
                Diff_Less_Signal(tf, indi_ma, indi, 1),
                Diff_Less_Signal(tf, indi, indi_ma, 1),
            ]),
            History_Signal(Diff_Greater_Signal(tf, indi, indi_ma, 1), 1),
            History_Signal(Diff_Greater_Signal(tf, indi, indi_ma, 1), 2),
        ])

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if indi has bounced back toward indi_ma.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: True if bounce pattern detected, False otherwise.
        """
        return self._inner.check(data_point, levels, action)

    def set_shift(self, shift: int) -> None:
        """Propagate shift to inner And_Signal.

        Args:
            shift: Number of candles to shift back.
        """
        self.shift = shift
        self._inner.set_shift(shift)


class Bounce_Short_Signal(BaseSignal):
    """Signal: indi_ma previously well above indi (gap > 1 for last 2 candles), gap now closed.

    Composed from existing signals:
        And_Signal([
            Or_Signal([
                Diff_Less_Signal(tf, indi_ma, indi, 1),  # indi_ma slightly above indi
                Diff_Less_Signal(tf, indi, indi_ma, 1),  # indi slightly above indi_ma
            ]),
            History_Signal(Diff_Greater_Signal(tf, indi_ma, indi, 1), 1),
            History_Signal(Diff_Greater_Signal(tf, indi_ma, indi, 1), 2),
        ])

    Delegates entirely to inner And_Signal. set_shift() propagates to inner signal.
    """

    def __init__(self, tf: int, indi: str, indi_ma: str) -> None:
        """Initialize Bounce_Short_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi: Indicator name (without tf prefix).
            indi_ma: Indicator moving average name (without tf prefix).
        """
        super().__init__()
        self._inner = And_Signal([
            Or_Signal([
                Diff_Less_Signal(tf, indi_ma, indi, 1),
                Diff_Less_Signal(tf, indi, indi_ma, 1),
            ]),
            History_Signal(Diff_Greater_Signal(tf, indi_ma, indi, 1), 1),
            History_Signal(Diff_Greater_Signal(tf, indi_ma, indi, 1), 2),
        ])

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if indi_ma has bounced back toward indi.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: True if bounce pattern detected, False otherwise.
        """
        return self._inner.check(data_point, levels, action)

    def set_shift(self, shift: int) -> None:
        """Propagate shift to inner And_Signal.

        Args:
            shift: Number of candles to shift back.
        """
        self.shift = shift
        self._inner.set_shift(shift)
