"""Candle pattern signals for Phase 04, Task 06."""

import math

from data import DataPoint
from signals_lib.base_signal import BaseSignal
from constants import CANDLE_HIGH, CANDLE_LOW


def _any_nan(*values: float) -> bool:
    """Check if any of the given float values is NaN.

    Args:
        *values: Variable number of float values to check.

    Returns:
        bool: True if any value is NaN, False otherwise.
    """
    return any(math.isnan(v) for v in values)


class BigCandle_Signal(BaseSignal):
    """Signal: natr_14 > coef * natr_ma (big candle detection)."""

    def __init__(self, tf: int, coef: float) -> None:
        """Initialize BigCandle_Signal.

        Args:
            tf: Timeframe (in minutes).
            coef: Coefficient to multiply natr_ma by for threshold.
        """
        super().__init__()
        self.tf = tf
        self.coef = coef

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if natr_14 > coef * natr_ma.

        Both are normalized ATR (%) — natr_14 vs natr_ma (EMA-5 of natr_14).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if natr_14 > coef * natr_ma, False otherwise.
                  Returns False if either value is NaN.
        """
        natr_14 = data_point.get("natr_14", self.tf, self.shift)
        natr_ma = data_point.get("natr_ma", self.tf, self.shift)

        if _any_nan(natr_14, natr_ma):
            return False

        return natr_14 > self.coef * natr_ma


class LongTale_Signal(BaseSignal):
    """Signal: long wick detection (lower or upper)."""

    def __init__(self, tf: int, tale_type: int) -> None:
        """Initialize LongTale_Signal.

        Args:
            tf: Timeframe (in minutes).
            tale_type: CANDLE_LOW (2) for lower wick, CANDLE_HIGH (1) for upper wick.
        """
        super().__init__()
        self.tf = tf
        self.tale_type = tale_type

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if wick exceeds 0.5 * candle_size.

        For CANDLE_LOW: (min(open, close) - low) > 0.5 * candle_size
        For CANDLE_HIGH: (high - max(open, close)) > 0.5 * candle_size

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if wick is long, False otherwise.
                  Returns False if candle_size == 0 or any OHLC is NaN.
        """
        open_price = data_point.get("open", self.tf, self.shift)
        close_price = data_point.get("close", self.tf, self.shift)
        high_price = data_point.get("high", self.tf, self.shift)
        low_price = data_point.get("low", self.tf, self.shift)

        if _any_nan(open_price, close_price, high_price, low_price):
            return False

        candle_size = high_price - low_price
        if candle_size == 0:
            return False

        if self.tale_type == CANDLE_LOW:
            wick = min(open_price, close_price) - low_price
            return wick > 0.5 * candle_size
        elif self.tale_type == CANDLE_HIGH:
            wick = high_price - max(open_price, close_price)
            return wick > 0.5 * candle_size
        else:
            return False


class ShortTale_Signal(BaseSignal):
    """Signal: short wick detection (lower or upper)."""

    def __init__(self, tf: int, tale_type: int) -> None:
        """Initialize ShortTale_Signal.

        Args:
            tf: Timeframe (in minutes).
            tale_type: CANDLE_LOW (2) for lower wick, CANDLE_HIGH (1) for upper wick.
        """
        super().__init__()
        self.tf = tf
        self.tale_type = tale_type

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if wick is less than 0.2 * candle_size.

        For CANDLE_LOW: (min(open, close) - low) < 0.2 * candle_size
        For CANDLE_HIGH: (high - max(open, close)) < 0.2 * candle_size

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if wick is short, False otherwise.
                  Returns False if candle_size == 0 or any OHLC is NaN.
        """
        open_price = data_point.get("open", self.tf, self.shift)
        close_price = data_point.get("close", self.tf, self.shift)
        high_price = data_point.get("high", self.tf, self.shift)
        low_price = data_point.get("low", self.tf, self.shift)

        if _any_nan(open_price, close_price, high_price, low_price):
            return False

        candle_size = high_price - low_price
        if candle_size == 0:
            return False

        if self.tale_type == CANDLE_LOW:
            wick = min(open_price, close_price) - low_price
            return wick < 0.2 * candle_size
        elif self.tale_type == CANDLE_HIGH:
            wick = high_price - max(open_price, close_price)
            return wick < 0.2 * candle_size
        else:
            return False


class CandleUP_Signal(BaseSignal):
    """Signal: bullish candle (close > open)."""

    def __init__(self, tf: int) -> None:
        """Initialize CandleUP_Signal.

        Args:
            tf: Timeframe (in minutes).
        """
        super().__init__()
        self.tf = tf

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if open < close (bullish candle).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if close > open, False otherwise.
                  Returns False if either value is NaN.
        """
        open_price = data_point.get("open", self.tf, self.shift)
        close_price = data_point.get("close", self.tf, self.shift)

        if _any_nan(open_price, close_price):
            return False

        return open_price < close_price


class CandleDOWN_Signal(BaseSignal):
    """Signal: bearish candle (open > close)."""

    def __init__(self, tf: int) -> None:
        """Initialize CandleDOWN_Signal.

        Args:
            tf: Timeframe (in minutes).
        """
        super().__init__()
        self.tf = tf

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if open > close (bearish candle).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if open > close, False otherwise.
                  Returns False if either value is NaN.
        """
        open_price = data_point.get("open", self.tf, self.shift)
        close_price = data_point.get("close", self.tf, self.shift)

        if _any_nan(open_price, close_price):
            return False

        return open_price > close_price
