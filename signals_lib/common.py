"""Atomic comparison signals for Phase 04, Task 02."""

import math

from data import DataPoint
from signals_lib.base_signal import BaseSignal


class Less_Signal(BaseSignal):
    """Signal: indi_1 < indi_2."""

    def __init__(self, tf: int, indi_1: str, indi_2: str) -> None:
        """Initialize Less_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: First indicator name (without tf prefix).
            indi_2: Second indicator name (without tf prefix).
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.indi_2 = indi_2

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if indi_1 < indi_2.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if indi_1 < indi_2, False otherwise.
                  Returns False if either value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)
        val_2 = data_point.get(self.indi_2, self.tf, self.shift)

        if math.isnan(val_1) or math.isnan(val_2):
            return False

        return val_1 < val_2


class Greater_Signal(BaseSignal):
    """Signal: indi_1 > indi_2."""

    def __init__(self, tf: int, indi_1: str, indi_2: str) -> None:
        """Initialize Greater_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: First indicator name (without tf prefix).
            indi_2: Second indicator name (without tf prefix).
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.indi_2 = indi_2

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if indi_1 > indi_2.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if indi_1 > indi_2, False otherwise.
                  Returns False if either value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)
        val_2 = data_point.get(self.indi_2, self.tf, self.shift)

        if math.isnan(val_1) or math.isnan(val_2):
            return False

        return val_1 > val_2


class Less_Val_Signal(BaseSignal):
    """Signal: indi_1 < val."""

    def __init__(self, tf: int, indi_1: str, val: float) -> None:
        """Initialize Less_Val_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            val: Threshold value to compare against.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.val = val

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if indi_1 < val.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if indi_1 < val, False otherwise.
                  Returns False if value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)

        if math.isnan(val_1):
            return False

        return val_1 < self.val


class Greater_Val_Signal(BaseSignal):
    """Signal: indi_1 > val."""

    def __init__(self, tf: int, indi_1: str, val: float) -> None:
        """Initialize Greater_Val_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            val: Threshold value to compare against.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.val = val

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if indi_1 > val.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if indi_1 > val, False otherwise.
                  Returns False if value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)

        if math.isnan(val_1):
            return False

        return val_1 > self.val


class Rising_Signal(BaseSignal):
    """Signal: indi_1[shift] > indi_1[shift+1] (value is rising)."""

    def __init__(self, tf: int, indi_1: str) -> None:
        """Initialize Rising_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if current value > previous value.

        When shift=2, compares get(..., shift=2) > get(..., shift=3).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if current > previous, False otherwise.
                  Returns False if either value is NaN.
        """
        val_current = data_point.get(self.indi_1, self.tf, self.shift)
        val_previous = data_point.get(self.indi_1, self.tf, self.shift + 1)

        if math.isnan(val_current) or math.isnan(val_previous):
            return False

        return val_current > val_previous


class Falling_Signal(BaseSignal):
    """Signal: indi_1[shift] < indi_1[shift+1] (value is falling)."""

    def __init__(self, tf: int, indi_1: str) -> None:
        """Initialize Falling_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if current value < previous value.

        When shift=2, compares get(..., shift=2) < get(..., shift=3).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if current < previous, False otherwise.
                  Returns False if either value is NaN.
        """
        val_current = data_point.get(self.indi_1, self.tf, self.shift)
        val_previous = data_point.get(self.indi_1, self.tf, self.shift + 1)

        if math.isnan(val_current) or math.isnan(val_previous):
            return False

        return val_current < val_previous


class Diff_Greater_Signal(BaseSignal):
    """Signal: (indi_1 - indi_2) > val."""

    def __init__(self, tf: int, indi_1: str, indi_2: str, val: float) -> None:
        """Initialize Diff_Greater_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: First indicator name (without tf prefix).
            indi_2: Second indicator name (without tf prefix).
            val: Threshold for the difference.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.indi_2 = indi_2
        self.val = val

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if (indi_1 - indi_2) > val.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if difference > val, False otherwise.
                  Returns False if either value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)
        val_2 = data_point.get(self.indi_2, self.tf, self.shift)

        if math.isnan(val_1) or math.isnan(val_2):
            return False

        return (val_1 - val_2) > self.val


class Diff_Less_Signal(BaseSignal):
    """Signal: (indi_1 - indi_2) < val."""

    def __init__(self, tf: int, indi_1: str, indi_2: str, val: float) -> None:
        """Initialize Diff_Less_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: First indicator name (without tf prefix).
            indi_2: Second indicator name (without tf prefix).
            val: Threshold for the difference.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.indi_2 = indi_2
        self.val = val

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if (indi_1 - indi_2) < val.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if difference < val, False otherwise.
                  Returns False if either value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)
        val_2 = data_point.get(self.indi_2, self.tf, self.shift)

        if math.isnan(val_1) or math.isnan(val_2):
            return False

        return (val_1 - val_2) < self.val


class Diff_LessIndi_Signal(BaseSignal):
    """Signal: (indi_1 - indi_2) > 0 AND (indi_1 - indi_2) < indi_dist."""

    def __init__(self, tf: int, indi_1: str, indi_2: str, indi_dist: str) -> None:
        """Initialize Diff_LessIndi_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: First indicator name (without tf prefix).
            indi_2: Second indicator name (without tf prefix).
            indi_dist: Distance threshold indicator name (without tf prefix).
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.indi_2 = indi_2
        self.indi_dist = indi_dist

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if (indi_1 - indi_2) > 0 AND (indi_1 - indi_2) < indi_dist.

        Returns False if diff is negative (prevents firing on negative diffs).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if diff > 0 AND diff < threshold, False otherwise.
                  Returns False if any value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf, self.shift)
        val_2 = data_point.get(self.indi_2, self.tf, self.shift)
        val_dist = data_point.get(self.indi_dist, self.tf, self.shift)

        if math.isnan(val_1) or math.isnan(val_2) or math.isnan(val_dist):
            return False

        diff = val_1 - val_2
        return diff > 0 and diff < val_dist


class Diff_GreaterIndi_Signal(BaseSignal):
    """Signal: (indi_1 - indi_2) > 0 AND (indi_1 - indi_2) > indi_dist (cross-timeframe)."""

    def __init__(
        self,
        tf_1: int,
        indi_1: str,
        tf_2: int,
        indi_2: str,
        tf_dist: int,
        indi_dist: str,
    ) -> None:
        """Initialize Diff_GreaterIndi_Signal.

        Args:
            tf_1: Timeframe for indi_1.
            indi_1: First indicator name (without tf prefix).
            tf_2: Timeframe for indi_2.
            indi_2: Second indicator name (without tf prefix).
            tf_dist: Timeframe for indi_dist.
            indi_dist: Distance threshold indicator name (without tf prefix).
        """
        super().__init__()
        self.tf_1 = tf_1
        self.indi_1 = indi_1
        self.tf_2 = tf_2
        self.indi_2 = indi_2
        self.tf_dist = tf_dist
        self.indi_dist = indi_dist

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if (indi_1 - indi_2) > 0 AND (indi_1 - indi_2) > indi_dist.

        self.shift is applied uniformly to all three get() calls.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if diff > 0 AND diff > threshold, False otherwise.
                  Returns False if any value is NaN.
        """
        val_1 = data_point.get(self.indi_1, self.tf_1, self.shift)
        val_2 = data_point.get(self.indi_2, self.tf_2, self.shift)
        val_dist = data_point.get(self.indi_dist, self.tf_dist, self.shift)

        if math.isnan(val_1) or math.isnan(val_2) or math.isnan(val_dist):
            return False

        diff = val_1 - val_2
        return diff > 0 and diff > val_dist
