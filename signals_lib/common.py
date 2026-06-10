"""Atomic comparison signals for Phase 04, Task 02."""

import math

from data import DataPoint
from signals_lib.base_signal import BaseSignal


def _any_nan(*values: float) -> bool:
    """Check if any of the given float values is NaN.

    Args:
        *values: Variable number of float values to check.

    Returns:
        bool: True if any value is NaN, False otherwise.
    """
    return any(math.isnan(v) for v in values)


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

        if _any_nan(val_1, val_2):
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

        if _any_nan(val_1, val_2):
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

        if _any_nan(val_1):
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

        if _any_nan(val_1):
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

        if _any_nan(val_current, val_previous):
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

        if _any_nan(val_current, val_previous):
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

        if _any_nan(val_1, val_2):
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

        if _any_nan(val_1, val_2):
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

        if _any_nan(val_1, val_2, val_dist):
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

        if _any_nan(val_1, val_2, val_dist):
            return False

        diff = val_1 - val_2
        return diff > 0 and diff > val_dist


# ============================================================================
# CROSSOVER SIGNALS (Phase 04, Task 03)
# ============================================================================


class Cross_Up_Signal(BaseSignal):
    """Signal: current indi_1 > indi_2 AND previous indi_1 < indi_2."""

    def __init__(self, tf: int, indi_1: str, indi_2: str) -> None:
        """Initialize Cross_Up_Signal.

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
        """Check if current indi_1 > indi_2 AND previous indi_1 < indi_2.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if crossover detected, False otherwise.
                  Returns False if any value is NaN.
        """
        val_1_current = data_point.get(self.indi_1, self.tf, self.shift)
        val_2_current = data_point.get(self.indi_2, self.tf, self.shift)
        val_1_previous = data_point.get(self.indi_1, self.tf, self.shift + 1)
        val_2_previous = data_point.get(self.indi_2, self.tf, self.shift + 1)

        if _any_nan(val_1_current, val_2_current, val_1_previous, val_2_previous):
            return False

        return val_1_current > val_2_current and val_1_previous < val_2_previous


class Cross_Down_Signal(BaseSignal):
    """Signal: current indi_1 < indi_2 AND previous indi_1 > indi_2."""

    def __init__(self, tf: int, indi_1: str, indi_2: str) -> None:
        """Initialize Cross_Down_Signal.

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
        """Check if current indi_1 < indi_2 AND previous indi_1 > indi_2.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if crossover detected, False otherwise.
                  Returns False if any value is NaN.
        """
        val_1_current = data_point.get(self.indi_1, self.tf, self.shift)
        val_2_current = data_point.get(self.indi_2, self.tf, self.shift)
        val_1_previous = data_point.get(self.indi_1, self.tf, self.shift + 1)
        val_2_previous = data_point.get(self.indi_2, self.tf, self.shift + 1)

        if _any_nan(val_1_current, val_2_current, val_1_previous, val_2_previous):
            return False

        return val_1_current < val_2_current and val_1_previous > val_2_previous


class Cross_Up_Val_Signal(BaseSignal):
    """Signal: current indi_1 > val AND previous indi_1 < val."""

    def __init__(self, tf: int, indi_1: str, val: float) -> None:
        """Initialize Cross_Up_Val_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            val: Threshold value to cross above.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.val = val

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if current indi_1 > val AND previous indi_1 < val.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if crossover detected, False otherwise.
                  Returns False if any value is NaN.
        """
        val_current = data_point.get(self.indi_1, self.tf, self.shift)
        val_previous = data_point.get(self.indi_1, self.tf, self.shift + 1)

        if _any_nan(val_current, val_previous):
            return False

        return val_current > self.val and val_previous < self.val


class Cross_Down_Val_Signal(BaseSignal):
    """Signal: current indi_1 < val AND previous indi_1 > val."""

    def __init__(self, tf: int, indi_1: str, val: float) -> None:
        """Initialize Cross_Down_Val_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            val: Threshold value to cross below.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.val = val

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if current indi_1 < val AND previous indi_1 > val.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if crossover detected, False otherwise.
                  Returns False if any value is NaN.
        """
        val_current = data_point.get(self.indi_1, self.tf, self.shift)
        val_previous = data_point.get(self.indi_1, self.tf, self.shift + 1)

        if _any_nan(val_current, val_previous):
            return False

        return val_current < self.val and val_previous > self.val


# ============================================================================
# LEVEL-BASED SIGNALS (Phase 04, Task 03)
# ============================================================================


class Near_Level_Signal(BaseSignal):
    """Signal: abs(price - level_val) <= buffer for any level in levels[level_type]."""

    def __init__(self, tf: int, indi_1: str, level_type: int, buffer: float) -> None:
        """Initialize Near_Level_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            level_type: Key to lookup in levels dict.
            buffer: Distance threshold for proximity.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.level_type = level_type
        self.buffer = buffer
        self._matched_level = None

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if price is within buffer of any level.

        Args:
            data_point: Current market data point.
            levels: Dictionary mapping level_type to list of level values.
            action: The action being evaluated (unused).

        Returns:
            bool: True if price within buffer of any level, False otherwise.
                  Returns False if price is NaN or no levels for type.
        """
        price = data_point.get(self.indi_1, self.tf, self.shift)

        if math.isnan(price):
            return False

        level_list = levels.get(self.level_type, [])
        if not level_list:
            return False

        self._matched_level = None
        for level_val in level_list:
            if abs(price - level_val) <= self.buffer:
                self._matched_level = level_val

        return self._matched_level is not None

    def get_data(self) -> dict:
        """Return matched level value.

        Returns:
            dict: {"level_val": level_val} if a level matched, {} otherwise.
        """
        if self._matched_level is not None:
            return {"level_val": self._matched_level}
        return {}


class Near_Price_Level_Signal(BaseSignal):
    """Signal: abs(price - level_val) <= (buffer × buffer_indi) for any level."""

    def __init__(
        self, tf: int, indi_1: str, level_type: int, buffer: float, buffer_indi: str
    ) -> None:
        """Initialize Near_Price_Level_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            level_type: Key to lookup in levels dict.
            buffer: Coefficient multiplied by buffer_indi.
            buffer_indi: Indicator name for dynamic buffer (without tf prefix).
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.level_type = level_type
        self.buffer = buffer
        self.buffer_indi = buffer_indi
        # Create internal Near_Level_Signal for delegation
        self._inner_signal = Near_Level_Signal(tf, indi_1, level_type, 0.0)

    def set_shift(self, shift: int) -> None:
        """Propagate shift to inner Near_Level_Signal.

        Args:
            shift: Number of candles to shift back.
        """
        super().set_shift(shift)
        self._inner_signal.set_shift(shift)

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if price within dynamic buffer of any level.

        Dynamic buffer = buffer × data_point.get(buffer_indi, tf, shift)

        Args:
            data_point: Current market data point.
            levels: Dictionary mapping level_type to list of level values.
            action: The action being evaluated (unused).

        Returns:
            bool: True if price within dynamic buffer, False otherwise.
                  Returns False if buffer_indi is NaN.
        """
        buffer_val = data_point.get(self.buffer_indi, self.tf, self.shift)

        if math.isnan(buffer_val):
            return False

        effective_buffer = self.buffer * buffer_val
        self._inner_signal.buffer = effective_buffer
        return self._inner_signal.check(data_point, levels, action)

    def get_data(self) -> dict:
        """Delegate to inner signal's get_data().

        Returns:
            dict: Signal metadata from inner signal.
        """
        return self._inner_signal.get_data()


class Over_Level_Signal(BaseSignal):
    """Signal: indi_1 > level_val for any level in levels[level_type]."""

    def __init__(self, tf: int, indi_1: str, level_type: int) -> None:
        """Initialize Over_Level_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            level_type: Key to lookup in levels dict.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.level_type = level_type

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if price is above any level.

        Args:
            data_point: Current market data point.
            levels: Dictionary mapping level_type to list of level values.
            action: The action being evaluated (unused).

        Returns:
            bool: True if price above any level, False otherwise.
                  Returns False if price is NaN or no levels for type.
        """
        price = data_point.get(self.indi_1, self.tf, self.shift)

        if math.isnan(price):
            return False

        level_list = levels.get(self.level_type, [])
        if not level_list:
            return False

        return any(price > level_val for level_val in level_list)


class Under_Level_Signal(BaseSignal):
    """Signal: indi_1 < level_val for any level in levels[level_type]."""

    def __init__(self, tf: int, indi_1: str, level_type: int) -> None:
        """Initialize Under_Level_Signal.

        Args:
            tf: Timeframe (in minutes).
            indi_1: Indicator name (without tf prefix).
            level_type: Key to lookup in levels dict.
        """
        super().__init__()
        self.tf = tf
        self.indi_1 = indi_1
        self.level_type = level_type

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if price is below any level.

        Args:
            data_point: Current market data point.
            levels: Dictionary mapping level_type to list of level values.
            action: The action being evaluated (unused).

        Returns:
            bool: True if price below any level, False otherwise.
                  Returns False if price is NaN or no levels for type.
        """
        price = data_point.get(self.indi_1, self.tf, self.shift)

        if math.isnan(price):
            return False

        level_list = levels.get(self.level_type, [])
        if not level_list:
            return False

        return any(price < level_val for level_val in level_list)
