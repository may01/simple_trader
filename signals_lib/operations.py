"""Combinator and temporal signals for Phase 04, Task 04.

This module provides:
- Boolean combinators: And_Signal, Or_Signal, Not_Signal
- Temporal signals: History_Signal, Continuous_Signal
- Boolean utility signals: True_Signal, False_Signal, BoolValue_Signal,
  PersistentCounter_Signal, DataValue_Signal
"""

import math

from data import DataPoint
from signals_lib.base_signal import BaseSignal


# ============================================================================
# BOOLEAN COMBINATORS
# ============================================================================


class And_Signal(BaseSignal):
    """Signal: all nested signals are True.

    IMPORTANT: evaluates ALL nested signals (no short-circuit) so that
    get_data() can be collected from every child regardless of result.
    """

    def __init__(self, signals: list) -> None:
        """Initialize And_Signal.

        Args:
            signals: List of BaseSignal instances to combine with AND logic.
        """
        super().__init__()
        self.signals = signals

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if all nested signals are True (no short-circuit evaluation).

        All signals are always evaluated so that get_data() can be collected
        from each child.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: True if all nested signals return True, False otherwise.
        """
        results = [s.check(data_point, levels, action) for s in self.signals]
        return all(results)

    def set_shift(self, shift: int) -> None:
        """Propagate shift to all nested signals.

        Args:
            shift: Number of candles to shift back.
        """
        self.shift = shift
        for s in self.signals:
            s.set_shift(shift)


class Or_Signal(BaseSignal):
    """Signal: any nested signal is True."""

    def __init__(self, signals: list) -> None:
        """Initialize Or_Signal.

        Args:
            signals: List of BaseSignal instances to combine with OR logic.
        """
        super().__init__()
        self.signals = signals

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if any nested signal is True.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: True if any nested signal returns True, False otherwise.
        """
        return any(s.check(data_point, levels, action) for s in self.signals)

    def set_shift(self, shift: int) -> None:
        """Propagate shift to all nested signals.

        Args:
            shift: Number of candles to shift back.
        """
        self.shift = shift
        for s in self.signals:
            s.set_shift(shift)


class Not_Signal(BaseSignal):
    """Signal: negation of a nested signal."""

    def __init__(self, signal: BaseSignal) -> None:
        """Initialize Not_Signal.

        Args:
            signal: The BaseSignal instance to negate.
        """
        super().__init__()
        self.signal = signal

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Return the logical NOT of the nested signal.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: True if nested signal returns False, and vice versa.
        """
        return not self.signal.check(data_point, levels, action)

    def set_shift(self, shift: int) -> None:
        """Propagate shift to nested signal.

        Args:
            shift: Number of candles to shift back.
        """
        self.shift = shift
        self.signal.set_shift(shift)


# ============================================================================
# TEMPORAL SIGNALS
# ============================================================================


class History_Signal(BaseSignal):
    """Signal: evaluate inner signal at a historical offset.

    The inner signal's effective shift is: self.history + self.shift.

    Note: Does NOT restore child shift after check() — restoration is moot
    because set_shift() is always called before the next check().
    """

    def __init__(self, signal: BaseSignal, history: int) -> None:
        """Initialize History_Signal.

        Args:
            signal: The BaseSignal instance to evaluate historically.
            history: Number of additional candles to look back.
        """
        super().__init__()
        self.signal = signal
        self.history = history

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Evaluate inner signal at history + shift offset.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: Result of inner signal evaluated at historical offset.
        """
        self.signal.set_shift(self.history + self.shift)
        return self.signal.check(data_point, levels, action)

    def set_shift(self, shift: int) -> None:
        """Update self.shift; child's effective shift is computed on next check().

        Args:
            shift: Number of candles to shift back.
        """
        self.shift = shift


class Continuous_Signal(BaseSignal):
    """Signal: inner signal is True at least `multiply` times across `history` candles.

    Evaluates the inner signal at each shift in range(history), counts True
    results, and returns True if count >= multiply. Restores the inner
    signal's original shift after evaluation.
    """

    def __init__(self, signal: BaseSignal, history: int, multiply: int = 1) -> None:
        """Initialize Continuous_Signal.

        Args:
            signal: The BaseSignal instance to evaluate across history.
            history: Number of candles to look back (evaluates shifts 0..history-1).
            multiply: Minimum number of True results required (default 1).
        """
        super().__init__()
        self.signal = signal
        self.history = history
        self.multiply = multiply

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if inner signal fires at least multiply times across history candles.

        Saves and restores inner signal's original shift.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels.
            action: The action being evaluated.

        Returns:
            bool: True if count of True results >= multiply.
        """
        original_shift = self.signal.shift
        count = 0
        for i in range(self.history):
            self.signal.set_shift(i)
            if self.signal.check(data_point, levels, action):
                count += 1
        # Restore original shift
        self.signal.set_shift(original_shift)
        return count >= self.multiply


# ============================================================================
# BOOLEAN UTILITY SIGNALS
# ============================================================================


class True_Signal(BaseSignal):
    """Signal: always returns True."""

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Always return True.

        Args:
            data_point: Current market data point (unused).
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: Always True.
        """
        return True


class False_Signal(BaseSignal):
    """Signal: always returns False."""

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Always return False.

        Args:
            data_point: Current market data point (unused).
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: Always False.
        """
        return False


class BoolValue_Signal(BaseSignal):
    """Signal: returns True if data_point.get(value, tf, shift) != 0.

    data_point.get() returns float; any nonzero value is treated as True.
    NaN values are treated as False.
    """

    def __init__(self, value: str, tf: int) -> None:
        """Initialize BoolValue_Signal.

        Args:
            value: Indicator name (without tf prefix).
            tf: Timeframe (in minutes).
        """
        super().__init__()
        self.value = value
        self.tf = tf

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Return True if data value is nonzero (NaN → False).

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if data value != 0, False otherwise.
                  Returns False for NaN values.
        """
        val = data_point.get(self.value, self.tf, self.shift)
        if math.isnan(val):
            return False
        return val != 0


class PersistentCounter_Signal(BaseSignal):
    """Signal: fires every fire_each calls.

    A global counter is incremented each time check() is called.
    Returns True when count % fire_each == 0. Counter starts at 1
    (the first call is call #1).
    """

    def __init__(self, fire_each: int) -> None:
        """Initialize PersistentCounter_Signal.

        Args:
            fire_each: Fire (return True) every this many calls.
        """
        super().__init__()
        self.fire_each = fire_each
        self._count = 0

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Increment counter and fire if count % fire_each == 0.

        Args:
            data_point: Current market data point (unused).
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True every fire_each calls.
        """
        self._count += 1
        return self._count % self.fire_each == 0


class DataValue_Signal(BaseSignal):
    """Signal: armed/fired flag with associated data payload.

    check() returns the value_set flag (armed or not).
    set_value() arms the signal.
    get_data() returns the stored value once armed.

    FOOTGUN: Breaks when wrapped in And_Signal/Or_Signal because the
    armed/fired state is set externally (not via check()). Place directly
    in a chain, not inside combinators.
    """

    def __init__(self, data_name: str, value, value_set: bool) -> None:
        """Initialize DataValue_Signal.

        Args:
            data_name: Key used in get_data() output.
            value: Initial stored value (used only if value_set is True).
            value_set: Initial armed state.
        """
        super().__init__()
        self.data_name = data_name
        self.value = value
        self.value_set = value_set

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Return the armed/fired flag.

        Args:
            data_point: Current market data point (unused).
            levels: Dictionary of price levels (unused).
            action: The action being evaluated (unused).

        Returns:
            bool: True if armed (value_set is True), False otherwise.
        """
        return self.value_set

    def set_value(self, val) -> None:
        """Arm the signal with a value.

        Args:
            val: Value to store; also sets value_set to True.
        """
        self.value = val
        self.value_set = True

    def get_data(self) -> dict:
        """Return data payload when armed.

        Returns:
            dict: {data_name: value} if armed, {} otherwise.
        """
        if self.value_set:
            return {self.data_name: self.value}
        return {}
