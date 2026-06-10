"""BaseSignal abstract interface for signal definitions."""

from abc import ABC, abstractmethod
from data import DataPoint


class BaseSignal(ABC):
    """Abstract base class for all signals.

    A signal represents a condition that can be checked against market data.
    Signals are typically combined in chains (SignalChain) to form trading rules.
    """

    def __init__(self) -> None:
        """Initialize BaseSignal with default shift of 0."""
        self.shift: int = 0

    @abstractmethod
    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Check if the signal condition is met.

        This method must be overridden by subclasses. If a subclass does not
        override it, an AssertionError will be raised.

        Args:
            data_point: Current market data point.
            levels: Dictionary of price levels (context-specific).
            action: The action being evaluated (context-specific).

        Returns:
            bool: True if the signal condition is met, False otherwise.

        Raises:
            AssertionError: If not overridden by a subclass.
        """
        assert False, "check() must be implemented by subclass"

    def reset(self) -> bool:
        """Reset signal state and return abort flag.

        Called by SignalChain when the chain completes. Subclasses can
        override to implement stateful signal behavior.

        Returns:
            bool: True to abort the parent SignalChain, False to continue.
                  Default is False (do not abort).
        """
        return False

    def get_data(self) -> dict:
        """Return signal-specific metadata when chain completes.

        Subclasses can override to provide custom metadata (e.g., indicator
        values, calculated thresholds, etc.).

        Returns:
            dict: Signal-specific metadata. Default is empty dict.
        """
        return {}

    def get_marker_pos(self, data_point: DataPoint) -> float:
        """Return price for chart marker.

        Called when visualizing the signal for charting purposes.
        Subclasses can override to use a custom price (e.g., close, high, low).

        Args:
            data_point: Current market data point.

        Returns:
            float: Price value for the marker. Default is close price at shift=1
                   (previous closed candle): data_point.get("close", 1, 1).
        """
        return data_point.get("close", 1, 1)

    def set_shift(self, shift: int) -> None:
        """Set the historical offset applied in check().

        When set, all data_point.get() calls in check() should add this shift
        to their explicit shift parameter.

        Args:
            shift: Number of candles to shift back (0 = current, 1+ = past).
        """
        self.shift = shift
