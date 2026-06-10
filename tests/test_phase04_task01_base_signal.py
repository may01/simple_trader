"""Tests for BaseSignal abstract interface (Phase 04, Task 01)."""

import math

import pytest
from data import DataPoint
from signals_lib.base_signal import BaseSignal


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


class ConcreteSignal(BaseSignal):
    """Concrete implementation for testing."""

    def check(self, data_point: DataPoint, levels: dict, action) -> bool:
        """Simple implementation that checks if close > 100."""
        close_price = data_point.get("close", 1, shift=self.shift)
        return close_price > 100.0


class TestBaseSignalAbstract:
    """Test that BaseSignal is abstract."""

    def test_base_signal_is_abstract(self):
        """BaseSignal cannot be instantiated directly."""
        # Python's ABC mechanism prevents instantiation
        with pytest.raises(TypeError, match="abstract"):
            BaseSignal()

    def test_concrete_signal_can_be_instantiated(self):
        """Concrete subclass can be instantiated."""
        signal = ConcreteSignal()
        assert signal is not None

    def test_concrete_signal_check_works(self):
        """ConcreteSignal.check() works as expected."""
        signal = ConcreteSignal()
        data_point = MockDataPoint({("close", 1, 0): 150.0})
        assert signal.check(data_point, {}, None) is True


class TestShiftAttribute:
    """Test shift attribute initialization and modification."""

    def test_shift_default_value(self):
        """shift should default to 0."""
        signal = ConcreteSignal()
        assert signal.shift == 0

    def test_set_shift(self):
        """set_shift() should set self.shift."""
        signal = ConcreteSignal()
        signal.set_shift(5)
        assert signal.shift == 5

    def test_shift_applied_in_get_call(self):
        """Shift should be applied when calling data_point.get()."""
        signal = ConcreteSignal()
        signal.set_shift(2)

        # When check() is called, it should use shift=0 + self.shift = 2
        # Simulate: data_point.get("close", 1, shift=0 + self.shift)
        # So we need to provide data at (close, 1, 2)
        data_point = MockDataPoint({("close", 1, 2): 150.0})

        # With shift=2, it should read from shift=2
        assert signal.check(data_point, {}, None) is True


class TestResetMethod:
    """Test reset() method."""

    def test_reset_default_returns_false(self):
        """reset() should return False by default."""
        signal = ConcreteSignal()
        assert signal.reset() is False


class TestGetDataMethod:
    """Test get_data() method."""

    def test_get_data_default_returns_empty_dict(self):
        """get_data() should return {} by default."""
        signal = ConcreteSignal()
        assert signal.get_data() == {}


class TestGetMarkerPosMethod:
    """Test get_marker_pos() method."""

    def test_get_marker_pos_default_value(self):
        """get_marker_pos() should return data_point.get('close', 1, 1) by default."""
        signal = ConcreteSignal()
        data_point = MockDataPoint({("close", 1, 1): 99.5})

        pos = signal.get_marker_pos(data_point)
        assert pos == 99.5

    def test_get_marker_pos_returns_nan_when_not_available(self):
        """get_marker_pos() should return NaN if data not available."""
        signal = ConcreteSignal()
        data_point = MockDataPoint()

        pos = signal.get_marker_pos(data_point)
        assert math.isnan(pos)


class TestIntegration:
    """Integration tests for BaseSignal usage patterns."""

    def test_signal_workflow(self):
        """Test typical signal workflow: create, configure, check, get metadata."""
        signal = ConcreteSignal()

        # Configure shift
        signal.set_shift(1)
        assert signal.shift == 1

        # Check condition
        data_point = MockDataPoint({("close", 1, 1): 120.0})
        result = signal.check(data_point, {}, None)
        assert result is True

        # Get metadata
        metadata = signal.get_data()
        assert isinstance(metadata, dict)

        # Check reset behavior
        should_abort = signal.reset()
        assert should_abort is False

        # Get marker position
        data_point = MockDataPoint({("close", 1, 1): 125.0})
        pos = signal.get_marker_pos(data_point)
        assert pos == 125.0
