"""Shared test fixtures for signals testing."""

import pytest
from data import DataPoint


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


@pytest.fixture
def make_data_point():
    """Fixture to create MockDataPoint instances with test data.

    Returns:
        Callable that takes a dict of test data and returns a MockDataPoint.
    """
    def _make(data: dict) -> MockDataPoint:
        return MockDataPoint(data)
    return _make
