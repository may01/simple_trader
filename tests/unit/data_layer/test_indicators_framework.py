"""Unit tests for indicators.py — IndicatorField framework and Indicators orchestrator."""

import os

import pandas as pd
import pytest

from indicators import (
    IndicatorField,
    Indicators,
    _PlaceholderField,
    build_indicator_input,
)
from data import WideDataPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wide_df(n: int = 20) -> pd.DataFrame:
    """Build a minimal wide DataFrame covering tf=1 and tf=5 with is_closed columns.

    n rows at 1-minute intervals starting at 2024-01-01 00:00 UTC.
    tf=1: every row is closed.
    tf=5: rows at positions 4, 9, 14, 19, ... are closed (minute % 5 == 4).
    """
    start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    idx = pd.date_range(start, periods=n, freq="1min")
    data: dict = {}

    for tf in (1, 5):
        data[f"{tf}_open"]  = [float(i + 1) for i in range(n)]
        data[f"{tf}_close"] = [float(i + 10) for i in range(n)]
        data[f"{tf}_high"]  = [float(i + 11) for i in range(n)]
        data[f"{tf}_low"]   = [float(i + 0.5) for i in range(n)]
        data[f"{tf}_volume"] = [float(i + 100) for i in range(n)]
        if tf == 1:
            data[f"{tf}_is_closed"] = [True] * n
        else:  # tf == 5
            data[f"{tf}_is_closed"] = [idx[i].minute % 5 == 4 for i in range(n)]

    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# ConstantField — a minimal real IndicatorField used in test 4
# ---------------------------------------------------------------------------

class ConstantField(IndicatorField):
    """A concrete IndicatorField that always returns [1.0, 2.0] for testing."""

    name = "test_const"
    group = "test"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to = [5]
    params: dict = {}

    def compute(self, dp, tf: int) -> pd.Series:
        return pd.Series([1.0, 2.0], name=f"{tf}_{self.name}")


# ---------------------------------------------------------------------------
# Test: _sorted_fields returns fields for a TF
# ---------------------------------------------------------------------------

class TestSortedFields:
    def setup_method(self):
        """Reset the registry so each test starts fresh."""
        Indicators._registry = None

    def test_sorted_fields_returns_fields_for_tf(self):
        """_sorted_fields(5) returns a non-empty list."""
        fields = Indicators._sorted_fields(5)
        assert len(fields) > 0

    def test_sorted_fields_filters_by_tf(self):
        """Fields with applies_to=[15,60,240,1440] (classification group) NOT in _sorted_fields(1)."""
        fields_1 = Indicators._sorted_fields(1)
        names_1 = {f.name for f in fields_1}

        fields_15 = Indicators._sorted_fields(15)
        names_15 = {f.name for f in fields_15}

        # Classification group only applies to [15, 60, 240, 1440].
        classification_in_15 = {f.name for f in fields_15 if f.group == "classification"}
        assert len(classification_in_15) > 0, "Expected classification fields at tf=15"

        # Those same fields must not appear at tf=1.
        for name in classification_in_15:
            assert name not in names_1, (
                f"Classification field '{name}' should not appear in _sorted_fields(1)"
            )

    def test_sorted_fields_filters_by_group(self):
        """_sorted_fields(5, groups=['momentum']) returns only momentum fields."""
        fields = Indicators._sorted_fields(5, groups=["momentum"])
        assert len(fields) > 0
        for f in fields:
            assert f.group == "momentum", (
                f"Field '{f.name}' has group '{f.group}', expected 'momentum'"
            )


# ---------------------------------------------------------------------------
# Test: compute() writes to the DataFrame
# ---------------------------------------------------------------------------

class TestIndicatorsCompute:
    def setup_method(self):
        """Reset the registry and inject a controlled ConstantField for tf=5."""
        Indicators._registry = None

    def test_compute_writes_to_dataframe(self):
        """Indicators.compute() with a ConstantField writes the expected column."""
        # Override the registry with a single known field
        Indicators._registry = [ConstantField()]

        df = _make_wide_df()
        ts = df.index[-1]

        class _SimpleMutableDP:
            """Minimal DataPoint that returns a mutable DataFrame from get_df()."""
            def __init__(self, frame):
                self._df = frame

            def get_df(self, tf):
                return self._df

        dp = _SimpleMutableDP(df)
        Indicators.compute(dp, tf=5)

        assert "5_test_const" in df.columns, (
            "Indicators.compute() must write '5_test_const' column"
        )


# ---------------------------------------------------------------------------
# Test: build_indicator_input
# ---------------------------------------------------------------------------

class TestBuildIndicatorInput:
    def test_build_indicator_input_at_boundary(self):
        """At a 5_is_closed=True row, result includes that row and last row is closed."""
        df = _make_wide_df(10)
        # Row at position 4 is closed for tf=5 (minute 04)
        ts = df.index[4]
        result = build_indicator_input(df, ts, tf=5)

        assert not result.empty, "Result should not be empty"
        assert bool(result.iloc[-1]["5_is_closed"]), (
            "Last row should be closed when ts itself is a closed candle"
        )
        assert result.index[-1] == ts

    def test_build_indicator_input_mid_candle(self):
        """At a non-closed row, result's last row is the partial row."""
        df = _make_wide_df(10)
        # Row at position 1 is NOT closed for tf=5 (minute 01)
        ts = df.index[1]
        assert not df.loc[ts, "5_is_closed"], "Precondition: ts should not be closed"

        result = build_indicator_input(df, ts, tf=5)
        assert not result.empty
        assert result.index[-1] == ts, (
            "Last row of result should be the partial current candle"
        )
        # The partial row is NOT closed
        assert not bool(result.iloc[-1]["5_is_closed"])

    def test_build_indicator_input_max_105_rows(self):
        """With 200+ rows in the input, result has at most 105 rows."""
        # Build a df where all rows are closed for tf=1
        n = 200
        start = pd.Timestamp("2024-01-01", tz="UTC")
        idx = pd.date_range(start, periods=n, freq="1min")
        df = pd.DataFrame(
            {
                "1_close": [float(i) for i in range(n)],
                "1_is_closed": [True] * n,
            },
            index=idx,
        )
        ts = idx[-1]
        result = build_indicator_input(df, ts, tf=1)
        assert len(result) <= 105, f"Expected at most 105 rows, got {len(result)}"
        assert len(result) == 105


# ---------------------------------------------------------------------------
# Test: WideDataPoint.get_df() uses build_indicator_input
# ---------------------------------------------------------------------------

class TestWideDataPointGetDf:
    def test_wide_data_point_get_df_uses_build_indicator_input(self):
        """WideDataPoint(df, ts).get_df(5) returns same result as build_indicator_input(df, ts, 5)."""
        df = _make_wide_df(20)
        ts = df.index[9]  # A closed row for tf=5 (minute 09 → 09 % 5 = 4 → closed)

        wdp = WideDataPoint(df, ts)
        result = wdp.get_df(5)
        expected = build_indicator_input(df, ts, 5)

        assert result.shape == expected.shape, (
            f"Shape mismatch: {result.shape} vs {expected.shape}"
        )
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# Test: resource dependency checking
# ---------------------------------------------------------------------------

class TestResourceDependencies:
    def setup_method(self):
        Indicators._registry = None

    def test_resource_dep_unavailable_skips_field(self, monkeypatch, tmp_path):
        """Field with resource_dependencies=["nonexistent.json"] excluded when check_resources=True."""
        # Point stats_folder() to tmp_path so we control the filesystem
        monkeypatch.setenv("DATA_ROOT", "test_data")
        monkeypatch.setenv("PAIR", "BTCUSDT")

        class _FieldWithResource(IndicatorField):
            name = "with_resource"
            group = "test"
            dependencies: list[str] = []
            resource_dependencies = ["nonexistent.json"]
            applies_to = [5]
            params: dict = {}

            def compute(self, dp, tf):
                return pd.Series(dtype=float)

        class _FieldNoResource(IndicatorField):
            name = "no_resource"
            group = "test"
            dependencies: list[str] = []
            resource_dependencies: list[str] = []
            applies_to = [5]
            params: dict = {}

            def compute(self, dp, tf):
                return pd.Series(dtype=float)

        Indicators._registry = [_FieldWithResource(), _FieldNoResource()]

        # With check_resources=True: field with missing dep is excluded
        fields_checked = Indicators._sorted_fields(5, check_resources=True)
        names_checked = [f.name for f in fields_checked]
        assert "with_resource" not in names_checked, (
            "Field with missing resource dep should be excluded when check_resources=True"
        )
        assert "no_resource" in names_checked, (
            "Field with no resource dep should be included"
        )

        # With check_resources=False: both fields are included
        fields_unchecked = Indicators._sorted_fields(5, check_resources=False)
        names_unchecked = [f.name for f in fields_unchecked]
        assert "with_resource" in names_unchecked, (
            "Field with missing resource dep should be included when check_resources=False"
        )
        assert "no_resource" in names_unchecked
