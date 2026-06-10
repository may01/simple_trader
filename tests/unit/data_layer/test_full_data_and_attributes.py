"""Unit tests for FullData (data.py) and DataAttributes (indicators.py)."""

import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from data import FullData, _build_wide_df
from indicators import DataAttributes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_df():
    """Build a synthetic wide DataFrame covering ~2 days at 1-min frequency.

    Uses _build_wide_df so all tf_is_closed, tf_open_index, etc. columns are
    present for tfs [1, 5, 15, 60, 240, 1440].
    """
    np.random.seed(99)
    # Use 2880 rows = 2 full days so we have multiple closed candles at all TFs
    idx = pd.date_range("2024-01-01 00:00", periods=2880, freq="1min", tz="UTC")
    raw = pd.DataFrame(
        {
            "open":          np.random.uniform(40000, 41000, 2880),
            "high":          np.random.uniform(41000, 42000, 2880),
            "low":           np.random.uniform(39000, 40000, 2880),
            "close":         np.random.uniform(40000, 41000, 2880),
            "volume":        np.random.uniform(1, 100, 2880),
            "taker_base_vol": np.random.uniform(0.5, 50, 2880),
        },
        index=idx,
    )
    return _build_wide_df(raw)


@pytest.fixture
def wide_df_with_rsi(wide_df):
    """Add synthetic rsi_ma8 columns for the TFs used by DataAttributes."""
    df = wide_df.copy()
    np.random.seed(77)
    for tf in [15, 60, 240, 1440]:
        df[f"{tf}_rsi_ma8"] = np.random.uniform(20, 80, len(df))
    return df


# ---------------------------------------------------------------------------
# FullData tests
# ---------------------------------------------------------------------------

class TestFullData:
    def test_get_returns_only_closed_rows(self, wide_df):
        """All returned rows must have {tf}_is_closed == True."""
        fd = FullData(wide_df)
        result = fd.get(tf=5)
        assert result["5_is_closed"].all()

    def test_get_returns_only_tf_columns(self, wide_df):
        """All returned column names must start with the tf prefix."""
        fd = FullData(wide_df)
        result = fd.get(tf=5)
        for col in result.columns:
            assert col.startswith("5_"), f"Column '{col}' does not start with '5_'"

    def test_get_tf5_closes_at_minute_4(self, wide_df):
        """For tf=5, every closed-candle timestamp has minute in {4, 9, 14, 19, 24, 29, ...}."""
        fd = FullData(wide_df)
        result = fd.get(tf=5)
        # Each minute index should be 4 mod 5 (i.e. 4, 9, 14, 19, ...)
        assert len(result) > 0, "Expected at least some tf=5 closed candles"
        for ts in result.index:
            assert ts.minute % 5 == 4, (
                f"Timestamp {ts} has minute {ts.minute}, expected minute % 5 == 4"
            )

    def test_get_candle_returns_series(self, wide_df):
        """get_candle() returns a pd.Series for a valid open_time."""
        fd = FullData(wide_df)
        # Find a valid open_time for tf=15 from the df
        closed_mask = wide_df["15_is_closed"] == True
        closed_rows = wide_df[closed_mask]
        assert len(closed_rows) > 0
        # open_time is stored in 15_open_index column
        valid_open_time = closed_rows["15_open_index"].iloc[0]
        result = fd.get_candle(tf=15, open_time=valid_open_time)
        assert isinstance(result, pd.Series)

    def test_get_candle_raises_on_missing(self, wide_df):
        """get_candle() raises KeyError if open_time is not present."""
        fd = FullData(wide_df)
        bogus_ts = pd.Timestamp("1990-01-01", tz="UTC")
        with pytest.raises(KeyError):
            fd.get_candle(tf=15, open_time=bogus_ts)


# ---------------------------------------------------------------------------
# DataAttributes tests
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_stats_folder(tmp_path, monkeypatch):
    """Monkeypatch helpers.stats_folder() to return tmp_path subdir.

    DataAttributes uses ``from helpers import stats_folder`` locally, so
    patching the function on the ``helpers`` module is sufficient.
    """
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    import helpers
    monkeypatch.setattr(helpers, "stats_folder", lambda: str(stats_dir) + "/")
    return str(stats_dir) + "/"


class TestDataAttributes:
    def test_compute_creates_rsi_classification_file(
        self, wide_df_with_rsi, patched_stats_folder
    ):
        """compute() creates rsi_classification.json in stats_folder()."""
        da = DataAttributes()
        da.compute(wide_df_with_rsi)
        path = patched_stats_folder + "rsi_classification.json"
        assert os.path.exists(path), f"Expected file at {path}"

    def test_compute_idempotent(self, wide_df_with_rsi, patched_stats_folder):
        """Calling compute() twice does not raise and produces same file."""
        da = DataAttributes()
        da.compute(wide_df_with_rsi)
        path = patched_stats_folder + "rsi_classification.json"
        mtime_first = os.path.getmtime(path)
        # Second call should be a no-op (file already exists)
        da.compute(wide_df_with_rsi)
        mtime_second = os.path.getmtime(path)
        assert mtime_first == mtime_second, "compute() should not overwrite existing files"

    def test_load_rsi_classification_raises_if_absent(
        self, tmp_path, monkeypatch
    ):
        """load_rsi_classification() raises FileNotFoundError when file is missing."""
        stats_dir = tmp_path / "empty_stats"
        stats_dir.mkdir()
        import helpers
        monkeypatch.setattr(helpers, "stats_folder", lambda: str(stats_dir) + "/")
        with pytest.raises(FileNotFoundError):
            DataAttributes.load_rsi_classification()

    def test_compute_nn_stats_populates_column_stats(
        self, wide_df_with_rsi, patched_stats_folder
    ):
        """compute_nn_stats() populates column_stats for each feature_col."""
        da = DataAttributes()
        # Use a column that exists in the df and is tied to a tf with is_closed data
        feature_cols = ["15_rsi_ma8"]
        da.compute_nn_stats(wide_df_with_rsi, feature_cols)
        assert "15_rsi_ma8" in da.column_stats
        entry = da.column_stats["15_rsi_ma8"]
        assert "mean" in entry
        assert "std" in entry

    def test_get_stats_returns_mean_std(self, wide_df_with_rsi, patched_stats_folder):
        """get_stats() returns (mean, std) tuple for a known column."""
        da = DataAttributes()
        feature_cols = ["15_rsi_ma8"]
        da.compute_nn_stats(wide_df_with_rsi, feature_cols)
        mean, std = da.get_stats("15_rsi_ma8")
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_get_stats_raises_on_unknown_col(self):
        """get_stats() raises KeyError for a column not in column_stats."""
        da = DataAttributes()
        with pytest.raises(KeyError):
            da.get_stats("99_nonexistent_col")

    def test_save_and_load_roundtrip(self, wide_df_with_rsi, patched_stats_folder, tmp_path):
        """save() then load() preserves column_stats."""
        da = DataAttributes()
        feature_cols = ["15_rsi_ma8", "60_rsi_ma8"]
        da.compute_nn_stats(wide_df_with_rsi, feature_cols)

        save_path = str(tmp_path / "da.pkl")
        da.save(save_path)

        loaded = DataAttributes.load(save_path)
        assert loaded.column_stats == da.column_stats
        for col in feature_cols:
            assert col in loaded.column_stats
