"""Unit tests for training/graber.py — Graber class."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from training.graber import Graber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame matching Phase 02 schema."""
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    idx.name = "open_time"
    return pd.DataFrame(
        {
            "o": np.ones(n, dtype=float),
            "h": np.ones(n, dtype=float) * 1.1,
            "l": np.ones(n, dtype=float) * 0.9,
            "c": np.ones(n, dtype=float),
            "v": np.ones(n, dtype=float) * 100,
            "close_time": idx,
            "taker_base_vol": np.ones(n, dtype=float) * 50,
        },
        index=idx,
    )


def _ts_ms(dt_str: str) -> int:
    """Convert UTC datetime string to Unix milliseconds."""
    ts = pd.Timestamp(dt_str, tz="UTC")
    return int(ts.timestamp() * 1000)


def _make_mock_stock(return_df: pd.DataFrame = None) -> MagicMock:
    """Return a mock StockInterface whose get_candles_range returns return_df."""
    stock = MagicMock()
    stock.get_candles_range.return_value = return_df if return_df is not None else pd.DataFrame()
    return stock


# ---------------------------------------------------------------------------
# ensure_data — no-op when data is current
# ---------------------------------------------------------------------------

def test_ensure_data_noop_when_current(tmp_path):
    """If file exists and covers the requested range, no fetch is triggered."""
    start_ms = _ts_ms("2024-01-01")
    end_ms = _ts_ms("2024-01-01 00:09")
    df = _make_df(10, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")
    df.to_pickle(path)

    stock = _make_mock_stock(df)
    graber = Graber(stock, path)
    graber.ensure_data("LINKUSDT", start_ms, end_ms)

    stock.get_candles_range.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_data — fetch when file is missing
# ---------------------------------------------------------------------------

def test_ensure_data_fetches_when_file_missing(tmp_path):
    """If file does not exist, fetches full range and saves."""
    start_ms = _ts_ms("2024-01-01")
    end_ms = _ts_ms("2024-01-01 00:09")
    new_df = _make_df(10, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")

    stock = _make_mock_stock(new_df)
    graber = Graber(stock, path)
    graber.ensure_data("LINKUSDT", start_ms, end_ms)

    stock.get_candles_range.assert_called_once_with("LINKUSDT", start_ms, end_ms)
    assert os.path.exists(path)
    loaded = pd.read_pickle(path)
    assert len(loaded) == 10


# ---------------------------------------------------------------------------
# ensure_data — partial data: fetches only missing range
# ---------------------------------------------------------------------------

def test_ensure_data_fetches_missing_range_only(tmp_path):
    """If existing data ends before end_ms, only missing tail is fetched."""
    # Existing: first 5 minutes
    existing_df = _make_df(5, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")
    existing_df.to_pickle(path)

    # New data covers the tail: minutes 5-9
    tail_df = _make_df(5, "2024-01-01 00:05")
    stock = _make_mock_stock(tail_df)

    start_ms = _ts_ms("2024-01-01")
    end_ms = _ts_ms("2024-01-01 00:09")  # inclusive of minute 9

    graber = Graber(stock, path)
    graber.ensure_data("LINKUSDT", start_ms, end_ms)

    # Should have fetched only from existing end onwards
    call_args = stock.get_candles_range.call_args
    assert call_args is not None
    fetched_start = call_args[0][1]
    existing_end_ms = int(existing_df.index[-1].timestamp() * 1000)
    assert fetched_start == existing_end_ms

    # Merged file should contain more rows than original
    loaded = pd.read_pickle(path)
    assert len(loaded) > len(existing_df)


# ---------------------------------------------------------------------------
# ensure_data — raises when stock returns empty for non-empty range
# ---------------------------------------------------------------------------

def test_ensure_data_raises_on_empty_fetch(tmp_path):
    """If stock returns empty DataFrame for a non-empty range, raise ValueError."""
    path = str(tmp_path / "graber_data.pkl")
    stock = _make_mock_stock(pd.DataFrame())  # empty result

    start_ms = _ts_ms("2024-01-01")
    end_ms = _ts_ms("2024-01-01 00:09")

    graber = Graber(stock, path)
    with pytest.raises(ValueError, match="empty"):
        graber.ensure_data("LINKUSDT", start_ms, end_ms)


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_returns_dataframe(tmp_path):
    """load() returns the DataFrame stored at output_path."""
    df = _make_df(5)
    path = str(tmp_path / "graber_data.pkl")
    df.to_pickle(path)

    stock = _make_mock_stock()
    graber = Graber(stock, path)
    loaded = graber.load()

    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 5


def test_load_raises_file_not_found(tmp_path):
    """load() raises FileNotFoundError when output_path does not exist."""
    path = str(tmp_path / "nonexistent.pkl")
    stock = _make_mock_stock()
    graber = Graber(stock, path)

    with pytest.raises(FileNotFoundError):
        graber.load()


# ---------------------------------------------------------------------------
# is_current()
# ---------------------------------------------------------------------------

def test_is_current_true_when_last_row_meets_end_ms(tmp_path):
    """is_current() returns True if last row open_time >= end_ms."""
    df = _make_df(10, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")
    df.to_pickle(path)

    stock = _make_mock_stock()
    graber = Graber(stock, path)

    # last row is minute 9 → 2024-01-01 00:09 UTC
    end_ms = _ts_ms("2024-01-01 00:09")
    assert graber.is_current(end_ms) is True


def test_is_current_false_when_last_row_before_end_ms(tmp_path):
    """is_current() returns False if last row open_time < end_ms."""
    df = _make_df(5, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")
    df.to_pickle(path)

    stock = _make_mock_stock()
    graber = Graber(stock, path)

    # last row is minute 4; request end is minute 9
    end_ms = _ts_ms("2024-01-01 00:09")
    assert graber.is_current(end_ms) is False


def test_is_current_false_when_file_missing(tmp_path):
    """is_current() returns False (not raises) when file does not exist."""
    path = str(tmp_path / "nonexistent.pkl")
    stock = _make_mock_stock()
    graber = Graber(stock, path)

    end_ms = _ts_ms("2024-01-01 00:09")
    assert graber.is_current(end_ms) is False


# ---------------------------------------------------------------------------
# Atomic save: no .tmp file left after ensure_data
# ---------------------------------------------------------------------------

def test_atomic_save_no_tmp_file_remains(tmp_path):
    """No .tmp file should be left on disk after ensure_data completes."""
    new_df = _make_df(5, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")

    stock = _make_mock_stock(new_df)
    graber = Graber(stock, path)

    start_ms = _ts_ms("2024-01-01")
    end_ms = _ts_ms("2024-01-01 00:04")
    graber.ensure_data("LINKUSDT", start_ms, end_ms)

    assert not os.path.exists(path + ".tmp")
    assert os.path.exists(path)
