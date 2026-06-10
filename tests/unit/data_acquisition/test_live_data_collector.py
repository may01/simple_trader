"""Unit tests for training/live_data_collector.py — LiveDataCollector class."""

import os
import threading

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call

from training.live_data_collector import LiveDataCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a DataFrame with raw column names (open, high, low, close, volume)
    as returned by get_candles_history."""
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    idx.name = "open_time"
    return pd.DataFrame(
        {
            "open": np.ones(n, dtype=float),
            "high": np.ones(n, dtype=float) * 1.1,
            "low": np.ones(n, dtype=float) * 0.9,
            "close": np.ones(n, dtype=float),
            "volume": np.ones(n, dtype=float) * 100,
            "close_time": idx,
            "taker_base_vol": np.ones(n, dtype=float) * 50,
        },
        index=idx,
    )


def _make_renamed_df(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a DataFrame with renamed columns (o, h, l, c, v) matching graber_data.pkl."""
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


def _make_mock_stock(raw_df: pd.DataFrame = None) -> MagicMock:
    """Return a mock StockInterface whose get_candles_history returns {1: raw_df}."""
    stock = MagicMock()
    if raw_df is None:
        raw_df = _make_raw_df(5)
    stock.get_candles_history.return_value = {1: raw_df}
    return stock


# ---------------------------------------------------------------------------
# 1. Constructor stores attributes, running=False
# ---------------------------------------------------------------------------

def test_constructor_stores_attributes():
    """Constructor sets stock, output_path, coin, interval_seconds, running=False."""
    stock = _make_mock_stock()
    collector = LiveDataCollector(
        stock=stock,
        output_path="/tmp/test.pkl",
        coin="link",
        interval_seconds=30,
    )
    assert collector.stock is stock
    assert collector.output_path == "/tmp/test.pkl"
    assert collector.coin == "link"
    assert collector.interval_seconds == 30
    assert collector.running is False


def test_constructor_default_interval():
    """Default interval_seconds is 60."""
    stock = _make_mock_stock()
    collector = LiveDataCollector(stock=stock, output_path="/tmp/test.pkl", coin="btc")
    assert collector.interval_seconds == 60


# ---------------------------------------------------------------------------
# 2. stop() sets running=False
# ---------------------------------------------------------------------------

def test_stop_sets_running_false():
    """stop() sets running to False."""
    stock = _make_mock_stock()
    collector = LiveDataCollector(stock=stock, output_path="/tmp/test.pkl", coin="link")
    collector.running = True
    collector.stop()
    assert collector.running is False


# ---------------------------------------------------------------------------
# 3. _poll() — no-op when no new rows (all old timestamps)
# ---------------------------------------------------------------------------

def test_poll_noop_when_no_new_rows(tmp_path):
    """_poll() does nothing when stock returns data with no rows newer than last_timestamp."""
    existing = _make_renamed_df(5, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")
    existing.to_pickle(path)

    # Stock returns same 5 rows (all old)
    raw_df = _make_raw_df(5, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    # File should be unchanged (same 5 rows)
    loaded = pd.read_pickle(path)
    assert len(loaded) == 5


def test_poll_noop_does_not_modify_file_mtime(tmp_path):
    """_poll() with no new rows does not rewrite the file."""
    existing = _make_renamed_df(5, "2024-01-01")
    path = str(tmp_path / "graber_data.pkl")
    existing.to_pickle(path)
    mtime_before = os.path.getmtime(path)

    raw_df = _make_raw_df(5, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    mtime_after = os.path.getmtime(path)
    assert mtime_after == mtime_before


# ---------------------------------------------------------------------------
# 4. _poll() — filters and renames columns when new rows present
# ---------------------------------------------------------------------------

def test_poll_appends_new_rows(tmp_path):
    """_poll() appends new rows to existing file when newer candles are available."""
    existing = _make_renamed_df(3, "2024-01-01 00:00")
    path = str(tmp_path / "graber_data.pkl")
    existing.to_pickle(path)

    # Stock returns 5 rows: 3 old + 2 new
    raw_df = _make_raw_df(5, "2024-01-01 00:00")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    loaded = pd.read_pickle(path)
    assert len(loaded) == 5


def test_poll_renames_columns(tmp_path):
    """_poll() renames open→o, high→h, low→l, close→c, volume→v in saved data."""
    path = str(tmp_path / "graber_data.pkl")
    # No existing file

    raw_df = _make_raw_df(3, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    loaded = pd.read_pickle(path)
    assert "o" in loaded.columns
    assert "h" in loaded.columns
    assert "l" in loaded.columns
    assert "c" in loaded.columns
    assert "v" in loaded.columns
    assert "open" not in loaded.columns
    assert "high" not in loaded.columns


def test_poll_preserves_close_time_and_taker_base_vol(tmp_path):
    """_poll() preserves close_time and taker_base_vol columns as-is."""
    path = str(tmp_path / "graber_data.pkl")
    raw_df = _make_raw_df(3, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    loaded = pd.read_pickle(path)
    assert "close_time" in loaded.columns
    assert "taker_base_vol" in loaded.columns


def test_poll_calls_get_candles_history_with_correct_args(tmp_path):
    """_poll() calls stock.get_candles_history([1], coin)."""
    path = str(tmp_path / "graber_data.pkl")
    raw_df = _make_raw_df(3, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    stock.get_candles_history.assert_called_once_with([1], "link")


# ---------------------------------------------------------------------------
# 5. _poll() — handles empty existing file (loads empty DataFrame)
# ---------------------------------------------------------------------------

def test_poll_creates_file_when_missing(tmp_path):
    """_poll() creates the output file when it does not exist."""
    path = str(tmp_path / "graber_data.pkl")
    raw_df = _make_raw_df(5, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._poll()

    assert os.path.exists(path)
    loaded = pd.read_pickle(path)
    assert len(loaded) == 5


def test_poll_all_rows_taken_when_existing_is_empty(tmp_path):
    """_poll() takes all rows from stock when no existing file present."""
    path = str(tmp_path / "graber_data.pkl")
    raw_df = _make_raw_df(3, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="btc")
    collector._poll()

    loaded = pd.read_pickle(path)
    assert len(loaded) == 3


# ---------------------------------------------------------------------------
# 6. _append_save() — deduplicates on open_time
# ---------------------------------------------------------------------------

def test_append_save_deduplicates_open_time(tmp_path):
    """_append_save() drops duplicate open_time rows, keeping last."""
    path = str(tmp_path / "graber_data.pkl")

    existing = _make_renamed_df(3, "2024-01-01")
    # new_rows overlaps: 2 duplicate + 2 new
    new_rows = _make_renamed_df(4, "2024-01-01 00:02")

    stock = _make_mock_stock()
    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._append_save(existing, new_rows)

    loaded = pd.read_pickle(path)
    # 00:00, 00:01, 00:02, 00:03, 00:04, 00:05 → 6 unique minutes
    # existing: 00:00, 00:01, 00:02 — new_rows: 00:02, 00:03, 00:04, 00:05
    assert len(loaded) == 6
    # No duplicate index
    assert loaded.index.is_unique


def test_append_save_keep_last_on_duplicate(tmp_path):
    """_append_save() keeps last value when open_time is duplicated."""
    path = str(tmp_path / "graber_data.pkl")

    idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
    idx.name = "open_time"
    existing = pd.DataFrame({"o": [1.0, 2.0], "h": [1.0, 2.0], "l": [1.0, 2.0],
                              "c": [1.0, 2.0], "v": [100.0, 200.0],
                              "close_time": idx, "taker_base_vol": [50.0, 50.0]},
                             index=idx)

    # new_rows has duplicate of second row with updated value
    new_rows = pd.DataFrame({"o": [9.9], "h": [9.9], "l": [9.9],
                              "c": [9.9], "v": [999.0],
                              "close_time": [idx[1]], "taker_base_vol": [99.0]},
                             index=pd.DatetimeIndex([idx[1]], name="open_time"))

    stock = _make_mock_stock()
    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._append_save(existing, new_rows)

    loaded = pd.read_pickle(path)
    assert len(loaded) == 2
    # The last (new_rows) value should be kept
    assert loaded.loc[idx[1], "o"] == pytest.approx(9.9)


# ---------------------------------------------------------------------------
# 7. _append_save() — saves atomically (no .tmp left)
# ---------------------------------------------------------------------------

def test_append_save_no_tmp_left(tmp_path):
    """_append_save() leaves no .tmp file after saving."""
    path = str(tmp_path / "graber_data.pkl")
    existing = _make_renamed_df(2, "2024-01-01")
    new_rows = _make_renamed_df(2, "2024-01-01 00:02")

    stock = _make_mock_stock()
    collector = LiveDataCollector(stock=stock, output_path=path, coin="link")
    collector._append_save(existing, new_rows)

    assert not os.path.exists(path + ".tmp")
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 8. start() — calls _poll() and stops when running=False after first iteration
# ---------------------------------------------------------------------------

def test_start_calls_poll_and_exits_when_stopped(tmp_path):
    """start() sleeps, calls _poll(), then exits when running=False."""
    path = str(tmp_path / "graber_data.pkl")
    raw_df = _make_raw_df(3, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link", interval_seconds=1)

    poll_call_count = []

    original_poll = collector._poll.__func__ if hasattr(collector._poll, '__func__') else None

    def mock_poll():
        poll_call_count.append(1)
        collector.running = False  # stop after first poll

    collector._poll = mock_poll

    with patch("time.sleep") as mock_sleep:
        collector.start()

    assert collector.running is False
    assert len(poll_call_count) == 1
    mock_sleep.assert_called_once_with(1)


def test_start_sets_running_true_initially(tmp_path):
    """start() sets running=True before entering the loop."""
    path = str(tmp_path / "graber_data.pkl")
    raw_df = _make_raw_df(2, "2024-01-01")
    stock = _make_mock_stock(raw_df)

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link", interval_seconds=1)

    running_states = []

    def mock_poll():
        running_states.append(collector.running)
        collector.running = False

    collector._poll = mock_poll

    with patch("time.sleep"):
        collector.start()

    assert True in running_states  # running was True during poll


def test_start_handles_keyboard_interrupt(tmp_path):
    """start() exits cleanly on KeyboardInterrupt, setting running=False."""
    path = str(tmp_path / "graber_data.pkl")
    stock = _make_mock_stock()

    collector = LiveDataCollector(stock=stock, output_path=path, coin="link", interval_seconds=60)

    def mock_sleep(_):
        raise KeyboardInterrupt

    with patch("time.sleep", side_effect=mock_sleep):
        collector.start()

    assert collector.running is False
