"""Unit tests for grabers/grab_binance.py."""

import os
import time
import pickle

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from grabers.grab_binance import (
    grab_data,
    load_existing,
    merge_incremental,
    save_atomic,
    validate_1min_spacing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "o": np.ones(n),
            "h": np.ones(n) * 1.1,
            "l": np.ones(n) * 0.9,
            "c": np.ones(n),
            "v": np.ones(n) * 100,
            "close_time": idx,
            "taker_base_vol": np.ones(n) * 50,
        },
        index=idx,
    )


def _klines(n: int) -> list:
    """Build a list of fake Binance kline rows (n rows, 1-min spaced)."""
    now = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC in ms
    return [
        [
            now + i * 60_000,  # open_time
            "1.0",             # o
            "2.0",             # h
            "0.5",             # l
            "1.5",             # c
            "100",             # v
            now + i * 60_000 + 59_999,  # close_time
            "0",               # qav
            "0",               # num_trades
            "50",              # taker_base_vol
            "0",               # taker_quote_vol
            "0",               # ignore
        ]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# validate_1min_spacing
# ---------------------------------------------------------------------------

def test_validate_1min_spacing_pass():
    df = _make_df(10)
    validate_1min_spacing(df)  # should not raise


def test_validate_1min_spacing_empty_noop():
    validate_1min_spacing(pd.DataFrame())  # should not raise


def test_validate_1min_spacing_fail_gap():
    df = _make_df(10)
    df = df.drop(df.index[5])  # create a 2-min gap
    with pytest.raises(ValueError, match="1-minute spacing violated"):
        validate_1min_spacing(df)


def test_validate_1min_spacing_fail_duplicate():
    df = _make_df(5)
    # Duplicate a row by concatenating it
    dup = pd.concat([df, df.iloc[[2]]])
    dup.sort_index(inplace=True)
    with pytest.raises(ValueError, match="1-minute spacing violated"):
        validate_1min_spacing(dup)


# ---------------------------------------------------------------------------
# merge_incremental
# ---------------------------------------------------------------------------

def test_merge_incremental_no_duplicates():
    df1 = _make_df(5, "2024-01-01 00:00")
    df2 = _make_df(5, "2024-01-01 00:03")  # 2-row overlap
    merged = merge_incremental(df1, df2)
    assert len(merged) == 8  # 5 + 5 - 2 overlap


def test_merge_incremental_no_overlap():
    df1 = _make_df(3, "2024-01-01 00:00")
    df2 = _make_df(3, "2024-01-01 00:03")
    merged = merge_incremental(df1, df2)
    assert len(merged) == 6


def test_merge_incremental_sorted():
    df1 = _make_df(3, "2024-01-01 00:05")
    df2 = _make_df(3, "2024-01-01 00:00")
    merged = merge_incremental(df1, df2)
    assert merged.index.is_monotonic_increasing


def test_merge_incremental_full_overlap():
    df1 = _make_df(5, "2024-01-01 00:00")
    df2 = _make_df(5, "2024-01-01 00:00")  # identical range
    merged = merge_incremental(df1, df2)
    assert len(merged) == 5


# ---------------------------------------------------------------------------
# save_atomic / load_existing
# ---------------------------------------------------------------------------

def test_save_atomic_creates_file(tmp_path):
    df = _make_df(5)
    path = str(tmp_path / "test.pkl")
    save_atomic(df, path)
    assert os.path.exists(path)


def test_save_atomic_no_tmp_file_remains(tmp_path):
    df = _make_df(5)
    path = str(tmp_path / "test.pkl")
    save_atomic(df, path)
    assert not os.path.exists(path + ".tmp")


def test_save_atomic_roundtrip(tmp_path):
    df = _make_df(5)
    path = str(tmp_path / "test.pkl")
    save_atomic(df, path)
    loaded = pd.read_pickle(path)
    assert len(loaded) == 5
    assert list(loaded.columns) == list(df.columns)


def test_load_existing_returns_none_when_missing(tmp_path):
    path = str(tmp_path / "nonexistent.pkl")
    result = load_existing(path)
    assert result is None


def test_load_existing_returns_df(tmp_path):
    df = _make_df(5)
    path = str(tmp_path / "data.pkl")
    df.to_pickle(path)
    loaded = load_existing(path)
    assert loaded is not None
    assert len(loaded) == 5


# ---------------------------------------------------------------------------
# grab_data
# ---------------------------------------------------------------------------

def test_output_columns(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")

    with patch("grabers.grab_binance.Client") as mock_client_cls:
        mock_client_cls.return_value.get_historical_klines.return_value = _klines(60)
        df = grab_data("link_usdt", 0, 60 * 60 * 1000)

    for col in ["o", "h", "l", "c", "v", "taker_base_vol"]:
        assert col in df.columns, f"Missing column: {col}"


def test_output_dtypes_float64(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")

    with patch("grabers.grab_binance.Client") as mock_client_cls:
        mock_client_cls.return_value.get_historical_klines.return_value = _klines(10)
        df = grab_data("link_usdt", 0, 60 * 60 * 1000)

    for col in ["o", "h", "l", "c", "v", "taker_base_vol"]:
        assert df[col].dtype == np.float64, f"Column {col} not float64: {df[col].dtype}"


def test_symbol_conversion(monkeypatch):
    """Ensure pair string is converted to correct Binance symbol."""
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")

    with patch("grabers.grab_binance.Client") as mock_client_cls:
        mock_instance = mock_client_cls.return_value
        mock_instance.get_historical_klines.return_value = []
        grab_data("link_usdt", 0, 1000)

    call_args = mock_instance.get_historical_klines.call_args
    assert call_args[0][0] == "LINKUSDT"


def test_empty_klines_returns_empty_df(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")

    with patch("grabers.grab_binance.Client") as mock_client_cls:
        mock_client_cls.return_value.get_historical_klines.return_value = []
        df = grab_data("link_usdt", 0, 1000)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_index_is_datetime_utc(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")

    with patch("grabers.grab_binance.Client") as mock_client_cls:
        mock_client_cls.return_value.get_historical_klines.return_value = _klines(5)
        df = grab_data("link_usdt", 0, 60 * 60 * 1000)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
