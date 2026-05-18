import pytest
import pandas as pd
import numpy as np
from simple_trader.data.indicators import Indicators


def _make_ohlcv(tf: int, rows: int = 100) -> pd.DataFrame:
    """Build OHLCV DataFrame with tf-prefixed columns."""
    np.random.seed(42)
    closes = 15.0 + np.cumsum(np.random.randn(rows) * 0.1)
    return pd.DataFrame({
        f"{tf}_open":   closes * 0.999,
        f"{tf}_high":   closes * 1.001,
        f"{tf}_low":    closes * 0.998,
        f"{tf}_close":  closes,
        f"{tf}_volume": np.abs(np.random.randn(rows)) * 1000,
    })


def test_indicators_compute_adds_rsi_column():
    df = _make_ohlcv(tf=1, rows=100)
    ind = Indicators()
    ind.compute(df, tf=1)
    assert "1_rsi_14" in df.columns


def test_indicators_compute_adds_ema_columns():
    df = _make_ohlcv(tf=1, rows=100)
    ind = Indicators()
    ind.compute(df, tf=1)
    assert "1_ema_25" in df.columns
    assert "1_ema_50" in df.columns


def test_indicators_rsi_values_are_between_0_and_100():
    df = _make_ohlcv(tf=1, rows=100)
    ind = Indicators()
    ind.compute(df, tf=1)
    rsi = df["1_rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_indicators_drop_na_removes_incomplete_leading_rows():
    df = _make_ohlcv(tf=1, rows=100)
    original_len = len(df)
    ind = Indicators()
    ind.compute(df, tf=1)
    ind.drop_na(df, tf=1)
    assert len(df) < original_len


def test_indicators_compute_twice_does_not_duplicate_columns():
    df = _make_ohlcv(tf=1, rows=100)
    ind = Indicators()
    ind.compute(df, tf=1)
    ind.compute(df, tf=1)
    assert df.columns.duplicated().sum() == 0
