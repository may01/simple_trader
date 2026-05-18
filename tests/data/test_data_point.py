import pytest
import pandas as pd
import numpy as np
from simple_trader.data.data_point import LiveDataPoint, WideDataPoint


def _make_ohlc(tf: int, rows: int = 5) -> pd.DataFrame:
    """Build minimal OHLC DataFrame with tf-prefixed columns."""
    data = {
        f"{tf}_close": np.arange(rows, dtype=float),
        f"{tf}_rsi_14": np.arange(rows, dtype=float) * 2,
        f"{tf}_open": np.ones(rows),
    }
    return pd.DataFrame(data)


def test_live_data_point_get_returns_last_row():
    ohlc = {1: _make_ohlc(1, rows=5)}
    point = LiveDataPoint(ohlc)
    # Last row of 1_close is 4.0
    assert point.get("close", tf=1) == 4.0


def test_live_data_point_shift_1_returns_second_to_last():
    ohlc = {1: _make_ohlc(1, rows=5)}
    point = LiveDataPoint(ohlc)
    assert point.get("close", tf=1, shift=1) == 3.0


def test_live_data_point_get_different_tf():
    ohlc = {
        1: _make_ohlc(1, rows=5),
        15: _make_ohlc(15, rows=3),
    }
    point = LiveDataPoint(ohlc)
    # Last row of 15_close is 2.0
    assert point.get("close", tf=15) == 2.0


def test_live_data_point_get_df_returns_dataframe():
    ohlc = {1: _make_ohlc(1, rows=5)}
    point = LiveDataPoint(ohlc)
    df = point.get_df(tf=1)
    assert isinstance(df, pd.DataFrame)
    assert "1_close" in df.columns


def test_live_data_point_missing_tf_raises_key_error():
    ohlc = {1: _make_ohlc(1)}
    point = LiveDataPoint(ohlc)
    with pytest.raises(KeyError):
        point.get("close", tf=99)


def test_wide_data_point_get_returns_scalar():
    df = pd.DataFrame({
        "1_close": [10.0, 20.0, 30.0],
        "15_rsi_14": [50.0, 55.0, 60.0],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    ts = df.index[2]
    point = WideDataPoint(df, ts)
    assert point.get("close", tf=1) == 30.0


def test_wide_data_point_shift_reads_earlier_row():
    df = pd.DataFrame({
        "1_close": [10.0, 20.0, 30.0],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    ts = df.index[2]
    point = WideDataPoint(df, ts)
    assert point.get("close", tf=1, shift=1) == 20.0
    assert point.get("close", tf=1, shift=2) == 10.0
