# tests/unit/data_layer/test_wide_df_builder.py
# Tests for _build_wide_df() in data.py

import numpy as np
import pandas as pd
import pytest

from config_loader import CANDLES
from data import _build_wide_df


def make_fixture() -> pd.DataFrame:
    np.random.seed(42)
    idx = pd.date_range("2023-09-01", periods=1440, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.random.uniform(20, 21, 1440),
            "high": np.random.uniform(21, 22, 1440),
            "low": np.random.uniform(19, 20, 1440),
            "close": np.random.uniform(20, 21, 1440),
            "volume": np.random.uniform(100, 200, 1440),
            "taker_base_vol": np.random.uniform(50, 100, 1440),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# is_closed tests
# ---------------------------------------------------------------------------


def test_is_closed_tf1_always_true():
    df = _build_wide_df(make_fixture())
    assert df["1_is_closed"].all(), "tf=1 is_closed must be True for every row"


def test_is_closed_tf5_at_minute_4():
    df = _build_wide_df(make_fixture())
    # minutes 4, 9, 14 within the first hour should be True
    for minute in [4, 9, 14]:
        ts = pd.Timestamp("2023-09-01", tz="UTC") + pd.Timedelta(minutes=minute)
        assert df.loc[ts, "5_is_closed"], f"5_is_closed should be True at minute {minute}"


def test_is_closed_tf5_false_mid_candle():
    df = _build_wide_df(make_fixture())
    # minutes 0, 1, 2, 3 are mid-candle — should be False
    for minute in [0, 1, 2, 3]:
        ts = pd.Timestamp("2023-09-01", tz="UTC") + pd.Timedelta(minutes=minute)
        assert not df.loc[ts, "5_is_closed"], f"5_is_closed should be False at minute {minute}"


def test_is_closed_tf60_at_minute_59():
    df = _build_wide_df(make_fixture())
    # minute 59 of any hour should be True
    for hour in range(24):
        ts = pd.Timestamp("2023-09-01", tz="UTC") + pd.Timedelta(hours=hour, minutes=59)
        assert df.loc[ts, "60_is_closed"], f"60_is_closed should be True at hour={hour} minute=59"


def test_is_closed_tf240_rule():
    df = _build_wide_df(make_fixture())
    # True at minute==59 AND hour % 4 == 3  → hours 3, 7, 11, 15, 19, 23
    for hour in range(24):
        ts = pd.Timestamp("2023-09-01", tz="UTC") + pd.Timedelta(hours=hour, minutes=59)
        expected = (hour % 4 == 3)
        assert df.loc[ts, "240_is_closed"] == expected, (
            f"240_is_closed mismatch at hour={hour}: expected {expected}"
        )


def test_is_closed_tf1440_rule():
    df = _build_wide_df(make_fixture())
    # True only at minute==59 AND hour==23
    ts_true = pd.Timestamp("2023-09-01 23:59", tz="UTC")
    assert df.loc[ts_true, "1440_is_closed"], "1440_is_closed should be True at 23:59"

    # All other minute-59 rows (hours 0-22) should be False
    for hour in range(23):
        ts = pd.Timestamp("2023-09-01", tz="UTC") + pd.Timedelta(hours=hour, minutes=59)
        assert not df.loc[ts, "1440_is_closed"], (
            f"1440_is_closed should be False at hour={hour} minute=59"
        )


# ---------------------------------------------------------------------------
# open_index / open tests
# ---------------------------------------------------------------------------


def test_open_index_same_within_candle():
    df = _build_wide_df(make_fixture())
    # rows 09:00 – 09:04 all belong to the same 5-min candle
    start = pd.Timestamp("2023-09-01 09:00", tz="UTC")
    rows = df.loc[start : start + pd.Timedelta(minutes=4), "5_open_index"]
    assert rows.nunique() == 1, "5_open_index must be identical for rows 09:00–09:04"
    assert rows.iloc[0] == start


# ---------------------------------------------------------------------------
# expanding high / low / volume tests
# ---------------------------------------------------------------------------


def test_high_expands_within_candle():
    df = _build_wide_df(make_fixture())
    ts01 = pd.Timestamp("2023-09-01 09:01", tz="UTC")
    ts03 = pd.Timestamp("2023-09-01 09:03", tz="UTC")
    assert df.loc[ts03, "5_high"] >= df.loc[ts01, "5_high"], (
        "5_high at 09:03 must be >= 5_high at 09:01 (expanding max)"
    )


def test_close_is_1min_close():
    df = _build_wide_df(make_fixture())
    # For every TF, {tf}_close must equal the raw 1-min close
    for tf in CANDLES:
        pd.testing.assert_series_equal(
            df[f"{tf}_close"].rename("close"),
            df["close"].rename("close"),
            check_names=True,
            obj=f"{tf}_close vs close",
        )


def test_volume_accumulates():
    df = _build_wide_df(make_fixture())
    ts00 = pd.Timestamp("2023-09-01 09:00", tz="UTC")
    ts04 = pd.Timestamp("2023-09-01 09:04", tz="UTC")
    assert df.loc[ts04, "5_volume"] >= df.loc[ts00, "5_volume"], (
        "5_volume at 09:04 must be >= 5_volume at 09:00 (expanding sum)"
    )


# ---------------------------------------------------------------------------
# tf=1 identity tests
# ---------------------------------------------------------------------------


def test_tf1_columns_equal_raw():
    df = _build_wide_df(make_fixture())
    pd.testing.assert_series_equal(
        df["1_high"].rename("high"),
        df["high"].rename("high"),
        check_names=True,
        obj="1_high vs high",
    )
    pd.testing.assert_series_equal(
        df["1_close"].rename("close"),
        df["close"].rename("close"),
        check_names=True,
        obj="1_close vs close",
    )


# ---------------------------------------------------------------------------
# column existence test
# ---------------------------------------------------------------------------


def test_all_candles_have_columns():
    df = _build_wide_df(make_fixture())
    expected_suffixes = [
        "open_index",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "is_closed",
    ]
    for tf in CANDLES:
        for suffix in expected_suffixes:
            col = f"{tf}_{suffix}"
            assert col in df.columns, f"Missing column: {col}"


def test_buy_volume_accumulates():
    df = _build_wide_df(make_fixture())
    ts00 = pd.Timestamp("2023-09-01 09:00", tz="UTC")
    ts04 = pd.Timestamp("2023-09-01 09:04", tz="UTC")
    assert df.loc[ts04, "5_buy_volume"] >= df.loc[ts00, "5_buy_volume"], (
        "5_buy_volume at 09:04 must be >= 5_buy_volume at 09:00 (expanding sum)"
    )


def test_volume_resets_at_candle_boundary():
    df = _build_wide_df(make_fixture())
    ts_last = pd.Timestamp("2023-09-01 09:04", tz="UTC")
    ts_next = pd.Timestamp("2023-09-01 09:05", tz="UTC")
    assert df.loc[ts_next, "5_volume"] < df.loc[ts_last, "5_volume"], (
        "5_volume at 09:05 (first row of new candle) must be < 5_volume at 09:04 "
        "(last row of previous candle, confirming reset)"
    )
