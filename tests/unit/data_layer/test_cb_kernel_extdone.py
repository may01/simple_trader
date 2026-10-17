"""Tests for the CB kernel: per-minute extdone state and the filter flags (G6)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds.kernel import candle_progress, extdone, filters, ladder_threshold
from indicators.candle_bounds.params import ATR_COL, EXTDONE_LADDER

LADDER_15_HIGH = ((0.25, 5.0), (0.50, 2.5), (0.75, 1.5))


def _idx(*minutes: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(f"2025-01-01 {m}", tz="UTC") for m in minutes])


def _range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(f"2025-01-01 {start}", f"2025-01-01 {end}", freq="1min", tz="UTC")


def _frame(idx: pd.DatetimeIndex, tf: int, *, high=110.0, low=90.0, close=100.0,
           atr=1.0, closed_last: bool = True) -> pd.DataFrame:
    n = len(idx)
    closed = np.zeros(n, bool)
    if closed_last:
        closed[-1] = True
    return pd.DataFrame(
        {
            f"{tf}_high": np.broadcast_to(np.asarray(high, float), (n,)).copy(),
            f"{tf}_low": np.broadcast_to(np.asarray(low, float), (n,)).copy(),
            "1_close": np.broadcast_to(np.asarray(close, float), (n,)).copy(),
            ATR_COL: np.broadcast_to(np.asarray(atr, float), (n,)).copy(),
            f"{tf}_is_closed": closed,
        },
        index=idx,
    )


def _gap_frame() -> pd.DataFrame:
    idx = _range("10:00", "10:03").append(_range("10:08", "10:14"))
    return _frame(idx, 15)


# --------------------------------------------------------------------------- progress

def test_progress_wall_clock():
    df = _frame(_range("10:00", "10:14"), 15)
    np.testing.assert_allclose(candle_progress(df, 15), np.arange(1, 16) / 15)


def test_progress_wall_clock_with_gap():
    df = _gap_frame()
    p = pd.Series(candle_progress(df, 15), index=df.index)
    assert p[pd.Timestamp("2025-01-01 10:08", tz="UTC")] == pytest.approx(9 / 15)
    assert p.iloc[-1] == pytest.approx(1.0)


def test_progress_by_rows_with_gap():
    df = _gap_frame()
    p = pd.Series(candle_progress(df, 15, by_rows=True), index=df.index)
    assert p[pd.Timestamp("2025-01-01 10:08", tz="UTC")] == pytest.approx(5 / 15)


def test_progress_by_rows_candles_delimited_by_close():
    # two candles: closes at 10:14 and 10:29; the row after a close restarts at 1
    df = _frame(_range("10:00", "10:29"), 15, closed_last=False)
    df.loc[[pd.Timestamp("2025-01-01 10:14", tz="UTC"),
            pd.Timestamp("2025-01-01 10:29", tz="UTC")], "15_is_closed"] = True
    p = candle_progress(df, 15, by_rows=True)
    np.testing.assert_allclose(p, np.r_[np.arange(1, 16), np.arange(1, 16)] / 15)


def test_progress_wall_clock_240_alignment():
    df = _frame(_idx("04:00", "07:59"), 240)
    np.testing.assert_allclose(candle_progress(df, 240), [1 / 240, 1.0])


# --------------------------------------------------------------------------- ladder

def test_ladder_threshold_steps():
    p = np.array([0.2, 0.25, 0.49, 0.50, 0.75, 1.0])
    np.testing.assert_array_equal(
        ladder_threshold(p, LADDER_15_HIGH), [np.inf, 5.0, 5.0, 2.5, 1.5, 1.5])


def test_ladder_single_step_240():
    np.testing.assert_array_equal(
        ladder_threshold(np.array([0.74, 0.75]), ((0.75, 8.0),)), [np.inf, 8.0])


# --------------------------------------------------------------------------- extdone

def _at(minutes: list[str], tf: int = 15, **kw) -> pd.DataFrame:
    """Rows of one tf-15 candle opened at 10:00 (no closing row needed)."""
    return _frame(_idx(*minutes), tf, closed_last=False, **kw)


def test_extdone_true_at_threshold():
    # 10:08 -> progress 9/15 = 0.6 -> d = 2.5; ATR 1, 15_high 110
    df = _at(["10:08", "10:08"], close=[107.5, 107.6])
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert ext["15_extdone_high"].tolist() == [True, False]


def test_extdone_not_sticky():
    df = _at(["10:08", "10:09"], close=[107.0, 109.0])  # pull 3.0, then 1.0; d = 2.5
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert ext["15_extdone_high"].tolist() == [True, False]


def test_extdone_can_reactivate():
    df = _at(["10:08", "10:09", "10:10"], close=[107.0, 109.0, 107.0])
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert ext["15_extdone_high"].tolist() == [True, False, True]


def test_extdone_never_before_first_step():
    # 10:02 -> progress 3/15 = 0.2 < 0.25
    df = _at(["10:00", "10:02"], close=10.0, high=110.0, low=-1000.0)
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert ext["15_extdone_high"].tolist() == [False, False]
    assert ext["15_extdone_low"].tolist() == [False, False]


def test_extdone_low_mirrored():
    # low ladder for tf 15 at progress 0.6: d = 2.5; pull = (close - 15_low) / atr
    df = _at(["10:08", "10:09"], high=93.0, low=90.0, close=[92.5, 92.4], atr=1.0)
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert ext["15_extdone_low"].tolist() == [True, False]
    assert ext["15_extdone_high"].tolist() == [False, False]


def test_extdone_per_side_ladder():
    # tf 60 at progress 0.6 (minute 36 of the candle): high d = 8.0, low d = 6.0
    df = _frame(_idx("10:35"), 60, high=107.0, low=93.0, close=100.0, closed_last=False)
    assert candle_progress(df, 60)[0] == pytest.approx(0.6)
    ext = extdone(df, 60, EXTDONE_LADDER[60])
    assert ext["60_extdone_low"].tolist() == [True]
    assert ext["60_extdone_high"].tolist() == [False]


def test_extdone_bad_atr_false():
    df = _at(["10:08", "10:09", "10:10", "10:11"], close=90.0, high=110.0, low=80.0,
             atr=[np.nan, 0.0, np.inf, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert not ext["15_extdone_high"].any()
    assert not ext["15_extdone_low"].any()
    assert ext["15_extdone_high_lvl"].isna().all()


def test_lvl_only_when_true():
    df = _at(["10:08", "10:09", "10:10"], close=[107.0, 109.0, 107.0], high=[110.0, 110.5, 111.0],
             low=108.0)
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    hi = ext["15_extdone_high"].to_numpy()
    lvl = ext["15_extdone_high_lvl"]
    np.testing.assert_array_equal(lvl[hi].to_numpy(), df["15_high"][hi].to_numpy())
    assert lvl[~hi].isna().all()
    assert ext["15_extdone_low_lvl"].isna().all()


def test_extdone_warmup_nan_extreme_false():
    df = _at(["10:08", "10:09"], high=np.nan, low=np.nan, close=100.0)
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert not ext["15_extdone_high"].any() and not ext["15_extdone_low"].any()


def test_extdone_columns_and_dtypes():
    df = _at(["10:08"])
    ext = extdone(df, 15, EXTDONE_LADDER[15])
    assert list(ext.columns) == ["15_extdone_high", "15_extdone_low",
                                 "15_extdone_high_lvl", "15_extdone_low_lvl"]
    assert ext["15_extdone_high"].dtype == bool and ext["15_extdone_low"].dtype == bool
    assert ext["15_extdone_high_lvl"].dtype == float
    assert ext.index.equals(df.index)


def test_extdone_by_rows_uses_row_count():
    # gap frame: 10:08 is row 5 of the candle -> progress 5/15 = 0.33 -> d = 5.0 (wall clock: 9/15 -> 2.5)
    df = _gap_frame()
    df["1_close"] = 107.0  # pull 3.0
    t = pd.Timestamp("2025-01-01 10:08", tz="UTC")
    assert bool(extdone(df, 15, EXTDONE_LADDER[15])["15_extdone_high"][t]) is True
    assert bool(extdone(df, 15, EXTDONE_LADDER[15], by_rows=True)["15_extdone_high"][t]) is False


# --------------------------------------------------------------------------- filters

def _class_frame(tf: int, trend: list, mcx: list) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 10:00", periods=len(trend), freq="1min", tz="UTC")
    return pd.DataFrame({f"{tf}_ema_25_slope_class": trend, f"{tf}_move_class": mcx}, index=idx)


def _ext(index: pd.Index, tf: int, high: list, low: list) -> pd.DataFrame:
    return pd.DataFrame({f"{tf}_extdone_high": np.array(high, bool),
                         f"{tf}_extdone_low": np.array(low, bool)}, index=index)


def test_filter_extdone_is_negation():
    df = _class_frame(15, [0] * 4, [0] * 4)
    ext = _ext(df.index, 15, [True, False, True, False], [True, True, False, False])
    f = filters(df, ext, 15)
    np.testing.assert_array_equal(f["15_cb_f_extdone_long"], ~ext["15_extdone_high"].to_numpy())
    np.testing.assert_array_equal(f["15_cb_f_extdone_short"], ~ext["15_extdone_low"].to_numpy())


def test_filter_extdone_warmup_passes():
    # warm-up rows: NaN extreme/ATR -> extdone False -> filter True ("nothing blocks")
    df = _at(["10:00", "10:08"], high=np.nan, low=np.nan, atr=np.nan)
    df["15_ema_25_slope_class"] = np.nan
    df["15_move_class"] = np.nan
    f = filters(df, extdone(df, 15, EXTDONE_LADDER[15]), 15)
    assert f["15_cb_f_extdone_long"].all() and f["15_cb_f_extdone_short"].all()


def test_filter_trend_truth_table():
    df = _class_frame(15, [-1, 0, 1, np.nan], [0] * 4)
    f = filters(df, _ext(df.index, 15, [False] * 4, [False] * 4), 15)
    assert f["15_cb_f_trend_long"].tolist() == [False, True, True, True]
    assert f["15_cb_f_trend_short"].tolist() == [True, True, False, True]


def test_filter_mcx_truth_table():
    mcx = [-3, 3, -2, -1, 0, 1, 2, np.nan]
    df = _class_frame(60, [0] * len(mcx), mcx)
    f = filters(df, _ext(df.index, 60, [False] * len(mcx), [False] * len(mcx)), 60)
    assert f["60_cb_f_mcx_long"].tolist() == [True, False, False, False, False, False, False, False]
    assert f["60_cb_f_mcx_short"].tolist() == [False, True, False, False, False, False, False, False]


def test_filter_dtypes_bool():
    df = _class_frame(240, [-1, np.nan, 1], [-3, np.nan, 3])
    f = filters(df, _ext(df.index, 240, [True, False, False], [False, False, True]), 240)
    assert list(f.columns) == [
        "240_cb_f_extdone_long", "240_cb_f_extdone_short",
        "240_cb_f_trend_long", "240_cb_f_trend_short",
        "240_cb_f_mcx_long", "240_cb_f_mcx_short",
    ]
    assert all(f[c].dtype == bool for c in f.columns)
    assert f.index.equals(df.index)
