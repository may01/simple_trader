"""Tests for the CB kernel: per-minute extdone state, the overshoot z and the filter flags (G6; phase 19 G1/G2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds.kernel import candle_progress, extdone, filters, ladder_threshold, overshoot
from indicators.candle_bounds.params import ATR_COL, EXTDONE_LADDER, OVERSHOOT_Q, OVERSHOOT_THR

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


def _over(index: pd.Index, tf: int, long: list | None = None, short: list | None = None) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame({f"{tf}_cb_overshoot_long": np.array([np.nan] * n if long is None else long, float),
                         f"{tf}_cb_overshoot_short": np.array([np.nan] * n if short is None else short, float)},
                        index=index)


def test_filter_extdone_is_negation():
    df = _class_frame(15, [0] * 4, [0] * 4)
    ext = _ext(df.index, 15, [True, False, True, False], [True, True, False, False])
    f = filters(df, ext, _over(df.index, 15), 15)
    np.testing.assert_array_equal(f["15_cb_f_extdone_long"], ~ext["15_extdone_high"].to_numpy())
    np.testing.assert_array_equal(f["15_cb_f_extdone_short"], ~ext["15_extdone_low"].to_numpy())


def test_filter_extdone_warmup_passes():
    # warm-up rows: NaN extreme/ATR -> extdone False -> filter True ("nothing blocks")
    df = _at(["10:00", "10:08"], high=np.nan, low=np.nan, atr=np.nan)
    df["15_ema_25_slope_class"] = np.nan
    df["15_move_class"] = np.nan
    f = filters(df, extdone(df, 15, EXTDONE_LADDER[15]), _over(df.index, 15), 15)
    assert f["15_cb_f_extdone_long"].all() and f["15_cb_f_extdone_short"].all()


def test_filter_trend_truth_table():
    df = _class_frame(15, [-1, 0, 1, np.nan], [0] * 4)
    f = filters(df, _ext(df.index, 15, [False] * 4, [False] * 4), _over(df.index, 15), 15)
    assert f["15_cb_f_trend_long"].tolist() == [False, True, True, True]
    assert f["15_cb_f_trend_short"].tolist() == [True, True, False, True]


def test_filter_mcx_truth_table():
    mcx = [-3, 3, -2, -1, 0, 1, 2, np.nan]
    df = _class_frame(60, [0] * len(mcx), mcx)
    f = filters(df, _ext(df.index, 60, [False] * len(mcx), [False] * len(mcx)), _over(df.index, 60), 60)
    assert f["60_cb_f_mcx_long"].tolist() == [True, False, False, False, False, False, False, False]
    assert f["60_cb_f_mcx_short"].tolist() == [False, True, False, False, False, False, False, False]


def test_filter_dtypes_bool():
    df = _class_frame(240, [-1, np.nan, 1], [-3, np.nan, 3])
    f = filters(df, _ext(df.index, 240, [True, False, False], [False, False, True]), _over(df.index, 240), 240)
    assert list(f.columns) == [
        "240_cb_f_extdone_long", "240_cb_f_extdone_short",
        "240_cb_f_trend_long", "240_cb_f_trend_short",
        "240_cb_f_mcx_long", "240_cb_f_mcx_short",
        "240_cb_f_overshoot_long", "240_cb_f_overshoot_short",
    ]
    assert all(f[c].dtype == bool for c in f.columns)
    assert f.index.equals(df.index)


# --------------------------------------------------------------------------- overshoot (phase 19)

def _cb_frame(tf: int, *, cb_low=100.0, low_std=2.0, cb_high=200.0, high_std=4.0,
              one_low=99.0, one_high=201.0, n: int | None = None):
    """(cb, df) pair: closed bounds + the 1-min extremes, broadcast to ``n`` rows."""
    arrays = {k: np.atleast_1d(np.asarray(v, float)) for k, v in
              dict(cb_low=cb_low, low_std=low_std, cb_high=cb_high, high_std=high_std,
                   one_low=one_low, one_high=one_high).items()}
    n = n or max(len(a) for a in arrays.values())
    arrays = {k: np.broadcast_to(a, (n,)).copy() for k, a in arrays.items()}
    idx = pd.date_range("2025-01-01 10:00", periods=n, freq="1min", tz="UTC")
    cb = pd.DataFrame({f"{tf}_cb_low": arrays["cb_low"], f"{tf}_cb_low_std": arrays["low_std"],
                       f"{tf}_cb_high": arrays["cb_high"], f"{tf}_cb_high_std": arrays["high_std"]}, index=idx)
    df = pd.DataFrame({"1_low": arrays["one_low"], "1_high": arrays["one_high"]}, index=idx)
    return cb, df


def test_overshoot_z_long_exact():
    cb, df = _cb_frame(15, one_low=[99.0, 100.0, 101.0])
    z = overshoot(cb, df, 15)
    np.testing.assert_allclose(z["15_cb_overshoot_long"], [0.5, 0.0, -0.5], atol=1e-12)


def test_overshoot_z_short_exact():
    cb, df = _cb_frame(15, one_high=[201.0, 200.0, 199.0])
    z = overshoot(cb, df, 15)
    np.testing.assert_allclose(z["15_cb_overshoot_short"], [0.25, 0.0, -0.25], atol=1e-12)


@pytest.mark.parametrize("kw", [
    dict(low_std=np.nan, high_std=np.nan), dict(low_std=0.0, high_std=0.0),
    dict(low_std=-1.0, high_std=-1.0), dict(cb_low=np.nan, cb_high=np.nan),
    dict(one_low=np.nan, one_high=np.nan),
])
def test_overshoot_bad_input_is_nan_never_inf(kw):
    cb, df = _cb_frame(60, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z = overshoot(cb, df, 60)
    assert z["60_cb_overshoot_long"].isna().all() and z["60_cb_overshoot_short"].isna().all()
    assert not np.isinf(z.to_numpy()).any()


def test_overshoot_columns_and_dtype():
    cb, df = _cb_frame(240, n=3)
    z = overshoot(cb, df, 240)
    assert list(z.columns) == ["240_cb_overshoot_long", "240_cb_overshoot_short"]
    assert all(z[c].dtype == np.float64 for c in z.columns)
    assert z.index.equals(df.index)


@pytest.mark.parametrize("tf", [15, 60, 240])
@pytest.mark.parametrize("side", ["long", "short"])
def test_filter_overshoot_false_iff_above_threshold(tf, side):
    thr = OVERSHOOT_THR[tf][side]
    zs = [thr - 1e-6, thr, thr + 1e-6, np.nan]
    df = _class_frame(tf, [0] * 4, [0] * 4)
    over = _over(df.index, tf, **{side: zs})
    f = filters(df, _ext(df.index, tf, [False] * 4, [False] * 4), over, tf)
    assert f[f"{tf}_cb_f_overshoot_{side}"].tolist() == [True, True, False, True]
    other = "short" if side == "long" else "long"
    assert f[f"{tf}_cb_f_overshoot_{other}"].all()          # NaN z on the other side passes


def test_filter_overshoot_reads_params_at_call_time(monkeypatch):
    df = _class_frame(15, [0], [0])
    over = _over(df.index, 15, long=[0.2])
    monkeypatch.setitem(OVERSHOOT_THR[15], "long", 0.1)
    f = filters(df, _ext(df.index, 15, [False], [False]), over, 15)
    assert f["15_cb_f_overshoot_long"].tolist() == [False]


def test_filter_overshoot_leaves_old_flags_unchanged():
    df = _class_frame(15, [-1, 0, 1, np.nan], [-3, 3, 0, np.nan])
    ext = _ext(df.index, 15, [True, False, True, False], [False, True, False, True])
    f_nan = filters(df, ext, _over(df.index, 15), 15)
    f_hit = filters(df, ext, _over(df.index, 15, long=[9.0] * 4, short=[9.0] * 4), 15)
    old = [c for c in f_nan.columns if "overshoot" not in c]
    pd.testing.assert_frame_equal(f_nan[old], f_hit[old])
    assert f_nan["15_cb_f_trend_long"].tolist() == [False, True, True, True]


def test_overshoot_thresholds_frozen():
    assert OVERSHOOT_Q == 0.75
    assert OVERSHOOT_THR == {
        15: {"long": 0.3477, "short": 0.3808},
        60: {"long": 0.3756, "short": 0.3796},
        240: {"long": 0.4693, "short": 0.5178},
    }
