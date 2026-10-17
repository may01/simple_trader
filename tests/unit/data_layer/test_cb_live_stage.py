"""Tests for the live CB stage ``LiveCbStage`` and its wiring into ``data.LiveData``.

``ohlc_from_wide`` rebuilds, from a ``cb_fixtures.make_wide`` frame, the per-tf
frames live holds at minute ``t`` (closed candles at their open time + the
forming candle as the last row), so every live value can be compared with the
prep row ``compute_cb(wide, models).columns.loc[t]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import Indicators
from indicators.candle_bounds import stage
from indicators.candle_bounds.params import TFS
from indicators.candle_bounds.stage import LiveCbStage, compute_cb, required_columns
from tests.unit.data_layer.cb_fixtures import _raw_ohlc, make_models, make_wide

ONE_MIN = pd.Timedelta(minutes=1)


# --------------------------------------------------------------------------- helpers

def ohlc_from_wide(wide: pd.DataFrame, t: pd.Timestamp) -> dict[int, pd.DataFrame]:
    upto = wide.loc[:t]
    ohlc: dict[int, pd.DataFrame] = {
        1: upto[[c for c in wide.columns if c.startswith("1_")]].tail(120).copy()
    }
    for tf in TFS:
        cols = [c for c in wide.columns if c.startswith(f"{tf}_")]
        start = t.floor(f"{tf}min")
        mask = (upto[f"{tf}_is_closed"] == True) & (upto.index < start)  # noqa: E712
        closed = upto.loc[mask, cols].copy()
        closed.index = closed.index - pd.Timedelta(minutes=tf - 1)
        forming = upto.loc[[t], cols].copy()
        forming.index = pd.DatetimeIndex([start])
        frame = pd.concat([closed, forming]).tail(120)
        frame[f"{tf}_is_closed"] = True          # live flags every row closed
        ohlc[tf] = frame
    return ohlc


def _tf_blank(ohlc: dict[int, pd.DataFrame], tf: int) -> None:
    """Every CB column of ``tf`` (bools included): last row NaN."""
    for name in stage._tf_names(tf):
        v = ohlc[tf][name].iloc[-1]
        assert np.isnan(v), (name, v)


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return make_wide(6000)


@pytest.fixture(scope="module")
def models():
    return make_models()


@pytest.fixture(scope="module")
def prep(wide, models) -> pd.DataFrame:
    return compute_cb(wide, models).columns


@pytest.fixture
def live_stage(monkeypatch, models) -> LiveCbStage:
    monkeypatch.setattr(stage, "load_models", lambda pair: models)
    return LiveCbStage("link_usdt")


@pytest.fixture(scope="module")
def sample_minutes(wide) -> list[pd.Timestamp]:
    rng = np.random.default_rng(7)
    idx = wide.index
    picks = set(rng.choice(np.arange(4000, len(idx)), 24, replace=False).tolist())
    # include candle-closing minutes of every tf and a candle-opening minute
    forced = [i for i in range(4000, len(idx)) if idx[i].minute % 15 == 14][:2]
    forced += [i for i in range(4000, len(idx))
               if idx[i].minute == 59 and idx[i].hour % 4 == 3][:2]
    forced += [i for i in range(4000, len(idx)) if idx[i].minute == 0][:2]
    return [idx[i] for i in sorted(picks | set(forced))]


# --------------------------------------------------------------------------- build_wide

def _first_index(wide, cond, start=5000) -> int:
    return next(i for i in range(start, len(wide)) if cond(wide.index[i]))


def test_build_wide_rows_and_flags(wide):
    i240 = _first_index(wide, lambda t: t.minute == 59 and t.hour % 4 == 3)
    i60 = _first_index(wide, lambda t: t.minute == 59 and t.hour % 4 != 3)
    for i in (5014, 5020, i60, i240):                   # 5014 % 15 == 14
        t = wide.index[i]
        ohlc = ohlc_from_wide(wide, t)
        w = LiveCbStage.build_wide(ohlc)
        assert w.index[-1] == t
        assert w.index.is_monotonic_increasing and w.index.is_unique
        for tf in TFS:
            frame = ohlc[tf]
            closing = frame.index[:-1] + pd.Timedelta(minutes=tf - 1)
            assert (w.loc[closing, f"{tf}_is_closed"] == True).all()  # noqa: E712
            np.testing.assert_allclose(w.loc[closing, f"{tf}_high"].to_numpy(dtype=float),
                                       frame[f"{tf}_high"].iloc[:-1].to_numpy(dtype=float))
            assert w[f"{tf}_high"].iloc[-1] == frame[f"{tf}_high"].iloc[-1]
        assert bool(w["15_is_closed"].iloc[-1]) == (t.minute % 15 == 14)
        assert bool(w["60_is_closed"].iloc[-1]) == (t.minute == 59)
        assert bool(w["240_is_closed"].iloc[-1]) == (t.minute == 59 and t.hour % 4 == 3)
        assert w["1_low"].iloc[-1] == ohlc[1]["1_low"].iloc[-1]


def test_build_wide_has_required_columns(wide):
    w = LiveCbStage.build_wide(ohlc_from_wide(wide, wide.index[5000]))
    for c in required_columns():
        assert c in w.columns, c


# --------------------------------------------------------------------------- parity with prep

def test_closed_bound_matches_prep(wide, prep, live_stage, sample_minutes):
    for t in sample_minutes:
        ohlc = ohlc_from_wide(wide, t)
        live_stage.update(ohlc)
        for tf, col in ((15, "15_cb_high"), (60, "60_cb_low"), (240, "240_cb_zone_long")):
            exp = prep[col].loc[t]
            assert np.isfinite(exp), (t, col)
            np.testing.assert_allclose(ohlc[tf][col].iloc[-1], exp, rtol=1e-9, err_msg=f"{t} {col}")


def test_marker_extdone_filters_match_prep(wide, prep, live_stage, sample_minutes):
    names = ("cb_inzone_long", "extdone_high", "cb_f_extdone_long", "cb_f_trend_short",
             "cb_f_mcx_long")
    for t in sample_minutes:
        ohlc = ohlc_from_wide(wide, t)
        live_stage.update(ohlc)
        for tf in TFS:
            for n in names:
                col = f"{tf}_{n}"
                assert ohlc[tf][col].iloc[-1] == float(bool(prep[col].loc[t])), (t, col)


def test_overshoot_matches_prep(wide, prep, live_stage, sample_minutes):
    """Phase 19: z and pass flag per tf/side equal compute_cb's row (NaN where prep is NaN)."""
    for t in sample_minutes:
        ohlc = ohlc_from_wide(wide, t)
        live_stage.update(ohlc)
        for tf in TFS:
            for side in ("long", "short"):
                zc, fc = f"{tf}_cb_overshoot_{side}", f"{tf}_cb_f_overshoot_{side}"
                exp_z, got_z = prep[zc].loc[t], ohlc[tf][zc].iloc[-1]
                if np.isnan(exp_z):
                    assert np.isnan(got_z), (t, zc)
                else:
                    np.testing.assert_allclose(got_z, exp_z, rtol=1e-9, err_msg=f"{t} {zc}")
                assert ohlc[tf][fc].iloc[-1] == float(bool(prep[fc].loc[t])), (t, fc)


def test_overshoot_unusable_tf_is_nan(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    ohlc[60] = ohlc[60].iloc[:-1]        # stale -> unusable
    live_stage.update(ohlc)
    for n in ("cb_overshoot_long", "cb_overshoot_short", "cb_f_overshoot_long", "cb_f_overshoot_short"):
        assert np.isnan(ohlc[60][f"60_{n}"].iloc[-1]), n
    assert ohlc[15]["15_cb_f_overshoot_long"].iloc[-1] in (0.0, 1.0)


# --------------------------------------------------------------------------- forming bound hold

def test_cbnc_first_minute_nan(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    live_stage.update(ohlc)
    assert np.isnan(ohlc[15]["15_cbnc_high"].iloc[-1])
    assert ohlc[15]["15_cbnc_inzone_long"].iloc[-1] == 0.0
    assert ohlc[15]["15_cbx_inzone_long"].iloc[-1] == 0.0


def test_cbnc_held_from_previous_minute_matches_prep(wide, prep, live_stage):
    i0 = 5000
    live_stage.update(ohlc_from_wide(wide, wide.index[i0 - 1]))
    for i in range(i0, i0 + 20):
        t = wide.index[i]
        ohlc = ohlc_from_wide(wide, t)
        live_stage.update(ohlc)
        for tf in TFS:
            for col in (f"{tf}_cbnc_high", f"{tf}_cbnc_low", f"{tf}_cbnc_high_up"):
                exp = prep[col].loc[t]
                assert np.isfinite(exp), (t, col)
                np.testing.assert_allclose(ohlc[tf][col].iloc[-1], exp, rtol=1e-9,
                                           err_msg=f"{t} {col}")


def test_cbnc_stable_within_minute(wide, live_stage):
    i = 5003
    live_stage.update(ohlc_from_wide(wide, wide.index[i - 1]))
    t = wide.index[i]
    seen = []
    for bump in (1.0, 1.002, 0.997):
        ohlc = ohlc_from_wide(wide, t)
        for tf in (1, *TFS):
            ohlc[tf].loc[ohlc[tf].index[-1], f"{tf}_close"] *= bump
        live_stage.update(ohlc)
        seen.append(ohlc[15]["15_cbnc_high"].iloc[-1])
    assert np.isfinite(seen[0])
    assert seen[0] == seen[1] == seen[2]


def test_gap_longer_than_one_minute_resets_held(wide, live_stage):
    i = 5000
    live_stage.update(ohlc_from_wide(wide, wide.index[i]))
    ohlc = ohlc_from_wide(wide, wide.index[i + 5])
    live_stage.update(ohlc)
    assert np.isnan(ohlc[15]["15_cbnc_high"].iloc[-1])
    assert np.isnan(ohlc[60]["60_cbnc_low"].iloc[-1])
    ohlc = ohlc_from_wide(wide, wide.index[i + 6])
    live_stage.update(ohlc)
    assert np.isfinite(ohlc[15]["15_cbnc_high"].iloc[-1])
    assert np.isfinite(ohlc[60]["60_cbnc_low"].iloc[-1])


def test_held_survives_stale_last_tick_of_closing_minute(wide, prep, live_stage):
    """F1 (i): the 15-min frame already shows the new candle on the last tick
    of the closing minute (fetch skew) -> next minute still holds the earlier
    same-minute prediction."""
    i = _first_index(wide, lambda t: t.minute % 15 == 14)
    t_prev, t = wide.index[i], wide.index[i + 1]
    live_stage.update(ohlc_from_wide(wide, wide.index[i - 1]))
    good = ohlc_from_wide(wide, t_prev)
    live_stage.update(good)
    expected = compute_cb(LiveCbStage.build_wide(good), live_stage._models).nc_pred_last[(15, "high")][0]
    skewed = ohlc_from_wide(wide, t_prev)
    skewed[15] = ohlc_from_wide(wide, t)[15]           # one candle ahead -> stale
    live_stage.update(skewed)
    _tf_blank(skewed, 15)
    ohlc = ohlc_from_wide(wide, t)
    live_stage.update(ohlc)
    got = ohlc[15]["15_cbnc_high"].iloc[-1]
    assert np.isfinite(got)
    assert got == expected
    np.testing.assert_allclose(got, prep["15_cbnc_high"].loc[t], rtol=1e-9)


def test_held_nan_when_previous_minute_all_unusable(wide, live_stage):
    """F1 (ii): t-2 good, every tick of t-1 without a usable 15 frame -> NaN at
    t (no stale t-2 prediction); other tfs keep their held value."""
    i = 5005
    live_stage.update(ohlc_from_wide(wide, wide.index[i - 2]))
    for _ in range(2):
        ohlc = ohlc_from_wide(wide, wide.index[i - 1])
        del ohlc[15]
        live_stage.update(ohlc)
    ohlc = ohlc_from_wide(wide, wide.index[i])
    live_stage.update(ohlc)
    assert np.isnan(ohlc[15]["15_cbnc_high"].iloc[-1])
    assert np.isnan(ohlc[15]["15_cbnc_low"].iloc[-1])
    assert np.isfinite(ohlc[60]["60_cbnc_high"].iloc[-1])


def test_held_is_latest_finite_tick_of_previous_minute(wide, live_stage):
    """F1 (iii): three ticks at t-1 with different closes -> t holds the 3rd."""
    i = 5007
    preds = []
    for bump in (1.0, 1.002, 0.997):
        ohlc = ohlc_from_wide(wide, wide.index[i - 1])
        for tf in (1, *TFS):
            ohlc[tf].loc[ohlc[tf].index[-1], f"{tf}_close"] *= bump
        preds.append(compute_cb(LiveCbStage.build_wide(ohlc), live_stage._models)
                     .nc_pred_last[(15, "high")][0])
        live_stage.update(ohlc)
    assert preds[0] != preds[2]
    ohlc = ohlc_from_wide(wide, wide.index[i])
    live_stage.update(ohlc)
    assert ohlc[15]["15_cbnc_high"].iloc[-1] == preds[2]


def test_clock_backwards_resets_held(wide, live_stage):
    """F1 (iv)."""
    live_stage.update(ohlc_from_wide(wide, wide.index[5010]))
    live_stage.update(ohlc_from_wide(wide, wide.index[5011]))
    ohlc = ohlc_from_wide(wide, wide.index[5008])
    live_stage.update(ohlc)
    assert np.isnan(ohlc[15]["15_cbnc_high"].iloc[-1])
    assert np.isfinite(ohlc[15]["15_cb_high"].iloc[-1])


def test_update_equals_compute_cb_on_build_wide(wide, live_stage):
    """The cached per-row path gives exactly ``compute_cb(build_wide)``'s last
    row, across candle changes of every tf and degraded frames."""
    i240 = _first_index(wide, lambda t: t.minute == 59 and t.hour % 4 == 3)
    for i in range(i240 - 3, i240 + 4):
        ohlc = ohlc_from_wide(wide, wide.index[i])
        if i == i240 + 2:
            ohlc[60] = ohlc[60].iloc[:-1]
        live_stage.update(ohlc)
        exp = compute_cb(LiveCbStage.build_wide(ohlc), live_stage._models,
                         nc_pred_prev=live_stage._held).columns.iloc[-1]
        for tf in TFS:
            if tf == 60 and i == i240 + 2:
                _tf_blank(ohlc, 60)
                continue
            for name in stage._tf_names(tf):
                np.testing.assert_allclose(ohlc[tf][name].iloc[-1], float(exp[name]),
                                           rtol=1e-12, equal_nan=True, err_msg=f"{i} {name}")


# --------------------------------------------------------------------------- degraded inputs

def test_disabled_warns_once(monkeypatch):
    warned = []
    monkeypatch.setattr(stage, "load_models", lambda pair: None)
    monkeypatch.setattr(stage, "_WARNED_PAIRS", set())
    monkeypatch.setattr(stage, "log_warning", lambda msg: warned.append(msg))
    LiveCbStage("nope_usdt")
    LiveCbStage("nope_usdt")
    assert len(warned) == 1 and "nope_usdt" in warned[0]


def test_missing_one_min_input_warns_once(monkeypatch, wide, live_stage):
    warned = []
    monkeypatch.setattr(stage, "log_warning", lambda msg: warned.append(msg))
    for i in (5000, 5001):
        ohlc = ohlc_from_wide(wide, wide.index[i])
        ohlc[1] = ohlc[1].drop(columns=["1_atr_14_ma_5"])
        live_stage.update(ohlc)
    assert ohlc[15]["15_extdone_high"].iloc[-1] == 0.0
    assert len(warned) == 1 and "1_atr_14_ma_5" in warned[0]



def test_disabled_without_model(monkeypatch, wide):
    monkeypatch.setattr(stage, "load_models", lambda pair: None)
    s = LiveCbStage("nope_usdt")
    assert s.enabled is False
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    before = {tf: list(df.columns) for tf, df in ohlc.items()}
    s.update(ohlc)
    assert {tf: list(df.columns) for tf, df in ohlc.items()} == before


def test_short_history_tf_is_nan_others_ok(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    ohlc[240] = ohlc[240].tail(5)
    live_stage.update(ohlc)
    _tf_blank(ohlc, 240)
    assert np.isfinite(ohlc[15]["15_cb_high"].iloc[-1])
    assert np.isfinite(ohlc[60]["60_cb_high"].iloc[-1])


def test_stale_tf_frame_is_nan(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    ohlc[60] = ohlc[60].iloc[:-1]
    live_stage.update(ohlc)
    _tf_blank(ohlc, 60)
    assert np.isfinite(ohlc[15]["15_cb_high"].iloc[-1])
    assert np.isfinite(ohlc[240]["240_cb_high"].iloc[-1])


def test_missing_required_column_tf_is_nan(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    ohlc[15] = ohlc[15].drop(columns=["15_rsi_ma8"])
    live_stage.update(ohlc)
    _tf_blank(ohlc, 15)
    assert np.isfinite(ohlc[60]["60_cb_high"].iloc[-1])


def test_missing_tf_frame(monkeypatch, wide, live_stage):
    warned = []
    monkeypatch.setattr(stage, "log_warning", lambda msg: warned.append(msg))
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    del ohlc[240]
    live_stage.update(ohlc)
    assert 240 not in ohlc
    assert np.isfinite(ohlc[15]["15_cb_high"].iloc[-1])
    assert np.isfinite(ohlc[60]["60_cb_high"].iloc[-1])
    assert len(warned) == 1 and "240" in warned[0] and "missing" in warned[0]


def test_unusable_tf_bools_are_nan_not_zero(wide, live_stage):
    """0.0 is a value (published as a change); an unusable tf has none."""
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    ohlc[60] = ohlc[60].iloc[:-1]
    live_stage.update(ohlc)
    for n in ("cb_inzone_long", "extdone_high", "cb_f_trend_short", "cbx_inzone_long"):
        assert np.isnan(ohlc[60][f"60_{n}"].iloc[-1]), n
    assert ohlc[15]["15_cb_inzone_long"].iloc[-1] in (0.0, 1.0)


def test_stale_tf_warns_once(monkeypatch, wide, live_stage):
    warned = []
    monkeypatch.setattr(stage, "log_warning", lambda msg: warned.append(msg))
    for i in range(5000, 5005):
        ohlc = ohlc_from_wide(wide, wide.index[i])
        ohlc[60] = ohlc[60].iloc[:-1]
        live_stage.update(ohlc)
    assert len(warned) == 1 and "60" in warned[0] and "stale" in warned[0]
    ohlc = ohlc_from_wide(wide, wide.index[5006])
    ohlc[240] = ohlc[240].tail(5)
    live_stage.update(ohlc)
    assert len(warned) == 2 and "240" in warned[1]


def test_update_writes_nothing_on_mid_loop_exception(monkeypatch, wide, live_stage):
    """All tfs are computed before any is written: no partially written tick."""
    real = stage.tf_columns

    def flaky(df, models, tf, cb, **kw):
        if tf == 240:
            raise RuntimeError("boom")
        return real(df, models, tf, cb, **kw)

    monkeypatch.setattr(stage, "tf_columns", flaky)
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    with pytest.raises(RuntimeError):
        live_stage.update(ohlc)
    for tf in TFS:
        assert "{}_cb_high".format(tf) not in ohlc[tf].columns, tf


def test_writes_only_last_row(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    live_stage.update(ohlc)
    for tf in TFS:
        for name in stage._tf_names(tf):
            assert ohlc[tf][name].iloc[:-1].isna().all(), name
    assert ohlc[15]["15_cb_inzone_long"].iloc[-1] in (0.0, 1.0)


def test_does_not_touch_live_inputs(wide, live_stage):
    ohlc = ohlc_from_wide(wide, wide.index[5000])
    before = {tf: df.copy() for tf, df in ohlc.items()}
    live_stage.update(ohlc)
    for tf, df in before.items():
        pd.testing.assert_frame_equal(ohlc[tf][df.columns], df)


# --------------------------------------------------------------------------- LiveData wiring

class _ConsistentStock:
    """Stock mock whose per-tf candles all end at the same minute (as live)."""

    def __init__(self, n_minutes: int = 240 * 121 + 97) -> None:
        self._raw = _raw_ohlc(n_minutes, seed=3, start="2024-01-01 00:00")

    def get_candles_history(self, time_list, coin, time_point=0):
        from stocks.base_stock import StockInterface
        out = {}
        for tf in (1, 15, 60, 240):
            if tf not in time_list:
                continue
            df = self._raw if tf == 1 else StockInterface._resample_to_tf(None, self._raw, 1, tf)
            out[tf] = df.dropna(how="all").tail(120).copy()
        return out


@pytest.fixture
def consistent_stock(monkeypatch):
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("DATA_ROOT", "data")
    monkeypatch.setenv("DATA_SET_NAME", "test")
    Indicators._registry = None
    monkeypatch.setattr("data.stock_holder.item", _ConsistentStock())
    yield
    Indicators._registry = None


def test_live_data_survives_stage_exception(monkeypatch, consistent_stock):
    import data

    def boom(self, ohlc):
        raise RuntimeError("cb boom")

    logged = []
    monkeypatch.setattr(LiveCbStage, "update", boom)
    monkeypatch.setattr(data, "log_error", lambda msg: logged.append(msg))
    live = data.LiveData()
    live.build_candles()
    live.build_candles()
    dp = live.get_data_point()
    assert np.isfinite(dp.get("close", 15))
    assert "15_cb_high" not in live.ohlc[15].columns
    assert len(logged) == 1 and "RuntimeError" in logged[0]


def test_live_data_point_get_returns_cb_value(consistent_stock):
    import data
    live = data.LiveData()
    live.build_candles()
    dp = live.get_data_point()
    for tf in TFS:
        v = dp.get("cb_zone_long", tf)
        assert isinstance(v, float)
        assert np.isfinite(v), tf
        assert dp.get("cb_inzone_long", tf) in (0.0, 1.0)
    assert np.isfinite(dp.get("rsi_14", 15))


@pytest.mark.parametrize("env_pair", [None, ""])
def test_live_data_without_pair_builds_no_stage(monkeypatch, env_pair):
    import data
    if env_pair is None:
        monkeypatch.delenv("PAIR", raising=False)
    else:
        monkeypatch.setenv("PAIR", env_pair)
    built = []
    monkeypatch.setattr(data, "LiveCbStage", lambda pair: built.append(pair))
    live = data.LiveData()
    assert live._cb_stage is None and built == []


def test_live_data_with_pair_builds_stage(monkeypatch):
    import data
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setattr(data, "LiveCbStage", lambda pair: ("stage", pair))
    assert data.LiveData()._cb_stage == ("stage", "link_usdt")
