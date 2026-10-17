"""Tests for the CB kernel: closed bounds, closed zone, raw marker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds.kernel import closed_bounds, closed_frame, closed_zones, to_rows
from indicators.candle_bounds.params import FEATURES
from tests.unit.data_layer.cb_fixtures import make_models, make_wide

TF = 15


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return make_wide(6000)


@pytest.fixture(scope="module")
def models():
    return make_models()


@pytest.fixture(scope="module")
def cb_high(wide, models) -> pd.DataFrame:
    return closed_bounds(wide, models[(TF, "high")])


def _closing_ts(wide: pd.DataFrame, tf: int) -> pd.DatetimeIndex:
    return wide.index[(wide[f"{tf}_is_closed"] == True).to_numpy()]  # noqa: E712


# --------------------------------------------------------------------------- to_rows

def test_to_rows_strictly_before():
    idx = pd.date_range("2025-01-01 00:00", "2025-01-01 00:44", freq="1min", tz="UTC")
    s = pd.Series([1.0, 2.0], index=[idx[14], idx[29]])
    out = to_rows(s, idx)
    assert out.index.equals(idx)
    assert out.iloc[0:15].isna().all()          # 00:00–00:14, closing row itself NaN
    assert (out.iloc[15:30] == 1.0).all()       # 00:15–00:29
    assert (out.iloc[30:45] == 2.0).all()       # 00:30–00:44


# --------------------------------------------------------------------------- closed_frame

def _hand_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01 00:00", periods=60, freq="1min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["15_is_closed"] = pd.Series([None] * 60, index=idx, dtype=object)
    df.loc[idx.minute % 15 == 14, "15_is_closed"] = True
    highs = [100.0, 102.0, 101.0, 103.5]
    lows = [98.0, 99.0, 99.5, 100.0]
    opens = [99.0, 100.0, 101.0, 100.5]
    closes = [99.5, 101.5, 100.0, 103.0]
    emas = [99.0, 100.0, 100.5, 101.0]
    for k in range(4):
        sl = slice(15 * k, 15 * k + 15)
        df.loc[idx[sl], "15_high"] = highs[k] - 0.5  # running extreme, below final
        df.loc[idx[sl], "15_low"] = lows[k] + 0.5
        df.loc[idx[15 * k + 14], "15_high"] = highs[k]
        df.loc[idx[15 * k + 14], "15_low"] = lows[k]
        df.loc[idx[sl], "15_open"] = opens[k]
        df.loc[idx[sl], "15_close"] = 0.0
        df.loc[idx[15 * k + 14], "15_close"] = closes[k]
        df.loc[idx[sl], "15_ema_25"] = emas[k]
    for c in ("15_rsi_14", "15_rsi_ma8", "15_macd_12_26_9", "15_macd_hist_12_26_9"):
        df[c] = 50.0
    return df


def test_closed_frame_features_by_hand():
    df = _hand_frame()
    cf = closed_frame(df, 15, "high")
    assert list(cf.columns) == FEATURES + ["d", "ext"]
    assert len(cf) == 4
    assert cf.index.equals(df.index[df.index.minute % 15 == 14])
    highs = np.array([100.0, 102.0, 101.0, 103.5])
    lows = np.array([98.0, 99.0, 99.5, 100.0])
    opens = np.array([99.0, 100.0, 101.0, 100.5])
    closes = np.array([99.5, 101.5, 100.0, 103.0])
    emas = np.array([99.0, 100.0, 100.5, 101.0])
    exp_d = [np.nan, 2.0, (101.0 / 102.0 - 1) * 100, (103.5 / 101.0 - 1) * 100]
    np.testing.assert_allclose(cf["d"].to_numpy(), exp_d, rtol=1e-12)
    np.testing.assert_allclose(cf["ext"].to_numpy(), highs)
    np.testing.assert_allclose(cf["V_range"].to_numpy(), (highs - lows) / lows * 100)
    np.testing.assert_allclose(cf["V_body"].to_numpy(), (closes - opens) / opens * 100)
    np.testing.assert_allclose(cf["A_position"].to_numpy(), (closes - emas) / closes * 100)
    np.testing.assert_allclose(cf["A_slope"].to_numpy(), [np.nan, 1.0, 0.5, 0.5])


# --------------------------------------------------------------------------- closed_bounds

def test_closed_bounds_columns(cb_high, wide):
    assert cb_high.index.equals(wide.index)
    assert list(cb_high.columns) == [
        f"{TF}_cb_high", f"{TF}_cb_high_std", f"{TF}_cb_high_up", f"{TF}_cb_high_dn"]


def test_closed_bounds_constant_within_candle(cb_high, wide):
    s = cb_high[f"{TF}_cb_high"]
    nun = s.groupby(wide[f"{TF}_open_index"]).nunique(dropna=True)
    cnt = s.groupby(wide[f"{TF}_open_index"]).count()
    # every candle: either all NaN (warm-up) or one unique value on all its rows
    assert (nun <= 1).all()
    assert ((cnt == 0) | (cnt == 15)).all()
    assert (nun == 1).sum() > 300


def test_closed_bounds_no_lookahead(wide, models):
    cut = 3000
    mod = wide.copy()
    scaled = [c for c in mod.columns
              if any(c == f"{tf}_{k}" for tf in (1, 5, 15, 60, 240, 1440)
                     for k in ("open", "high", "low", "close", "rsi_14", "rsi_ma8",
                               "macd_12_26_9", "macd_hist_12_26_9", "ema_25"))]
    assert f"{TF}_high" in scaled and f"{TF}_rsi_ma8" in scaled
    mod.iloc[cut:, [mod.columns.get_loc(c) for c in scaled]] *= 2.0
    for (tf, side), m in models.items():
        a = closed_bounds(wide, m).iloc[:cut]
        b = closed_bounds(mod, m).iloc[:cut]
        pd.testing.assert_frame_equal(a, b)
        if tf != 240:  # tf-240 warm-up (13 candles = 3120 min) ends after the cut
            assert a.iloc[:, 0].notna().any()
        # the scaled tail does change the outputs (the test is not vacuous)
        assert not closed_bounds(wide, m).iloc[cut:].equals(closed_bounds(mod, m).iloc[cut:])


def test_closed_bounds_value_formula(cb_high, wide, models):
    m = models[(TF, "high")]
    cf = closed_frame(wide, TF, "high")
    k = 200
    ts_k, ts_k1 = cf.index[k], cf.index[k + 1]
    row = cf.loc[ts_k]
    assert row[FEATURES].notna().all()
    pred = float(m.predict_pct(row[FEATURES].to_numpy(dtype=float)[None, :])[0])
    ext = float(row["ext"])
    rows = cb_high.loc[(cb_high.index > ts_k) & (cb_high.index <= ts_k1)]
    assert len(rows) == TF
    np.testing.assert_allclose(rows[f"{TF}_cb_high"], ext * (1 + pred / 100), rtol=1e-12)
    np.testing.assert_allclose(rows[f"{TF}_cb_high_std"], ext * m.band_pct / 100, rtol=1e-12)
    np.testing.assert_allclose(rows[f"{TF}_cb_high_up"] - rows[f"{TF}_cb_high_dn"],
                               2 * rows[f"{TF}_cb_high_std"], rtol=1e-9)


def test_closed_bounds_warmup_nan(cb_high, wide):
    cf = closed_frame(wide, TF, "high")
    first_valid = cf.dropna(subset=FEATURES).index[0]
    assert first_valid > cf.index[0]
    before = cb_high.loc[cb_high.index <= first_valid, f"{TF}_cb_high"]
    assert before.isna().all()
    after = cb_high.loc[cb_high.index > first_valid, f"{TF}_cb_high"]
    assert after.notna().all()


def _load_cbnc():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "candle_bounds_nc" / "cbnc.py"
    spec = importlib.util.spec_from_file_location("_cbnc_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_closed_bounds_match_reference(wide, models):
    """Port parity: identical to cbnc.infer_closed with the same linear model."""
    cbnc = _load_cbnc()

    class _Shim:
        def __init__(self, m):
            self.tf, self.side, self.band_pct, self._m = m.tf, m.side, m.band_pct, m

        def predict_pct(self, X):
            return self._m.predict_pct(X[FEATURES].to_numpy())

    for m in models.values():
        ref = cbnc.infer_closed(wide, _Shim(m))
        pd.testing.assert_frame_equal(closed_bounds(wide, m), ref, check_exact=True)


# --------------------------------------------------------------------------- closed_zones

def _zone_inputs(high, low, low1, high1):
    idx = pd.date_range("2025-01-01", periods=len(high), freq="1min", tz="UTC")
    cb = pd.DataFrame({"15_cb_high": high, "15_cb_low": low}, index=idx, dtype=float)
    df = pd.DataFrame({"1_low": low1, "1_high": high1}, index=idx, dtype=float)
    return cb, df


def test_closed_zone_formulas():
    cb, df = _zone_inputs([110.0], [100.0], [105.0], [105.0])
    z = closed_zones(cb, df, 15)
    assert z.loc[z.index[0], "15_cb_zone_long"] == pytest.approx(100.5)
    assert z.loc[z.index[0], "15_cb_zone_short"] == pytest.approx(109.5)
    assert list(z.columns) == ["15_cb_zone_long", "15_cb_zone_short",
                               "15_cb_inzone_long", "15_cb_inzone_short"]


def test_inzone_boundaries():
    zl = 100.0 + 0.05 * 10.0
    zs = 110.0 - 0.05 * 10.0
    tick = 1e-4
    cb, df = _zone_inputs([110.0, 110.0], [100.0, 100.0], [zl, zl + tick], [zs, zs - tick])
    z = closed_zones(cb, df, 15)
    assert z["15_cb_inzone_long"].tolist() == [True, False]
    assert z["15_cb_inzone_short"].tolist() == [True, False]


def test_inzone_nan_level_false():
    cb, df = _zone_inputs([np.nan, 110.0], [100.0, np.nan], [0.0, 0.0], [1e9, 1e9])
    z = closed_zones(cb, df, 15)
    assert z["15_cb_inzone_long"].dtype == bool
    assert z["15_cb_inzone_short"].dtype == bool
    assert not z["15_cb_inzone_long"].any()
    assert not z["15_cb_inzone_short"].any()


def test_dtypes(wide, models):
    cb = pd.concat([closed_bounds(wide, models[(TF, s)]) for s in ("high", "low")], axis=1)
    for c in cb.columns:
        assert cb[c].dtype == np.float64, c
    z = closed_zones(cb, wide, TF)
    assert z[f"{TF}_cb_zone_long"].dtype == np.float64
    assert z[f"{TF}_cb_zone_short"].dtype == np.float64
    assert z[f"{TF}_cb_inzone_long"].dtype == bool
    assert z[f"{TF}_cb_inzone_short"].dtype == bool
    assert z[f"{TF}_cb_inzone_long"].any() and z[f"{TF}_cb_inzone_short"].any()
