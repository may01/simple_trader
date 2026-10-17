"""Tests for the CB kernel: forming bounds, hybrid zone, cbx intersection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds.kernel import (
    closed_bounds,
    closed_frame,
    closed_zones,
    forming_bounds,
    forming_features,
    forming_prediction,
    forming_zones,
)
from indicators.candle_bounds.params import FEATURES, SIDES, TFS
from tests.unit.data_layer.cb_fixtures import make_models, make_wide

TF = 15


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return make_wide(6000)


@pytest.fixture(scope="module")
def models():
    return make_models()


def _closing_ts(wide: pd.DataFrame, tf: int) -> pd.DatetimeIndex:
    return wide.index[(wide[f"{tf}_is_closed"] == True).to_numpy()]  # noqa: E712


def _cbnc(df: pd.DataFrame, m) -> pd.DataFrame:
    return forming_bounds(forming_prediction(df, m), m.tf, m.side)


# --------------------------------------------------------------------------- forming_features

def test_forming_features_columns(wide):
    ff = forming_features(wide, TF, "high")
    assert ff.index.equals(wide.index)
    assert list(ff.columns) == FEATURES + ["ext_f"]


def test_forming_features_equal_closed_on_closing_row(wide):
    ff = forming_features(wide, TF, "high")
    cf = closed_frame(wide, TF, "high")
    ts = _closing_ts(wide, TF)
    a = ff.loc[ts, FEATURES]
    b = cf.loc[ts, FEATURES]
    both = a.notna() & b.notna()
    assert both.all(axis=1).sum() > 300
    for c in FEATURES:
        m = both[c].to_numpy()
        np.testing.assert_allclose(a[c].to_numpy()[m], b[c].to_numpy()[m], rtol=1e-9, err_msg=c)
    np.testing.assert_array_equal(ff.loc[ts, "ext_f"].to_numpy(), cf["ext"].to_numpy())


# --------------------------------------------------------------------------- forming_prediction / bounds

def test_forming_prediction_columns_and_nan(wide, models):
    m = models[(TF, "high")]
    pred = forming_prediction(wide, m)
    assert pred.index.equals(wide.index)
    assert list(pred.columns) == ["bound", "band"]
    ff = forming_features(wide, TF, "high")
    incomplete = ff[FEATURES].isna().any(axis=1)
    assert incomplete.any() and (~incomplete).any()
    assert pred.loc[incomplete, "bound"].isna().all()
    assert pred.loc[~incomplete, "bound"].notna().all()


def test_forming_prediction_converges_to_closed_bound(wide, models):
    """Experiment gate G3: on the first row of each candle the forming bound
    (prediction from the closing minute) equals the closed bound."""
    for tf in TFS:
        first_rows = wide.index[(wide[f"{tf}_is_closed"] == True).shift(1, fill_value=False)  # noqa: E712
                                .to_numpy()]
        for side in SIDES:
            m = models[(tf, side)]
            cbnc = _cbnc(wide, m).loc[first_rows]
            cb = closed_bounds(wide, m).loc[first_rows]
            for suf in ("", "_std", "_up", "_dn"):
                a = cbnc[f"{tf}_cbnc_{side}{suf}"].to_numpy()
                b = cb[f"{tf}_cb_{side}{suf}"].to_numpy()
                ok = ~np.isnan(a) & ~np.isnan(b)
                assert ok.sum() > 0, (tf, side, suf)
                np.testing.assert_allclose(a[ok], b[ok], rtol=1e-6, err_msg=f"{tf} {side}{suf}")
            np.testing.assert_array_equal(np.isnan(cbnc[f"{tf}_cbnc_{side}"].to_numpy()),
                                          np.isnan(cb[f"{tf}_cb_{side}"].to_numpy()))


def test_forming_bounds_is_prediction_shifted_one_row(wide, models):
    pred = forming_prediction(wide, models[(TF, "high")])
    fb = forming_bounds(pred, TF, "high")
    assert list(fb.columns) == [f"{TF}_cbnc_high", f"{TF}_cbnc_high_std",
                                f"{TF}_cbnc_high_up", f"{TF}_cbnc_high_dn"]
    s = fb[f"{TF}_cbnc_high"]
    assert np.isnan(s.iloc[0])
    np.testing.assert_array_equal(s.iloc[1:].to_numpy(), pred["bound"].iloc[:-1].to_numpy())
    np.testing.assert_array_equal(fb[f"{TF}_cbnc_high_std"].iloc[1:].to_numpy(),
                                  pred["band"].iloc[:-1].to_numpy())


def test_forming_no_lookahead(wide, models):
    cut = 3000
    mod = wide.copy()
    scaled = [c for c in mod.columns
              if any(c == f"{tf}_{k}" for tf in (1, 5, 15, 60, 240, 1440)
                     for k in ("open", "high", "low", "close", "rsi_14", "rsi_ma8",
                               "macd_12_26_9", "macd_hist_12_26_9", "ema_25"))]
    mod.iloc[cut:, [mod.columns.get_loc(c) for c in scaled]] *= 2.0
    for (tf, side), m in models.items():
        a_full, b_full = _cbnc(wide, m), _cbnc(mod, m)
        pd.testing.assert_frame_equal(a_full.iloc[:cut + 1], b_full.iloc[:cut + 1])
        if tf != 240:
            assert a_full.iloc[:cut + 1, 0].notna().any()
        assert not a_full.iloc[cut + 1:].equals(b_full.iloc[cut + 1:])


# --------------------------------------------------------------------------- hybrid zone

def _zone_frames(cbnc_high, cbnc_low, cb_high, cb_low, low1, high1,
                 cb_in_long=None, cb_in_short=None):
    n = len(cbnc_high)
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    cbnc = pd.DataFrame({"15_cbnc_high": cbnc_high, "15_cbnc_low": cbnc_low},
                        index=idx, dtype=float)
    cb = pd.DataFrame({"15_cb_high": cb_high, "15_cb_low": cb_low}, index=idx, dtype=float)
    cb["15_cb_inzone_long"] = cb_in_long if cb_in_long is not None else [True] * n
    cb["15_cb_inzone_short"] = cb_in_short if cb_in_short is not None else [True] * n
    df = pd.DataFrame({"1_low": low1, "1_high": high1}, index=idx, dtype=float)
    return cbnc, cb, df


def test_hybrid_zone_formulas():
    cbnc, cb, df = _zone_frames([200.0], [90.0], [150.0], [100.0], [120.0], [120.0])
    z = forming_zones(cbnc, cb, df, 15)
    assert list(z.columns) == [
        "15_cbnc_tgt_long", "15_cbnc_tgt_short", "15_cbnc_zone_long", "15_cbnc_zone_short",
        "15_cbnc_inzone_long", "15_cbnc_inzone_short", "15_cbx_inzone_long", "15_cbx_inzone_short"]
    r = z.iloc[0]
    assert r["15_cbnc_tgt_long"] == pytest.approx(199.7, rel=1e-12)
    assert r["15_cbnc_zone_long"] == pytest.approx(100 + 0.05 * 99.7, rel=1e-12)
    assert r["15_cbnc_tgt_short"] == pytest.approx(90.135, rel=1e-12)
    assert r["15_cbnc_zone_short"] == pytest.approx(150 - 0.05 * (150 - 90.135), rel=1e-12)
    # 1_low 120 above zone_long (~104.985); 1_high 120 below zone_short (~146.99)
    assert not r["15_cbnc_inzone_long"] and not r["15_cbnc_inzone_short"]


def test_hybrid_zone_inzone_boundaries():
    zl = 100 + 0.05 * (199.7 - 100)
    zs = 150 - 0.05 * (150 - 90.135)
    cbnc, cb, df = _zone_frames([200.0] * 2, [90.0] * 2, [150.0] * 2, [100.0] * 2,
                                [zl, zl + 1e-3], [zs, zs - 1e-3])
    z = forming_zones(cbnc, cb, df, 15)
    assert z["15_cbnc_inzone_long"].tolist() == [True, False]
    assert z["15_cbnc_inzone_short"].tolist() == [True, False]


def test_inverted_span_no_zone():
    # cbnc_high 95 below cb_low 100 -> tgt_long < SL; cbnc_low 160 above cb_high 150
    cbnc, cb, df = _zone_frames([95.0], [160.0], [150.0], [100.0], [1.0], [1e9])
    z = forming_zones(cbnc, cb, df, 15)
    assert not z["15_cbnc_inzone_long"].iloc[0]
    assert not z["15_cbx_inzone_long"].iloc[0]
    assert not z["15_cbnc_inzone_short"].iloc[0]
    assert not z["15_cbx_inzone_short"].iloc[0]


def test_cbx_is_intersection():
    # rows: (cb_inzone, cbnc_inzone) = (F,F), (F,T), (T,F), (T,T)
    cb_in = [False, False, True, True]
    low1 = [1e9, 1.0, 1e9, 1.0]       # long: reaches zone on rows 1, 3
    high1 = [0.0, 1e9, 0.0, 1e9]      # short: reaches zone on rows 1, 3
    cbnc, cb, df = _zone_frames([200.0] * 4, [90.0] * 4, [150.0] * 4, [100.0] * 4,
                                low1, high1, cb_in, cb_in)
    z = forming_zones(cbnc, cb, df, 15)
    for d in ("long", "short"):
        assert z[f"15_cbnc_inzone_{d}"].tolist() == [False, True, False, True]
        assert z[f"15_cbx_inzone_{d}"].tolist() == [False, False, False, True]


def test_bool_dtypes_and_nan_false():
    cbnc, cb, df = _zone_frames([np.nan, 200.0], [np.nan, 90.0], [150.0, np.nan],
                                [100.0, np.nan], [1.0, 1.0], [1e9, 1e9])
    z = forming_zones(cbnc, cb, df, 15)
    for c in ("15_cbnc_inzone_long", "15_cbnc_inzone_short",
              "15_cbx_inzone_long", "15_cbx_inzone_short"):
        assert z[c].dtype == bool, c
        assert not z[c].any(), c
    for c in ("15_cbnc_tgt_long", "15_cbnc_tgt_short", "15_cbnc_zone_long", "15_cbnc_zone_short"):
        assert z[c].dtype == np.float64, c


# --------------------------------------------------------------------------- reference parity

def _load_cbnc():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "candle_bounds_nc" / "cbnc.py"
    spec = importlib.util.spec_from_file_location("_cbnc_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Shim:
    def __init__(self, m):
        self.tf, self.side, self.band_pct, self._m = m.tf, m.side, m.band_pct, m

    def predict_pct(self, X):
        return self._m.predict_pct(X[FEATURES].to_numpy())


def test_forming_matches_reference(wide, models):
    """Port parity: identical to cbnc.forming_features / infer_forming / add_zones."""
    ref = _load_cbnc()
    for tf in TFS:
        for side in SIDES:
            pd.testing.assert_frame_equal(forming_features(wide, tf, side),
                                          ref.forming_features(wide, tf, side), check_exact=True)
        nc = pd.concat([_cbnc(wide, models[(tf, s)]) for s in SIDES], axis=1)
        ref_nc = pd.concat([ref.infer_forming(wide, _Shim(models[(tf, s)])) for s in SIDES], axis=1)
        pd.testing.assert_frame_equal(nc, ref_nc, check_exact=True)

        cb = pd.concat([closed_bounds(wide, models[(tf, s)]) for s in SIDES], axis=1)
        cb = pd.concat([cb, closed_zones(cb, wide, tf)], axis=1)
        ref_out = ref_nc.copy()
        ref.add_zones(ref_out, wide, cb, tf)
        z = forming_zones(nc, cb, wide, tf)
        pd.testing.assert_frame_equal(z, ref_out[z.columns], check_exact=True,
                                      check_dtype=False)
        assert z[f"{tf}_cbx_inzone_long"].any() or z[f"{tf}_cbx_inzone_short"].any()
