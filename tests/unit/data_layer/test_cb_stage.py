"""Tests for the CB stage entry point: ``compute_cb`` / ``compute_cb_columns``."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds import kernel, stage
from indicators.candle_bounds.params import ATR_COL, EXTDONE_LADDER, SIDES, TFS
from indicators.candle_bounds.stage import (
    CbResult,
    compute_cb,
    compute_cb_columns,
    expected_columns,
    required_columns,
)
from tests.unit.data_layer.cb_fixtures import make_models, make_wide


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return make_wide(6000)


@pytest.fixture(scope="module")
def models():
    return make_models()


@pytest.fixture(scope="module")
def result(wide, models) -> CbResult:
    return compute_cb(wide, models)


def _all_six(value: tuple[float, float]) -> dict[tuple[int, str], tuple[float, float]]:
    return {(tf, side): value for tf in TFS for side in SIDES}


# --------------------------------------------------------------------------- contract

def test_expected_columns_count_and_unique():
    cols = expected_columns()
    assert len(cols) == 114
    assert len(set(cols)) == 114
    assert all(c.startswith(("15_", "60_", "240_")) for c in cols)


def test_expected_columns_per_tf_order():
    cols = expected_columns()
    per_tf = [c[len("15_"):] for c in cols[:38]]
    assert per_tf[:12] == [
        "cb_high", "cb_high_std", "cb_high_up", "cb_high_dn",
        "cb_low", "cb_low_std", "cb_low_up", "cb_low_dn",
        "cb_zone_long", "cb_zone_short", "cb_inzone_long", "cb_inzone_short",
    ]
    assert per_tf[-6:] == [
        "cb_f_extdone_long", "cb_f_extdone_short", "cb_f_trend_long",
        "cb_f_trend_short", "cb_f_mcx_long", "cb_f_mcx_short",
    ]
    assert [c.split("_", 1)[1] for c in cols[38:76]] == per_tf
    assert all(c.startswith("60_") for c in cols[38:76])


def test_required_columns_content():
    req = required_columns()
    assert req[:4] == ["1_high", "1_low", "1_close", ATR_COL]
    assert "240_ema_25_slope_class" in req
    assert len(req) == 4 + 12 * 3


def test_compute_cb_columns_match_expected(wide, result):
    assert list(result.columns.columns) == expected_columns()
    assert result.columns.index.equals(wide.index)


def test_compute_cb_equals_kernel_pieces(wide, models, result):
    cb = kernel.closed_bounds(wide, models[(15, "high")])
    pd.testing.assert_series_equal(result.columns["15_cb_high"], cb["15_cb_high"])
    ext = kernel.extdone(wide, 15, EXTDONE_LADDER[15])
    pd.testing.assert_series_equal(result.columns["15_extdone_low"], ext["15_extdone_low"])
    pd.testing.assert_series_equal(result.columns["15_extdone_high"], ext["15_extdone_high"])
    nc = kernel.forming_bounds(kernel.forming_prediction(wide, models[(60, "low")]), 60, "low")
    pd.testing.assert_series_equal(result.columns["60_cbnc_low_dn"], nc["60_cbnc_low_dn"])
    # the result is not degenerate
    assert result.columns["15_cb_high"].notna().sum() > 1000
    assert result.columns["15_cbnc_high"].notna().sum() > 1000


def _hand_row_frame() -> pd.DataFrame:
    """One row at 10:35 (tf 60 progress 0.6), every required column present."""
    idx = pd.DatetimeIndex([pd.Timestamp("2025-01-01 10:35", tz="UTC")])
    data: dict[str, list] = {c: [np.nan] for c in required_columns()}
    data["1_high"], data["1_low"], data["1_close"], data[ATR_COL] = [100.5], [99.5], [100.0], [1.0]
    for tf in TFS:
        data[f"{tf}_is_closed"] = [False]
        data[f"{tf}_high"], data[f"{tf}_low"] = [100.5], [99.5]
    data["60_high"], data["60_low"] = [107.0], [93.0]
    return pd.DataFrame(data, index=idx)


def test_uses_production_ladder_per_side(models):
    # tf 60 at progress 0.6: high d = 8.0 (pull 7 -> False), low d = 6.0 (pull 7 -> True)
    res = compute_cb(_hand_row_frame(), models)
    assert bool(res.columns["60_extdone_low"].iloc[0]) is True
    assert bool(res.columns["60_extdone_high"].iloc[0]) is False


def test_input_not_mutated(models):
    df = make_wide(1500)
    df["extra_col"] = 1.0
    before = df.copy(deep=True)
    compute_cb(df, models)
    pd.testing.assert_frame_equal(df, before)


def test_missing_input_raises_naming_columns(wide, models):
    df = wide.drop(columns=["15_ema_25_slope_class", "60_move_class"])
    with pytest.raises(ValueError, match="candle_bounds: missing input columns") as ei:
        compute_cb(df, models)
    assert "15_ema_25_slope_class" in str(ei.value)
    assert "60_move_class" in str(ei.value)


def test_empty_frame(wide, models):
    df = wide.iloc[:0]
    res = compute_cb(df, models)
    assert len(res.columns) == 0
    assert list(res.columns.columns) == expected_columns()
    assert res.nc_pred_last == {}


# --------------------------------------------------------------------------- forming hold

def test_nc_pred_last_is_unshifted_last_row(wide, models, result):
    assert set(result.nc_pred_last) == {(tf, s) for tf in TFS for s in SIDES}
    for (tf, side), (bound, band) in result.nc_pred_last.items():
        pred = kernel.forming_prediction(wide, models[(tf, side)])
        assert bound == pred["bound"].iloc[-1]
        assert band == pred["band"].iloc[-1]
    assert np.isfinite(result.nc_pred_last[(15, "high")][0])


def test_nc_pred_last_nan_on_incomplete_features(models):
    res = compute_cb(_hand_row_frame(), models)
    assert set(res.nc_pred_last) == {(tf, s) for tf in TFS for s in SIDES}
    assert np.isnan(res.nc_pred_last[(15, "high")][0])


def test_nc_pred_prev_overrides_last_row_only(wide, models, result):
    res = compute_cb(wide, models, nc_pred_prev=_all_six((123.0, 1.0)))
    last = res.columns.iloc[-1]
    assert last["15_cbnc_high"] == 123.0
    assert last["15_cbnc_high_std"] == 1.0
    assert last["15_cbnc_high_up"] == 124.0
    assert last["15_cbnc_high_dn"] == 122.0
    assert last["15_cbnc_tgt_long"] == pytest.approx(123.0 * (1 - 0.0015))
    assert last["240_cbnc_low"] == 123.0
    pd.testing.assert_frame_equal(res.columns.iloc[:-1], result.columns.iloc[:-1])
    # non-cbnc columns of the last row are unaffected
    for c in ("15_cb_high", "15_extdone_high", "15_cb_inzone_long", "15_cb_f_trend_long"):
        assert last[c] == result.columns[c].iloc[-1]


def test_nc_pred_prev_empty_gives_nan_last_row(wide, models, result):
    res = compute_cb(wide, models, nc_pred_prev={})
    last = res.columns.iloc[-1]
    for tf in TFS:
        for side in SIDES:
            for suf in ("", "_std", "_up", "_dn"):
                assert np.isnan(last[f"{tf}_cbnc_{side}{suf}"])
        for d in ("long", "short"):
            assert np.isnan(last[f"{tf}_cbnc_tgt_{d}"])
            assert np.isnan(last[f"{tf}_cbnc_zone_{d}"])
            assert not last[f"{tf}_cbnc_inzone_{d}"]
            assert not last[f"{tf}_cbx_inzone_{d}"]
    pd.testing.assert_frame_equal(res.columns.iloc[:-1], result.columns.iloc[:-1])


def test_nc_pred_prev_partial_key_and_nan(wide, models):
    prev = {(15, "high"): (123.0, 1.0), (15, "low"): (np.nan, np.nan)}
    res = compute_cb(wide, models, nc_pred_prev=prev)
    last = res.columns.iloc[-1]
    assert last["15_cbnc_high"] == 123.0
    assert np.isnan(last["15_cbnc_low"])
    assert np.isnan(last["60_cbnc_high"])


def test_single_row_with_prev_uses_prev(wide, models):
    res = compute_cb(wide.iloc[-1:], models, nc_pred_prev={(15, "high"): (123.0, 1.0)})
    assert res.columns["15_cbnc_high"].iloc[0] == 123.0


# --------------------------------------------------------------------------- compute_cb_columns

def test_compute_cb_columns_no_model_returns_none(wide, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(stage, "log_warning", calls.append)
    monkeypatch.setattr(stage, "_WARNED_PAIRS", set())
    assert compute_cb_columns(wide, pair="btc_usdt") is None
    assert compute_cb_columns(wide, pair="btc_usdt") is None
    assert len(calls) == 1
    assert "btc_usdt" in calls[0]


def test_compute_cb_columns_default_pair_from_env(wide, monkeypatch):
    monkeypatch.setenv("PAIR", "link_usdt")
    out = compute_cb_columns(wide)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == expected_columns()
    assert out.index.equals(wide.index)


def test_no_sklearn_imported():
    code = "import sys, indicators.candle_bounds.stage; print('sklearn' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
