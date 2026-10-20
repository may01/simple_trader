"""Phase 20 task 02: the run-stats driver — masks, edge rule, loaders, report."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds import kernel
from indicators.candle_bounds.params import ATR_COL, TFS
from indicators.candle_bounds.stage import compute_cb
from tests.unit.data_layer.cb_fixtures import make_models, make_wide

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
for _p in (os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "notebooks", "zone_profitability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cb_zone_run_stats as drv  # noqa: E402
import zone_runs  # noqa: E402
from run_zone_ext import atr1_ma5  # noqa: E402


@pytest.fixture(scope="module")
def wide():
    return make_wide(3000, seed=3)


@pytest.fixture(scope="module")
def cb(wide):
    return compute_cb(wide, make_models()).columns


def _flag_frame():
    idx = pd.date_range("2025-01-01", periods=6, freq="1min", tz="UTC")
    return pd.DataFrame({
        "15_cb_inzone_long": [True, True, False, True, True, True],
        "15_cb_f_extdone_long": [True, False, True, True, True, True],
        "15_cb_f_overshoot_long": [True, True, True, False, True, np.nan],
        "15_cb_f_trend_long": [False, True, True, True, True, True],
    }, index=idx)


def test_masks_from_columns_and_sets():
    m = drv.masks_from_columns(_flag_frame(), 15, "long")
    assert set(m) == set(drv.SETS)
    assert m["raw"].tolist() == [True, True, False, True, True, True]
    assert m["extdone"].tolist() == [True, False, False, True, True, True]
    assert m["overshoot"].tolist() == [True, True, False, False, True, True]   # NaN flag passes
    assert m["entry"].tolist() == [False, True, False, False, True, True]


def test_masks_from_kernel_equals_masks_from_columns_on_fixture(wide, cb):
    df_with_cb = pd.concat([wide, cb], axis=1)
    bounds = cb[[c for c in cb.columns if "_cb_" in c and "_cb_f_" not in c and "overshoot" not in c]]
    for side in ("long", "short"):
        a = drv.masks_from_columns(df_with_cb, 15, side)
        b = drv.masks_from_kernel(wide, bounds, 15, side)
        for name in drv.SETS:
            assert a[name].tolist() == b[name].tolist(), (side, name)


def test_edge_mask_drops_first_last_candle_and_nan_bound():
    n = 60
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    closed = np.zeros(n, dtype=bool)
    closed[14::15] = True                              # 4 candles of 15 rows
    df = pd.DataFrame({"15_is_closed": closed}, index=idx)
    bound = np.full(n, 100.0)
    cbf = pd.DataFrame({"15_cb_low": bound, "15_cb_high": bound + 1}, index=idx)
    e = drv.edge_mask(df, cbf, 15, "long")
    assert not e[:15].any() and not e[45:].any()
    assert e[15:45].all()
    bound[20] = np.nan
    cbf2 = pd.DataFrame({"15_cb_low": bound, "15_cb_high": bound + 1}, index=idx)
    e2 = drv.edge_mask(df, cbf2, 15, "long")
    assert not e2[15:30].any() and e2[30:45].all()


def test_load_train_rebuilds_atr_when_absent(tmp_path, capsys, wide, cb):
    ind = wide.drop(columns=[ATR_COL])
    ind.to_pickle(tmp_path / "ind.pkl")
    cb.to_pickle(tmp_path / "cb.pkl")
    df, cbo = drv.load_train(str(tmp_path / "ind.pkl"), str(tmp_path / "cb.pkl"), False, "link_usdt")
    assert ATR_COL in df.columns
    np.testing.assert_allclose(df[ATR_COL].to_numpy(), atr1_ma5(wide), equal_nan=True)
    assert "rebuilt" in capsys.readouterr().out
    assert "15_cb_inzone_long" in cbo.columns


def test_load_train_backfills_slope_class_and_nan_move_class(tmp_path, wide, cb):
    ind = wide.drop(columns=[f"{tf}_{c}" for tf in TFS for c in ("ema_25_slope_class", "move_class")])
    ind.to_pickle(tmp_path / "ind.pkl")
    cb.to_pickle(tmp_path / "cb.pkl")
    df, _ = drv.load_train(str(tmp_path / "ind.pkl"), str(tmp_path / "cb.pkl"), False, "link_usdt")
    for tf in TFS:
        assert f"{tf}_ema_25_slope_class" in df.columns
        assert df[f"{tf}_move_class"].isna().all()


def test_driver_writes_json_matching_zone_runs_on_synthetic_oos(tmp_path, wide, cb):
    oos = tmp_path / "oos"
    oos.mkdir()
    full = pd.concat([wide, cb], axis=1)
    full.to_pickle(oos / "df_with_indicators.pkl")
    out = tmp_path / "out"
    assert drv.main(["--skip-train", "--oos-dir", str(oos), "--out", str(out)]) == 0
    report = json.loads((out / "cb_zone_run_stats.json").read_text())
    cell = report["oos2m"]["raw"]["15_long"]
    m = drv.masks_from_columns(full, 15, "long")["raw"] & drv.edge_mask(full, full, 15, "long")
    cid = drv.candle_id(full, 15)
    minute = drv.minutes_of(full.index)
    lengths = zone_runs.blocks(m, cid, minute)
    expect = zone_runs.summarize(lengths, zone_runs.per_candle(m, cid, minute))
    expect["remaining_given_streak"] = zone_runs.remaining_given_streak(lengths, 12)
    assert cell == json.loads(json.dumps(expect))
    md = (out / "cb_zone_run_stats.md").read_text()
    assert "## Freeze table" in md
    assert "CB_STOP_EXPECTED_POINTS=" in md and "n/a" in md


def test_minutes_of_is_unit_independent():
    # The stored frames carry a datetime64[ms, UTC] index; a ns-based divisor made every minute 0.
    ms = pd.DatetimeIndex(np.arange(5, dtype=np.int64) * 60_000, dtype="datetime64[ms]").tz_localize("UTC")
    ns = pd.date_range("1970-01-01", periods=5, freq="1min", tz="UTC")
    assert drv.minutes_of(ms).tolist() == [0, 1, 2, 3, 4]
    assert drv.minutes_of(ns).tolist() == [0, 1, 2, 3, 4]


def test_kernel_parity_reports_zero_mismatches_on_a_frame_with_stored_flags(wide, cb):
    # The 2y path (kernel) against the oos2m path (stored flags), on one frame
    # that has both; the real-data run of this is recorded in the results doc.
    full = pd.concat([wide, cb], axis=1)
    report = drv.kernel_parity(full)
    assert set(report) == {f"{tf}_{side}_{name}" for tf in TFS for side in ("long", "short") for name in drv.SETS}
    assert all(v == 0 for v in report.values()), report
