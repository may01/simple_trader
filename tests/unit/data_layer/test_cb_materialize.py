# tests/unit/data_layer/test_cb_materialize.py
# Materialize CB columns into an existing dataset (phase 18, task 07).
#
# The prepared frame is built through DataPreparer.prepare() with the heavy
# passes stubbed (as in test_data_preparer_cb.py), except that the EMA-25
# family is computed by the REAL per-row field pass (EMAField,
# EmaDiffPrcField, EmaSlopeClassField over build_indicator_input slices), so
# the backfill is checked against genuine field output. CB columns come from
# the real frozen LINK/USDT model file.

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MAIN_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
if str(MAIN_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(MAIN_DIR / "scripts"))

import cb_markers_materialize as cbm  # noqa: E402
from cb_markers_materialize import backfill_ema_slope, materialize  # noqa: E402
from constants import INDICATOR_WINDOW_ROWS  # noqa: E402
from indicators.candle_bounds import stage  # noqa: E402
from indicators.candle_bounds.params import TFS  # noqa: E402
from indicators.candle_bounds.stage import expected_columns  # noqa: E402
from indicators.framework import build_indicator_input  # noqa: E402
from indicators.library.classification import EmaSlopeClassField  # noqa: E402
from indicators.library.trend import EMAField, EmaDiffPrcField  # noqa: E402
from tests.unit.data_layer.cb_fixtures import make_wide  # noqa: E402
from training.data_preparer import DataPreparer, _SliceDataPoint  # noqa: E402

DAY = 86_400_000
N_DAYS = 5           # 480 x 15m candles (past INDICATOR_WINDOW_ROWS); 30 x 240m (> EMA 25)
EMA_COLS = [f"{tf}_{n}" for tf in TFS for n in ("ema_25_diff_prc", "ema_25_slope_class")]


def _field_pass(df: pd.DataFrame, fields, start_ts=None) -> None:
    """The production per-row mechanism (_compute_tf_rows) for *fields* only."""
    rows = df.index if start_ts is None else df.index[df.index >= start_ts]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for tf in TFS:
            names = [f"{tf}_{f.name}" for f in fields]
            res: dict[str, list] = {n: [] for n in names}
            for ts in rows:
                s = build_indicator_input(df, ts, tf)
                dp = _SliceDataPoint(s, ts)
                for f in fields:
                    s[f"{tf}_{f.name}"] = f.compute(dp, tf)
                last = s.iloc[-1]
                for n in names:
                    res[n].append(last[n])
            df[names] = pd.DataFrame(res, index=rows)


class _DummyAttrs:
    def save(self, path):
        with open(path, "w") as f:
            f.write("attrs")


def _prepare_real_ema(tmp_dir: Path) -> pd.DataFrame:
    wide = make_wide(n_minutes=N_DAYS * 1440, seed=3, start="1970-01-01 00:00")
    raw = pd.DataFrame({"c": range(len(wide))}, index=wide.index.copy())
    dp = DataPreparer(
        "configs/indicators_config.yaml",
        str(tmp_dir / "df_with_indicators.pkl"),
        str(tmp_dir / "data_attributes.pkl"),
    )
    dp.num_workers = 1
    dp._load_raw_data = lambda path: raw
    dp._build_base_dataframe = lambda r: wide.loc[r.index].copy()
    dp._compute_base_indicators = lambda df, start_ts=None: _field_pass(
        df, [EMAField(25), EmaDiffPrcField(25)], start_ts)
    dp._compute_base_attributes = lambda df: None
    dp._compute_class_indicators = lambda df, start_ts=None: _field_pass(
        df, [EmaSlopeClassField()], start_ts)
    dp._compute_profit_labels = lambda df: None
    dp._data_attributes_container = lambda df: _DummyAttrs()
    dp.prepare("raw.pkl", data_start_ms=None)
    return pd.read_pickle(dp.output_path)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> pd.DataFrame:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PAIR", "link_usdt")
        return _prepare_real_ema(tmp_path_factory.mktemp("prep"))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setattr(stage, "_WARNED_PAIRS", set())


def _stripped(prepared: pd.DataFrame) -> pd.DataFrame:
    return prepared.drop(columns=expected_columns() + EMA_COLS)


# ---------------------------------------------------------------------------


def test_prepared_fixture_is_real(prepared):
    """Non-vacuous: CB columns present, 15m history past the indicator window."""
    assert all(c in prepared.columns for c in expected_columns() + EMA_COLS)
    assert int(prepared["15_is_closed"].sum()) > INDICATOR_WINDOW_ROWS + 50
    assert prepared["15_cb_high"].notna().sum() > 0
    assert set(np.unique(prepared["15_ema_25_slope_class"].dropna())) == {-1, 0, 1}


def test_backfill_matches_field_output(prepared):
    src = prepared.drop(columns=EMA_COLS)
    out = backfill_ema_slope(src)
    for tf in TFS:
        orig = prepared[f"{tf}_ema_25_diff_prc"]
        m = orig.notna()
        assert m.sum() > 0
        assert out.loc[m, f"{tf}_ema_25_diff_prc"].notna().all()
        np.testing.assert_allclose(
            out.loc[m, f"{tf}_ema_25_diff_prc"].to_numpy(float),
            orig[m].to_numpy(float), rtol=1e-9, atol=0.0)
        np.testing.assert_array_equal(
            out.loc[m, f"{tf}_ema_25_slope_class"].to_numpy(int),
            prepared.loc[m, f"{tf}_ema_25_slope_class"].to_numpy(int))


def test_backfill_trimmed_frame_exact_after_window(prepared):
    """A dataset trimmed at DATA_START lacks the warm-up candles the field's
    EMA window reached into: exact once INDICATOR_WINDOW_ROWS - 1 closed candles
    of the tf precede the row inside the frame, close (abs < 1e-3 %) before."""
    start = pd.Timestamp(1 * DAY, unit="ms", tz="UTC")
    trimmed = prepared.loc[prepared.index >= start]
    out = backfill_ema_slope(trimmed.drop(columns=EMA_COLS))
    for tf in TFS:
        orig = trimmed[f"{tf}_ema_25_diff_prc"].to_numpy(float)
        ours = out[f"{tf}_ema_25_diff_prc"].to_numpy(float)
        n_prev = np.concatenate(
            ([0], np.cumsum(trimmed[f"{tf}_is_closed"].to_numpy(bool))[:-1]))
        full = n_prev >= INDICATOR_WINDOW_ROWS - 1
        ok = ~np.isnan(orig)
        if tf == 15:
            assert (ok & full).sum() > 1000          # non-vacuous exact region
        np.testing.assert_allclose(ours[ok & full], orig[ok & full], rtol=1e-9, atol=0.0)
        # before the window fills: close, NaN only before the frame's first close
        if (ok & ~full).any():
            assert np.nanmax(np.abs(ours[ok & ~full] - orig[ok & ~full])) < 1e-3
        assert np.isnan(ours[ok]).sum() == int((ok & (n_prev == 0)).sum())


def test_backfill_does_not_overwrite(prepared):
    src = prepared.drop(columns=EMA_COLS)
    src["15_ema_25_slope_class"] = 7
    out = backfill_ema_slope(src)
    assert (out["15_ema_25_slope_class"] == 7).all()
    assert "15_ema_25_diff_prc" in out.columns          # absent ones still added
    assert out.columns.is_unique


def test_materialize_equals_prep(prepared):
    """G7: materialize on a stripped prepared frame == prep's CB columns."""
    out = materialize(_stripped(prepared), "link_usdt")
    cols = expected_columns()
    pd.testing.assert_frame_equal(
        out[cols], prepared[cols], check_exact=False, rtol=1e-12)
    assert prepared["60_cbnc_low"].notna().sum() > 0


def test_materialize_idempotent(prepared):
    once = materialize(_stripped(prepared), "link_usdt")
    twice = materialize(once, "link_usdt")
    assert once.columns.is_unique and twice.columns.is_unique
    pd.testing.assert_frame_equal(once, twice)


def test_materialize_keeps_existing_columns(prepared):
    src = _stripped(prepared)
    out = materialize(src, "link_usdt")
    pd.testing.assert_frame_equal(out[src.columns], src)
    assert len(out.columns) == len(src.columns) + len(expected_columns()) + len(EMA_COLS)


def test_materialize_missing_move_class_raises(prepared):
    src = _stripped(prepared).drop(columns=["60_move_class"])
    with pytest.raises(ValueError, match="move_class7_materialize"):
        materialize(src, "link_usdt")


def test_materialize_no_model_raises(prepared):
    with pytest.raises(RuntimeError):
        materialize(_stripped(prepared), "btc_usdt")


# --- main(): backup / atomic write on a tmp dataset dir ---------------------


@pytest.fixture
def ds(tmp_path, monkeypatch, prepared):
    path = tmp_path / "df_with_indicators.pkl"
    _stripped(prepared).to_pickle(path)
    monkeypatch.setattr(cbm, "wide_df_path", lambda: str(path))
    return path


def _bak(path: Path) -> Path:
    return Path(str(path) + ".precb_bak")


def _tmp(path: Path) -> Path:
    return Path(str(path) + ".tmp")


def test_main_backup_before_write_and_rename(ds):
    orig_bytes, orig_ino = ds.read_bytes(), os.stat(ds).st_ino
    cbm.main()
    bak = _bak(ds)
    assert bak.exists()
    assert os.stat(bak).st_ino == orig_ino            # hard link to the original inode
    assert bak.read_bytes() == orig_bytes
    assert os.stat(ds).st_ino != orig_ino             # original replaced by rename
    assert not _tmp(ds).exists()
    out = pd.read_pickle(ds)
    assert all(c in out.columns for c in expected_columns() + EMA_COLS)


def test_main_existing_backup_never_overwritten(ds):
    _bak(ds).write_bytes(b"older backup")
    cbm.main()
    assert _bak(ds).read_bytes() == b"older backup"
    assert "15_cb_high" in pd.read_pickle(ds).columns


def test_main_empty_backup_aborts_without_writing(ds):
    _bak(ds).write_bytes(b"")
    before = ds.read_bytes()
    with pytest.raises(RuntimeError, match="empty"):
        cbm.main()
    assert ds.read_bytes() == before
    assert not _tmp(ds).exists()


def test_main_second_run_idempotent(ds):
    cbm.main()
    first = pd.read_pickle(ds)
    cbm.main()
    pd.testing.assert_frame_equal(pd.read_pickle(ds), first)


def test_main_guard_lost_column_aborts(ds, monkeypatch):
    real = cbm.materialize
    monkeypatch.setattr(cbm, "materialize",
                        lambda df, pair: real(df, pair).drop(columns=["1_close"]))
    before = ds.read_bytes()
    with pytest.raises(RuntimeError, match="lost columns"):
        cbm.main()
    assert ds.read_bytes() == before
    assert not _bak(ds).exists() and not _tmp(ds).exists()


def test_main_removes_partial_tmp_on_failure(ds, monkeypatch):
    before = ds.read_bytes()

    def boom(src, dst):
        raise OSError("rename failed")

    monkeypatch.setattr(cbm.os, "rename", boom)
    with pytest.raises(OSError, match="rename failed"):
        cbm.main()
    assert not _tmp(ds).exists()
    assert ds.read_bytes() == before
