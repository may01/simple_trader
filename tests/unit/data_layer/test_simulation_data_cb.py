# tests/unit/data_layer/test_simulation_data_cb.py
# Backtest read path for the candle-bounds (CB) columns (phase 18, task 08).
# No production change: SimulationData / WideDataPoint already expose any
# "{tf}_{col}" column of the prepared df_with_indicators.pkl.

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MAIN_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from data import SimulationData, WideDataPoint  # noqa: E402
from indicators.candle_bounds import stage  # noqa: E402
from indicators.candle_bounds.stage import expected_columns  # noqa: E402
from tests.unit.data_layer.cb_fixtures import make_models, make_wide  # noqa: E402
from tests.unit.data_layer.test_data_preparer_cb import _make_dp  # noqa: E402

DAY = 86_400_000
N_DAYS = 5
WARMUP_ROWS = 2 * 1440  # steps start here (prepared frame starts at day 1)
WALK = 200
BOOL_COLS = (
    (15, "cb_inzone_long"),
    (15, "extdone_high"),
    (15, "cb_f_trend_long"),
    (15, "cb_f_mcx_short"),
    (15, "cbx_inzone_long"),
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> pd.DataFrame:
    """A DataPreparer.prepare()-built frame (models monkeypatched in)."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("PAIR", "link_usdt")
        mp.setattr(stage, "_WARNED_PAIRS", set())
        models = make_models()
        mp.setattr(stage, "load_models", lambda pair: models)
        wide = make_wide(n_minutes=N_DAYS * 1440, seed=3, start="1970-01-01 00:00")
        dp = _make_dp(tmp_path_factory.mktemp("simcb"), "ds", wide)
        dp.prepare("raw.pkl", data_start_ms=1 * DAY)
        return pd.read_pickle(dp.output_path)
    finally:
        mp.undo()


def _make_sim(monkeypatch, tmp_path, prepared, start_row: int) -> SimulationData:
    path = tmp_path / "df_with_indicators.pkl"
    prepared.to_pickle(str(path))
    monkeypatch.setattr("data._wide_df_path_for_pair", lambda p: str(path))
    begin = int(prepared.index[start_row].timestamp())
    end = int(prepared.index[-1].timestamp())
    return SimulationData("link_usdt", begin, end, 1)


@pytest.fixture
def sim(monkeypatch, tmp_path, prepared) -> SimulationData:
    return _make_sim(monkeypatch, tmp_path, prepared, WARMUP_ROWS)


def _bool_frame(prepared: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"{tf}_{n}": prepared[f"{tf}_{n}"].astype(bool)
                         for tf, n in BOOL_COLS})


@pytest.fixture(scope="module")
def mixed_start(prepared) -> int:
    """First row (past warm-up) of a WALK-row window in which the three-way
    marker combination fires on some but not all steps and every listed bool
    column takes both values. The fixture's ema_25_slope_class is -1 for the
    first ~200 post-warm-up rows (a downtrend of the seeded walk), so the
    default start would give a constant cb_f_trend_long == False."""
    b = _bool_frame(prepared)
    three = (prepared["15_cb_inzone_long"].astype(bool)
             & prepared["15_cb_f_extdone_long"].astype(bool)
             & prepared["15_cb_f_trend_long"].astype(bool))
    for start in range(WARMUP_ROWS, len(prepared) - WALK, 20):
        w = b.iloc[start:start + WALK]
        n3 = int(three.iloc[start:start + WALK].sum())
        if 0 < n3 < WALK and all(w[c].nunique() == 2 for c in w.columns):
            return start
    pytest.fail("fixture has no window with mixed marker values")


def _split(col: str) -> tuple[int, str]:
    tf, name = col.split("_", 1)
    return int(tf), name


def _walk(sim: SimulationData, n: int):
    for _ in range(n):
        assert not sim.is_end()
        yield sim.get()
        sim.next()


def test_simulation_point_exposes_all_cb_columns(sim, prepared):
    cols = expected_columns()
    assert cols
    point = sim.get()
    assert isinstance(point, WideDataPoint)
    for col in cols:
        tf, name = _split(col)
        val = point.get(name, tf)
        assert isinstance(val, float), (col, type(val))
    # past warm-up the 15-tf family carries real (finite) values; extdone_*_lvl
    # is NaN by design while the extdone flag is not set, so it is excluded.
    for col in (c for c in cols if c.startswith("15_") and not c.endswith("_lvl")):
        val = point.get(col.split("_", 1)[1], 15)
        assert math.isfinite(val), (col, val)


def test_bool_columns_read_as_zero_or_one(monkeypatch, tmp_path, prepared, mixed_start):
    sim = _make_sim(monkeypatch, tmp_path, prepared, mixed_start)
    seen = {c: set() for c in BOOL_COLS}
    for point in _walk(sim, WALK):
        for tf, name in BOOL_COLS:
            v = point.get(name, tf)
            assert v in (0.0, 1.0), (name, v)
            seen[(tf, name)].add(v)
    for c, vals in seen.items():
        assert vals == {0.0, 1.0}, (c, vals)


def test_values_match_frame(sim, prepared):
    checked = 0
    stride = (sim.steps - 1) // 19
    for k in range(20):
        point = sim.get()
        got = point.get("cb_zone_long", 60)
        want = float(prepared.loc[point.timestamp, "60_cb_zone_long"])
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == want
            checked += 1
        for _ in range(stride):
            sim.next()
    assert checked > 0  # non-vacuous: some sampled steps carry real values


def test_marker_filter_combination_is_computable(monkeypatch, tmp_path, prepared, mixed_start):
    sim = _make_sim(monkeypatch, tmp_path, prepared, mixed_start)
    count = 0
    ts_list = []
    for point in _walk(sim, WALK):
        ts_list.append(point.timestamp)
        if (
            point.get("cb_inzone_long", 15)
            and point.get("cb_f_extdone_long", 15)
            and point.get("cb_f_trend_long", 15)
        ):
            count += 1
    sub = prepared.loc[ts_list]
    want = int(
        (
            sub["15_cb_inzone_long"].astype(bool)
            & sub["15_cb_f_extdone_long"].astype(bool)
            & sub["15_cb_f_trend_long"].astype(bool)
        ).sum()
    )
    assert 0 < want < len(ts_list)  # combination fires on some, not all steps
    assert count == want


def test_shift_reads_previous_closed_row(monkeypatch, tmp_path, prepared):
    closed = prepared["15_is_closed"].astype(bool)
    start = WARMUP_ROWS + 60
    sim = _make_sim(monkeypatch, tmp_path, prepared, start)
    compared = 0
    for point in _walk(sim, 40):  # spans mid-candle and boundary rows
        ts = point.timestamp
        prior = prepared.loc[(prepared.index <= ts) & closed, "15_cb_high"]
        for shift in (1, 2):
            want = float(prior.iloc[-shift])
            got = point.get("cb_high", 15, shift=shift)
            assert (math.isnan(got) and math.isnan(want)) or got == want
            compared += not math.isnan(want)
    assert compared > 0
