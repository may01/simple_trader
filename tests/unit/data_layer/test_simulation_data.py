"""Unit tests for SimulationData in data.py."""

import os
import numpy as np
import pandas as pd
import pytest

from data import WideDataPoint, _build_wide_df, SimulationData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim_df():
    """Build minimal wide df with 1440 rows (24h) and required columns."""
    np.random.seed(42)
    idx = pd.date_range("2023-09-01", periods=1440, freq="1min", tz="UTC")
    raw = pd.DataFrame(
        {
            "open": np.random.uniform(25000, 26000, 1440),
            "high": np.random.uniform(26000, 27000, 1440),
            "low": np.random.uniform(24000, 25000, 1440),
            "close": np.random.uniform(25000, 26000, 1440),
            "volume": np.random.uniform(1, 100, 1440),
            "taker_base_vol": np.random.uniform(0.5, 50, 1440),
        },
        index=idx,
    )
    return _build_wide_df(raw)


@pytest.fixture
def sim_pkl(tmp_path, sim_df):
    path = tmp_path / "df_with_indicators.pkl"
    sim_df.to_pickle(str(path))
    return path, sim_df


# ---------------------------------------------------------------------------
# Helper: build SimulationData with monkeypatched path
# ---------------------------------------------------------------------------

def _make_sim(monkeypatch, sim_pkl, pair="btc_usdt", begin_ts=None, end_ts=None, step_min=1):
    pkl_path, sim_df = sim_pkl
    # Patch the path function used inside SimulationData.__init__
    monkeypatch.setattr("data._wide_df_path_for_pair", lambda p: str(pkl_path))

    idx = sim_df.index
    if begin_ts is None:
        begin_ts = int(idx[0].timestamp())
    if end_ts is None:
        end_ts = int(idx[-1].timestamp())

    return SimulationData(pair, begin_ts, end_ts, step_min)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_loads_df(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    assert isinstance(sim._df, pd.DataFrame)
    assert len(sim._df) > 0


def test_steps_count(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl, step_min=1)
    # All 1440 rows should match
    assert sim.steps == 1440


def test_get_returns_wide_data_point(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    dp = sim.get()
    assert isinstance(dp, WideDataPoint)


def test_next_advances_index(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    ts_before = sim.current_ts
    sim.next()
    ts_after = sim.current_ts
    assert ts_after != ts_before


def test_is_end_false_initially(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    assert sim.is_end() is False


def test_is_end_true_at_end(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    for _ in range(sim.steps):
        sim.next()
    assert sim.is_end() is True


def test_full_iteration(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    count = 0
    while not sim.is_end():
        dp = sim.get()
        assert isinstance(dp, WideDataPoint)
        sim.next()
        count += 1
    assert count == sim.steps


def test_split_returns_n_instances(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    slices = sim.split(3)
    assert len(slices) == 3
    for s in slices:
        assert isinstance(s, SimulationData)


def test_split_shares_df_reference(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    slices = sim.split(3)
    for s in slices:
        assert s._df is sim._df


def test_split_independent_cursors(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    slices = sim.split(3)
    # Advance first slice several times
    for _ in range(5):
        slices[0].next()
    # Other slices should be unaffected
    assert slices[1]._cur_idx == 0
    assert slices[2]._cur_idx == 0
    assert slices[0]._cur_idx == 5


def test_missing_timestamps_warning(monkeypatch, sim_pkl, capsys):
    """If only ~50% of the requested timestamps exist in df, a warning is logged."""
    pkl_path, sim_df = sim_pkl
    monkeypatch.setattr("data._wide_df_path_for_pair", lambda p: str(pkl_path))

    idx = sim_df.index
    begin_ts = int(idx[0].timestamp())
    # Request 2x the range so only ~50% of timestamps will exist in df
    end_ts = int(idx[0].timestamp()) + (1440 * 2 * 60) - 60

    sim = SimulationData("btc_usdt", begin_ts, end_ts, step_min=1)
    captured = capsys.readouterr()
    assert "WARN" in captured.out


def test_split_total_coverage(monkeypatch, sim_pkl):
    """Sum of slice steps == total steps."""
    sim = _make_sim(monkeypatch, sim_pkl)
    total = sim.steps
    slices = sim.split(4)
    assert sum(s.steps for s in slices) == total


def test_current_ts_is_timestamp(monkeypatch, sim_pkl):
    sim = _make_sim(monkeypatch, sim_pkl)
    assert isinstance(sim.current_ts, pd.Timestamp)


def test_step_min_5_reduces_steps(monkeypatch, sim_pkl):
    """step_min=5 should yield ~1/5 the timestamps of step_min=1."""
    sim1 = _make_sim(monkeypatch, sim_pkl, step_min=1)
    sim5 = _make_sim(monkeypatch, sim_pkl, step_min=5)
    # At 5-min steps over 24h we expect 288 aligned timestamps
    # (only those that land on 5-min boundaries in the df)
    assert sim5.steps < sim1.steps
    assert sim5.steps > 0
