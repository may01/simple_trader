import pytest
import pandas as pd
import numpy as np
from simple_trader.data.simulation_data import SimulationData
from simple_trader.data.data_point import WideDataPoint


def _make_wide_df(rows: int = 10) -> pd.DataFrame:
    """Wide DataFrame: 1-min indexed, all tf-prefixed columns pre-computed."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    return pd.DataFrame({
        "1_close":   np.arange(rows, dtype=float),
        "1_rsi_14":  np.linspace(30, 70, rows),
        "5_close":   np.arange(rows, dtype=float) * 2,
        "5_rsi_14":  np.linspace(40, 60, rows),
    }, index=idx)


def test_simulation_data_step_returns_wide_data_point():
    df = _make_wide_df(10)
    sim = SimulationData(df, step_min=1)
    point = sim.step()
    assert isinstance(point, WideDataPoint)


def test_simulation_data_step_advances_cursor():
    df = _make_wide_df(10)
    sim = SimulationData(df, step_min=1)
    p1 = sim.step()
    p2 = sim.step()
    assert p1.get("close", tf=1) != p2.get("close", tf=1)


def test_simulation_data_is_done_after_last_row():
    df = _make_wide_df(3)
    sim = SimulationData(df, step_min=1)
    sim.step()
    sim.step()
    sim.step()
    assert sim.is_done()


def test_simulation_data_reset_restarts_cursor():
    df = _make_wide_df(3)
    sim = SimulationData(df, step_min=1)
    sim.step()
    sim.step()
    sim.reset()
    assert not sim.is_done()
    p = sim.step()
    assert p.get("close", tf=1) == 0.0


def test_simulation_data_step_raises_when_done():
    df = _make_wide_df(2)
    sim = SimulationData(df, step_min=1)
    sim.step()
    sim.step()
    with pytest.raises(StopIteration, match="SimulationData exhausted"):
        sim.step()
