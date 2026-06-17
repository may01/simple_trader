"""Phase 14 Task 07 — action overlay on the full_view chart."""

import pandas as pd
from plotly.subplots import make_subplots

from frontend.data_viewer import DataViewer, FullData


def _viewer():
    idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"1_close": [20.0] * 5}, index=idx)
    return DataViewer(FullData(df), tf=1), idx


def _fig():
    fig = make_subplots(rows=1, cols=1)
    fig._subplot_rows = {"price": 1}
    return fig


def test_open_and_close_markers_drawn(monkeypatch):
    dv, idx = _viewer()
    ts = idx[2].timestamp()
    monkeypatch.setattr(dv, "_load_latest_actions", lambda: [
        {"event": "OPEN", "position_type": "POSITION_TYPE_LONG",
         "timestamp": ts, "executed_price": 20.0, "target_price": 20.0},
        {"event": "CLOSE", "position_type": "POSITION_TYPE_LONG",
         "timestamp": ts, "executed_price": 20.1, "target_price": 20.1},
        {"event": "SIGNAL_FIRED", "position_type": "POSITION_TYPE_LONG",
         "timestamp": ts, "executed_price": 0.0, "target_price": 20.0},
    ])
    fig = _fig()
    window = pd.DataFrame({"1_close": [20.0] * 5}, index=idx)
    dv._draw_action_markers(fig, window)
    names = {t.name for t in fig.data}
    assert any("OPEN" in n for n in names)
    assert "CLOSE" in names
    # SIGNAL_FIRED must never be plotted
    assert not any("SIGNAL" in n for n in names)


def test_skip_if_no_actions(monkeypatch):
    dv, idx = _viewer()
    monkeypatch.setattr(dv, "_load_latest_actions", lambda: [])
    fig = _fig()
    window = pd.DataFrame({"1_close": [20.0] * 5}, index=idx)
    dv._draw_action_markers(fig, window)
    assert len(fig.data) == 0


def test_actions_outside_window_skipped(monkeypatch):
    dv, idx = _viewer()
    far = pd.Timestamp("2025-01-01", tz="UTC").timestamp()
    monkeypatch.setattr(dv, "_load_latest_actions", lambda: [
        {"event": "OPEN", "position_type": "POSITION_TYPE_LONG",
         "timestamp": far, "executed_price": 20.0, "target_price": 20.0},
    ])
    fig = _fig()
    window = pd.DataFrame({"1_close": [20.0] * 5}, index=idx)
    dv._draw_action_markers(fig, window)
    assert len(fig.data) == 0
