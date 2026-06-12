"""Tests for timeframe selection in HistoryDashboard (Phase 12, Task 06)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

import plotly.graph_objects as go
from dash import dcc

from frontend.data_viewer import DataViewer, FullData
from frontend.history_dashboard import HistoryDashboard, _tf_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multi_tf_df(n: int = 60 * 24 * 2) -> pd.DataFrame:
    """Two days of 1-min rows carrying columns for TFs 15 and 60.

    Higher-TF values repeat on every 1-min row, mimicking the wide df.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {}
    for tf in (15, 60):
        period = (np.arange(n) // tf).astype(float)
        data[f"{tf}_open"] = period + 100
        data[f"{tf}_high"] = period + 105
        data[f"{tf}_low"] = period + 95
        data[f"{tf}_close"] = period + 102
        data[f"{tf}_volume"] = period + 1000
        data[f"{tf}_rsi_14"] = np.linspace(40, 60, n)
        data[f"{tf}_cci_14"] = np.linspace(-100, 100, n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def df():
    return _make_multi_tf_df()


@pytest.fixture
def full_data(df):
    return FullData(df)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {"price": 1, "volume": 2, "rsi": 3, "cci": 4}
    r.create_figure.return_value = mock_fig
    return r


@pytest.fixture
def viewer(full_data, mock_renderer):
    v = DataViewer(full_data, tf=15)
    v.renderer = mock_renderer
    return v


def _find_component(layout, target_id):
    stack = [layout]
    while stack:
        node = stack.pop()
        if getattr(node, "id", None) == target_id:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    raise AssertionError(f"component {target_id!r} not found")


# ---------------------------------------------------------------------------
# TF labels
# ---------------------------------------------------------------------------

class TestTfLabel:
    @pytest.mark.parametrize(
        "tf,label",
        [(1, "1m"), (5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"), (1440, "1d")],
    )
    def test_known_labels(self, tf, label):
        assert _tf_label(tf) == label

    def test_fallback_minutes(self):
        assert _tf_label(3) == "3m"

    def test_fallback_hours(self):
        assert _tf_label(120) == "2h"

    def test_fallback_days(self):
        assert _tf_label(2880) == "2d"


# ---------------------------------------------------------------------------
# DataViewer.available_tfs
# ---------------------------------------------------------------------------

class TestAvailableTfs:
    def test_discovers_tfs_from_close_columns(self, viewer):
        assert viewer.available_tfs() == [15, 60]

    def test_sorted_ascending(self, full_data):
        df = full_data.df.copy()
        df["1440_close"] = 1.0
        df["5_close"] = 1.0
        v = DataViewer(FullData(df))
        assert v.available_tfs() == [5, 15, 60, 1440]

    def test_ignores_non_close_columns(self, viewer):
        # rsi/cci/volume columns must not create extra TFs
        assert len(viewer.available_tfs()) == 2


# ---------------------------------------------------------------------------
# DataViewer per-TF figures
# ---------------------------------------------------------------------------

class TestBuildWindowFigurePerTf:
    def test_tf_overrides_columns(self, viewer, mock_renderer, df):
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1, tf=60)
        args = mock_renderer.draw_candles.call_args[0]
        _fig, times, opens, highs, lows, closes = args
        # 60-min TF over 1 day → 24 deduped candles
        assert len(opens) == 24

    def test_tf_none_uses_primary(self, viewer, mock_renderer):
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1)
        args = mock_renderer.draw_candles.call_args[0]
        # 15-min TF over 1 day → 96 deduped candles
        assert len(args[2]) == 96

    def test_candle_plotted_at_period_open_timestamp(self, viewer, mock_renderer, df):
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1, tf=60)
        times = mock_renderer.draw_candles.call_args[0][1]
        assert times[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert times[1] == pd.Timestamp("2024-01-01 01:00:00")

    def test_candles_evenly_spaced_one_tf_apart(self, viewer, mock_renderer, df):
        """Even spacing makes plotly render each candle across the full period."""
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1, tf=60)
        times = mock_renderer.draw_candles.call_args[0][1]
        deltas = {b - a for a, b in zip(times, times[1:])}
        assert deltas == {pd.Timedelta(minutes=60)}

    def test_dedup_keeps_last_row_values_per_period(self, viewer, mock_renderer, df):
        """Timestamp moves to period open, values stay from the period's last row."""
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1, tf=60)
        args = mock_renderer.draw_candles.call_args[0]
        closes = args[5]
        # _make_multi_tf_df: 60_close = (row // 60) + 102; last row of hour 0
        # is row 59 → period value 0 + 102
        assert closes[0] == df["60_close"].iloc[59]

    def test_dedup_drops_nan_close_rows(self, mock_renderer):
        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        df = pd.DataFrame(
            {
                "15_open": 1.0, "15_high": 1.0, "15_low": 1.0,
                "15_close": [np.nan] * 30 + [1.0] * 30,
                "15_volume": 1.0,
            },
            index=idx,
        )
        v = DataViewer(FullData(df))
        v.renderer = mock_renderer
        v.build_window_figure(pd.Timestamp("2024-01-01"), 1)
        times = mock_renderer.draw_candles.call_args[0][1]
        assert min(times) >= pd.Timestamp("2024-01-01 00:30:00")


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

class TestTimeframeChecklist:
    def test_checklist_present_with_discovered_options(self, full_data):
        d = HistoryDashboard(full_data)
        cl = _find_component(d._app.layout, "timeframes")
        assert [o["value"] for o in cl.options] == [15, 60]
        assert [o["label"] for o in cl.options] == ["15m", "1h"]

    def test_initial_value_is_primary_tf(self, full_data):
        d = HistoryDashboard(full_data)
        cl = _find_component(d._app.layout, "timeframes")
        assert cl.value == [15]

    def test_initial_value_falls_back_to_first_tf(self, full_data):
        d = HistoryDashboard(full_data, tf=5)  # 5 not in df
        cl = _find_component(d._app.layout, "timeframes")
        assert cl.value == [15]


# ---------------------------------------------------------------------------
# _render_groups multi-TF behavior
# ---------------------------------------------------------------------------

class TestRenderGroupsMultiTf:
    def test_one_graph_per_selected_tf(self, full_data):
        d = HistoryDashboard(full_data)
        graphs = d._render_groups("2024-01-01", 1, [15, 60])
        assert len(graphs) == 2
        assert all(isinstance(g, dcc.Graph) for g in graphs)

    def test_ascending_tf_order_regardless_of_click_order(self, full_data):
        d = HistoryDashboard(full_data)
        graphs = d._render_groups("2024-01-01", 1, [60, 15])
        assert [g.id["tf"] for g in graphs] == [15, 60]

    def test_graph_ids_carry_tf(self, full_data):
        d = HistoryDashboard(full_data)
        graphs = d._render_groups("2024-01-01", 1, [60])
        assert graphs[0].id == {"type": "tf-chart", "tf": 60}

    def test_empty_selection_returns_empty_list(self, full_data):
        d = HistoryDashboard(full_data)
        assert d._render_groups("2024-01-01", 1, []) == []

    def test_each_graph_built_with_its_tf(self, full_data):
        d = HistoryDashboard(full_data)
        d.viewer = MagicMock()
        d.viewer.build_window_figure.return_value = go.Figure()
        d._render_groups("2024-01-01", 1, [60, 15])
        called_tfs = [
            c.kwargs["tf"] for c in d.viewer.build_window_figure.call_args_list
        ]
        assert called_tfs == [15, 60]
