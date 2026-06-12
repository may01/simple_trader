"""Tests for oscillator-subplot selection checkboxes (Phase 12, Task 12)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

import plotly.graph_objects as go
from dash import dcc

from frontend.data_viewer import DataViewer, FullData
from frontend.history_dashboard import HistoryDashboard


def _make_df(n: int = 60 * 24, tfs=(15, 60), extras=("rsi_14", "cci_14", "adx_14")) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {}
    for tf in tfs:
        period = (np.arange(n) // tf).astype(float)
        data[f"{tf}_open"] = period + 100
        data[f"{tf}_high"] = period + 105
        data[f"{tf}_low"] = period + 95
        data[f"{tf}_close"] = period + 102
        data[f"{tf}_volume"] = period + 1000
        for name in extras:
            data[f"{tf}_{name}"] = np.linspace(0, 1, n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {"price": 1, "volume": 2}
    r.create_figure.return_value = mock_fig
    return r


def _viewer(df, mock_renderer):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = mock_renderer
    return v


def _drawn_subplots(mock_renderer):
    return {c[0][1] for c in mock_renderer.draw_line.call_args_list}


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
# DataViewer.available_subplots
# ---------------------------------------------------------------------------

class TestAvailableSubplots:
    def test_discovers_from_columns(self):
        v = DataViewer(FullData(_make_df()))
        assert v.available_subplots() == ["rsi", "cci", "adx"]

    def test_skips_absent_indicator_groups(self):
        v = DataViewer(FullData(_make_df(extras=("rsi_14",))))
        assert v.available_subplots() == ["rsi"]

    def test_includes_derivative_groups(self):
        v = DataViewer(FullData(_make_df(extras=("rsi_14", "close_diff_prc"))))
        assert v.available_subplots() == ["rsi", "close_diff"]

    def test_dedups_indicators_sharing_subplot(self):
        v = DataViewer(FullData(_make_df(extras=("rsi_14", "rsi_ma8"))))
        assert v.available_subplots() == ["rsi"]


# ---------------------------------------------------------------------------
# build_window_figure subplot filter
# ---------------------------------------------------------------------------

class TestSubplotFilter:
    def test_only_selected_subplots_created(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, subplots=["rsi"])
        assert mock_renderer.create_figure.call_args[0][0] == [
            "price", "volume", "rsi"
        ]

    def test_unselected_indicators_not_drawn(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, subplots=["rsi"])
        drawn = _drawn_subplots(mock_renderer)
        assert "cci" not in drawn
        assert "adx" not in drawn

    def test_none_means_all(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        assert mock_renderer.create_figure.call_args[0][0] == [
            "price", "volume", "rsi", "cci", "adx"
        ]

    def test_empty_list_leaves_price_and_volume(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, subplots=[])
        assert mock_renderer.create_figure.call_args[0][0] == ["price", "volume"]

    def test_price_overlays_unaffected(self, mock_renderer):
        df = _make_df(extras=("rsi_14", "ema_7"))
        v = _viewer(df, mock_renderer)
        v.build_window_figure("2024-01-01", 1, subplots=[])
        labels = [
            c.kwargs.get("label")
            for c in mock_renderer.draw_line.call_args_list
            if c[0][1] == "price"
        ]
        assert "ema_7" in labels

    def test_filter_applies_to_explicit_indicator_list(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure(
            "2024-01-01", 1, indicators=["rsi_14", "cci_14"], subplots=["cci"]
        )
        assert mock_renderer.create_figure.call_args[0][0] == [
            "price", "volume", "cci"
        ]


# ---------------------------------------------------------------------------
# Dashboard checklist
# ---------------------------------------------------------------------------

class TestSubplotChecklist:
    def test_checklist_present_with_discovered_options(self):
        d = HistoryDashboard(FullData(_make_df()))
        cl = _find_component(d._app.layout, "subplots")
        assert [o["value"] for o in cl.options] == ["rsi", "cci", "adx"]

    def test_all_checked_by_default(self):
        d = HistoryDashboard(FullData(_make_df()))
        cl = _find_component(d._app.layout, "subplots")
        assert cl.value == ["rsi", "cci", "adx"]

    def test_selection_passed_to_every_tf(self):
        d = HistoryDashboard(FullData(_make_df()))
        d.viewer = MagicMock()
        d.viewer.build_window_figure.return_value = go.Figure()
        d._render_groups("2024-01-01", 1, [15, 60], ["rsi", "adx"])
        calls = d.viewer.build_window_figure.call_args_list
        assert len(calls) == 2
        assert all(c.kwargs["subplots"] == ["rsi", "adx"] for c in calls)

    def test_empty_selection_still_renders_graphs(self):
        d = HistoryDashboard(FullData(_make_df()))
        graphs = d._render_groups("2024-01-01", 1, [15], [])
        assert len(graphs) == 1
        assert isinstance(graphs[0], dcc.Graph)
