"""Tests for HistoryDashboard and DataViewer.build_window_figure (Phase 12, Task 05)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

import plotly.graph_objects as go

from frontend.data_viewer import DataViewer, FullData, empty_figure
from frontend.history_dashboard import HistoryDashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 60 * 24 * 3, tf: int = 15) -> pd.DataFrame:
    """Three days of 1-min rows with columns for the given TF."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
        f"{tf}_rsi_14": np.linspace(40, 60, n),
        f"{tf}_cci_14": np.linspace(-100, 100, n),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def df():
    return _make_df()


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


# ---------------------------------------------------------------------------
# empty_figure
# ---------------------------------------------------------------------------

class TestEmptyFigure:
    def test_returns_figure(self):
        assert isinstance(empty_figure(), go.Figure)

    def test_has_annotation(self):
        fig = empty_figure()
        assert fig.layout.annotations
        assert fig.layout.annotations[0].text == "no data in range"


# ---------------------------------------------------------------------------
# DataViewer.build_window_figure
# ---------------------------------------------------------------------------

class TestBuildWindowFigure:
    def test_returns_figure_without_show(self, viewer, mock_renderer):
        fig = viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1)
        assert fig is mock_renderer.create_figure.return_value
        fig.show.assert_not_called()

    def test_slices_by_date_window(self, viewer, mock_renderer, df):
        viewer.build_window_figure(pd.Timestamp("2024-01-02"), 1)
        args = mock_renderer.draw_candles.call_args[0]
        _fig, times, opens, highs, lows, closes = args
        # One deduped candle per 15-min period over 1 day
        assert len(opens) == 96
        assert min(times) >= pd.Timestamp("2024-01-02")
        assert max(times) < pd.Timestamp("2024-01-03")

    def test_end_is_exclusive(self, viewer, mock_renderer):
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1)
        args = mock_renderer.draw_candles.call_args[0]
        times = args[1]
        assert max(times) < pd.Timestamp("2024-01-02")

    def test_empty_window_returns_annotated_figure(self, viewer, mock_renderer):
        fig = viewer.build_window_figure(pd.Timestamp("2030-01-01"), 5)
        assert isinstance(fig, go.Figure)
        assert fig.layout.annotations[0].text == "no data in range"
        mock_renderer.create_figure.assert_not_called()

    def test_default_indicators_drawn(self, viewer, mock_renderer):
        viewer.build_window_figure(pd.Timestamp("2024-01-01"), 1)
        call_subplots = [c[0][1] for c in mock_renderer.draw_line.call_args_list]
        assert "rsi" in call_subplots
        assert "cci" in call_subplots

    def test_accepts_date_string(self, viewer, mock_renderer):
        fig = viewer.build_window_figure("2024-01-01", 1)
        assert fig is mock_renderer.create_figure.return_value

    def test_naive_start_against_tz_aware_index(self, df, mock_renderer):
        """Real wide df index is tz-aware UTC; date-picker dates are naive."""
        df_utc = df.tz_localize("UTC")
        v = DataViewer(FullData(df_utc), tf=15)
        v.renderer = mock_renderer
        fig = v.build_window_figure("2024-01-02", 1)
        assert fig is mock_renderer.create_figure.return_value
        times = mock_renderer.draw_candles.call_args[0][1]
        assert len(times) == 96

    def test_legacy_view_full_unchanged(self, viewer, mock_renderer, df):
        viewer.view_full(start_idx=2, end_idx=5)
        args = mock_renderer.draw_candles.call_args[0]
        assert len(args[2]) == 3  # iloc rows 2,3,4


# ---------------------------------------------------------------------------
# HistoryDashboard
# ---------------------------------------------------------------------------

def _layout_ids(layout) -> set:
    """Collect all component ids in a Dash layout tree."""
    ids = set()
    stack = [layout]
    while stack:
        node = stack.pop()
        node_id = getattr(node, "id", None)
        if node_id is not None:
            ids.add(node_id if isinstance(node_id, str) else tuple(sorted(node_id.items())))
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return ids


class TestHistoryDashboardInit:
    def test_stores_full_data_and_tf(self, full_data):
        d = HistoryDashboard(full_data, tf=60)
        assert d.full_data is full_data
        assert d.tf == 60

    def test_default_tf_is_15(self, full_data):
        d = HistoryDashboard(full_data)
        assert d.tf == 15

    def test_has_data_viewer(self, full_data):
        d = HistoryDashboard(full_data)
        assert isinstance(d.viewer, DataViewer)
        assert d.viewer.tf == d.tf

    def test_builds_dash_app(self, full_data):
        import dash
        d = HistoryDashboard(full_data)
        assert isinstance(d._app, dash.Dash)


class TestHistoryDashboardLayout:
    def test_layout_has_required_ids(self, full_data):
        d = HistoryDashboard(full_data)
        ids = _layout_ids(d._app.layout)
        assert "start-date" in ids
        assert "days" in ids
        assert "chart-groups" in ids

    def test_date_picker_bounded_by_index(self, full_data, df):
        d = HistoryDashboard(full_data)
        picker = _find_component(d._app.layout, "start-date")
        assert str(picker.min_date_allowed) == str(df.index.min().date())
        assert str(picker.max_date_allowed) == str(df.index.max().date())
        assert str(picker.date) == str(df.index.min().date())

    def test_days_input_defaults(self, full_data):
        d = HistoryDashboard(full_data)
        days = _find_component(d._app.layout, "days")
        assert days.value == 7
        assert days.min == 1
        assert days.type == "number"


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


class TestRenderGroups:
    def test_delegates_to_viewer(self, full_data):
        d = HistoryDashboard(full_data)
        d.viewer = MagicMock()
        sentinel = go.Figure()
        d.viewer.build_window_figure.return_value = sentinel
        graphs = d._render_groups("2024-01-01", 2, [15])
        assert len(graphs) == 1
        assert graphs[0].figure is sentinel
        d.viewer.build_window_figure.assert_called_once_with(
            pd.Timestamp("2024-01-01"), 2, tf=15
        )

    def test_none_start_date_returns_no_groups(self, full_data):
        d = HistoryDashboard(full_data)
        assert d._render_groups(None, 5, [15]) == []

    def test_none_days_returns_no_groups(self, full_data):
        d = HistoryDashboard(full_data)
        assert d._render_groups("2024-01-01", None, [15]) == []

    def test_days_below_one_returns_no_groups(self, full_data):
        d = HistoryDashboard(full_data)
        assert d._render_groups("2024-01-01", 0, [15]) == []

    def test_viewer_exception_never_propagates(self, full_data):
        d = HistoryDashboard(full_data)
        d.viewer = MagicMock()
        d.viewer.build_window_figure.side_effect = RuntimeError("boom")
        result = d._render_groups("2024-01-01", 2, [15])
        assert isinstance(result, list)


class TestEntryWiring:
    def test_view_full_uses_history_dashboard(self):
        import inspect
        import view_full
        src = inspect.getsource(view_full.main)
        assert "HistoryDashboard" in src
        assert ".run()" in src or "dashboard.run" in src
