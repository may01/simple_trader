"""Tests for price-overlay selection checkboxes (Phase 12, Task 15).

Mirrors the oscillator-subplot selection of Task 12, but for price-axis
overlays: Bollinger band groups, EMAs and SAR become toggleable, with
``sar``, ``bb_x_10_15`` and ``bb_x_20_3`` unchecked on first load. Target/
stop-loss overlays are always drawn and never toggleable.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

import plotly.graph_objects as go
from dash import dcc

from frontend.data_viewer import DataViewer, FullData
from frontend.history_dashboard import HistoryDashboard


# Every field backing the toggleable overlay groups, plus a target overlay.
_ALL_OVERLAY_FIELDS = (
    "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2",
    "bb_upper_10_15", "bb_lower_10_15",
    "bb_upper_20_3", "bb_lower_20_3",
    "ema_7", "ema_14", "ema_25", "ema_50", "ema_100",
    "sar_002_02",
    "tgt_long",
)


def _make_df(overlays=_ALL_OVERLAY_FIELDS, n: int = 60, tf: int = 15) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
        f"{tf}_rsi_14": np.linspace(40, 60, n),
    }
    for name in overlays:
        data[f"{tf}_{name}"] = np.linspace(100, 110, n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {"price": 1, "volume": 2, "rsi": 3}
    r.create_figure.return_value = mock_fig
    return r


def _viewer(df, mock_renderer):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = mock_renderer
    return v


def _price_line_labels(mock_renderer):
    return [
        c.kwargs.get("label")
        for c in mock_renderer.draw_line.call_args_list
        if c[0][1] == "price"
    ]


def _marker_labels(mock_renderer):
    return [c.kwargs.get("label") for c in mock_renderer.draw_marker.call_args_list]


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
# DataViewer.available_overlays / default_overlays
# ---------------------------------------------------------------------------

class TestAvailableOverlays:
    def test_discovers_groups_from_columns(self):
        v = DataViewer(FullData(_make_df()))
        assert v.available_overlays() == [
            "bb_x_20_2", "bb_x_10_15", "bb_x_20_3", "ema", "sar"
        ]

    def test_skips_absent_groups(self):
        v = DataViewer(FullData(_make_df(overlays=("bb_upper_20_2", "ema_7"))))
        assert v.available_overlays() == ["bb_x_20_2", "ema"]

    def test_default_excludes_off_groups(self):
        v = DataViewer(FullData(_make_df()))
        assert v.default_overlays() == ["bb_x_20_2", "ema"]

    def test_default_is_subset_of_available(self):
        v = DataViewer(FullData(_make_df(overlays=("bb_upper_10_15", "sar_002_02"))))
        # only off-by-default groups present → nothing checked
        assert v.default_overlays() == []
        assert v.available_overlays() == ["bb_x_10_15", "sar"]


# ---------------------------------------------------------------------------
# build_window_figure overlay filter
# ---------------------------------------------------------------------------

class TestOverlayFilter:
    def test_none_means_all(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        labels = _price_line_labels(mock_renderer)
        assert "bb_upper_10_15" in labels
        assert "bb_upper_20_3" in labels
        assert "sar_002_02" in _marker_labels(mock_renderer)

    def test_only_selected_groups_drawn(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, overlays=["bb_x_20_2"])
        labels = _price_line_labels(mock_renderer)
        assert "bb_upper_20_2" in labels
        assert "bb_upper_10_15" not in labels
        assert "ema_7" not in labels
        assert "sar_002_02" not in _marker_labels(mock_renderer)

    def test_empty_list_drops_all_toggleable(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, overlays=[])
        labels = _price_line_labels(mock_renderer)
        for name in ("bb_upper_20_2", "ema_7", "bb_upper_10_15"):
            assert name not in labels
        assert "sar_002_02" not in _marker_labels(mock_renderer)

    def test_targets_always_drawn(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, overlays=[])
        assert "tgt_long" in _price_line_labels(mock_renderer)

    def test_sar_group_drawn_as_marker(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, overlays=["sar"])
        assert "sar_002_02" in _marker_labels(mock_renderer)
        assert "sar_002_02" not in _price_line_labels(mock_renderer)


# ---------------------------------------------------------------------------
# Dashboard checklist
# ---------------------------------------------------------------------------

class TestOverlayChecklist:
    def test_checklist_present_with_discovered_options(self):
        d = HistoryDashboard(FullData(_make_df()))
        cl = _find_component(d._app.layout, "overlays")
        assert [o["value"] for o in cl.options] == [
            "bb_x_20_2", "bb_x_10_15", "bb_x_20_3", "ema", "sar"
        ]

    def test_off_groups_unchecked_by_default(self):
        d = HistoryDashboard(FullData(_make_df()))
        cl = _find_component(d._app.layout, "overlays")
        assert cl.value == ["bb_x_20_2", "ema"]
        for off in ("sar", "bb_x_10_15", "bb_x_20_3"):
            assert off not in cl.value

    def test_selection_passed_to_every_tf(self):
        d = HistoryDashboard(FullData(_make_df()))
        d.viewer = MagicMock()
        d.viewer.build_window_figure.return_value = go.Figure()
        d._render_groups("2024-01-01", 1, [15], None, ["sar", "ema"])
        calls = d.viewer.build_window_figure.call_args_list
        assert all(c.kwargs["overlays"] == ["sar", "ema"] for c in calls)

    def test_empty_overlay_selection_still_renders(self):
        d = HistoryDashboard(FullData(_make_df()))
        graphs = d._render_groups("2024-01-01", 1, [15], None, [])
        assert len(graphs) == 1
        assert isinstance(graphs[0], dcc.Graph)
