"""Tests for target/stop-loss price overlays — tgt_long, sl_long, tgt_short, sl_short (Phase 12, Task 11)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.data_viewer import DataViewer, FullData, _indicator_subplot

TARGET_FIELDS = ["tgt_long", "sl_long", "tgt_short", "sl_short"]


def _make_df(overlays: list[str], n: int = 60, tf: int = 15) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
    }
    for name in overlays:
        data[f"{tf}_{name}"] = np.linspace(100, 110, n)
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


def _price_line_labels(mock_renderer):
    return [
        c.kwargs.get("label", c[0][4] if len(c[0]) > 4 else None)
        for c in mock_renderer.draw_line.call_args_list
        if c[0][1] == "price"
    ]


class TestTargetSLOverlays:
    def test_routed_to_price_axis(self):
        for name in TARGET_FIELDS:
            assert _indicator_subplot(name) is None

    def test_drawn_on_price_in_window_figure(self, mock_renderer):
        df = _make_df(TARGET_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        labels = _price_line_labels(mock_renderer)
        for name in TARGET_FIELDS:
            assert name in labels

    def test_absent_columns_skipped(self, mock_renderer):
        df = _make_df(["tgt_long"])  # only one of the four present
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        labels = _price_line_labels(mock_renderer)
        assert "tgt_long" in labels
        for name in ("sl_long", "tgt_short", "sl_short"):
            assert name not in labels

    def test_no_extra_subplot_created(self, mock_renderer):
        df = _make_df(TARGET_FIELDS)
        v = _viewer(df, mock_renderer)
        v.build_window_figure("2024-01-01", 1, indicators=TARGET_FIELDS)
        subplots = mock_renderer.create_figure.call_args[0][0]
        assert subplots == ["price", "volume"]

    def test_targets_and_sls_have_distinct_colors(self, mock_renderer):
        df = _make_df(TARGET_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        colors = {}
        for c in mock_renderer.draw_line.call_args_list:
            if c[0][1] == "price":
                colors[c.kwargs.get("label")] = c.kwargs.get("color")
        assert len({colors[n] for n in TARGET_FIELDS}) == 4
