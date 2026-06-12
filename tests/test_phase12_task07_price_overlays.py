"""Tests for price-axis overlays — Bollinger, EMAs, SAR (Phase 12, Task 07)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.data_viewer import DataViewer, FullData


def _make_df(overlays: list[str], n: int = 60, tf: int = 15) -> pd.DataFrame:
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
    for name in overlays:
        data[f"{tf}_{name}"] = np.linspace(100, 110, n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {"price": 1, "volume": 2, "rsi": 3, "cci": 4}
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


class TestPriceOverlays:
    def test_bollinger_drawn_on_price(self, mock_renderer):
        df = _make_df(["bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        labels = _price_line_labels(mock_renderer)
        assert "bb_upper_20_2" in labels
        assert "bb_middle_20_2" in labels
        assert "bb_lower_20_2" in labels

    def test_emas_drawn_on_price(self, mock_renderer):
        df = _make_df(["ema_7", "ema_14", "ema_25", "ema_50", "ema_100"])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        labels = _price_line_labels(mock_renderer)
        for ema in ("ema_7", "ema_14", "ema_25", "ema_50", "ema_100"):
            assert ema in labels

    def test_sar_drawn_as_marker_not_line(self, mock_renderer):
        df = _make_df(["sar_002_02"])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        marker_labels = [
            c.kwargs.get("label", "")
            for c in mock_renderer.draw_marker.call_args_list
        ]
        assert "sar_002_02" in marker_labels
        assert "sar_002_02" not in _price_line_labels(mock_renderer)

    def test_absent_columns_skipped(self, mock_renderer):
        df = _make_df([])  # no overlay columns at all
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        assert _price_line_labels(mock_renderer) == []
        mock_renderer.draw_marker.assert_not_called()

    def test_bb_variant_shares_color(self, mock_renderer):
        df = _make_df(["bb_upper_20_2", "bb_lower_20_2", "bb_upper_10_15"])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        colors = {}
        for c in mock_renderer.draw_line.call_args_list:
            if c[0][1] == "price":
                label = c.kwargs.get("label")
                colors[label] = c.kwargs.get("color")
        assert colors["bb_upper_20_2"] == colors["bb_lower_20_2"]
        assert colors["bb_upper_20_2"] != colors["bb_upper_10_15"]

    def test_overlays_use_tf_prefixed_columns(self, mock_renderer):
        df = _make_df(["ema_7"])
        # Add a decoy 60-TF column with different values
        df["60_ema_7"] = 999.0
        v = _viewer(df, mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        ema_calls = [
            c for c in mock_renderer.draw_line.call_args_list
            if c.kwargs.get("label") == "ema_7"
        ]
        assert len(ema_calls) == 1
        assert 999.0 not in ema_calls[0][0][3]

    def test_legacy_view_full_has_no_overlays(self, mock_renderer):
        df = _make_df(["ema_7", "sar_002_02"])
        v = _viewer(df, mock_renderer)
        v.view_full()
        assert "ema_7" not in _price_line_labels(mock_renderer)
        mock_renderer.draw_marker.assert_not_called()
