"""Tests for price-derivative charts — diff fields with rolling means (Phase 12, Task 09)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.data_viewer import DataViewer, FullData, _indicator_subplot


_DERIV_FIELDS = [
    "close_diff_prc", "close_diff_prc_rm_20",
    "close_diff_prc_rm_20_mean_above", "close_diff_prc_rm_20_mean_below",
    "high_diff_prc", "high_diff_prc_rm_20",
    "high_diff_prc_rm_20_mean_above", "high_diff_prc_rm_20_mean_below",
    "low_diff_prc", "low_diff_prc_rm_20",
    "low_diff_prc_rm_20_mean_above", "low_diff_prc_rm_20_mean_below",
    "rsi_ma8_diff", "rsi_ma12_diff", "rsi_ma24_diff",
]


def _make_df(fields: list[str], n: int = 60, tf: int = 15) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
    }
    for name in fields:
        data[f"{tf}_{name}"] = np.linspace(-0.5, 0.5, n)
    return pd.DataFrame(data, index=idx)


def _mock_renderer(extra_subplots: list[str]):
    r = MagicMock()
    mock_fig = MagicMock()
    rows = {"price": 1, "volume": 2}
    for i, sp in enumerate(extra_subplots):
        rows[sp] = i + 3
    mock_fig._subplot_rows = rows
    r.create_figure.return_value = mock_fig
    return r


def _viewer(df, renderer):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = renderer
    return v


def _line_subplot_labels(renderer):
    return [
        (c[0][1], c.kwargs.get("label"))
        for c in renderer.draw_line.call_args_list
    ]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting:
    @pytest.mark.parametrize(
        "field,subplot",
        [
            ("close_diff_prc", "close_diff"),
            ("close_diff_prc_rm_20", "close_diff"),
            ("close_diff_prc_rm_20_mean_above", "close_diff"),
            ("close_diff_prc_rm_20_mean_below", "close_diff"),
            ("high_diff_prc", "high_diff"),
            ("high_diff_prc_rm_20", "high_diff"),
            ("low_diff_prc", "low_diff"),
            ("low_diff_prc_rm_20_mean_below", "low_diff"),
            ("rsi_ma8_diff", "rsi_diff"),
            ("rsi_ma12_diff", "rsi_diff"),
            ("rsi_ma24_diff", "rsi_diff"),
        ],
    )
    def test_field_routes_to_group_subplot(self, field, subplot):
        assert _indicator_subplot(field) == subplot

    def test_base_name_split_bypassed(self):
        # Without exact routing this would land on a subplot named "close"
        assert _indicator_subplot("close_diff_prc") != "close"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

class TestDerivativeDrawing:
    def test_each_group_on_own_subplot(self):
        r = _mock_renderer(["close_diff", "high_diff", "low_diff", "rsi_diff"])
        df = _make_df(_DERIV_FIELDS)
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        pairs = _line_subplot_labels(r)
        assert ("close_diff", "close_diff_prc") in pairs
        assert ("close_diff", "close_diff_prc_rm_20") in pairs
        assert ("high_diff", "high_diff_prc") in pairs
        assert ("low_diff", "low_diff_prc") in pairs
        assert ("rsi_diff", "rsi_ma8_diff") in pairs

    def test_subplot_order_close_high_low_rsi(self):
        r = _mock_renderer(["close_diff", "high_diff", "low_diff", "rsi_diff"])
        df = _make_df(_DERIV_FIELDS)
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        subplots = r.create_figure.call_args[0][0]
        d = [s for s in subplots if s.endswith("_diff")]
        assert d == ["close_diff", "high_diff", "low_diff", "rsi_diff"]

    def test_mean_bands_muted_colors(self):
        r = _mock_renderer(["close_diff"])
        df = _make_df(_DERIV_FIELDS[:4])
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        colors = {
            c.kwargs.get("label"): c.kwargs.get("color")
            for c in r.draw_line.call_args_list
            if c[0][1] == "close_diff"
        }
        assert colors["close_diff_prc_rm_20_mean_above"] != colors["close_diff_prc"]
        assert colors["close_diff_prc_rm_20_mean_below"] != colors["close_diff_prc"]

    def test_missing_group_produces_no_subplot(self):
        r = _mock_renderer(["close_diff"])
        df = _make_df(["close_diff_prc"])  # only close group, partial
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        subplots = r.create_figure.call_args[0][0]
        assert "high_diff" not in subplots
        assert "low_diff" not in subplots
        assert "rsi_diff" not in subplots


# ---------------------------------------------------------------------------
# Zero lines and figure height
# ---------------------------------------------------------------------------

class TestZeroLines:
    def test_zero_line_per_derivative_subplot(self):
        r = _mock_renderer(["close_diff", "high_diff"])
        df = _make_df(_DERIV_FIELDS[:8])
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        fig = r.create_figure.return_value
        zero_shapes = [
            c for c in fig.add_shape.call_args_list
            if c.kwargs.get("y0") == 0 and c.kwargs.get("y1") == 0
        ]
        assert len(zero_shapes) == 2

    def test_no_zero_line_on_price_or_oscillators(self):
        r = _mock_renderer(["rsi"])
        df = _make_df([])
        df["15_rsi_14"] = 50.0
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        fig = r.create_figure.return_value
        fig.add_shape.assert_not_called()

    def test_height_scales_with_row_count(self):
        r = _mock_renderer(["close_diff", "high_diff", "low_diff", "rsi_diff"])
        df = _make_df(_DERIV_FIELDS)
        _viewer(df, r).build_window_figure("2024-01-01", 1)
        fig = r.create_figure.return_value
        heights = [
            c.kwargs.get("height")
            for c in fig.update_layout.call_args_list
            if "height" in c.kwargs
        ]
        assert heights  # height was set
        assert heights[-1] == max(600, 220 * len(fig._subplot_rows))
