"""Tests for oscillator charts — RSI/CCI/MACD with MAs (Phase 12, Task 08)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.chart_renderer import ChartRenderer
from frontend.data_viewer import DataViewer, FullData, _indicator_subplot


_OSC_FIELDS = [
    "rsi_14", "rsi_ma8", "rsi_ma12", "rsi_ma24",
    "cci_14", "cci_14_ma_5",
    "macd_12_26_9", "macd_signal_12_26_9", "macd_hist_12_26_9",
    "macd_5_13_9", "macd_signal_5_13_9",
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
        data[f"{tf}_{name}"] = np.linspace(-1, 1, n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {
        "price": 1, "volume": 2, "rsi": 3, "cci": 4,
        "macd_12_26_9": 5, "macd_5_13_9": 6,
    }
    r.create_figure.return_value = mock_fig
    return r


def _viewer(df, mock_renderer):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = mock_renderer
    return v


def _line_subplot_labels(mock_renderer):
    """(subplot, label) pairs for every draw_line call."""
    return [
        (c[0][1], c.kwargs.get("label"))
        for c in mock_renderer.draw_line.call_args_list
    ]


# ---------------------------------------------------------------------------
# Subplot routing
# ---------------------------------------------------------------------------

class TestRouting:
    @pytest.mark.parametrize(
        "field,subplot",
        [
            ("rsi_14", "rsi"),
            ("rsi_ma8", "rsi"),
            ("rsi_ma12", "rsi"),
            ("rsi_ma24", "rsi"),
            ("cci_14", "cci"),
            ("cci_14_ma_5", "cci"),
            ("macd_12_26_9", "macd_12_26_9"),
            ("macd_signal_12_26_9", "macd_12_26_9"),
            ("macd_hist_12_26_9", "macd_12_26_9"),
            ("macd_5_13_9", "macd_5_13_9"),
            ("macd_signal_5_13_9", "macd_5_13_9"),
        ],
    )
    def test_field_routes_to_subplot(self, field, subplot):
        assert _indicator_subplot(field) == subplot


# ---------------------------------------------------------------------------
# ChartRenderer.draw_bar
# ---------------------------------------------------------------------------

class TestDrawBar:
    def test_adds_bar_trace_to_subplot_row(self):
        cr = ChartRenderer()
        fig = cr.create_figure(["price", "macd_12_26_9"])
        cr.draw_bar(fig, "macd_12_26_9", [1, 2], [0.1, -0.2], label="hist")
        bars = [t for t in fig.data if t.type == "bar"]
        assert len(bars) == 1
        assert bars[0].name == "hist"

    def test_stateless_additive(self):
        cr = ChartRenderer()
        fig = cr.create_figure(["price", "x"])
        cr.draw_bar(fig, "x", [1], [1.0], label="a")
        cr.draw_bar(fig, "x", [1], [2.0], label="b")
        assert len([t for t in fig.data if t.type == "bar"]) == 2


# ---------------------------------------------------------------------------
# Window figure oscillator content
# ---------------------------------------------------------------------------

class TestOscillatorDrawing:
    def test_rsi_mas_share_rsi_subplot(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        pairs = _line_subplot_labels(mock_renderer)
        for f in ("rsi_14", "rsi_ma8", "rsi_ma12", "rsi_ma24"):
            assert ("rsi", f) in pairs

    def test_cci_ma_shares_cci_subplot(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        pairs = _line_subplot_labels(mock_renderer)
        assert ("cci", "cci_14") in pairs
        assert ("cci", "cci_14_ma_5") in pairs

    def test_macd_hist_drawn_as_bar(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        bar_calls = [
            (c[0][1], c.kwargs.get("label"))
            for c in mock_renderer.draw_bar.call_args_list
        ]
        assert ("macd_12_26_9", "macd_hist_12_26_9") in bar_calls
        # hist must not also be a line
        assert ("macd_12_26_9", "macd_hist_12_26_9") not in _line_subplot_labels(
            mock_renderer
        )

    def test_macd_variants_get_own_subplots(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        subplots = mock_renderer.create_figure.call_args[0][0]
        assert "macd_12_26_9" in subplots
        assert "macd_5_13_9" in subplots

    def test_subplot_order(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        subplots = mock_renderer.create_figure.call_args[0][0]
        assert subplots == [
            "price", "volume", "rsi", "cci", "macd_12_26_9", "macd_5_13_9"
        ]

    def test_missing_fields_skipped(self, mock_renderer):
        df = _make_df(["rsi_14"])  # only RSI present
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        subplots = mock_renderer.create_figure.call_args[0][0]
        assert "cci" not in subplots
        assert "macd_12_26_9" not in subplots
        # Volume renders as a bar; no indicator (macd_hist) bars expected.
        bar_subplots = [c[0][1] for c in mock_renderer.draw_bar.call_args_list]
        assert bar_subplots == ["volume"]

    def test_explicit_indicators_respected(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).build_window_figure(
            "2024-01-01", 1, indicators=["rsi_14"]
        )
        pairs = _line_subplot_labels(mock_renderer)
        labels = [l for _s, l in pairs]
        assert "rsi_14" in labels
        assert "cci_14" not in labels

    def test_legacy_view_full_defaults_unchanged(self, mock_renderer):
        df = _make_df(_OSC_FIELDS)
        _viewer(df, mock_renderer).view_full()
        labels = [l for _s, l in _line_subplot_labels(mock_renderer)]
        assert "rsi_14" in labels
        assert "cci_14" in labels
        assert "rsi_ma8" not in labels
        assert "macd_12_26_9" not in labels
