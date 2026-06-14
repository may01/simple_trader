"""Tests for profit-label markers on the OHLC chart (Phase 12, Task 14)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.data_viewer import DataViewer, FullData


def _make_df(n: int = 60 * 24, labels=()) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    tf = 15
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
    }
    for name, vals in labels:
        data[f"{tf}_{name}"] = vals
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {"price": 1, "volume": 2, "range": 3}
    r.create_figure.return_value = mock_fig
    return r


def _viewer(df, mock_renderer):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = mock_renderer
    return v


def _price_marker_calls(mock_renderer):
    return [
        c for c in mock_renderer.draw_marker.call_args_list
        if c.kwargs.get("subplot", "price") == "price"
    ]


def _ones(n, every=4):
    vals = np.zeros(n, dtype=int)
    vals[::every] = 1
    return vals


N = 60 * 24


class TestLabelMarkers:
    def test_marker_trace_per_label_column(self, mock_renderer):
        df = _make_df(labels=[
            ("plong_n1_m1_x0p4", _ones(N)),
            ("pshort_n1_m1_x0p4", _ones(N)),
            ("pslong_n1_m1_x0p4_l15_y0p4", _ones(N)),
            ("psshort_n1_m1_x0p4_l15_y0p4", _ones(N)),
        ])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        labels = {c.kwargs.get("label") for c in _price_marker_calls(mock_renderer)}
        assert {
            "15_plong_n1_m1_x0p4", "15_pshort_n1_m1_x0p4",
            "15_pslong_n1_m1_x0p4_l15_y0p4", "15_psshort_n1_m1_x0p4_l15_y0p4",
        } <= labels

    def test_markers_only_on_label_one_rows(self, mock_renderer):
        df = _make_df(labels=[("plong_n1_m1_x0p4", _ones(N))])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        call = next(
            c for c in _price_marker_calls(mock_renderer)
            if c.kwargs.get("label") == "15_plong_n1_m1_x0p4"
        )
        times = call[0][1]
        # Per-minute: a marker on every raw row where the label is 1, not one
        # per 15m candle.
        flagged = set(df.index[df["15_plong_n1_m1_x0p4"].values == 1])
        assert set(times) == flagged
        assert len(times) > 0

    def test_long_markers_below_low_short_above_high(self, mock_renderer):
        df = _make_df(labels=[
            ("plong_n1_m1_x0p4", np.ones(N, dtype=int)),
            ("pshort_n1_m1_x0p4", np.ones(N, dtype=int)),
        ])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        calls = {c.kwargs.get("label"): c for c in _price_marker_calls(mock_renderer)}
        long_call = calls["15_plong_n1_m1_x0p4"]
        short_call = calls["15_pshort_n1_m1_x0p4"]
        # Per-minute markers: compare each marker y to the 15m low/high of its
        # own row, not to the deduped candle array.
        long_lows = np.array([df.loc[t, "15_low"] for t in long_call[0][1]])
        short_highs = np.array([df.loc[t, "15_high"] for t in short_call[0][1]])
        assert (np.array(long_call[0][2]) < long_lows).all()
        assert (np.array(short_call[0][2]) > short_highs).all()
        assert long_call.kwargs.get("marker_symbol", long_call[0][3] if len(long_call[0]) > 3 else None) == "triangle-up"
        assert short_call.kwargs.get("marker_symbol", short_call[0][3] if len(short_call[0]) > 3 else None) == "triangle-down"

    def test_variants_get_distinct_offsets(self, mock_renderer):
        df = _make_df(labels=[
            ("plong_n1_m1_x0p4", np.ones(N, dtype=int)),
            ("pslong_n1_m1_x0p4_l15_y0p4", np.ones(N, dtype=int)),
        ])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        calls = {c.kwargs.get("label"): c for c in _price_marker_calls(mock_renderer)}
        y1 = np.array(calls["15_plong_n1_m1_x0p4"][0][2])
        y2 = np.array(calls["15_pslong_n1_m1_x0p4_l15_y0p4"][0][2])
        assert (y1 != y2).all()

    def test_no_label_columns_no_markers(self, mock_renderer):
        df = _make_df()
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        assert _price_marker_calls(mock_renderer) == []

    def test_all_zero_labels_no_trace(self, mock_renderer):
        df = _make_df(labels=[("plong_n1_m1_x0p4", np.zeros(N, dtype=int))])
        _viewer(df, mock_renderer).build_window_figure("2024-01-01", 1)
        assert _price_marker_calls(mock_renderer) == []


class TestLabelsAcrossTimeframes:
    def test_15m_labels_drawn_on_60m_chart(self, mock_renderer):
        df = _make_df(labels=[("plong_n1_m1_x0p4", _ones(N))])
        # add 60-TF columns so the chart can render at tf=60
        period = (np.arange(N) // 60).astype(float)
        for col, off in (("open", 100), ("high", 105), ("low", 95),
                         ("close", 102), ("volume", 1000)):
            df[f"60_{col}"] = period + off
        v = _viewer(df, mock_renderer)
        v.build_window_figure("2024-01-01", 1, tf=60)
        labels = {c.kwargs.get("label") for c in _price_marker_calls(mock_renderer)}
        assert "15_plong_n1_m1_x0p4" in labels

    def test_marker_y_uses_label_tf_low(self, mock_renderer):
        df = _make_df(labels=[("plong_n1_m1_x0p4", _ones(N))])
        period = (np.arange(N) // 60).astype(float)
        for col, off in (("open", 100), ("high", 105), ("low", 95),
                         ("close", 102), ("volume", 1000)):
            df[f"60_{col}"] = period + off
        v = _viewer(df, mock_renderer)
        v.build_window_figure("2024-01-01", 1, tf=60)
        call = next(
            c for c in _price_marker_calls(mock_renderer)
            if c.kwargs.get("label") == "15_plong_n1_m1_x0p4"
        )
        ys = np.array(call[0][2])
        # offsets are below the 15m lows (95..105), far from 60m period lows
        assert (ys < np.array(
            [df.loc[t, "15_low"] for t in call[0][1]]
        )).all()
