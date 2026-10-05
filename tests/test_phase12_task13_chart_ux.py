"""Tests for chart UX: OHLC range row, volume bars, RSI class markers (Phase 12, Task 13)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.chart_renderer import ChartRenderer
from frontend.data_viewer import DataViewer, FullData


def _make_df(n: int = 60 * 24, extras=()) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    tf = 15
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
    }
    for name, vals in extras:
        data[f"{tf}_{name}"] = vals
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def mock_renderer():
    r = MagicMock()
    mock_fig = MagicMock()
    mock_fig._subplot_rows = {"price": 1, "volume": 2, "rsi": 3, "range": 4}
    r.create_figure.return_value = mock_fig
    return r


def _viewer(df, mock_renderer):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = mock_renderer
    return v


# ---------------------------------------------------------------------------
# ChartRenderer: dedicated range row
# ---------------------------------------------------------------------------

class TestRangeRow:
    def test_range_row_appended_last(self):
        fig = ChartRenderer().create_figure(["price", "volume"], range_row=True)
        assert fig._subplot_rows == {"price": 1, "volume": 2, "range": 3}

    def test_default_has_no_range_row(self):
        fig = ChartRenderer().create_figure(["price", "volume"])
        assert "range" not in fig._subplot_rows

    def test_rangeslider_only_on_range_row(self):
        fig = ChartRenderer().create_figure(["price", "volume"], range_row=True)
        fig.add_scatter(y=[1], row=1, col=1)
        fig.add_scatter(y=[1], row=2, col=1)
        fig.add_scatter(y=[1], row=3, col=1)
        sliders = {
            ax: fig.layout[ax].rangeslider.visible
            for ax in ("xaxis", "xaxis2", "xaxis3")
        }
        assert sliders == {"xaxis": False, "xaxis2": False, "xaxis3": True}

    def test_range_row_yaxis_hidden(self):
        fig = ChartRenderer().create_figure(["price", "volume"], range_row=True)
        assert fig.layout.yaxis3.visible is False

    def test_draw_candles_subplot_override(self):
        fig = ChartRenderer().create_figure(["price", "volume"], range_row=True)
        ChartRenderer().draw_candles(
            fig, ["2024-01-01"], [100.0], [105.0], [99.0], [102.0],
            subplot="range",
        )
        assert fig.data[0].xaxis == "x3"


# ---------------------------------------------------------------------------
# DataViewer: range row content + volume bars
# ---------------------------------------------------------------------------

class TestWindowFigureRangeAndVolume:
    def test_candles_drawn_on_price_and_range(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        subplots = [
            c.kwargs.get("subplot", "price")
            for c in mock_renderer.draw_candles.call_args_list
        ]
        assert sorted(subplots) == ["price", "range"]

    def test_create_figure_requests_range_row(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        assert mock_renderer.create_figure.call_args.kwargs.get("range_row") is True

    def test_no_indicators_on_range_row(self, mock_renderer):
        extras = [("rsi_14", np.linspace(40, 60, 60 * 24))]
        v = _viewer(_make_df(extras=extras), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        line_subplots = {c[0][1] for c in mock_renderer.draw_line.call_args_list}
        bar_subplots = {c[0][1] for c in mock_renderer.draw_bar.call_args_list}
        assert "range" not in line_subplots | bar_subplots

    def test_volume_drawn_as_bar(self, mock_renderer):
        v = _viewer(_make_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        bar_calls = [
            c for c in mock_renderer.draw_bar.call_args_list if c[0][1] == "volume"
        ]
        assert len(bar_calls) == 1
        line_volume = [
            c for c in mock_renderer.draw_line.call_args_list if c[0][1] == "volume"
        ]
        assert line_volume == []


# ---------------------------------------------------------------------------
# DataViewer: RSI class markers
# ---------------------------------------------------------------------------

def _class_df(n: int = 60 * 24):
    rsi_ma8 = np.linspace(40, 60, n)
    move = np.resize([-1, 0, 1, 2], n)
    zone = move + 1
    return _make_df(extras=[
        ("rsi_14", np.linspace(40, 60, n)),
        ("rsi_ma8", rsi_ma8),
        ("move_class", move),
        ("zone_class", zone),
    ])


class TestRsiClassMarkers:
    def test_markers_drawn_on_rsi_subplot(self, mock_renderer):
        v = _viewer(_class_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        marker_subplots = {
            c.kwargs.get("subplot", "price")
            for c in mock_renderer.draw_marker.call_args_list
        }
        assert "rsi" in marker_subplots

    def test_one_trace_per_class_value_and_field(self, mock_renderer):
        v = _viewer(_class_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        labels = {
            c.kwargs.get("label")
            for c in mock_renderer.draw_marker.call_args_list
            if c.kwargs.get("subplot") == "rsi"
        }
        assert {"move_class=-1", "move_class=0", "move_class=1",
                "move_class=2"} <= labels
        assert {"zone_class=0", "zone_class=1", "zone_class=2",
                "zone_class=3"} <= labels

    def test_marker_y_values_come_from_rsi_ma8(self, mock_renderer):
        v = _viewer(_class_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        calls = [
            c for c in mock_renderer.draw_marker.call_args_list
            if c.kwargs.get("subplot") == "rsi"
        ]
        df = v.full_data.df
        valid = set(np.round(df["15_rsi_ma8"].values, 6))
        for c in calls:
            ys = c[0][2]
            assert set(np.round(ys, 6)) <= valid

    def test_markers_drawn_per_minute_not_per_candle(self, mock_renderer):
        # _class_df cycles move_class over every 1m row; per-minute rendering
        # plots one marker per raw row (1440), not one per 15m candle (96).
        df = _class_df()
        v = _viewer(df, mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        move_times = sum(
            len(c[0][1])
            for c in mock_renderer.draw_marker.call_args_list
            if str(c.kwargs.get("label", "")).startswith("move_class=")
        )
        assert move_times == (df["15_move_class"].notna()).sum() == len(df)

    def test_skipped_without_class_columns(self, mock_renderer):
        extras = [("rsi_14", np.linspace(40, 60, 60 * 24)),
                  ("rsi_ma8", np.linspace(40, 60, 60 * 24))]
        v = _viewer(_make_df(extras=extras), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        rsi_markers = [
            c for c in mock_renderer.draw_marker.call_args_list
            if c.kwargs.get("subplot") == "rsi"
        ]
        assert rsi_markers == []

    def test_skipped_when_rsi_subplot_hidden(self, mock_renderer):
        mock_renderer.create_figure.return_value._subplot_rows = {
            "price": 1, "volume": 2, "range": 3
        }
        v = _viewer(_class_df(), mock_renderer)
        v.build_window_figure("2024-01-01", 1, subplots=[])
        rsi_markers = [
            c for c in mock_renderer.draw_marker.call_args_list
            if c.kwargs.get("subplot") == "rsi"
        ]
        assert rsi_markers == []


# ---------------------------------------------------------------------------
# task-05: new coexisting fields + side-aware tooltip
# ---------------------------------------------------------------------------

import frontend.data_viewer as dv_mod  # noqa: E402


def _class_df_new(n: int = 60 * 24):
    return _make_df(extras=[
        ("rsi_14", np.linspace(40, 60, n)),
        ("rsi_ma8", np.linspace(40, 60, n)),
        ("move_class", np.resize([-1, 0, 1, 2], n)),
        ("zone_class", np.resize([1, 2, 3, 4], n)),
        ("move_class_sym0", np.resize([-2, -1, 0, 1, 2], n)),
        ("zone_class_q", np.resize([0, 1, 2, 3, 4], n)),
    ])


FAKE_SIDE = {"15": {
    "zone_class_q": {"4": {"long_winrate": 0.55, "short_winrate": 0.44,
                           "long_lift": 0.05, "short_lift": -0.05,
                           "base_long": 0.5, "n": 10}},
    "move_class_sym0": {"-2": {"long_winrate": 0.60, "short_winrate": 0.39,
                              "long_lift": 0.10, "short_lift": -0.10,
                              "base_long": 0.5, "n": 10}},
}}


@pytest.fixture
def stub_side_stats(monkeypatch):
    import helpers
    monkeypatch.setattr(helpers, "stats_folder", lambda: "x/")
    monkeypatch.setattr(dv_mod, "_load_side_stats_file", lambda path: FAKE_SIDE)


@pytest.fixture
def no_side_stats(monkeypatch):
    import helpers
    monkeypatch.setattr(helpers, "stats_folder", lambda: "x/")
    monkeypatch.setattr(dv_mod, "_load_side_stats_file", lambda path: {})


def _rsi_labels(mock_renderer):
    return {
        c.kwargs.get("label")
        for c in mock_renderer.draw_marker.call_args_list
        if c.kwargs.get("subplot") == "rsi"
    }


class TestNewClassMarkersCoexist:
    def test_new_field_traces_drawn_beside_legacy(self, mock_renderer, no_side_stats):
        v = _viewer(_class_df_new(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        labels = _rsi_labels(mock_renderer)
        # legacy still present (A/B)
        assert {"zone_class=1", "move_class=0"} <= labels
        # new fields present
        assert {"zone_class_q=0", "zone_class_q=4"} <= labels
        assert {"move_class_sym0=-2", "move_class_sym0=2"} <= labels

    def test_side_tooltip_attached_when_stats_present(self, mock_renderer, stub_side_stats):
        v = _viewer(_class_df_new(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        zq4 = [c for c in mock_renderer.draw_marker.call_args_list
               if c.kwargs.get("label") == "zone_class_q=4"][0]
        ht = zq4.kwargs.get("hovertext")
        assert ht and "long 55%" in ht[0] and "short 44%" in ht[0]
        assert len(ht) == len(zq4[0][1])  # one tooltip per point

    def test_no_tooltip_when_side_stats_absent(self, mock_renderer, no_side_stats):
        v = _viewer(_class_df_new(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        new_calls = [c for c in mock_renderer.draw_marker.call_args_list
                     if str(c.kwargs.get("label", "")).startswith("zone_class_q=")]
        assert new_calls and all(c.kwargs.get("hovertext") is None for c in new_calls)

    def test_legacy_markers_have_no_tooltip(self, mock_renderer, stub_side_stats):
        v = _viewer(_class_df_new(), mock_renderer)
        v.build_window_figure("2024-01-01", 1)
        legacy = [c for c in mock_renderer.draw_marker.call_args_list
                  if str(c.kwargs.get("label", "")).startswith("zone_class=")]
        assert legacy and all(c.kwargs.get("hovertext") is None for c in legacy)

    def test_new_fields_skipped_when_columns_absent(self, mock_renderer, no_side_stats):
        v = _viewer(_class_df(), mock_renderer)  # legacy-only df
        v.build_window_figure("2024-01-01", 1)
        labels = _rsi_labels(mock_renderer)
        assert not any(str(l).startswith("zone_class_q=") for l in labels)
        assert {"zone_class=1"} <= labels  # legacy unaffected
