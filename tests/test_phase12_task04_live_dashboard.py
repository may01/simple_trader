"""Tests for LiveDashboard (Phase 12, Task 04)."""

from __future__ import annotations

import pickle
import os
import tempfile
from collections import deque

import pytest
import plotly.graph_objects as go

from frontend.live_dashboard import LiveDashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dashboard():
    return LiveDashboard(window_size=100, tf=15)


def _make_candles(n: int) -> list[dict]:
    """Return *n* minimal OHLCV candle dicts."""
    return [
        {
            "time": i,
            "open": 100.0 + i,
            "high": 102.0 + i,
            "low": 99.0 + i,
            "close": 101.0 + i,
            "volume": 1000.0,
        }
        for i in range(n)
    ]


@pytest.fixture
def base_state():
    return {
        "candles": _make_candles(10),
        "position": {"is_open": False, "direction": "flat", "stop_loss": None},
        "trade_markers": [],
        "revenue_history": [],
    }


@pytest.fixture
def long_state():
    return {
        "candles": _make_candles(10),
        "position": {"is_open": True, "direction": "long", "stop_loss": 98.5},
        "trade_markers": [],
        "revenue_history": [],
    }


@pytest.fixture
def short_state():
    return {
        "candles": _make_candles(10),
        "position": {"is_open": True, "direction": "short", "stop_loss": 105.0},
        "trade_markers": [],
        "revenue_history": [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_traces(figs: list[go.Figure]) -> list:
    return [t for fig in figs for t in fig.data]


def _all_shapes(figs: list[go.Figure]) -> list:
    return [s for fig in figs for s in (fig.layout.shapes or [])]


def _all_vrects(figs: list[go.Figure]) -> list:
    """vrects are stored as shapes with x0/x1 and xref='paper'."""
    return [s for s in _all_shapes(figs) if s.type == "rect"]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_attrs(self):
        ld = LiveDashboard()
        assert ld.window_size == 200
        assert ld.tf == 15
        assert isinstance(ld.candle_buffer, deque)
        assert isinstance(ld.trade_markers, list)

    def test_custom_attrs(self):
        ld = LiveDashboard(window_size=50, tf=5)
        assert ld.window_size == 50
        assert ld.tf == 5

    def test_candle_buffer_maxlen(self):
        ld = LiveDashboard(window_size=77)
        assert ld.candle_buffer.maxlen == 77

    def test_renderer_present(self):
        from frontend.chart_renderer import ChartRenderer
        ld = LiveDashboard()
        assert isinstance(ld.renderer, ChartRenderer)


# ---------------------------------------------------------------------------
# _update_charts — candlestick
# ---------------------------------------------------------------------------


class TestUpdateChartsCandles:
    def test_returns_list_of_figures(self, dashboard, base_state):
        figs = dashboard._update_charts(base_state)
        assert isinstance(figs, list)
        assert len(figs) > 0
        assert all(isinstance(f, go.Figure) for f in figs)

    def test_candlestick_trace_present(self, dashboard, base_state):
        figs = dashboard._update_charts(base_state)
        traces = _all_traces(figs)
        candlestick_traces = [t for t in traces if isinstance(t, go.Candlestick)]
        assert len(candlestick_traces) >= 1, "Expected at least one Candlestick trace"

    def test_candlestick_has_correct_data(self, dashboard, base_state):
        figs = dashboard._update_charts(base_state)
        traces = _all_traces(figs)
        cs = next((t for t in traces if isinstance(t, go.Candlestick)), None)
        assert cs is not None
        assert len(cs.open) == 10
        assert list(cs.open) == [100.0 + i for i in range(10)]

    def test_empty_candles_no_crash(self, dashboard):
        state = {
            "candles": [],
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)


# ---------------------------------------------------------------------------
# _update_charts — window trimming
# ---------------------------------------------------------------------------


class TestWindowTrimming:
    def test_trims_candles_to_window_size(self):
        ld = LiveDashboard(window_size=5)
        state = {
            "candles": _make_candles(20),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        figs = ld._update_charts(state)
        traces = _all_traces(figs)
        cs = next((t for t in traces if isinstance(t, go.Candlestick)), None)
        assert cs is not None
        assert len(cs.open) == 5, f"Expected 5 candles (window_size=5), got {len(cs.open)}"

    def test_fewer_candles_than_window_ok(self):
        ld = LiveDashboard(window_size=100)
        state = {
            "candles": _make_candles(3),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        figs = ld._update_charts(state)
        traces = _all_traces(figs)
        cs = next((t for t in traces if isinstance(t, go.Candlestick)), None)
        assert cs is not None
        assert len(cs.open) == 3

    def test_candle_buffer_updated(self):
        ld = LiveDashboard(window_size=5)
        state = {
            "candles": _make_candles(20),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        ld._update_charts(state)
        assert len(ld.candle_buffer) == 5


# ---------------------------------------------------------------------------
# _update_charts — position overlay (vrect)
# ---------------------------------------------------------------------------


class TestPositionOverlay:
    def test_long_position_adds_vrect(self, dashboard, long_state):
        figs = dashboard._update_charts(long_state)
        vrects = _all_vrects(figs)
        assert len(vrects) >= 1, "Expected at least one vrect for open long position"

    def test_long_position_vrect_is_green(self, dashboard, long_state):
        figs = dashboard._update_charts(long_state)
        vrects = _all_vrects(figs)
        assert len(vrects) >= 1
        fillcolor = vrects[0].fillcolor or ""
        assert "green" in fillcolor.lower() or "rgba(0,128,0" in fillcolor.lower() or "rgba(0, 128, 0" in fillcolor.lower() or "rgba(0,255,0" in fillcolor.lower(), (
            f"Expected green vrect for long position, got fillcolor={fillcolor!r}"
        )

    def test_short_position_adds_vrect(self, dashboard, short_state):
        figs = dashboard._update_charts(short_state)
        vrects = _all_vrects(figs)
        assert len(vrects) >= 1, "Expected at least one vrect for open short position"

    def test_short_position_vrect_is_red(self, dashboard, short_state):
        figs = dashboard._update_charts(short_state)
        vrects = _all_vrects(figs)
        assert len(vrects) >= 1
        fillcolor = vrects[0].fillcolor or ""
        assert "red" in fillcolor.lower() or "rgba(255,0,0" in fillcolor.lower() or "rgba(255, 0, 0" in fillcolor.lower(), (
            f"Expected red vrect for short position, got fillcolor={fillcolor!r}"
        )

    def test_no_position_vrect_gray_or_absent(self, dashboard, base_state):
        """Flat position: vrect absent or gray."""
        figs = dashboard._update_charts(base_state)
        vrects = _all_vrects(figs)
        # Acceptable: no vrect at all, or gray vrect
        if vrects:
            fillcolor = vrects[0].fillcolor or ""
            assert "gray" in fillcolor.lower() or "grey" in fillcolor.lower() or "rgba(128,128,128" in fillcolor.lower(), (
                f"Flat position vrect should be gray or absent, got {fillcolor!r}"
            )


# ---------------------------------------------------------------------------
# _update_charts — stop-loss level
# ---------------------------------------------------------------------------


class TestStopLossLevel:
    def test_stop_loss_line_present_when_position_open(self, dashboard, long_state):
        figs = dashboard._update_charts(long_state)
        shapes = _all_shapes(figs)
        lines = [s for s in shapes if s.type == "line"]
        assert len(lines) >= 1, "Expected at least one horizontal line for stop-loss"

    def test_stop_loss_line_at_correct_price(self, dashboard, long_state):
        figs = dashboard._update_charts(long_state)
        shapes = _all_shapes(figs)
        lines = [s for s in shapes if s.type == "line"]
        assert any(s.y0 == 98.5 and s.y1 == 98.5 for s in lines), (
            f"Expected stop-loss line at y=98.5, got lines: {[(s.y0, s.y1) for s in lines]}"
        )

    def test_stop_loss_absent_when_position_closed(self, dashboard, base_state):
        figs = dashboard._update_charts(base_state)
        shapes = _all_shapes(figs)
        lines = [s for s in shapes if s.type == "line"]
        assert len(lines) == 0, (
            f"Expected no stop-loss line when position closed, got: {lines}"
        )

    def test_stop_loss_absent_when_stop_loss_none(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": True, "direction": "long", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        shapes = _all_shapes(figs)
        lines = [s for s in shapes if s.type == "line"]
        assert len(lines) == 0, "No stop-loss line when stop_loss=None"


# ---------------------------------------------------------------------------
# _update_charts — trade markers
# ---------------------------------------------------------------------------


class TestTradeMarkers:
    def _get_marker_traces(self, figs: list[go.Figure]) -> list[go.Scatter]:
        return [
            t for t in _all_traces(figs)
            if isinstance(t, go.Scatter) and t.mode and "markers" in t.mode
        ]

    def test_buy_marker_triangle_up(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [{"type": "buy", "time": 2, "price": 101.0}],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        marker_traces = self._get_marker_traces(figs)
        symbols = [t.marker.symbol for t in marker_traces if t.marker]
        assert any("triangle" in str(s).lower() and "up" in str(s).lower() for s in symbols), (
            f"Expected triangle-up symbol for buy marker, got: {symbols}"
        )

    def test_buy_marker_color_green(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [{"type": "buy", "time": 2, "price": 101.0}],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        marker_traces = self._get_marker_traces(figs)
        colors = [t.marker.color for t in marker_traces if t.marker]
        assert any("green" in str(c).lower() for c in colors), (
            f"Expected green marker for buy, got: {colors}"
        )

    def test_sell_marker_triangle_down(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [{"type": "sell", "time": 3, "price": 105.0}],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        marker_traces = self._get_marker_traces(figs)
        symbols = [t.marker.symbol for t in marker_traces if t.marker]
        assert any("triangle" in str(s).lower() and "down" in str(s).lower() for s in symbols), (
            f"Expected triangle-down symbol for sell marker, got: {symbols}"
        )

    def test_sell_marker_color_red(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [{"type": "sell", "time": 3, "price": 105.0}],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        marker_traces = self._get_marker_traces(figs)
        colors = [t.marker.color for t in marker_traces if t.marker]
        assert any("red" in str(c).lower() for c in colors), (
            f"Expected red marker for sell, got: {colors}"
        )

    def test_stop_loss_marker_orange_x(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [{"type": "stop_loss", "time": 4, "price": 99.0}],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        marker_traces = self._get_marker_traces(figs)
        symbols = [t.marker.symbol for t in marker_traces if t.marker]
        colors = [t.marker.color for t in marker_traces if t.marker]
        assert any("x" in str(s).lower() for s in symbols), (
            f"Expected x symbol for stop_loss marker, got: {symbols}"
        )
        assert any("orange" in str(c).lower() for c in colors), (
            f"Expected orange color for stop_loss marker, got: {colors}"
        )

    def test_multiple_markers(self, dashboard):
        state = {
            "candles": _make_candles(10),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [
                {"type": "buy", "time": 1, "price": 100.0},
                {"type": "sell", "time": 5, "price": 105.0},
                {"type": "stop_loss", "time": 3, "price": 99.0},
            ],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        marker_traces = self._get_marker_traces(figs)
        assert len(marker_traces) >= 3, (
            f"Expected at least 3 marker traces, got {len(marker_traces)}"
        )

    def test_no_markers_no_crash(self, dashboard, base_state):
        figs = dashboard._update_charts(base_state)
        assert isinstance(figs, list)


# ---------------------------------------------------------------------------
# _update_charts — revenue / P&L
# ---------------------------------------------------------------------------


class TestRevenueFigure:
    def test_revenue_history_produces_figure(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [(1.5, 100.0), (-0.5, -30.0), (2.0, 150.0)],
        }
        figs = dashboard._update_charts(state)
        # At least a second figure for P&L
        assert len(figs) >= 2

    def test_empty_revenue_no_crash(self, dashboard, base_state):
        figs = dashboard._update_charts(base_state)
        assert isinstance(figs, list)


# ---------------------------------------------------------------------------
# Edge cases / robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_state_dict_no_crash(self, dashboard):
        figs = dashboard._update_charts({})
        assert isinstance(figs, list)
        assert all(isinstance(f, go.Figure) for f in figs)

    def test_missing_candles_key_no_crash(self, dashboard):
        state = {
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_missing_position_key_no_crash(self, dashboard):
        state = {
            "candles": _make_candles(5),
            "trade_markers": [],
            "revenue_history": [],
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_missing_state_file_returns_empty_figures(self, dashboard):
        """_load_state on a non-existent file must return {}."""
        dashboard._state_path = "/tmp/nonexistent_live_state_xyz.pkl"
        state = dashboard._load_state()
        assert state == {}

    def test_corrupt_state_file_no_crash(self, dashboard, tmp_path):
        bad_file = tmp_path / "bad.pkl"
        bad_file.write_bytes(b"not a valid pickle")
        dashboard._state_path = str(bad_file)
        state = dashboard._load_state()
        assert state == {}

    def test_non_dict_state_file_returns_empty(self, dashboard, tmp_path):
        state_file = tmp_path / "state.pkl"
        with open(state_file, "wb") as fh:
            pickle.dump([1, 2, 3], fh)
        dashboard._state_path = str(state_file)
        state = dashboard._load_state()
        assert state == {}

    def test_valid_state_file_loads_correctly(self, dashboard, tmp_path):
        state_file = tmp_path / "state.pkl"
        expected = {
            "candles": _make_candles(3),
            "position": {"is_open": False, "direction": "flat", "stop_loss": None},
            "trade_markers": [],
            "revenue_history": [],
        }
        with open(state_file, "wb") as fh:
            pickle.dump(expected, fh)
        dashboard._state_path = str(state_file)
        state = dashboard._load_state()
        assert state == expected
