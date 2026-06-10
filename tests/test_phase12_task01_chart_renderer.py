"""Tests for ChartRenderer base class (Phase 12, Task 01)."""

import pytest
import plotly.graph_objects as go
from unittest.mock import patch

from frontend.chart_renderer import ChartRenderer


@pytest.fixture
def renderer():
    return ChartRenderer(title="Test Chart")


class TestCreateFigure:
    def test_returns_go_figure(self, renderer):
        fig = renderer.create_figure(["price"])
        assert isinstance(fig, go.Figure)

    def test_single_subplot_price_only(self, renderer):
        fig = renderer.create_figure(["price"])
        assert fig._subplot_rows == {"price": 1}

    def test_two_subplots_price_and_volume(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        assert fig._subplot_rows == {"price": 1, "volume": 2}

    def test_three_subplots_order_preserved(self, renderer):
        fig = renderer.create_figure(["price", "volume", "rsi"])
        assert fig._subplot_rows == {"price": 1, "volume": 2, "rsi": 3}

    def test_subplot_rows_is_plain_dict(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        assert type(fig._subplot_rows) is dict

    def test_title_stored_on_renderer(self, renderer):
        assert renderer.title == "Test Chart"

    def test_default_title_is_empty_string(self):
        cr = ChartRenderer()
        assert cr.title == ""


class TestDrawCandles:
    def test_adds_candlestick_trace(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        times = ["2024-01-01", "2024-01-02"]
        opens = [100.0, 101.0]
        highs = [105.0, 106.0]
        lows = [99.0, 100.0]
        closes = [102.0, 103.0]
        renderer.draw_candles(fig, times, opens, highs, lows, closes)
        candlestick_traces = [t for t in fig.data if isinstance(t, go.Candlestick)]
        assert len(candlestick_traces) == 1

    def test_candlestick_on_price_row(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        renderer.draw_candles(
            fig,
            ["2024-01-01"],
            [100.0],
            [105.0],
            [99.0],
            [102.0],
        )
        trace = next((t for t in fig.data if isinstance(t, go.Candlestick)), None)
        assert trace is not None, "expected Candlestick trace not found"
        # price is row 1, which maps to yaxis / yaxis1
        assert trace.yaxis == "y"

    def test_candlestick_has_green_red_colors(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_candles(fig, ["2024-01-01"], [100.0], [105.0], [99.0], [102.0])
        trace = next((t for t in fig.data if isinstance(t, go.Candlestick)), None)
        assert trace is not None, "expected Candlestick trace not found"
        assert trace.increasing.line.color is not None
        assert trace.decreasing.line.color is not None

    def test_draw_candles_returns_none(self, renderer):
        fig = renderer.create_figure(["price"])
        result = renderer.draw_candles(fig, [], [], [], [], [])
        assert result is None


class TestDrawLine:
    def test_adds_scatter_trace(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        renderer.draw_line(fig, "volume", ["2024-01-01"], [1000.0], label="Vol")
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1

    def test_scatter_mode_is_lines(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        renderer.draw_line(fig, "volume", ["2024-01-01"], [1000.0], label="Vol")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.mode == "lines"

    def test_label_set_as_name(self, renderer):
        fig = renderer.create_figure(["price", "rsi"])
        renderer.draw_line(fig, "rsi", ["2024-01-01"], [55.0], label="RSI14")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.name == "RSI14"

    def test_color_applied(self, renderer):
        fig = renderer.create_figure(["price", "rsi"])
        renderer.draw_line(fig, "rsi", ["2024-01-01"], [55.0], label="RSI", color="red")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.line.color == "red"

    def test_draw_line_on_price_row(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        renderer.draw_line(fig, "price", ["2024-01-01"], [102.0], label="SMA")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.yaxis == "y"

    def test_draw_line_on_second_row(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        renderer.draw_line(fig, "volume", ["2024-01-01"], [1000.0], label="Vol")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.yaxis == "y2"

    def test_draw_line_returns_none(self, renderer):
        fig = renderer.create_figure(["price"])
        result = renderer.draw_line(fig, "price", [], [], label="x")
        assert result is None


class TestDrawMarker:
    def test_adds_scatter_trace(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_marker(fig, ["2024-01-01"], [100.0], "triangle-up", "green", "Buy")
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1

    def test_scatter_mode_is_markers(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_marker(fig, ["2024-01-01"], [100.0], "triangle-up", "green")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.mode == "markers"

    def test_marker_on_price_row(self, renderer):
        fig = renderer.create_figure(["price", "volume"])
        renderer.draw_marker(fig, ["2024-01-01"], [100.0], "triangle-up", "green")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.yaxis == "y"

    def test_marker_symbol_set(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_marker(fig, ["2024-01-01"], [100.0], "circle", "blue")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.marker.symbol == "circle"

    def test_marker_color_set(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_marker(fig, ["2024-01-01"], [100.0], "square", "purple")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.marker.color == "purple"

    def test_label_set_as_name(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_marker(fig, ["2024-01-01"], [100.0], "x", "red", label="Sell")
        trace = next((t for t in fig.data if isinstance(t, go.Scatter)), None)
        assert trace is not None, "expected Scatter trace not found"
        assert trace.name == "Sell"

    def test_draw_marker_returns_none(self, renderer):
        fig = renderer.create_figure(["price"])
        result = renderer.draw_marker(fig, [], [], "x", "red")
        assert result is None


class TestDrawLevel:
    def test_adds_hline_shape(self, renderer):
        fig = renderer.create_figure(["price"])
        before = len(fig.layout.shapes)
        renderer.draw_level(fig, price=50000.0, label="Support", color="green")
        assert len(fig.layout.shapes) == before + 1

    def test_hline_y_value(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_level(fig, price=42000.0, label="Level")
        shape = fig.layout.shapes[-1]
        assert shape.y0 == 42000.0
        assert shape.y1 == 42000.0

    def test_default_color_is_gray(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_level(fig, price=100.0, label="Line")
        shape = fig.layout.shapes[-1]
        assert shape.line.color == "gray"

    def test_custom_color(self, renderer):
        fig = renderer.create_figure(["price"])
        renderer.draw_level(fig, price=100.0, label="Line", color="red")
        shape = fig.layout.shapes[-1]
        assert shape.line.color == "red"

    def test_draw_level_returns_none(self, renderer):
        fig = renderer.create_figure(["price"])
        result = renderer.draw_level(fig, price=100.0, label="x")
        assert result is None


class TestSave:
    def test_save_calls_write_image_with_path(self, renderer):
        fig = renderer.create_figure(["price"])
        with patch.object(fig, "write_image") as mock_write:
            renderer.save(fig, "out.png")
            mock_write.assert_called_once_with("out.png")
