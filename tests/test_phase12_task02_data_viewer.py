"""Tests for DataViewer (Phase 12, Task 02)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

from frontend.data_viewer import DataViewer, FullData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 10, tf: int = 15) -> pd.DataFrame:
    """Build a minimal wide DataFrame with columns for the given TF."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
        f"{tf}_rsi_14": np.linspace(40, 60, n),
        f"{tf}_cci_14": np.linspace(-100, 100, n),
        f"{tf}_ema_20": np.linspace(101, 111, n),
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
# FullData
# ---------------------------------------------------------------------------

class TestFullData:
    def test_stores_df(self, df):
        fd = FullData(df)
        assert fd.df is df

    def test_df_not_modified(self, df):
        original_cols = list(df.columns)
        fd = FullData(df)
        assert list(fd.df.columns) == original_cols


# ---------------------------------------------------------------------------
# DataViewer construction
# ---------------------------------------------------------------------------

class TestDataViewerInit:
    def test_stores_full_data(self, full_data):
        v = DataViewer(full_data)
        assert v.full_data is full_data

    def test_default_tf_is_15(self, full_data):
        v = DataViewer(full_data)
        assert v.tf == 15

    def test_custom_tf(self, full_data):
        v = DataViewer(full_data, tf=60)
        assert v.tf == 60

    def test_has_renderer(self, full_data):
        v = DataViewer(full_data)
        from frontend.chart_renderer import ChartRenderer
        assert isinstance(v.renderer, ChartRenderer)


# ---------------------------------------------------------------------------
# view_full
# ---------------------------------------------------------------------------

class TestViewFull:
    def test_create_figure_called_with_price_volume_and_indicator_subplots(
        self, viewer, mock_renderer
    ):
        viewer.view_full()
        mock_renderer.create_figure.assert_called_once()
        args = mock_renderer.create_figure.call_args[0][0]
        assert "price" in args
        assert "volume" in args
        # Default indicators RSI and CCI get own subplots
        assert "rsi" in args
        assert "cci" in args

    def test_create_figure_subplot_order_price_first(self, viewer, mock_renderer):
        viewer.view_full()
        args = mock_renderer.create_figure.call_args[0][0]
        assert args[0] == "price"

    def test_draw_candles_called(self, viewer, mock_renderer):
        viewer.view_full()
        mock_renderer.draw_candles.assert_called_once()

    def test_draw_candles_uses_tf_prefixed_columns(self, viewer, mock_renderer, df):
        viewer.view_full()
        call_kwargs = mock_renderer.draw_candles.call_args
        # Extract positional args: (fig, times, opens, highs, lows, closes)
        args = call_kwargs[0]
        fig, times, opens, highs, lows, closes = args
        assert list(opens) == pytest.approx(list(df["15_open"]), rel=1e-5)
        assert list(closes) == pytest.approx(list(df["15_close"]), rel=1e-5)

    def test_draw_candles_uses_slice(self, viewer, mock_renderer, df):
        viewer.view_full(start_idx=2, end_idx=5)
        args = mock_renderer.draw_candles.call_args[0]
        _fig, times, opens, highs, lows, closes = args
        assert list(opens) == pytest.approx(list(df["15_open"].iloc[2:5]), rel=1e-5)

    def test_draw_line_called_for_rsi(self, viewer, mock_renderer):
        viewer.view_full()
        call_names = [c[0][1] for c in mock_renderer.draw_line.call_args_list]
        assert "rsi" in call_names

    def test_draw_line_called_for_cci(self, viewer, mock_renderer):
        viewer.view_full()
        call_names = [c[0][1] for c in mock_renderer.draw_line.call_args_list]
        assert "cci" in call_names

    def test_draw_line_rsi_uses_tf_prefixed_column(self, viewer, mock_renderer, df):
        viewer.view_full()
        rsi_calls = [
            c for c in mock_renderer.draw_line.call_args_list
            if c[0][1] == "rsi"
        ]
        assert len(rsi_calls) == 1
        values = rsi_calls[0][0][3]  # (fig, subplot, times, values, ...)
        assert list(values) == pytest.approx(list(df["15_rsi_14"]), rel=1e-5)

    def test_draw_line_cci_uses_tf_prefixed_column(self, viewer, mock_renderer, df):
        viewer.view_full()
        cci_calls = [
            c for c in mock_renderer.draw_line.call_args_list
            if c[0][1] == "cci"
        ]
        assert len(cci_calls) == 1
        values = cci_calls[0][0][3]
        assert list(values) == pytest.approx(list(df["15_cci_14"]), rel=1e-5)

    def test_fig_show_called(self, viewer, mock_renderer):
        viewer.view_full()
        fig = mock_renderer.create_figure.return_value
        fig.show.assert_called_once()

    def test_default_indicators_are_rsi_14_and_cci_14(self, viewer, mock_renderer):
        """When indicators=None, defaults to rsi_14 and cci_14."""
        viewer.view_full(indicators=None)
        call_subplots = [c[0][1] for c in mock_renderer.draw_line.call_args_list]
        # Both RSI and CCI subplots should have been drawn
        assert "rsi" in call_subplots
        assert "cci" in call_subplots

    def test_ema_drawn_on_price_subplot(self, viewer, mock_renderer):
        """EMAs are drawn on the price axis, not a separate subplot."""
        viewer.view_full(indicators=["rsi_14", "ema_20"])
        # Mock fig needs price in subplot_rows
        price_calls = [
            c for c in mock_renderer.draw_line.call_args_list
            if c[0][1] == "price"
        ]
        assert len(price_calls) == 1

    def test_ema_not_added_as_own_subplot(self, viewer, mock_renderer):
        """EMA should not get its own subplot entry."""
        viewer.view_full(indicators=["rsi_14", "ema_20"])
        subplots_arg = mock_renderer.create_figure.call_args[0][0]
        assert "ema" not in subplots_arg
        assert "ema_20" not in subplots_arg

    def test_end_idx_none_uses_full_slice(self, viewer, mock_renderer, df):
        viewer.view_full(start_idx=0, end_idx=None)
        args = mock_renderer.draw_candles.call_args[0]
        _fig, times, opens, highs, lows, closes = args
        assert len(opens) == len(df)

    def test_partial_slice(self, viewer, mock_renderer, df):
        viewer.view_full(start_idx=3, end_idx=7)
        args = mock_renderer.draw_candles.call_args[0]
        _fig, times, opens, highs, lows, closes = args
        assert len(opens) == 4  # rows 3,4,5,6


# ---------------------------------------------------------------------------
# view_full_levels
# ---------------------------------------------------------------------------

class TestViewFullLevels:
    def _make_levels_mock(self, active_levels_map: dict) -> MagicMock:
        """Create a mock Levels object.

        active_levels_map: {level_type: [(price, label), ...]}
        """
        mock_levels = MagicMock()

        def _get_active(level_type, cur_time):
            return active_levels_map.get(level_type, [])

        mock_levels.get_active_levels.side_effect = _get_active
        return mock_levels

    def test_levels_none_behaves_like_view_full(self, viewer, mock_renderer):
        viewer.view_full_levels(levels=None)
        mock_renderer.draw_level.assert_not_called()
        mock_renderer.draw_candles.assert_called_once()

    def test_draw_level_called_for_each_active_level(self, viewer, mock_renderer):
        from constants import LEVEL_TYPE_LONG_SUPPORT
        levels = self._make_levels_mock({
            LEVEL_TYPE_LONG_SUPPORT: [(50000.0, "sup1"), (51000.0, "sup2")],
        })
        viewer.view_full_levels(levels=levels)
        assert mock_renderer.draw_level.call_count == 2

    def test_draw_level_uses_last_row_timestamp(self, viewer, mock_renderer, df):
        """cur_time passed to get_active_levels is the Unix timestamp of the last slice row."""
        from constants import LEVEL_TYPE_LONG_SUPPORT
        captured = {}

        def _get_active(level_type, cur_time):
            captured["cur_time"] = cur_time
            return []

        mock_levels = MagicMock()
        mock_levels.get_active_levels.side_effect = _get_active

        viewer.view_full_levels(levels=mock_levels)

        # Last row's Unix timestamp (seconds)
        expected_ts = int(df.index[-1].timestamp())
        assert captured.get("cur_time") == expected_ts

    def test_draw_level_called_with_price_and_label(self, viewer, mock_renderer):
        from constants import LEVEL_TYPE_LONG_SUPPORT
        levels = self._make_levels_mock({
            LEVEL_TYPE_LONG_SUPPORT: [(42000.0, "Support")],
        })
        viewer.view_full_levels(levels=levels)
        call_args = mock_renderer.draw_level.call_args_list[0]
        # draw_level(fig, price, label, ...)
        _fig, price, label = call_args[0][:3]
        assert price == 42000.0
        assert label == "Support"

    def test_all_level_types_queried(self, viewer, mock_renderer):
        """get_active_levels is called for every LEVEL_TYPE_* constant."""
        import constants
        level_types = [
            v for k, v in vars(constants).items()
            if k.startswith("LEVEL_TYPE_")
        ]
        mock_levels = MagicMock()
        mock_levels.get_active_levels.return_value = []

        viewer.view_full_levels(levels=mock_levels)

        queried_types = {c[0][0] for c in mock_levels.get_active_levels.call_args_list}
        for lt in level_types:
            assert lt in queried_types

    def test_fig_show_called(self, viewer, mock_renderer):
        viewer.view_full_levels(levels=None)
        fig = mock_renderer.create_figure.return_value
        fig.show.assert_called_once()


# ---------------------------------------------------------------------------
# save_chart
# ---------------------------------------------------------------------------

class TestSaveChart:
    def test_renderer_save_called(self, viewer, mock_renderer):
        viewer.save_chart("out.png")
        mock_renderer.save.assert_called_once()

    def test_renderer_save_called_with_path(self, viewer, mock_renderer):
        viewer.save_chart("/tmp/chart.png")
        call_args = mock_renderer.save.call_args[0]
        assert call_args[1] == "/tmp/chart.png"

    def test_fig_show_not_called(self, viewer, mock_renderer):
        viewer.save_chart("out.png")
        fig = mock_renderer.create_figure.return_value
        fig.show.assert_not_called()

    def test_draw_candles_still_called(self, viewer, mock_renderer):
        viewer.save_chart("out.png")
        mock_renderer.draw_candles.assert_called_once()

    def test_slice_applied(self, viewer, mock_renderer, df):
        viewer.save_chart("out.png", start_idx=1, end_idx=4)
        args = mock_renderer.draw_candles.call_args[0]
        _fig, times, opens, highs, lows, closes = args
        assert len(opens) == 3  # rows 1,2,3
