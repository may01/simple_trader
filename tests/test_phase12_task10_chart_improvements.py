"""Tests for chart improvements (Phase 12, Task 10).

ADX indicator, diff std bands, rangeslider placement/size, subplot resize.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from frontend.chart_renderer import ChartRenderer
from frontend.data_viewer import DataViewer, FullData, _indicator_subplot
from indicators.library.trend import ADXField
from indicators.library.price_derivatives import (
    CloseDiffPrcRMStdAboveField,
    CloseDiffPrcRMStdBelowField,
    HighDiffPrcRMStdAboveField,
    LowDiffPrcRMStdBelowField,
)


class _DataPoint:
    def __init__(self, df):
        self._df = df

    def get_df(self, tf):
        return self._df


# ---------------------------------------------------------------------------
# ADXField
# ---------------------------------------------------------------------------

class TestADXField:
    def _dp(self, n=100, tf=15):
        idx = pd.date_range("2024-01-01", periods=n, freq="15min")
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        return _DataPoint(pd.DataFrame({
            f"{tf}_high": close + rng.uniform(0.1, 1, n),
            f"{tf}_low": close - rng.uniform(0.1, 1, n),
            f"{tf}_close": close,
        }, index=idx))

    def test_name_carries_period(self):
        assert ADXField().name == "adx_14"
        assert ADXField(period=20).name == "adx_20"

    def test_group_is_trend(self):
        assert ADXField().group == "trend"

    def test_compute_returns_series_with_df_index(self):
        dp = self._dp()
        out = ADXField().compute(dp, 15)
        assert isinstance(out, pd.Series)
        assert out.index.equals(dp.get_df(15).index)

    def test_compute_values_in_adx_range(self):
        out = ADXField().compute(self._dp(), 15).dropna()
        assert len(out) > 0
        assert ((out >= 0) & (out <= 100)).all()

    def test_registered(self):
        from indicators.registry import _FIELD_REGISTRY
        assert "adx_14" in _FIELD_REGISTRY
        cfg = MagicMock(params={"period": 14})
        field = _FIELD_REGISTRY["adx_14"](cfg)
        assert field.name == "adx_14"


# ---------------------------------------------------------------------------
# Diff std band fields
# ---------------------------------------------------------------------------

class TestDiffStdBands:
    def _dp(self, n=60, tf=15):
        idx = pd.date_range("2024-01-01", periods=n, freq="15min")
        rng = np.random.default_rng(3)
        diff = pd.Series(rng.normal(0, 0.5, n), index=idx)
        rm = diff.rolling(20).mean()
        return _DataPoint(pd.DataFrame({
            f"{tf}_close_diff_prc": diff,
            f"{tf}_close_diff_prc_rm_20": rm,
            f"{tf}_high_diff_prc": diff,
            f"{tf}_high_diff_prc_rm_20": rm,
            f"{tf}_low_diff_prc": diff,
            f"{tf}_low_diff_prc_rm_20": rm,
        }, index=idx))

    def test_names(self):
        assert CloseDiffPrcRMStdAboveField().name == "close_diff_prc_rm_20_std_above"
        assert CloseDiffPrcRMStdBelowField().name == "close_diff_prc_rm_20_std_below"
        assert HighDiffPrcRMStdAboveField().name == "high_diff_prc_rm_20_std_above"
        assert LowDiffPrcRMStdBelowField().name == "low_diff_prc_rm_20_std_below"

    @staticmethod
    def _sided_std(diff: pd.Series, above: bool, window: int = 20) -> pd.Series:
        def inner(x):
            m = x.mean()
            sel = x[x > m] if above else x[x < m]
            if len(sel) == 0:
                return 0.0
            return float(np.std(sel, ddof=1)) if len(sel) > 1 else 0.0
        return diff.rolling(window).apply(inner, raw=True)

    def test_dependencies(self):
        f = CloseDiffPrcRMStdAboveField()
        assert f.dependencies == ["close_diff_prc"]

    def test_above_is_subset_std(self):
        """std_above = std of the window's diff_prc values above the window mean."""
        dp = self._dp()
        df = dp.get_df(15)
        out = CloseDiffPrcRMStdAboveField().compute(dp, 15)
        expected = self._sided_std(df["15_close_diff_prc"], above=True)
        pd.testing.assert_series_equal(out, expected, check_names=False)

    def test_below_is_subset_std(self):
        """std_below = std of the window's diff_prc values below the window mean."""
        dp = self._dp()
        df = dp.get_df(15)
        out = CloseDiffPrcRMStdBelowField().compute(dp, 15)
        expected = self._sided_std(df["15_close_diff_prc"], above=False)
        pd.testing.assert_series_equal(out, expected, check_names=False)

    def test_sided_stds_non_negative(self):
        dp = self._dp()
        above = CloseDiffPrcRMStdAboveField().compute(dp, 15).dropna()
        below = CloseDiffPrcRMStdBelowField().compute(dp, 15).dropna()
        assert (above >= 0).all()
        assert (below >= 0).all()

    def test_all_six_registered(self):
        from indicators.registry import _FIELD_REGISTRY
        for src in ("close", "high", "low"):
            for side in ("above", "below"):
                assert f"{src}_diff_prc_rm_20_std_{side}" in _FIELD_REGISTRY

    def test_config_declares_fields(self):
        import yaml
        cfg = yaml.safe_load(open("configs/indicators_config.yaml"))
        names = {f["name"] for f in cfg["fields"]}
        assert "adx_14" in names
        for src in ("close", "high", "low"):
            for side in ("above", "below"):
                assert f"{src}_diff_prc_rm_20_std_{side}" in names


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_adx_gets_own_subplot(self):
        assert _indicator_subplot("adx_14") == "adx"

    @pytest.mark.parametrize("src", ["close", "high", "low"])
    def test_std_bands_route_to_diff_subplot(self, src):
        assert _indicator_subplot(f"{src}_diff_prc_rm_20_std_above") == f"{src}_diff"
        assert _indicator_subplot(f"{src}_diff_prc_rm_20_std_below") == f"{src}_diff"

    @pytest.mark.parametrize("src", ["close", "high", "low"])
    def test_mean_bands_still_routable(self, src):
        assert _indicator_subplot(f"{src}_diff_prc_rm_20_mean_above") == f"{src}_diff"


# ---------------------------------------------------------------------------
# Rangeslider
# ---------------------------------------------------------------------------

class TestRangeslider:
    def test_disabled_on_price_axis(self):
        fig = ChartRenderer().create_figure(["price", "volume", "rsi"])
        assert fig.layout.xaxis.rangeslider.visible is False

    def test_enabled_slim_on_bottom_axis(self):
        fig = ChartRenderer().create_figure(["price", "volume", "rsi"])
        bottom = fig.layout.xaxis3.rangeslider
        assert bottom.visible is True
        assert bottom.thickness == 0.05

    def test_middle_axes_disabled(self):
        fig = ChartRenderer().create_figure(["price", "volume", "rsi"])
        assert fig.layout.xaxis2.rangeslider.visible is False

    def test_single_row_keeps_slim_slider(self):
        fig = ChartRenderer().create_figure(["price"])
        assert fig.layout.xaxis.rangeslider.visible is True
        assert fig.layout.xaxis.rangeslider.thickness == 0.05


# ---------------------------------------------------------------------------
# Row heights
# ---------------------------------------------------------------------------

class TestRowHeights:
    def test_non_price_rows_1_5x(self):
        n = 5
        fig = ChartRenderer().create_figure(["price", "volume", "rsi", "cci", "adx"])
        domains = sorted(
            (fig.layout[ax].domain for ax in ("yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5")),
            key=lambda d: d[0],
        )
        heights = [d[1] - d[0] for d in domains]
        price_h = max(heights)
        other_h = [h for h in heights if h != price_h]
        # Ratio price:other = 0.60 : 1.5*0.40/(n-1) = 0.60 : 0.15 = 4
        for h in other_h:
            assert price_h / h == pytest.approx(0.60 / (1.5 * 0.40 / (n - 1)), rel=0.05)

    def test_window_figure_height_scales_264_per_row(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        df = pd.DataFrame(
            {f"15_{c}": np.linspace(1, 2, 60)
             for c in ("open", "high", "low", "close", "volume", "rsi_14")},
            index=idx,
        )
        r = MagicMock()
        mock_fig = MagicMock()
        mock_fig._subplot_rows = {"price": 1, "volume": 2, "rsi": 3}
        r.create_figure.return_value = mock_fig
        v = DataViewer(FullData(df))
        v.renderer = r
        v.build_window_figure("2024-01-01", 1)
        heights = [
            c.kwargs.get("height")
            for c in mock_fig.update_layout.call_args_list
            if "height" in c.kwargs
        ]
        assert heights[-1] == max(600, 264 * 3)


# ---------------------------------------------------------------------------
# Window figure draws ADX + std bands
# ---------------------------------------------------------------------------

class TestWindowFigureContent:
    def test_adx_and_std_bands_drawn(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="1min")
        cols = ["open", "high", "low", "close", "volume", "adx_14",
                "close_diff_prc", "close_diff_prc_rm_20",
                "close_diff_prc_rm_20_std_above", "close_diff_prc_rm_20_std_below"]
        df = pd.DataFrame(
            {f"15_{c}": np.linspace(1, 2, 60) for c in cols}, index=idx
        )
        r = MagicMock()
        mock_fig = MagicMock()
        mock_fig._subplot_rows = {"price": 1, "volume": 2, "adx": 3, "close_diff": 4}
        r.create_figure.return_value = mock_fig
        v = DataViewer(FullData(df))
        v.renderer = r
        v.build_window_figure("2024-01-01", 1)
        pairs = [
            (c[0][1], c.kwargs.get("label"))
            for c in r.draw_line.call_args_list
        ]
        assert ("adx", "adx_14") in pairs
        assert ("close_diff", "close_diff_prc_rm_20_std_above") in pairs
        assert ("close_diff", "close_diff_prc_rm_20_std_below") in pairs
        # mean bands no longer drawn by default
        labels = [l for _s, l in pairs]
        assert "close_diff_prc_rm_20_mean_above" not in labels
