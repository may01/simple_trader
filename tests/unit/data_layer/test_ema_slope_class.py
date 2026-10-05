"""Tests for ema_25_diff_prc / ema_25_slope_class (ev_line experiment, 2026-10).

ema_25_diff_prc is the candle-to-candle change of ema_25 in %;
ema_25_slope_class splits it into fall (-1) / neutral (0) / rise (1) at
mean ± EMA_SLOPE_X·std, with mean/std frozen per TF from the 2y closed
candles (no stats file).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import LiveDataPoint
from indicators import EmaDiffPrcField, EmaSlopeClassField, _FIELD_REGISTRY
from indicators.library.classification import (
    EMA_SLOPE_STATS, EMA_SLOPE_X, ema_slope_class,
)


def _dp(tf: int, **cols):
    n = len(next(iter(cols.values())))
    idx = pd.date_range("2025-01-01", periods=n, freq=f"{tf}min", tz="UTC")
    df = pd.DataFrame({f"{tf}_{k}": v for k, v in cols.items()}, index=idx)
    return LiveDataPoint({tf: df})


class TestEmaDiffPrc:
    def test_pct_change_vs_previous_candle(self):
        out = EmaDiffPrcField().compute(_dp(15, ema_25=[20.0, 20.2, 20.0]), 15)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1] == pytest.approx((20.2 - 20.0) / 20.0 * 100.0)
        assert out.iloc[2] == pytest.approx((20.0 - 20.2) / 20.2 * 100.0)

    def test_name_and_dependency_follow_length(self):
        f = EmaDiffPrcField(length=50)
        assert f.name == "ema_50_diff_prc"
        assert f.dependencies == ["ema_50"]

    def test_does_not_shadow_the_nn_ols_slope(self):
        from indicators import NNSlopeField
        assert isinstance(_FIELD_REGISTRY["ema_25_slope"](type("C", (), {"params": {}})()), NNSlopeField)


class TestEmaSlopeClass:
    def test_frozen_stats_cover_the_class_tfs(self):
        assert set(EMA_SLOPE_STATS) == {15, 60, 240}
        assert EMA_SLOPE_X == 0.5
        assert EMA_SLOPE_STATS[15] == pytest.approx((0.0018696074433414, 0.0885475476512941))

    @pytest.mark.parametrize("tf", [15, 60, 240])
    def test_three_classes_at_mean_pm_x_std(self, tf):
        mean, std = EMA_SLOPE_STATS[tf]
        lo, hi = mean - EMA_SLOPE_X * std, mean + EMA_SLOPE_X * std
        slopes = [lo - 1e-6, lo, mean, hi, hi + 1e-6]
        out = EmaSlopeClassField().compute(_dp(tf, ema_25_diff_prc=slopes), tf)
        assert list(out) == [-1, 0, 0, 0, 1]      # boundaries are neutral

    def test_nan_slope_is_neutral(self):
        out = EmaSlopeClassField().compute(_dp(15, ema_25_diff_prc=[np.nan, 1.0]), 15)
        assert list(out) == [0, 1]
        assert out.dtype == int

    def test_helper_matches_field(self):
        s = pd.Series([-1.0, 0.0, 1.0])
        assert list(ema_slope_class(s, 60)) == [-1, 0, 1]

    def test_declares_slope_dependency_and_no_stats_file(self):
        f = EmaSlopeClassField()
        assert f.dependencies == ["ema_25_diff_prc"]
        assert f.resource_dependencies == []
        assert f.applies_to == [15, 60, 240]


class TestWiring:
    def test_registry_and_config(self):
        from config_loader import load_indicators_config

        assert "ema_25_diff_prc" in _FIELD_REGISTRY
        assert "ema_25_slope_class" in _FIELD_REGISTRY
        cfg = {f.name: f for f in load_indicators_config()}
        names = list(cfg)
        assert names.index("ema_25") < names.index("ema_25_diff_prc") < names.index("ema_25_slope_class")
        assert cfg["ema_25_slope_class"].applies_to == [15, 60, 240]
