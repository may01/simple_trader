"""indicators.library.price_derivatives — Close/high/low percentage-diff family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

class CloseDiffPrcField(IndicatorField):
    """Percentage change of close: (close - close.shift(1)) / close.shift(1) * 100."""

    name = "close_diff_prc"
    group = "price_derivatives"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return (close - close.shift(1)) / close.shift(1) * 100


class CloseDiffPrcRMField(IndicatorField):
    """20-period rolling mean of close_diff_prc."""

    name = "close_diff_prc_rm"
    group = "price_derivatives"
    dependencies: list[str] = ["close_diff_prc"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_close_diff_prc"].rolling(20).mean()


class CloseDiffPrcRMMeanAboveField(IndicatorField):
    """Rolling mean of positive values in a 20-window of close_diff_prc_rm."""

    name = "close_diff_prc_rm_mean_above"
    group = "price_derivatives"
    dependencies: list[str] = ["close_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_close_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
            raw=True,
        )


class CloseDiffPrcRMMeanBelowField(IndicatorField):
    """Rolling mean of negative values in a 20-window of close_diff_prc_rm."""

    name = "close_diff_prc_rm_mean_below"
    group = "price_derivatives"
    dependencies: list[str] = ["close_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_close_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0,
            raw=True,
        )


class HighDiffPrcField(IndicatorField):
    """Percentage change of high."""

    name = "high_diff_prc"
    group = "price_derivatives"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"]
        return (high - high.shift(1)) / high.shift(1) * 100


class HighDiffPrcRMField(IndicatorField):
    """20-period rolling mean of high_diff_prc."""

    name = "high_diff_prc_rm"
    group = "price_derivatives"
    dependencies: list[str] = ["high_diff_prc"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_high_diff_prc"].rolling(20).mean()


class HighDiffPrcRMMeanAboveField(IndicatorField):
    """Rolling mean of positive values in 20-window of high_diff_prc_rm."""

    name = "high_diff_prc_rm_mean_above"
    group = "price_derivatives"
    dependencies: list[str] = ["high_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_high_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
            raw=True,
        )


class HighDiffPrcRMMeanBelowField(IndicatorField):
    """Rolling mean of negative values in 20-window of high_diff_prc_rm."""

    name = "high_diff_prc_rm_mean_below"
    group = "price_derivatives"
    dependencies: list[str] = ["high_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_high_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0,
            raw=True,
        )


class LowDiffPrcField(IndicatorField):
    """Percentage change of low."""

    name = "low_diff_prc"
    group = "price_derivatives"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        low = df[f"{tf}_low"]
        return (low - low.shift(1)) / low.shift(1) * 100


class LowDiffPrcRMField(IndicatorField):
    """20-period rolling mean of low_diff_prc."""

    name = "low_diff_prc_rm"
    group = "price_derivatives"
    dependencies: list[str] = ["low_diff_prc"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_low_diff_prc"].rolling(20).mean()


class LowDiffPrcRMMeanAboveField(IndicatorField):
    """Rolling mean of positive values in 20-window of low_diff_prc_rm."""

    name = "low_diff_prc_rm_mean_above"
    group = "price_derivatives"
    dependencies: list[str] = ["low_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_low_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
            raw=True,
        )


class LowDiffPrcRMMeanBelowField(IndicatorField):
    """Rolling mean of negative values in 20-window of low_diff_prc_rm."""

    name = "low_diff_prc_rm_mean_below"
    group = "price_derivatives"
    dependencies: list[str] = ["low_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_low_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0,
            raw=True,
        )


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
