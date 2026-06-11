"""indicators.library.targets — Target / stop-loss fields."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField
from .classification import _load_diff_stats

def _get_tf_diff_stats(tf: int) -> dict:
    """Return per-tf entry from diff_stats.pkl (uses cached _load_diff_stats)."""
    from helpers import stats_folder
    path = os.path.join(stats_folder(), "diff_stats.pkl")
    data = _load_diff_stats(path)
    return data.get(str(tf), data.get(tf, {}))


class TgtLongField(IndicatorField):
    """Long target: close * (1 + mean_long_diff)."""

    name = "tgt_long"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_diff = float(stats.get("mean_long", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 + mean_diff)


class SLLongField(IndicatorField):
    """Long stop-loss: close * (1 - mean_long_sl)."""

    name = "sl_long"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_sl = float(stats.get("mean_long_sl", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 - mean_sl)


class TgtShortField(IndicatorField):
    """Short target: close * (1 - mean_short_diff)."""

    name = "tgt_short"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_diff = float(stats.get("mean_short", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 - mean_diff)


class SLShortField(IndicatorField):
    """Short stop-loss: close * (1 + mean_short_sl)."""

    name = "sl_short"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_sl = float(stats.get("mean_short_sl", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 + mean_sl)


class ZBField(IndicatorField):
    """Zone Buy classification from diff_stats."""

    name = "ZB"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        threshold = float(stats.get("zb_threshold", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return (close > threshold).astype(int)


class ZSField(IndicatorField):
    """Zone Sell classification from diff_stats."""

    name = "ZS"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        threshold = float(stats.get("zs_threshold", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return (close < threshold).astype(int)


# ---------------------------------------------------------------------------
# Trend Flags Group
# ---------------------------------------------------------------------------

