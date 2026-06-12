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
    """Long target: high[i-1] * (1 + (high_diff_prc_rm_6[i-1] - std_below[i-1])/100); diff_prc is percent."""

    name = "tgt_long"
    group = "targets"
    dependencies: list[str] = ["high_diff_prc_rm_6", "high_diff_prc_rm_6_std_below"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        prev_high = df[f"{tf}_high"].shift(1)
        rm = df[f"{tf}_high_diff_prc_rm_6"].shift(1)
        std = df[f"{tf}_high_diff_prc_rm_6_std_below"].shift(1)
        return prev_high * (1.0 + (rm - std) / 100.0)


class SLLongField(IndicatorField):
    """Long stop-loss: low[i-1] * (1 + (low_diff_prc_rm_6[i-1] - std_below[i-1])/100); diff_prc is percent."""

    name = "sl_long"
    group = "targets"
    dependencies: list[str] = ["low_diff_prc_rm_6", "low_diff_prc_rm_6_std_below"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        prev_low = df[f"{tf}_low"].shift(1)
        rm = df[f"{tf}_low_diff_prc_rm_6"].shift(1)
        std = df[f"{tf}_low_diff_prc_rm_6_std_below"].shift(1)
        return prev_low * (1.0 + (rm - std) / 100.0)


class TgtShortField(IndicatorField):
    """Short target: low[i-1] * (1 + (low_diff_prc_rm_6[i-1] + std_above[i-1])/100); diff_prc is percent."""

    name = "tgt_short"
    group = "targets"
    dependencies: list[str] = ["low_diff_prc_rm_6", "low_diff_prc_rm_6_std_above"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        prev_low = df[f"{tf}_low"].shift(1)
        rm = df[f"{tf}_low_diff_prc_rm_6"].shift(1)
        std = df[f"{tf}_low_diff_prc_rm_6_std_above"].shift(1)
        return prev_low * (1.0 + (rm + std) / 100.0)


class SLShortField(IndicatorField):
    """Short stop-loss: high[i-1] * (1 + (high_diff_prc_rm_6[i-1] + std_above[i-1])/100); diff_prc is percent."""

    name = "sl_short"
    group = "targets"
    dependencies: list[str] = ["high_diff_prc_rm_6", "high_diff_prc_rm_6_std_above"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        prev_high = df[f"{tf}_high"].shift(1)
        rm = df[f"{tf}_high_diff_prc_rm_6"].shift(1)
        std = df[f"{tf}_high_diff_prc_rm_6_std_above"].shift(1)
        return prev_high * (1.0 + (rm + std) / 100.0)


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

