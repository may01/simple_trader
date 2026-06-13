"""indicators.library.classification — Stats-dependent classification fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

import functools
import json
import pickle
import os


@functools.lru_cache(maxsize=None)
def _load_rsi_classification(stats_path: str) -> dict:
    """Load rsi_classification.json from stats_path (cached by path)."""
    with open(stats_path, "r") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def _load_diff_stats(stats_path: str) -> dict:
    """Load diff_stats.pkl from stats_path (cached by path)."""
    with open(stats_path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Classification Group
# ---------------------------------------------------------------------------

def _get_tf_classification(tf: int) -> dict:
    """Return the rsi_classification entry for *tf*.

    TFs whose dataset history was too short have no entry (DataAttributes
    skips them instead of writing NaN); fall back to the nearest available
    lower TF, then the nearest higher one, so class fields always compute.
    """
    from helpers import stats_folder
    path = os.path.join(stats_folder(), "rsi_classification.json")
    data = _load_rsi_classification(path)
    if str(tf) in data:
        return data[str(tf)]
    available = sorted(int(k) for k in data)
    lower = [t for t in available if t < tf]
    candidates = lower[::-1] + [t for t in available if t > tf]
    if not candidates:
        raise KeyError(f"rsi_classification has no entries (tf={tf})")
    return data[str(candidates[0])]


def _five_tiers(x: pd.Series, mean: float, std: float, base: int) -> np.ndarray:
    """Spec §5.7 tier split around mean with ±0.5·std / ±std boundaries.

    Returns base+0 .. base+4 low→high; NaN inputs land in the middle tier.
    """
    conditions = [
        x > mean + std,
        x > mean + 0.5 * std,
        x >= mean - 0.5 * std,
        x >= mean - std,
    ]
    choices = [base + 4, base + 3, base + 2, base + 1]
    result = np.select(conditions, choices, default=base)
    return np.where(x.isna(), base + 2, result)


class MoveClassField(IndicatorField):
    """RSI momentum classification: rsi_ma8_diff vs diff_mean ± diff_std → -2..2."""

    name = "move_class"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8", "rsi_ma8_diff"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        tf_data = _get_tf_classification(tf)
        mean = float(tf_data["diff_mean"])
        std = float(tf_data["diff_std"])

        df = data_point.get_df(tf)
        diff = df[f"{tf}_rsi_ma8_diff"]

        result = _five_tiers(diff, mean, std, base=-2)
        return pd.Series(result, index=df.index, dtype=int)


class ZoneClassField(IndicatorField):
    """RSI level classification: rsi_ma8 vs mean ± std → 0..4."""

    name = "zone_class"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        tf_data = _get_tf_classification(tf)
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]

        result = _five_tiers(rsi, mean, std, base=0)
        return pd.Series(result, index=df.index, dtype=int)


class OverLowField(IndicatorField):
    """Boolean: RSI above lower threshold (mean - std)."""

    name = "over_low"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        from helpers import stats_folder
        path = os.path.join(stats_folder(), "rsi_classification.json")
        data = _load_rsi_classification(path)
        tf_data = data[str(tf)]
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]
        return rsi > (mean - std)


class OverHighField(IndicatorField):
    """Boolean: RSI above upper threshold (mean + std)."""

    name = "over_high"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        from helpers import stats_folder
        path = os.path.join(stats_folder(), "rsi_classification.json")
        data = _load_rsi_classification(path)
        tf_data = data[str(tf)]
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]
        return rsi > (mean + std)


# ---------------------------------------------------------------------------
# Targets Group
# ---------------------------------------------------------------------------

