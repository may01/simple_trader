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


# Frozen 2y-closed-row sym0 cuts for move_class: 0 ± {0.3, 1.0, 2.0}·diff_std
# per TF, taken verbatim from the RSI parameters-selection experiment
# (external spec 2026-08-06-rsi-parameters-selection-design.md; winner cell
# window=8/sym0/7-class). Deliberately NOT refit from rsi_classification.json:
# its diff_std entries are stale smoke-set artifacts (e.g. TF15 1.27 vs the
# real 2y closed-row 1.364). No 1440 fit exists — 1440 is out of applies_to.
MOVE_CLASS_CUTS = {
    15: (-2.728564, -1.364282, -0.409285, 0.409285, 1.364282, 2.728564),
    60: (-2.819219, -1.409609, -0.422883, 0.422883, 1.409609, 2.819219),
    240: (-2.919142, -1.459571, -0.437871, 0.437871, 1.459571, 2.919142),
}


def _move_cuts(tf: int) -> tuple:
    """Cuts for *tf*, nearest-lower-then-higher fallback (mirrors
    _get_tf_classification so direct compute() calls on unfitted TFs work)."""
    if tf in MOVE_CLASS_CUTS:
        return MOVE_CLASS_CUTS[tf]
    available = sorted(MOVE_CLASS_CUTS)
    lower = [t for t in available if t < tf]
    candidates = lower[::-1] + [t for t in available if t > tf]
    return MOVE_CLASS_CUTS[candidates[0]]


class MoveClassField(IndicatorField):
    """RSI momentum classification: rsi_ma8_diff vs frozen sym0 cuts → -3..3.

    7 classes, 0 = neutral (flat slope), NaN → 0. Directional edge is
    INVERTED (experiment §6): -3 (steep fall) carries the long edge,
    +3 (steep rise) the short edge.
    """

    name = "move_class"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8", "rsi_ma8_diff"]
    resource_dependencies: list[str] = []  # frozen cuts; stats json unused
    applies_to: list[int] = [15, 60, 240]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        cuts = _move_cuts(tf)
        df = data_point.get_df(tf)
        diff = df[f"{tf}_rsi_ma8_diff"].to_numpy(dtype=float)
        # NaN → 0.0 digitizes into the middle tier (class 0), matching the
        # _five_tiers NaN policy.
        result = np.digitize(np.nan_to_num(diff, nan=0.0), cuts) - 3
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
        tf_data = _get_tf_classification(tf)
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
        tf_data = _get_tf_classification(tf)
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]
        return rsi > (mean + std)


# ---------------------------------------------------------------------------
# Targets Group
# ---------------------------------------------------------------------------

