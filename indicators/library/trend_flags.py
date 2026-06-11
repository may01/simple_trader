"""indicators.library.trend_flags — Trend flag fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

class TrendUpField(IndicatorField):
    """Boolean: close > ema_50 AND ema_50 is rising."""

    name = "trend_up"
    group = "trend_flags"
    dependencies: list[str] = ["ema_50"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        ema50 = df[f"{tf}_ema_50"]
        return (close > ema50) & (ema50 > ema50.shift(1))


class TrendDownField(IndicatorField):
    """Boolean: close < ema_50 AND ema_50 is falling."""

    name = "trend_down"
    group = "trend_flags"
    dependencies: list[str] = ["ema_50"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        ema50 = df[f"{tf}_ema_50"]
        return (close < ema50) & (ema50 < ema50.shift(1))


# ---------------------------------------------------------------------------
# NN Features Group
# ---------------------------------------------------------------------------

