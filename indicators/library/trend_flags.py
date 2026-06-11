"""indicators.library.trend_flags — Trend flag fields."""

from __future__ import annotations

import pandas as pd

from ..framework import IndicatorField


class TrendUpField(IndicatorField):
    """Boolean: close > EMA AND the EMA is rising (trend_up_50)."""

    group = "trend_flags"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, ema_length: int = 50) -> None:
        self.ema_length = ema_length
        self.params = {"ema_length": ema_length}
        self.name = f"trend_up_{ema_length}"
        self.dependencies = [f"ema_{ema_length}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        ema = df[f"{tf}_ema_{self.ema_length}"]
        return (close > ema) & (ema > ema.shift(1))


class TrendDownField(IndicatorField):
    """Boolean: close < EMA AND the EMA is falling (trend_down_50)."""

    group = "trend_flags"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, ema_length: int = 50) -> None:
        self.ema_length = ema_length
        self.params = {"ema_length": ema_length}
        self.name = f"trend_down_{ema_length}"
        self.dependencies = [f"ema_{ema_length}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        ema = df[f"{tf}_ema_{self.ema_length}"]
        return (close < ema) & (ema < ema.shift(1))
