"""indicators.library.momentum — RSI family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

# ---------------------------------------------------------------------------
# Momentum Group
# ---------------------------------------------------------------------------

class RSI14Field(IndicatorField):
    """RSI with period 14."""

    name = "rsi_14"
    group = "momentum"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []  # all timeframes
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.RSI(close, timeperiod=14)
        return pd.Series(result, index=df.index)


class RSI_MAField(IndicatorField):
    """EMA of an RSI series (generic: rsi_ma8, rsi_ma12, rsi_ma24)."""

    group = "momentum"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}
    dependencies: list[str]

    def __init__(self, source: str, length: int, name: str) -> None:
        self.name = name
        self.source = source
        self.length = length
        self.dependencies = [source]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source}"].values.astype(float)
        result = talib.EMA(series, timeperiod=self.length)
        return pd.Series(result, index=df.index)


class RSI_MA_DiffField(IndicatorField):
    """Difference (diff) of an RSI MA series."""

    group = "momentum"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}
    dependencies: list[str]

    def __init__(self, source_ma: str, name: str) -> None:
        self.name = name
        self.source_ma = source_ma
        self.dependencies = [source_ma]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source_ma}"]
        return series - series.shift(1)


# ---------------------------------------------------------------------------
# Trend Group
# ---------------------------------------------------------------------------

