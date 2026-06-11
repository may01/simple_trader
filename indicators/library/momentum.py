"""indicators.library.momentum — RSI family."""

from __future__ import annotations

import pandas as pd
import talib

from ..framework import IndicatorField


class RSI14Field(IndicatorField):
    """RSI; name carries the period (rsi_14)."""

    group = "momentum"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []  # all timeframes

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"rsi_{period}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.RSI(close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class RSI_MAField(IndicatorField):
    """EMA of an RSI series (generic: rsi_ma8, rsi_ma12, rsi_ma24)."""

    group = "momentum"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, source: str = "rsi_14", length: int = 8, name: str | None = None) -> None:
        self.source = source
        self.length = length
        self.params = {"source": source, "length": length}
        self.name = name if name is not None else f"rsi_ma{length}"
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

    def __init__(self, source_ma: str = "rsi_ma8", name: str | None = None) -> None:
        self.source_ma = source_ma
        self.params = {"source_ma": source_ma}
        self.name = name if name is not None else f"{source_ma}_diff"
        self.dependencies = [source_ma]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source_ma}"]
        return series - series.shift(1)
