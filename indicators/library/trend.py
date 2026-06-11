"""indicators.library.trend — EMA and MACD family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

class EMAField(IndicatorField):
    """Generic EMA on close price."""

    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def __init__(self, length: int, name: str) -> None:
        self.length = length
        self.name = name

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.EMA(close, timeperiod=self.length)
        return pd.Series(result, index=df.index)


class MACDField(IndicatorField):
    """MACD line (fast=12, slow=26, signal=9)."""

    name = "macd"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        macd, _signal, _hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.Series(macd, index=df.index)


class MACDSignalField(IndicatorField):
    """MACD signal line (fast=12, slow=26, signal=9)."""

    name = "macd_signal"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _macd, signal, _hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.Series(signal, index=df.index)


class MACDHistField(IndicatorField):
    """MACD histogram (fast=12, slow=26, signal=9)."""

    name = "macd_hist"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _macd, _signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.Series(hist, index=df.index)


class MACDFastField(IndicatorField):
    """Fast MACD line (fast=5, slow=13, signal=9)."""

    name = "macd_fast"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        macd, _signal, _hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=9)
        return pd.Series(macd, index=df.index)


class MACDFastSignalField(IndicatorField):
    """Fast MACD signal line (fast=5, slow=13, signal=9)."""

    name = "macd_fast_signal"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _macd, signal, _hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=9)
        return pd.Series(signal, index=df.index)


# ---------------------------------------------------------------------------
# Oscillators Group
# ---------------------------------------------------------------------------

