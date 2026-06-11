"""indicators.library.volatility — ATR, NATR and Bollinger bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

class ATR14Field(IndicatorField):
    """ATR with period 14."""

    name = "atr_14"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.ATR(high, low, close, timeperiod=14)
        return pd.Series(result, index=df.index)


class NATR14Field(IndicatorField):
    """Normalized ATR with period 14."""

    name = "natr_14"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.NATR(high, low, close, timeperiod=14)
        return pd.Series(result, index=df.index)


class ATR_MAField(IndicatorField):
    """20-period rolling SMA of atr_14."""

    name = "atr_ma"
    group = "volatility"
    dependencies: list[str] = ["atr_14"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_atr_14"].rolling(20).mean()


class BollingerUpperField(IndicatorField):
    """Bollinger Band upper (20, 2.0)."""

    name = "bb_upper"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        upper, _middle, _lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        return pd.Series(upper, index=df.index)


class BollingerMiddleField(IndicatorField):
    """Bollinger Band middle (20, 2.0)."""

    name = "bb_middle"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _upper, middle, _lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        return pd.Series(middle, index=df.index)


class BollingerLowerField(IndicatorField):
    """Bollinger Band lower (20, 2.0)."""

    name = "bb_lower"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _upper, _middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        return pd.Series(lower, index=df.index)


class BollingerFastUpperField(IndicatorField):
    """Fast Bollinger Band upper (10, 1.5)."""

    name = "bb_fast_upper"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        upper, _middle, _lower = talib.BBANDS(close, timeperiod=10, nbdevup=1.5, nbdevdn=1.5, matype=0)
        return pd.Series(upper, index=df.index)


class BollingerFastLowerField(IndicatorField):
    """Fast Bollinger Band lower (10, 1.5)."""

    name = "bb_fast_lower"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _upper, _middle, lower = talib.BBANDS(close, timeperiod=10, nbdevup=1.5, nbdevdn=1.5, matype=0)
        return pd.Series(lower, index=df.index)


class BollingerWideUpperField(IndicatorField):
    """Wide Bollinger Band upper (20, 3.0)."""

    name = "bb_wide_upper"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        upper, _middle, _lower = talib.BBANDS(close, timeperiod=20, nbdevup=3.0, nbdevdn=3.0, matype=0)
        return pd.Series(upper, index=df.index)


class BollingerWideLowerField(IndicatorField):
    """Wide Bollinger Band lower (20, 3.0)."""

    name = "bb_wide_lower"
    group = "volatility"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _upper, _middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=3.0, nbdevdn=3.0, matype=0)
        return pd.Series(lower, index=df.index)


# ---------------------------------------------------------------------------
# Volume Group
# ---------------------------------------------------------------------------
