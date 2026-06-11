"""indicators.library.volume — Volume moving averages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField


class VolMAField(IndicatorField):
    """20-period rolling mean of volume."""

    name = "vol_ma"
    group = "volume"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_volume"].rolling(20).mean()


class VolBuyMAField(IndicatorField):
    """20-period rolling mean of buy_volume."""

    name = "vol_buy_ma"
    group = "volume"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_buy_volume"].rolling(20).mean()


class VolSellMAField(IndicatorField):
    """vol_ma minus vol_buy_ma."""

    name = "vol_sell_ma"
    group = "volume"
    dependencies: list[str] = ["vol_ma", "vol_buy_ma"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_vol_ma"] - df[f"{tf}_vol_buy_ma"]


# ---------------------------------------------------------------------------
# Price Derivatives Group
# ---------------------------------------------------------------------------

