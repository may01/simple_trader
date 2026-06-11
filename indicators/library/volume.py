"""indicators.library.volume — Volume moving averages."""

from __future__ import annotations

import pandas as pd

from ..framework import IndicatorField


class VolMAField(IndicatorField):
    """Rolling mean of volume."""

    group = "volume"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"vol_ma_{period}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_volume"].rolling(self.period).mean()


class VolBuyMAField(IndicatorField):
    """Rolling mean of buy_volume."""

    group = "volume"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"vol_buy_ma_{period}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_buy_volume"].rolling(self.period).mean()


class VolSellMAField(IndicatorField):
    """vol_ma minus vol_buy_ma at the same period."""

    group = "volume"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 20) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"vol_sell_ma_{period}"
        self.dependencies = [f"vol_ma_{period}", f"vol_buy_ma_{period}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_vol_ma_{self.period}"] - df[f"{tf}_vol_buy_ma_{self.period}"]
