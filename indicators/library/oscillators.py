"""indicators.library.oscillators — SAR and CCI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

class SARField(IndicatorField):
    """Parabolic SAR."""

    name = "sar"
    group = "oscillators"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        if len(df) < 2:
            return pd.Series(float("nan"), index=df.index)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        result = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        return pd.Series(result, index=df.index)


class CCI14Field(IndicatorField):
    """CCI with period 14."""

    name = "cci_14"
    group = "oscillators"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.CCI(high, low, close, timeperiod=14)
        return pd.Series(result, index=df.index)


class CCI_MAField(IndicatorField):
    """20-period rolling SMA of cci_14."""

    name = "cci_ma"
    group = "oscillators"
    dependencies: list[str] = ["cci_14"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_cci_14"].rolling(20).mean()


# ---------------------------------------------------------------------------
# Volatility Group
# ---------------------------------------------------------------------------

