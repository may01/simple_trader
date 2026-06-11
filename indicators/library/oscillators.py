"""indicators.library.oscillators — SAR and CCI."""

from __future__ import annotations

import pandas as pd
import talib

from ..framework import IndicatorField


def _fmt(value: float) -> str:
    """Format a float param for a column name: 0.02 → '002', 1.5 → '15', 2.0 → '2'."""
    return ("%g" % value).replace(".", "")


class SARField(IndicatorField):
    """Parabolic SAR; name carries acceleration/maximum (sar_002_02)."""

    group = "oscillators"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2) -> None:
        self.acceleration = acceleration
        self.maximum = maximum
        self.params = {"acceleration": acceleration, "maximum": maximum}
        self.name = f"sar_{_fmt(acceleration)}_{_fmt(maximum)}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        if len(df) < 2:
            return pd.Series(float("nan"), index=df.index)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        result = talib.SAR(high, low, acceleration=self.acceleration, maximum=self.maximum)
        return pd.Series(result, index=df.index)


class CCI14Field(IndicatorField):
    """CCI; name carries the period (cci_14)."""

    group = "oscillators"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"cci_{period}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.CCI(high, low, close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class CCI_MAField(IndicatorField):
    """Rolling SMA of a CCI column (cci_14_ma_20)."""

    group = "oscillators"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 14, ma_length: int = 20) -> None:
        self.period = period
        self.ma_length = ma_length
        self.params = {"period": period, "ma_length": ma_length}
        self.name = f"cci_{period}_ma_{ma_length}"
        self.dependencies = [f"cci_{period}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_cci_{self.period}"].rolling(self.ma_length).mean()
