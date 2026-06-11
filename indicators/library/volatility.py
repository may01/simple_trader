"""indicators.library.volatility — ATR, NATR and Bollinger bands."""

from __future__ import annotations

import pandas as pd
import talib

from ..framework import IndicatorField
from .oscillators import _fmt


class ATR14Field(IndicatorField):
    """ATR; name carries the period (atr_14)."""

    group = "volatility"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"atr_{period}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.ATR(high, low, close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class NATR14Field(IndicatorField):
    """Normalized ATR; name carries the period (natr_14)."""

    group = "volatility"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.params = {"period": period}
        self.name = f"natr_{period}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"].values.astype(float)
        low = df[f"{tf}_low"].values.astype(float)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.NATR(high, low, close, timeperiod=self.period)
        return pd.Series(result, index=df.index)


class ATR_MAField(IndicatorField):
    """Rolling SMA of an ATR column (atr_14_ma_20)."""

    group = "volatility"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 14, ma_length: int = 20) -> None:
        self.period = period
        self.ma_length = ma_length
        self.params = {"period": period, "ma_length": ma_length}
        self.name = f"atr_{period}_ma_{ma_length}"
        self.dependencies = [f"atr_{period}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_atr_{self.period}"].rolling(self.ma_length).mean()


class NATR_MAField(IndicatorField):
    """EMA of a NATR column (natr_14_ma_5).

    Consumed by the BigCandle signal (signals_lib/candle.py), which documents
    natr_ma as "EMA-5 of natr_14" — no field produced it until now, so the
    signal NaN-guarded to False on every tick.
    """

    group = "volatility"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, period: int = 14, ma_length: int = 5) -> None:
        self.period = period
        self.ma_length = ma_length
        self.params = {"period": period, "ma_length": ma_length}
        self.name = f"natr_{period}_ma_{ma_length}"
        self.dependencies = [f"natr_{period}"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_natr_{self.period}"].values.astype(float)
        result = talib.EMA(series, timeperiod=self.ma_length)
        return pd.Series(result, index=df.index)


class _BollingerBase(IndicatorField):
    """Shared Bollinger machinery; subclasses pick band and defaults.

    Name pattern: bb_{band}_{period}_{nbdev formatted} — bb_upper_20_2,
    bb_upper_10_15 (nbdev 1.5), bb_upper_20_3.
    """

    group = "volatility"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    _band: int  # 0 = upper, 1 = middle, 2 = lower

    def __init__(self, period: int = 20, nbdev: float = 2.0) -> None:
        self.period = period
        self.nbdev = nbdev
        self.params = {"period": period, "nbdev": nbdev}
        band_name = ("upper", "middle", "lower")[self._band]
        self.name = f"bb_{band_name}_{period}_{_fmt(nbdev)}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        bands = talib.BBANDS(
            close, timeperiod=self.period, nbdevup=self.nbdev, nbdevdn=self.nbdev, matype=0
        )
        return pd.Series(bands[self._band], index=df.index)


class BollingerUpperField(_BollingerBase):
    """Bollinger upper band (bb_upper_20_2)."""

    _band = 0


class BollingerMiddleField(_BollingerBase):
    """Bollinger middle band (bb_middle_20_2)."""

    _band = 1


class BollingerLowerField(_BollingerBase):
    """Bollinger lower band (bb_lower_20_2)."""

    _band = 2


class BollingerFastUpperField(BollingerUpperField):
    """Fast Bollinger upper band (bb_upper_10_15)."""

    def __init__(self, period: int = 10, nbdev: float = 1.5) -> None:
        super().__init__(period=period, nbdev=nbdev)


class BollingerFastLowerField(BollingerLowerField):
    """Fast Bollinger lower band (bb_lower_10_15)."""

    def __init__(self, period: int = 10, nbdev: float = 1.5) -> None:
        super().__init__(period=period, nbdev=nbdev)


class BollingerWideUpperField(BollingerUpperField):
    """Wide Bollinger upper band (bb_upper_20_3)."""

    def __init__(self, period: int = 20, nbdev: float = 3.0) -> None:
        super().__init__(period=period, nbdev=nbdev)


class BollingerWideLowerField(BollingerLowerField):
    """Wide Bollinger lower band (bb_lower_20_3)."""

    def __init__(self, period: int = 20, nbdev: float = 3.0) -> None:
        super().__init__(period=period, nbdev=nbdev)
