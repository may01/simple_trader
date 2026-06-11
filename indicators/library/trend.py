"""indicators.library.trend — EMA and MACD family."""

from __future__ import annotations

import pandas as pd
import talib

from ..framework import IndicatorField


class EMAField(IndicatorField):
    """Generic EMA on close price; name carries the length (ema_50)."""

    group = "trend"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, length: int = 25, name: str | None = None) -> None:
        self.length = length
        self.params = {"length": length}
        self.name = name if name is not None else f"ema_{length}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.EMA(close, timeperiod=self.length)
        return pd.Series(result, index=df.index)


class _MACDBase(IndicatorField):
    """Shared MACD machinery; subclasses pick the output and default periods."""

    group = "trend"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    _output: int  # 0 = macd line, 1 = signal, 2 = histogram
    _stem: str    # name stem: "macd", "macd_signal", "macd_hist"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.params = {"fast": fast, "slow": slow, "signal": signal}
        self.name = f"{self._stem}_{fast}_{slow}_{signal}"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        outputs = talib.MACD(
            close, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal
        )
        return pd.Series(outputs[self._output], index=df.index)


class MACDField(_MACDBase):
    """MACD line; name carries the periods (macd_12_26_9)."""

    _output = 0
    _stem = "macd"


class MACDSignalField(_MACDBase):
    """MACD signal line (macd_signal_12_26_9)."""

    _output = 1
    _stem = "macd_signal"


class MACDHistField(_MACDBase):
    """MACD histogram (macd_hist_12_26_9)."""

    _output = 2
    _stem = "macd_hist"


class MACDFastField(MACDField):
    """MACD line with fast defaults (macd_5_13_9)."""

    def __init__(self, fast: int = 5, slow: int = 13, signal: int = 9) -> None:
        super().__init__(fast=fast, slow=slow, signal=signal)


class MACDFastSignalField(MACDSignalField):
    """MACD signal line with fast defaults (macd_signal_5_13_9)."""

    def __init__(self, fast: int = 5, slow: int = 13, signal: int = 9) -> None:
        super().__init__(fast=fast, slow=slow, signal=signal)
