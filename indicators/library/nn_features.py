"""indicators.library.nn_features — NN feature fields."""

from __future__ import annotations

import pandas as pd

from ..framework import IndicatorField


class NNRSINormField(IndicatorField):
    """Normalized RSI MA: (x - rolling_mean) / rolling_std over a window."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, source: str = "rsi_ma8", window: int = 20) -> None:
        self.source = source
        self.window = window
        self.params = {"source": source, "window": window}
        self.name = f"nn_{source}_norm_mean_{window}"
        self.dependencies = [source]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source}"]
        roll_mean = series.rolling(self.window).mean()
        roll_std = series.rolling(self.window).std().clip(lower=1e-8)
        return (series - roll_mean) / roll_std


class NNCloseDiffATRField(IndicatorField):
    """close_diff_prc divided by an ATR MA column (clipped against zero)."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, atr_period: int = 14, atr_ma_length: int = 20) -> None:
        self.atr_period = atr_period
        self.atr_ma_length = atr_ma_length
        self.params = {"atr_period": atr_period, "atr_ma_length": atr_ma_length}
        self._atr_ma_col = f"atr_{atr_period}_ma_{atr_ma_length}"
        self.name = f"nn_close_diff_{self._atr_ma_col}"
        self.dependencies = ["close_diff_prc", self._atr_ma_col]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        cdp = df[f"{tf}_close_diff_prc"]
        atr_ma = df[f"{tf}_{self._atr_ma_col}"].clip(lower=1e-8)
        return cdp / atr_ma
