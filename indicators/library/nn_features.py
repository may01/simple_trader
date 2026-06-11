"""indicators.library.nn_features — NN feature fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from ..framework import IndicatorField

class NNRSINormField(IndicatorField):
    """Normalized rsi_ma8: (rsi - rolling_mean) / rolling_std."""

    name = "nn_rsi_ma8_norm_mean"
    group = "nn_features"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]
        roll_mean = rsi.rolling(20).mean()
        roll_std = rsi.rolling(20).std().clip(lower=1e-8)
        return (rsi - roll_mean) / roll_std


class NNCloseDiffATRField(IndicatorField):
    """close_diff_prc divided by atr_ma (clipped to avoid division by zero)."""

    name = "nn_close_diff_atr_ma"
    group = "nn_features"
    dependencies: list[str] = ["close_diff_prc", "atr_ma"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        cdp = df[f"{tf}_close_diff_prc"]
        atr_ma = df[f"{tf}_atr_ma"].clip(lower=1e-8)
        return cdp / atr_ma


