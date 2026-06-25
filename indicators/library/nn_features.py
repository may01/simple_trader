"""indicators.library.nn_features — NN feature fields.

Engineered features materialised once by DataPreparer into the wide
``df_with_indicators.pkl`` as ordinary ``{tf}_``-prefixed indicator columns.
The NN module is a pure consumer — it never writes these columns.

Families:
- Group 1 (z-score on raw indicators): no field classes here; the existing
  base-indicator columns are selected into ``nn.feature_cols`` and the single
  global robust z-score layer standardises them.
- Group 2 differences (``NNDiffField``): raw ``{tf}_{left} - {tf}_{right}``.
- Group 3 slopes (``NNSlopeField``): trailing OLS slope (per-bar change) of a
  source column over ``window`` bars.
- Orthogonal engineered fields: log-returns, ATR-normalised range, candle
  body/wick ratios, rolling-percentile volatility regime, cyclical time-of-day
  / day-of-week, and cross-TF trend alignment.

No look-ahead: slopes use only trailing rows; differences are point-in-time;
cross-TF alignment reads the higher TF's last-closed value (no future leak);
cyclical encodings come point-in-time from the index. Warmup rows emit NaN and
are dropped at NNDataset build, never forward/back-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework import IndicatorField


def _rolling_ols_slope(series: pd.Series, window: int) -> pd.Series:
    """Trailing OLS slope (per-bar change) of *series* over *window* bars.

    Regresses y against the fixed integer abscissa x = [0, 1, ..., window-1].
    The slope is ``cov(x, y) / var(x)``; with a fixed x the denominator and the
    x-statistics are constants, so the rolling estimate is a closed-form ratio
    of rolling sums. Uses only the trailing window (no look-ahead). The first
    ``window - 1`` rows lack a full window and are NaN.
    """
    w = int(window)
    if w < 2:
        # A line through < 2 points has no defined slope.
        return pd.Series(np.nan, index=series.index)

    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    # Sxx = sum((x - x_mean)^2) — constant for a fixed integer abscissa.
    sxx = float(((x - x_mean) ** 2).sum())

    y = series.astype(float)
    # Sxy_t = sum_i (x_i - x_mean) * y_{t-w+1+i}; rolling dot of centred x with y.
    weights = x - x_mean

    def _slope(window_vals: np.ndarray) -> float:
        return float(np.dot(weights, window_vals) / sxx)

    return y.rolling(w).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Already-built fields (kept, unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Group 2 — indicator differences
# ---------------------------------------------------------------------------

class NNDiffField(IndicatorField):
    """Point-in-time difference ``{tf}_{left} - {tf}_{right}``.

    Raw difference only; the global robust z-score layer standardises it.
    For EMA pairs the convention is shorter − longer (positive ⇒ faster EMA
    above slower ⇒ up-momentum).
    """

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, left: str, right: str) -> None:
        self.left = left
        self.right = right
        self.params = {"left": left, "right": right}
        self.name = f"{left}_minus_{right}"
        self.dependencies = [left, right]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_{self.left}"] - df[f"{tf}_{self.right}"]


# ---------------------------------------------------------------------------
# Group 3 — slopes
# ---------------------------------------------------------------------------

class NNSlopeField(IndicatorField):
    """Trailing OLS slope (per-bar change) of *source* over *window* bars.

    Raw slope only; the global robust z-score layer standardises it. Uses only
    trailing data — no look-ahead. Warmup rows (< window) are NaN.
    """

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, source: str, window: int = 5) -> None:
        self.source = source
        self.window = window
        self.params = {"source": source, "window": window}
        self.name = f"{source}_slope"
        self.dependencies = [source]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source}"]
        return _rolling_ols_slope(series, self.window)


# ---------------------------------------------------------------------------
# Orthogonal engineered fields
# ---------------------------------------------------------------------------

class NNLogRetField(IndicatorField):
    """Log-return ``log(close_t / close_{t-1})``."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "logret"
        self.dependencies = ["close"]

    def compute(self, data_point, tf: int) -> pd.Series:
        close = data_point.get_df(tf)[f"{tf}_close"].astype(float)
        return np.log(close / close.shift(1))


class NNRangeATRField(IndicatorField):
    """ATR-normalised candle range ``(high - low) / atr.clip(lower=1e-8)``."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, atr_col: str = "atr_14") -> None:
        self.atr_col = atr_col
        self.params = {"atr_col": atr_col}
        self.name = "range_atr"
        self.dependencies = ["high", "low", atr_col]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"]
        low = df[f"{tf}_low"]
        atr = df[f"{tf}_{self.atr_col}"].clip(lower=1e-8)
        return (high - low) / atr


class NNBodyRatioField(IndicatorField):
    """Candle body ratio ``(close - open) / (high - low).clip(lower=1e-8)``."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "body_ratio"
        self.dependencies = ["open", "close", "high", "low"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        rng = (df[f"{tf}_high"] - df[f"{tf}_low"]).clip(lower=1e-8)
        return (df[f"{tf}_close"] - df[f"{tf}_open"]) / rng


class NNWickUpField(IndicatorField):
    """Upper-wick ratio ``(high - max(open, close)) / (high - low).clip(lower=1e-8)``."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "wick_up"
        self.dependencies = ["open", "close", "high", "low"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        body_top = df[[f"{tf}_open", f"{tf}_close"]].max(axis=1)
        rng = (df[f"{tf}_high"] - df[f"{tf}_low"]).clip(lower=1e-8)
        return (df[f"{tf}_high"] - body_top) / rng


class NNWickDnField(IndicatorField):
    """Lower-wick ratio ``(min(open, close) - low) / (high - low).clip(lower=1e-8)``."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "wick_dn"
        self.dependencies = ["open", "close", "high", "low"]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        body_bot = df[[f"{tf}_open", f"{tf}_close"]].min(axis=1)
        rng = (df[f"{tf}_high"] - df[f"{tf}_low"]).clip(lower=1e-8)
        return (body_bot - df[f"{tf}_low"]) / rng


class NNVolRegimeField(IndicatorField):
    """Bucketed rolling percentile of an ATR column → volatility regime.

    The trailing rolling percentile rank of ``atr_col`` over ``window`` bars is
    bucketed into ``0 .. buckets-1`` (float). Point-in-time: the rank uses only
    the trailing window. Warmup rows (< window) are NaN.
    """

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, atr_col: str = "atr_14", window: int = 200, buckets: int = 3) -> None:
        self.atr_col = atr_col
        self.window = window
        self.buckets = buckets
        self.params = {"atr_col": atr_col, "window": window, "buckets": buckets}
        self.name = "vol_regime"
        self.dependencies = [atr_col]

    def compute(self, data_point, tf: int) -> pd.Series:
        series = data_point.get_df(tf)[f"{tf}_{self.atr_col}"].astype(float)
        w = int(self.window)
        b = int(self.buckets)

        def _pct_rank(window_vals: np.ndarray) -> float:
            # Trailing percentile of the LAST value within its window [0, 1].
            last = window_vals[-1]
            return float((window_vals <= last).mean())

        pct = series.rolling(w).apply(_pct_rank, raw=True)
        # Map [0, 1] percentile to a bucket index 0 .. b-1.
        bucket = np.floor(pct * b)
        bucket = bucket.clip(upper=b - 1)  # pct == 1.0 would land on b → cap at b-1
        return bucket


class NNSinTodField(IndicatorField):
    """sin(2π · seconds_since_midnight / 86400) — point-in-time from the index."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "sin_tod"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        idx = data_point.get_df(tf).index
        secs = idx.hour * 3600 + idx.minute * 60 + idx.second
        return pd.Series(np.sin(2 * np.pi * secs / 86400.0), index=idx)


class NNCosTodField(IndicatorField):
    """cos(2π · seconds_since_midnight / 86400) — point-in-time from the index."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "cos_tod"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        idx = data_point.get_df(tf).index
        secs = idx.hour * 3600 + idx.minute * 60 + idx.second
        return pd.Series(np.cos(2 * np.pi * secs / 86400.0), index=idx)


class NNSinDowField(IndicatorField):
    """sin(2π · day_of_week / 7) — point-in-time from the index."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "sin_dow"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        idx = data_point.get_df(tf).index
        return pd.Series(np.sin(2 * np.pi * idx.dayofweek / 7.0), index=idx)


class NNCosDowField(IndicatorField):
    """cos(2π · day_of_week / 7) — point-in-time from the index."""

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self) -> None:
        self.params: dict = {}
        self.name = "cos_dow"
        self.dependencies: list[str] = []

    def compute(self, data_point, tf: int) -> pd.Series:
        idx = data_point.get_df(tf).index
        return pd.Series(np.cos(2 * np.pi * idx.dayofweek / 7.0), index=idx)


class NNCrossTFAlignField(IndicatorField):
    """Cross-TF trend-alignment: sign agreement of this-TF vs higher-TF slope.

    Computes the trailing slope of ``trend_col`` on this TF and on the higher
    ``other_tf`` column (already present, ffilled onto this TF's index — the
    last closed higher-TF value, never a future one), then emits +1 when the
    signs agree, -1 when they disagree, 0 when either is flat/NaN.

    ``applies_to`` is set to the lower member only (config: ``[15]`` / ``[60]``),
    so a missing higher TF is a config error, not a silent NaN.
    """

    group = "nn_features"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []

    def __init__(self, other_tf: int, trend_col: str = "ema_50", window: int = 5) -> None:
        self.other_tf = other_tf
        self.trend_col = trend_col
        self.window = window
        self.params = {"other_tf": other_tf, "trend_col": trend_col, "window": window}
        self.name = f"align_{other_tf}"
        self.dependencies = [trend_col]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        this_slope = _rolling_ols_slope(df[f"{tf}_{self.trend_col}"], self.window)

        # The higher-TF trend column is co-located on this TF's index only in the
        # wide-frame (DataPreparer) write path, where it carries the last-closed
        # higher-TF value forward-filled by candle alignment — never a future one.
        # In the per-TF live path the higher-TF frame is not present in this
        # slice; emit NaN (dropped at NNDataset build) rather than crash.
        other_key = f"{self.other_tf}_{self.trend_col}"
        if other_key not in df.columns:
            return pd.Series(np.nan, index=df.index)

        other_col = df[other_key].ffill()
        other_slope = _rolling_ols_slope(other_col, self.window)
        return np.sign(this_slope) * np.sign(other_slope)
