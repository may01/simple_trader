# indicators.py — IndicatorField ABC, Indicators orchestrator, and build_indicator_input.
#
# IndicatorField: abstract base class for a single computed indicator column.
# Indicators:     static orchestrator that runs all applicable fields on a DataPoint.
# build_indicator_input: slices a wide DataFrame to give Indicators.compute() its input.

from __future__ import annotations

import functools
import json
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import talib

from config_loader import CANDLES, load_indicators_config
from logs import log_warning

# Module-level set to track which missing resource deps have already been warned about.
_warned_resources: set[str] = set()


# ---------------------------------------------------------------------------
# IndicatorField — abstract base class
# ---------------------------------------------------------------------------

class IndicatorField(ABC):
    """Abstract base class for a single indicator field computation.

    Subclasses must set the class-level attributes and implement compute().
    """

    name: str                      # output column suffix, e.g. "rsi_14"
    group: str                     # config group, e.g. "momentum"
    dependencies: list[str]        # field names required before this
    resource_dependencies: list[str]  # paths relative to stats_folder() that must exist
    applies_to: list[int]          # TFs this field is valid for ([] = all CANDLES)
    params: dict                   # field-specific parameters from config

    @abstractmethod
    def compute(self, data_point, tf: int) -> pd.Series:
        """Read data_point.get_df(tf), return Series for column {tf}_{name}.

        Args:
            data_point: DataPoint instance providing access to OHLC data.
            tf: Timeframe in minutes.

        Returns:
            pd.Series with computed values for the column ``{tf}_{name}``.
        """

    def is_available(self) -> bool:
        """Check all resource_dependencies exist on disk.

        Reads stats_folder() at call time, checks os.path.exists for each dep.
        Returns True immediately when resource_dependencies is empty (no env vars needed).
        Returns False (without raising) if required env vars are not set.
        """
        if not self.resource_dependencies:
            return True

        from helpers import stats_folder  # local import: reads env vars at call time

        try:
            base = stats_folder()
        except (KeyError, ValueError):
            # Required env vars (DATA_ROOT, PAIR, etc.) not set — resource unavailable.
            return False
        for rel_path in self.resource_dependencies:
            full_path = os.path.join(base, rel_path)
            if not os.path.exists(full_path):
                global _warned_resources
                if full_path not in _warned_resources:
                    _warned_resources.add(full_path)
                    log_warning(
                        f"IndicatorField '{self.name}': resource dependency not found: {full_path}"
                    )
                return False
        return True


# ---------------------------------------------------------------------------
# _PlaceholderField — framework stub (replaced by Task 04 implementations)
# ---------------------------------------------------------------------------

class _PlaceholderField(IndicatorField):
    """Placeholder IndicatorField built from an IndicatorFieldConfig.

    compute() returns an empty Series. Replaced by real implementations in Task 04.
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: IndicatorFieldConfig instance from config_loader.
        """
        self.name = config.name
        self.group = config.group
        self.applies_to = list(config.applies_to)
        self.dependencies = list(config.depends_on)
        self.resource_dependencies = []
        self.params = dict(config.params)

    def compute(self, data_point, tf: int) -> pd.Series:
        """Return empty Series (placeholder — real logic comes in Task 04)."""
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# build_indicator_input
# ---------------------------------------------------------------------------

def build_indicator_input(df: pd.DataFrame, ts: pd.Timestamp, tf: int) -> pd.DataFrame:
    """Return indicator input slice at ts for tf (up to 105 rows).

    Selects all rows up to and including ts, keeps only closed-candle rows for
    the given tf, then appends the partial current candle if ts is not closed.
    Returns at most 105 rows (the tail).

    Args:
        df: Wide DataFrame with columns ``{tf}_is_closed`` and a DatetimeIndex.
        ts: The "now" timestamp — must be present in df.index.
        tf: Timeframe in minutes.

    Returns:
        DataFrame of at most 105 rows suitable for indicator computation.
    """
    closed_col = f"{tf}_is_closed"

    subset = df[:ts]

    closed = subset[subset[closed_col].astype(bool)]

    # If the current row is NOT closed, append it as a partial candle.
    if not subset.empty and not subset.iloc[-1][closed_col]:
        closed = pd.concat([closed, subset.iloc[[-1]]])

    return closed.tail(105)


# ---------------------------------------------------------------------------
# Indicators — static orchestrator
# ---------------------------------------------------------------------------

class Indicators:
    """Static orchestrator: loads the indicator registry and runs compute() calls.

    Registry is loaded lazily on first use from load_indicators_config().
    """

    _registry: list[IndicatorField] | None = None  # class variable, loaded lazily

    @classmethod
    def _get_registry(cls) -> list[IndicatorField]:
        """Load registry from config on first use.

        For each IndicatorFieldConfig, look up in _FIELD_REGISTRY (defined at module
        bottom after all field classes are declared). Falls back to _PlaceholderField
        for any field not yet implemented.
        """
        if cls._registry is None:
            configs = load_indicators_config()
            fields = []
            for cfg in configs:
                factory = _FIELD_REGISTRY.get(cfg.name)
                if factory is not None:
                    field = factory(cfg)
                else:
                    field = _PlaceholderField(cfg)
                fields.append(field)
            cls._registry = fields
        return cls._registry

    @classmethod
    def _sorted_fields(
        cls,
        tf: int,
        groups: list[str] | None = None,
        check_resources: bool = True,
    ) -> list[IndicatorField]:
        """Return fields applicable to tf in dependency order.

        The config is already topologically sorted so no re-sorting is needed.

        Args:
            tf: Timeframe in minutes.
            groups: If provided, only return fields whose group is in this list.
            check_resources: If True, exclude fields where is_available() is False.

        Returns:
            Filtered list of IndicatorField instances in dependency order.
        """
        result = []
        for field in cls._get_registry():
            # Filter: applies_to must include tf (empty list = all CANDLES)
            applies = field.applies_to if field.applies_to else CANDLES
            if tf not in applies:
                continue

            # Filter by group if provided
            if groups is not None and field.group not in groups:
                continue

            # Filter by resource availability
            if check_resources and not field.is_available():
                continue

            result.append(field)

        return result

    @classmethod
    def compute(cls, data_point, tf: int) -> None:
        """Run ALL applicable fields for tf and write results to data_point.get_df(tf).

        For each applicable field (in dependency order):
          1. Calls field.compute(data_point, tf) → pd.Series
          2. Writes data_point.get_df(tf)[f"{tf}_{field.name}"] = series

        Args:
            data_point: DataPoint providing mutable DataFrame access via get_df(tf).
            tf: Timeframe in minutes.
        """
        df = data_point.get_df(tf)
        for field in cls._sorted_fields(tf):
            series = field.compute(data_point, tf)
            df[f"{tf}_{field.name}"] = series

    @classmethod
    def compute_group(cls, data_point, tf: int, groups: list[str]) -> None:
        """Run only fields in specified groups, in dependency order.

        Args:
            data_point: DataPoint providing mutable DataFrame access via get_df(tf).
            tf: Timeframe in minutes.
            groups: Only fields whose group is in this list will be computed.
        """
        df = data_point.get_df(tf)
        for field in cls._sorted_fields(tf, groups=groups):
            series = field.compute(data_point, tf)
            df[f"{tf}_{field.name}"] = series


# ===========================================================================
# IndicatorField subclasses — Task 04 implementations
# ===========================================================================

# ---------------------------------------------------------------------------
# Momentum Group
# ---------------------------------------------------------------------------

class RSI14Field(IndicatorField):
    """RSI with period 14."""

    name = "rsi_14"
    group = "momentum"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []  # all timeframes
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.RSI(close, timeperiod=14)
        return pd.Series(result, index=df.index)


class RSI_MAField(IndicatorField):
    """EMA of an RSI series (generic: rsi_ma8, rsi_ma12, rsi_ma24)."""

    group = "momentum"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}
    dependencies: list[str]

    def __init__(self, source: str, length: int, name: str) -> None:
        self.name = name
        self.source = source
        self.length = length
        self.dependencies = [source]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source}"].values.astype(float)
        result = talib.EMA(series, timeperiod=self.length)
        return pd.Series(result, index=df.index)


class RSI_MA_DiffField(IndicatorField):
    """Difference (diff) of an RSI MA series."""

    group = "momentum"
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}
    dependencies: list[str]

    def __init__(self, source_ma: str, name: str) -> None:
        self.name = name
        self.source_ma = source_ma
        self.dependencies = [source_ma]

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_{self.source_ma}"]
        return series - series.shift(1)


# ---------------------------------------------------------------------------
# Trend Group
# ---------------------------------------------------------------------------

class EMAField(IndicatorField):
    """Generic EMA on close price."""

    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def __init__(self, length: int, name: str) -> None:
        self.length = length
        self.name = name

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        result = talib.EMA(close, timeperiod=self.length)
        return pd.Series(result, index=df.index)


class MACDField(IndicatorField):
    """MACD line (fast=12, slow=26, signal=9)."""

    name = "macd"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        macd, _signal, _hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.Series(macd, index=df.index)


class MACDSignalField(IndicatorField):
    """MACD signal line (fast=12, slow=26, signal=9)."""

    name = "macd_signal"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _macd, signal, _hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.Series(signal, index=df.index)


class MACDHistField(IndicatorField):
    """MACD histogram (fast=12, slow=26, signal=9)."""

    name = "macd_hist"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _macd, _signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.Series(hist, index=df.index)


class MACDFastField(IndicatorField):
    """Fast MACD line (fast=5, slow=13, signal=9)."""

    name = "macd_fast"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        macd, _signal, _hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=9)
        return pd.Series(macd, index=df.index)


class MACDFastSignalField(IndicatorField):
    """Fast MACD signal line (fast=5, slow=13, signal=9)."""

    name = "macd_fast_signal"
    group = "trend"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"].values.astype(float)
        _macd, signal, _hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=9)
        return pd.Series(signal, index=df.index)


# ---------------------------------------------------------------------------
# Oscillators Group
# ---------------------------------------------------------------------------

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

class CloseDiffPrcField(IndicatorField):
    """Percentage change of close: (close - close.shift(1)) / close.shift(1) * 100."""

    name = "close_diff_prc"
    group = "price_derivatives"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return (close - close.shift(1)) / close.shift(1) * 100


class CloseDiffPrcRMField(IndicatorField):
    """20-period rolling mean of close_diff_prc."""

    name = "close_diff_prc_rm"
    group = "price_derivatives"
    dependencies: list[str] = ["close_diff_prc"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_close_diff_prc"].rolling(20).mean()


class CloseDiffPrcRMMeanAboveField(IndicatorField):
    """Rolling mean of positive values in a 20-window of close_diff_prc_rm."""

    name = "close_diff_prc_rm_mean_above"
    group = "price_derivatives"
    dependencies: list[str] = ["close_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_close_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
            raw=True,
        )


class CloseDiffPrcRMMeanBelowField(IndicatorField):
    """Rolling mean of negative values in a 20-window of close_diff_prc_rm."""

    name = "close_diff_prc_rm_mean_below"
    group = "price_derivatives"
    dependencies: list[str] = ["close_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_close_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0,
            raw=True,
        )


class HighDiffPrcField(IndicatorField):
    """Percentage change of high."""

    name = "high_diff_prc"
    group = "price_derivatives"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        high = df[f"{tf}_high"]
        return (high - high.shift(1)) / high.shift(1) * 100


class HighDiffPrcRMField(IndicatorField):
    """20-period rolling mean of high_diff_prc."""

    name = "high_diff_prc_rm"
    group = "price_derivatives"
    dependencies: list[str] = ["high_diff_prc"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_high_diff_prc"].rolling(20).mean()


class HighDiffPrcRMMeanAboveField(IndicatorField):
    """Rolling mean of positive values in 20-window of high_diff_prc_rm."""

    name = "high_diff_prc_rm_mean_above"
    group = "price_derivatives"
    dependencies: list[str] = ["high_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_high_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
            raw=True,
        )


class HighDiffPrcRMMeanBelowField(IndicatorField):
    """Rolling mean of negative values in 20-window of high_diff_prc_rm."""

    name = "high_diff_prc_rm_mean_below"
    group = "price_derivatives"
    dependencies: list[str] = ["high_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_high_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0,
            raw=True,
        )


class LowDiffPrcField(IndicatorField):
    """Percentage change of low."""

    name = "low_diff_prc"
    group = "price_derivatives"
    dependencies: list[str] = []
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        low = df[f"{tf}_low"]
        return (low - low.shift(1)) / low.shift(1) * 100


class LowDiffPrcRMField(IndicatorField):
    """20-period rolling mean of low_diff_prc."""

    name = "low_diff_prc_rm"
    group = "price_derivatives"
    dependencies: list[str] = ["low_diff_prc"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        return df[f"{tf}_low_diff_prc"].rolling(20).mean()


class LowDiffPrcRMMeanAboveField(IndicatorField):
    """Rolling mean of positive values in 20-window of low_diff_prc_rm."""

    name = "low_diff_prc_rm_mean_above"
    group = "price_derivatives"
    dependencies: list[str] = ["low_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_low_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
            raw=True,
        )


class LowDiffPrcRMMeanBelowField(IndicatorField):
    """Rolling mean of negative values in 20-window of low_diff_prc_rm."""

    name = "low_diff_prc_rm_mean_below"
    group = "price_derivatives"
    dependencies: list[str] = ["low_diff_prc_rm"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        series = df[f"{tf}_low_diff_prc_rm"]
        return series.rolling(20).apply(
            lambda x: x[x < 0].mean() if (x < 0).any() else 0.0,
            raw=True,
        )


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _load_rsi_classification(stats_path: str) -> dict:
    """Load rsi_classification.json from stats_path (cached by path)."""
    with open(stats_path, "r") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def _load_diff_stats(stats_path: str) -> dict:
    """Load diff_stats.pkl from stats_path (cached by path)."""
    with open(stats_path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Classification Group
# ---------------------------------------------------------------------------

class MoveClassField(IndicatorField):
    """RSI-based move classification: -1, 0, 1, 2 relative to mean ± std."""

    name = "move_class"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        from helpers import stats_folder
        path = os.path.join(stats_folder(), "rsi_classification.json")
        data = _load_rsi_classification(path)
        tf_data = data[str(tf)]
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]

        conditions = [
            rsi > mean + std,
            rsi > mean,
            rsi > mean - std,
        ]
        choices = [2, 1, 0]
        result = np.select(conditions, choices, default=-1)
        return pd.Series(result, index=df.index, dtype=int)


class ZoneClassField(IndicatorField):
    """RSI-based zone classification: 0–3 based on RSI position relative to mean±std."""

    name = "zone_class"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        from helpers import stats_folder
        path = os.path.join(stats_folder(), "rsi_classification.json")
        data = _load_rsi_classification(path)
        tf_data = data[str(tf)]
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]

        conditions = [
            rsi > mean + std,
            rsi > mean,
            rsi > mean - std,
        ]
        choices = [3, 2, 1]
        result = np.select(conditions, choices, default=0)
        return pd.Series(result, index=df.index, dtype=int)


class OverLowField(IndicatorField):
    """Boolean: RSI above lower threshold (mean - std)."""

    name = "over_low"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        from helpers import stats_folder
        path = os.path.join(stats_folder(), "rsi_classification.json")
        data = _load_rsi_classification(path)
        tf_data = data[str(tf)]
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]
        return rsi > (mean - std)


class OverHighField(IndicatorField):
    """Boolean: RSI above upper threshold (mean + std)."""

    name = "over_high"
    group = "classification"
    dependencies: list[str] = ["rsi_ma8"]
    resource_dependencies: list[str] = ["rsi_classification.json"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        from helpers import stats_folder
        path = os.path.join(stats_folder(), "rsi_classification.json")
        data = _load_rsi_classification(path)
        tf_data = data[str(tf)]
        mean = float(tf_data["mean"])
        std = float(tf_data["std"])

        df = data_point.get_df(tf)
        rsi = df[f"{tf}_rsi_ma8"]
        return rsi > (mean + std)


# ---------------------------------------------------------------------------
# Targets Group
# ---------------------------------------------------------------------------

def _get_tf_diff_stats(tf: int) -> dict:
    """Return per-tf entry from diff_stats.pkl (uses cached _load_diff_stats)."""
    from helpers import stats_folder
    path = os.path.join(stats_folder(), "diff_stats.pkl")
    data = _load_diff_stats(path)
    return data.get(str(tf), data.get(tf, {}))


class TgtLongField(IndicatorField):
    """Long target: close * (1 + mean_long_diff)."""

    name = "tgt_long"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_diff = float(stats.get("mean_long", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 + mean_diff)


class SLLongField(IndicatorField):
    """Long stop-loss: close * (1 - mean_long_sl)."""

    name = "sl_long"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_sl = float(stats.get("mean_long_sl", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 - mean_sl)


class TgtShortField(IndicatorField):
    """Short target: close * (1 - mean_short_diff)."""

    name = "tgt_short"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_diff = float(stats.get("mean_short", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 - mean_diff)


class SLShortField(IndicatorField):
    """Short stop-loss: close * (1 + mean_short_sl)."""

    name = "sl_short"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        mean_sl = float(stats.get("mean_short_sl", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return close * (1.0 + mean_sl)


class ZBField(IndicatorField):
    """Zone Buy classification from diff_stats."""

    name = "ZB"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        threshold = float(stats.get("zb_threshold", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return (close > threshold).astype(int)


class ZSField(IndicatorField):
    """Zone Sell classification from diff_stats."""

    name = "ZS"
    group = "targets"
    dependencies: list[str] = []
    resource_dependencies: list[str] = ["diff_stats.pkl"]
    applies_to: list[int] = [15, 60, 240, 1440]
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        stats = _get_tf_diff_stats(tf)
        threshold = float(stats.get("zs_threshold", 0.0))
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        return (close < threshold).astype(int)


# ---------------------------------------------------------------------------
# Trend Flags Group
# ---------------------------------------------------------------------------

class TrendUpField(IndicatorField):
    """Boolean: close > ema_50 AND ema_50 is rising."""

    name = "trend_up"
    group = "trend_flags"
    dependencies: list[str] = ["ema_50"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        ema50 = df[f"{tf}_ema_50"]
        return (close > ema50) & (ema50 > ema50.shift(1))


class TrendDownField(IndicatorField):
    """Boolean: close < ema_50 AND ema_50 is falling."""

    name = "trend_down"
    group = "trend_flags"
    dependencies: list[str] = ["ema_50"]
    resource_dependencies: list[str] = []
    applies_to: list[int] = []
    params: dict = {}

    def compute(self, data_point, tf: int) -> pd.Series:
        df = data_point.get_df(tf)
        close = df[f"{tf}_close"]
        ema50 = df[f"{tf}_ema_50"]
        return (close < ema50) & (ema50 < ema50.shift(1))


# ---------------------------------------------------------------------------
# NN Features Group
# ---------------------------------------------------------------------------

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


# ===========================================================================
# DataAttributes — stats computation and storage
# ===========================================================================

class DataAttributes:
    """Compute and persist dataset statistics for classification and normalisation.

    Two types of stats are managed:
    - ``rsi_classification.json``: RSI mean/std per TF, used by classification fields.
    - ``diff_stats.pkl``: Price diff mean/std per TF, used by target fields.
    - ``column_stats``: per-column mean/std for NN feature normalisation (in-memory).
    """

    _STAT_TFS: list[int] = [15, 60, 240, 1440]

    def __init__(self) -> None:
        self.column_stats: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public compute entry-point
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> None:
        """Compute and save all stats files if absent. Idempotent.

        Calls ``_compute_rsi_classification`` if ``rsi_classification.json``
        is absent.  Calls ``_compute_diff_stats`` if ``diff_stats.pkl`` is
        absent.

        Args:
            df: Wide DataFrame with ``{tf}_rsi_ma8`` and ``{tf}_is_closed``
                columns.
        """
        from helpers import stats_folder  # local import: reads env at call time
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        rsi_path = base + "rsi_classification.json"
        if not os.path.exists(rsi_path):
            self._compute_rsi_classification(df)

        diff_path = base + "diff_stats.pkl"
        if not os.path.exists(diff_path):
            self._compute_diff_stats(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rsi_classification(self, df: pd.DataFrame) -> None:
        """Compute mean/std of ``{tf}_rsi_ma8`` for closed-candle rows per TF.

        Saves to ``stats_folder() + 'rsi_classification.json'``.
        """
        from helpers import stats_folder
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        result: dict = {}
        for tf in self._STAT_TFS:
            closed_col = f"{tf}_is_closed"
            rsi_col = f"{tf}_rsi_ma8"
            if closed_col not in df.columns or rsi_col not in df.columns:
                continue
            closed_rows = df[df[closed_col] == True][rsi_col].dropna()  # noqa: E712
            result[str(tf)] = {
                "mean": float(closed_rows.mean()),
                "std": float(closed_rows.std()),
            }

        out_path = base + "rsi_classification.json"
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as fh:
            json.dump(result, fh)
        os.rename(tmp_path, out_path)

    def _compute_diff_stats(self, df: pd.DataFrame) -> None:
        """Compute price differential stats per TF.

        For each TF in ``_STAT_TFS``, computes mean and std of the percentage
        close change over closed-candle rows.  Saves to
        ``stats_folder() + 'diff_stats.pkl'``.
        """
        from helpers import stats_folder
        base = stats_folder()
        os.makedirs(base, exist_ok=True)

        result: dict = {}
        for tf in self._STAT_TFS:
            closed_col = f"{tf}_is_closed"
            close_col = f"{tf}_close"
            if closed_col not in df.columns or close_col not in df.columns:
                continue
            closed_close = df[df[closed_col] == True][close_col].dropna()  # noqa: E712
            pct_changes = closed_close.pct_change().dropna()
            result[str(tf)] = {
                "mean_diff": float(pct_changes.mean()),
                "std_diff": float(pct_changes.std()),
            }

        out_path = base + "diff_stats.pkl"
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as fh:
            pickle.dump(result, fh)
        os.rename(tmp_path, out_path)

    # ------------------------------------------------------------------
    # Class-level loaders
    # ------------------------------------------------------------------

    @classmethod
    def load_rsi_classification(cls) -> dict:
        """Load ``rsi_classification.json`` from ``stats_folder()``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        from helpers import stats_folder
        path = stats_folder() + "rsi_classification.json"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"rsi_classification.json not found at: {path}"
            )
        with open(path, "r") as fh:
            return json.load(fh)

    @classmethod
    def load_diff_stats(cls) -> dict:
        """Load ``diff_stats.pkl`` from ``stats_folder()``.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        from helpers import stats_folder
        path = stats_folder() + "diff_stats.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"diff_stats.pkl not found at: {path}"
            )
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # NN column stats
    # ------------------------------------------------------------------

    def compute_nn_stats(self, df: pd.DataFrame, feature_cols: list) -> None:
        """Compute mean/std for each feature column using only closed-candle rows.

        The TF is parsed from the column name prefix (e.g. ``"15_nn_rsi_ma8"``
        → tf=15).  Only rows where ``{tf}_is_closed == True`` are used.

        Args:
            df:           Wide DataFrame.
            feature_cols: List of column names to compute stats for.
        """
        for col in feature_cols:
            # Parse TF from column prefix: "15_something" → tf=15
            parts = col.split("_", 1)
            try:
                tf = int(parts[0])
            except (ValueError, IndexError):
                # Cannot parse TF — fall back to all rows
                series = df[col].dropna()
            else:
                closed_col = f"{tf}_is_closed"
                if closed_col in df.columns:
                    series = df[df[closed_col] == True][col].dropna()  # noqa: E712
                else:
                    series = df[col].dropna()

            self.column_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
            }

    def get_stats(self, col: str) -> tuple:
        """Return ``(mean, std)`` for *col*.

        Raises:
            KeyError: If *col* is not in ``column_stats``.
        """
        if col not in self.column_stats:
            raise KeyError(f"Column '{col}' not found in column_stats")
        entry = self.column_stats[col]
        return float(entry["mean"]), float(entry["std"])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle self to *path* using an atomic write (write .tmp then rename).

        Args:
            path: Destination file path.
        """
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh)
        os.rename(tmp, path)

    @classmethod
    def load(cls, path: str) -> "DataAttributes":
        """Unpickle and return a :class:`DataAttributes` instance from *path*.

        Args:
            path: Source file path.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ===========================================================================
# Factory registry — maps field names → factory callables
# ===========================================================================

_FIELD_REGISTRY: dict[str, object] = {
    # Momentum
    "rsi_14": lambda cfg: RSI14Field(),
    "rsi_ma8": lambda cfg: RSI_MAField("rsi_14", 8, "rsi_ma8"),
    "rsi_ma12": lambda cfg: RSI_MAField("rsi_14", 12, "rsi_ma12"),
    "rsi_ma24": lambda cfg: RSI_MAField("rsi_14", 24, "rsi_ma24"),
    "rsi_ma8_diff": lambda cfg: RSI_MA_DiffField("rsi_ma8", "rsi_ma8_diff"),
    "rsi_ma12_diff": lambda cfg: RSI_MA_DiffField("rsi_ma12", "rsi_ma12_diff"),
    "rsi_ma24_diff": lambda cfg: RSI_MA_DiffField("rsi_ma24", "rsi_ma24_diff"),
    # Trend
    "ema_7": lambda cfg: EMAField(7, "ema_7"),
    "ema_14": lambda cfg: EMAField(14, "ema_14"),
    "ema_25": lambda cfg: EMAField(25, "ema_25"),
    "ema_50": lambda cfg: EMAField(50, "ema_50"),
    "ema_100": lambda cfg: EMAField(100, "ema_100"),
    "macd": lambda cfg: MACDField(),
    "macd_signal": lambda cfg: MACDSignalField(),
    "macd_hist": lambda cfg: MACDHistField(),
    "macd_fast": lambda cfg: MACDFastField(),
    "macd_fast_signal": lambda cfg: MACDFastSignalField(),
    # Oscillators
    "sar": lambda cfg: SARField(),
    "cci_14": lambda cfg: CCI14Field(),
    "cci_ma": lambda cfg: CCI_MAField(),
    # Volatility
    "atr_14": lambda cfg: ATR14Field(),
    "natr_14": lambda cfg: NATR14Field(),
    "atr_ma": lambda cfg: ATR_MAField(),
    "bb_upper": lambda cfg: BollingerUpperField(),
    "bb_middle": lambda cfg: BollingerMiddleField(),
    "bb_lower": lambda cfg: BollingerLowerField(),
    "bb_fast_upper": lambda cfg: BollingerFastUpperField(),
    "bb_fast_lower": lambda cfg: BollingerFastLowerField(),
    "bb_wide_upper": lambda cfg: BollingerWideUpperField(),
    "bb_wide_lower": lambda cfg: BollingerWideLowerField(),
    # Volume
    "vol_ma": lambda cfg: VolMAField(),
    "vol_buy_ma": lambda cfg: VolBuyMAField(),
    "vol_sell_ma": lambda cfg: VolSellMAField(),
    # Price derivatives
    "close_diff_prc": lambda cfg: CloseDiffPrcField(),
    "close_diff_prc_rm": lambda cfg: CloseDiffPrcRMField(),
    "close_diff_prc_rm_mean_above": lambda cfg: CloseDiffPrcRMMeanAboveField(),
    "close_diff_prc_rm_mean_below": lambda cfg: CloseDiffPrcRMMeanBelowField(),
    "high_diff_prc": lambda cfg: HighDiffPrcField(),
    "high_diff_prc_rm": lambda cfg: HighDiffPrcRMField(),
    "high_diff_prc_rm_mean_above": lambda cfg: HighDiffPrcRMMeanAboveField(),
    "high_diff_prc_rm_mean_below": lambda cfg: HighDiffPrcRMMeanBelowField(),
    "low_diff_prc": lambda cfg: LowDiffPrcField(),
    "low_diff_prc_rm": lambda cfg: LowDiffPrcRMField(),
    "low_diff_prc_rm_mean_above": lambda cfg: LowDiffPrcRMMeanAboveField(),
    "low_diff_prc_rm_mean_below": lambda cfg: LowDiffPrcRMMeanBelowField(),
    # Classification
    "move_class": lambda cfg: MoveClassField(),
    "zone_class": lambda cfg: ZoneClassField(),
    "over_low": lambda cfg: OverLowField(),
    "over_high": lambda cfg: OverHighField(),
    # Targets
    "tgt_long": lambda cfg: TgtLongField(),
    "sl_long": lambda cfg: SLLongField(),
    "tgt_short": lambda cfg: TgtShortField(),
    "sl_short": lambda cfg: SLShortField(),
    "ZB": lambda cfg: ZBField(),
    "ZS": lambda cfg: ZSField(),
    # Trend flags
    "trend_up": lambda cfg: TrendUpField(),
    "trend_down": lambda cfg: TrendDownField(),
    # NN features
    "nn_rsi_ma8_norm_mean": lambda cfg: NNRSINormField(),
    "nn_close_diff_atr_ma": lambda cfg: NNCloseDiffATRField(),
}
