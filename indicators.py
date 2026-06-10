# indicators.py — IndicatorField ABC, Indicators orchestrator, and build_indicator_input.
#
# IndicatorField: abstract base class for a single computed indicator column.
# Indicators:     static orchestrator that runs all applicable fields on a DataPoint.
# build_indicator_input: slices a wide DataFrame to give Indicators.compute() its input.

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import pandas as pd

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
        """
        if not self.resource_dependencies:
            return True

        from helpers import stats_folder  # local import: reads env vars at call time

        base = stats_folder()
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
        """Load registry from config on first use."""
        if cls._registry is None:
            configs = load_indicators_config()
            cls._registry = [_PlaceholderField(cfg) for cfg in configs]
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
