"""indicators.framework — IndicatorField ABC, Indicators orchestrator, build_indicator_input."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from config_loader import CANDLES, load_indicators_config
from constants import INDICATOR_WINDOW_ROWS
from logs import log_warning

# Module-level set to track which missing resource deps have already been warned about.
_warned_resources: set[str] = set()

# Cache of closed-row integer positions per (id(df), tf) for build_indicator_input.
# Invalidated by frame length change; id() collisions after gc are guarded by the
# length check and the fact that {tf}_is_closed never changes for a given frame.
_closed_pos_cache: dict[tuple[int, int], tuple[int, np.ndarray]] = {}


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
    """Return indicator input slice at ts for tf (up to INDICATOR_WINDOW_ROWS rows).

    Selects all rows up to and including ts, keeps only closed-candle rows for
    the given tf, then appends the partial current candle if ts is not closed.
    Returns at most INDICATOR_WINDOW_ROWS rows (the tail).

    Args:
        df: Wide DataFrame with columns ``{tf}_is_closed`` and a DatetimeIndex.
        ts: The "now" timestamp — must be present in df.index.
        tf: Timeframe in minutes.

    Returns:
        DataFrame of at most INDICATOR_WINDOW_ROWS rows suitable for indicator
        computation.
    """
    closed_col = f"{tf}_is_closed"

    # Positional fast path: label-slicing the prefix (df[:ts]) and boolean-
    # filtering it copies O(prefix) data per call, which makes the offline
    # per-row indicator pass O(n²). Cache the closed-row positions per
    # (frame, tf) and select just the needed window with iloc instead.
    i = df.index.get_loc(ts)
    key = (id(df), tf)
    cached = _closed_pos_cache.get(key)
    if cached is None or cached[0] != len(df):
        pos = np.flatnonzero(df[closed_col].to_numpy(dtype=bool))
        _closed_pos_cache[key] = (len(df), pos)
    else:
        pos = cached[1]

    k = np.searchsorted(pos, i, side="right")
    closed_pos = pos[:k]

    if len(closed_pos) > 0 and closed_pos[-1] == i:
        # Current row is a closed candle — window is the last N closed rows.
        sel = closed_pos[-INDICATOR_WINDOW_ROWS:]
    else:
        # Append the current row as a partial candle (N-1 closed + partial,
        # matching the previous tail(N)-after-append behavior).
        sel = np.append(closed_pos[-(INDICATOR_WINDOW_ROWS - 1):], i)

    return df.iloc[sel]


# ---------------------------------------------------------------------------
# Indicators — static orchestrator
# ---------------------------------------------------------------------------

def _field_registry() -> dict:
    """Lazy import of the registry — registry imports the library, which
    imports this framework module; importing eagerly here would be circular."""
    from .registry import _FIELD_REGISTRY  # noqa: WPS433
    return _FIELD_REGISTRY


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
                factory = _field_registry().get(cfg.name)
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
        cls._run_fields(data_point, tf, cls._sorted_fields(tf))

    @classmethod
    def compute_group(cls, data_point, tf: int, groups: list[str]) -> None:
        """Run only fields in specified groups, in dependency order.

        Args:
            data_point: DataPoint providing mutable DataFrame access via get_df(tf).
            tf: Timeframe in minutes.
            groups: Only fields whose group is in this list will be computed.
        """
        cls._run_fields(data_point, tf, cls._sorted_fields(tf, groups=groups))

    @classmethod
    def _run_fields(cls, data_point, tf: int, fields: list[IndicatorField]) -> None:
        """Compute fields in order, writing each into data_point.get_df(tf).

        Sequential single-column inserts are intentional: dependent fields
        must see earlier results, and on the small per-timestamp slices this
        is faster than pre-allocating (a shared multi-column block gets
        copy-on-write duplicated on every subsequent field write). Callers
        looping per timestamp should suppress pandas' fragmentation
        PerformanceWarning around the loop.
        """
        df = data_point.get_df(tf)
        for field in fields:
            series = field.compute(data_point, tf)
            df[f"{tf}_{field.name}"] = series

