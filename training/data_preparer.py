# training/data_preparer.py — DataPreparer: orchestrates the full data-preparation
# pipeline from raw OHLCV → resampled wide DataFrame → indicators → attributes.
#
# Dependency order (must be respected):
#   raw OHLCV → wide multi-TF df
#       → base indicators (momentum/trend/volatility/oscillators/volume/price_derivatives/trend_flags/nn_features)
#       → base attributes (rsi_classification.json, diff_stats.pkl)
#       → class indicators (classification, targets — only TFs [15,60,240,1440])
#       → merge df_with_nn.pkl (if present)
#       → nn attributes (column_stats for normalization)
#       → save df_with_indicators.pkl + data_attributes.pkl

from __future__ import annotations

import os
import warnings

import pandas as pd

# Heavy imports (talib-dependent) are deferred to method bodies to allow
# test-time patching without triggering talib at module import time.
# config_loader is safe to import eagerly.
from config_loader import CANDLES, load_nn_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_GROUPS: list[str] = [
    "momentum",
    "trend",
    "volatility",
    "oscillators",
    "volume",
    "price_derivatives",
    "trend_flags",
    "nn_features",
]

CLASS_GROUPS: list[str] = ["classification", "targets"]

CLASS_TFS: list[int] = [15, 60, 240, 1440]

# Required columns in the raw graber_data.pkl file
_REQUIRED_RAW_COLS: list[str] = ["open_time", "o", "h", "l", "c", "v"]

# Rename map for short graber column names → standard names
_RENAME_MAP: dict[str, str] = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}


# ---------------------------------------------------------------------------
# Parallel indicator-pass helpers
# ---------------------------------------------------------------------------

# Wide df shared with fork()ed pool workers via copy-on-write — set right
# before the pool is created, cleared after. Read-only in workers; passing it
# through task args would pickle gigabytes per task instead.
_PARALLEL_DF: "pd.DataFrame | None" = None


class _SliceDataPoint:
    """Holds ONE indicator-input slice so successive fields share it.

    WideDataPoint.get_df builds a fresh slice per call, so writes from
    earlier fields would be lost to later (dependent) fields and never
    reach the wide df. This wrapper pins the slice for the (ts, tf)
    computation; the slice's last row is then copied back.
    """

    def __init__(self, slice_df: pd.DataFrame, ts: pd.Timestamp) -> None:
        self._df = slice_df
        self._ts = ts

    def get_df(self, tf: int) -> pd.DataFrame:
        return self._df

    @property
    def timestamp(self) -> pd.Timestamp:
        return self._ts


def _split_index_chunks(index: pd.Index, n_chunks: int) -> list[pd.Index]:
    """Split *index* into at most *n_chunks* contiguous, order-preserving parts."""
    if len(index) == 0:
        return [index]
    n_chunks = max(1, min(n_chunks, len(index)))
    size = -(-len(index) // n_chunks)  # ceil division
    return [index[i:i + size] for i in range(0, len(index), size)]


def _compute_tf_rows(
    df: pd.DataFrame,
    tf: int,
    rows: pd.Index,
    groups: list[str],
) -> dict[str, list[float]]:
    """Compute indicator values of *groups* for one tf over *rows* of *df*.

    Pure with respect to df row order: each (ts, tf) computation reads only
    the lookback slice ending at ts, never another row's computed values —
    which is what makes chunked parallel execution safe.
    """
    from indicators import Indicators, build_indicator_input  # lazy import

    fields = Indicators._sorted_fields(tf, groups=groups)
    out_cols = [f"{tf}_{field.name}" for field in fields]

    # Collect per-row results in plain lists; bulk-assign once per tf
    # (per-cell .loc writes on a 20k×100+ frame are prohibitively slow).
    # Fragmentation of the throwaway slices is intentional — sequential
    # inserts beat pre-allocation there (see Indicators._run_fields) —
    # so silence pandas' PerformanceWarning for the loop.
    results: dict[str, list[float]] = {col: [] for col in out_cols}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for ts in rows:
            slice_df = build_indicator_input(df, ts, tf)
            data_point = _SliceDataPoint(slice_df, ts)
            Indicators.compute_group(data_point, tf, groups=groups)
            last = slice_df.iloc[-1]
            for col in out_cols:
                results[col].append(last[col] if col in slice_df.columns else float("nan"))
    return results


def _pool_compute_chunk(args: tuple[int, pd.Index, list[str]]) -> dict[str, list[float]]:
    """Pool worker: compute one (tf, row-chunk) against the fork-shared df."""
    tf, rows, groups = args
    return _compute_tf_rows(_PARALLEL_DF, tf, rows, groups)


# ---------------------------------------------------------------------------
# DataPreparer
# ---------------------------------------------------------------------------

class DataPreparer:
    """Orchestrates the full data-preparation pipeline.

    Args:
        config_path:            Path to indicators_config.yaml.
        output_path:            Path for df_with_indicators.pkl (plain DataFrame).
        attributes_output_path: Path for data_attributes.pkl.
        nn_output_path:         Path to df_with_nn.pkl; checked by _merge_nn_output().
    """

    def __init__(
        self,
        config_path: str,
        output_path: str,
        attributes_output_path: str,
        nn_output_path: str = "df_with_nn.pkl",
    ) -> None:
        self.config_path = config_path
        self.output_path = output_path
        self.attributes_output_path = attributes_output_path
        self.nn_output_path = nn_output_path
        self.num_workers: int = int(
            os.environ.get("NUM_WORKERS", os.environ.get("AVAIABLE_THREADS", "4"))
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def prepare(self, raw_data_path: str, data_start_ms: int | None = None) -> None:
        """Execute the full pipeline end-to-end.

        Steps (in order):
          1. Load raw 1-min OHLCV → raw_df
          2. Build wide multi-TF DataFrame from raw_df
          3. Compute base indicators (fills all non-stats-dependent columns)
          4. Compute base attributes (writes rsi_classification.json + diff_stats.pkl)
          5. Compute class indicators (classification + targets; only TFs [15,60,240,1440])
          6. Trim warmup rows (drop everything before data_start_ms)
          7. Compute lookahead profit labels (labels: config section)
          8. Merge df_with_nn.pkl columns (left-join on index; no-op if absent)
          9. Compute NN normalisation stats → data_attributes
         10. Save wide_df atomically to output_path
         11. Save data_attributes to attributes_output_path

        Args:
            raw_data_path:  Path to graber_data.pkl.
            data_start_ms:  DATA_START as Unix milliseconds. Rows before this
                            point are warmup history: they feed indicator
                            lookback windows (steps 3+5 compute nothing for
                            them) and are dropped in step 6 so the saved frame
                            starts at DATA_START. None = compute and keep the
                            full range (legacy behavior).
        """
        start_ts = (
            pd.Timestamp(data_start_ms, unit="ms", tz="UTC")
            if data_start_ms is not None
            else None
        )

        # Step 1
        raw_df = self._load_raw_data(raw_data_path)

        # Step 2
        wide_df = self._build_base_dataframe(raw_df)

        # Step 3 — base indicators (no classification/target stats required)
        self._compute_base_indicators(wide_df, start_ts)

        # Step 4 — derive rsi_classification.json + diff_stats.pkl
        self._compute_base_attributes(wide_df)

        # Step 5 — classification + target columns (require stats from step 4)
        self._compute_class_indicators(wide_df, start_ts)

        # Step 6 — drop warmup rows: indicators are filled from start_ts on,
        # so the leading history has served its purpose as lookback input.
        if start_ts is not None:
            wide_df = wide_df.loc[wide_df.index >= start_ts]

        # Step 7 — lookahead profit labels on the trimmed frame (labels need
        # future rows inside the simulation window, never warmup history)
        self._compute_profit_labels(wide_df)

        # Step 8 — optional NN output merge
        self._merge_nn_output(wide_df)

        # Step 9 — NN normalisation stats
        data_attributes = self._compute_nn_attributes(wide_df)

        # Step 10 — atomic save of wide_df
        tmp_out = self.output_path + ".tmp"
        wide_df.to_pickle(tmp_out)
        os.rename(tmp_out, self.output_path)

        # Step 11 — save DataAttributes
        data_attributes.save(self.attributes_output_path)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _load_raw_data(self, path: str) -> pd.DataFrame:
        """Load graber_data.pkl, validate columns, set open_time as index.

        Args:
            path: Path to graber_data.pkl.

        Returns:
            DataFrame with standard column names (open/high/low/close/volume)
            and open_time as the index.

        Raises:
            ValueError: If any required column is missing.
        """
        raw: pd.DataFrame = pd.read_pickle(path)

        # Phase-02 contract: open_time is the index of graber_data.pkl.
        # Normalize to a column so validation/rename below has one shape to handle.
        if "open_time" not in raw.columns and raw.index.name == "open_time":
            raw = raw.reset_index()

        # Validate required columns are present
        missing = [col for col in _REQUIRED_RAW_COLS if col not in raw.columns]
        if missing:
            raise ValueError(
                f"graber_data.pkl is missing required columns: {missing}. "
                f"Available columns: {list(raw.columns)}"
            )

        # Rename short names to standard names
        actual_rename = {k: v for k, v in _RENAME_MAP.items() if k in raw.columns}
        if actual_rename:
            raw = raw.rename(columns=actual_rename)

        # Also add taker_base_vol placeholder if absent (needed by _build_wide_df)
        if "taker_base_vol" not in raw.columns:
            raw["taker_base_vol"] = 0.0

        # Set open_time as index
        if "open_time" in raw.columns:
            raw = raw.set_index("open_time")

        return raw

    def _build_base_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Build wide multi-TF DataFrame from the raw 1-min OHLCV DataFrame.

        Calls _build_wide_df() from data.py directly (does NOT call get_stock_data,
        which reads from env-based fixed paths).

        Args:
            raw_df: 1-min OHLCV DataFrame with standard column names and
                    DatetimeIndex (output of _load_raw_data).

        Returns:
            Wide DataFrame with {tf}_{col} columns for all TFs in CANDLES.
        """
        from data import _build_wide_df  # lazy: avoids circular dep / talib at module level
        return _build_wide_df(raw_df)

    def _compute_base_indicators(
        self, df: pd.DataFrame, start_ts: pd.Timestamp | None = None
    ) -> None:
        """Compute base indicator groups for every TF from start_ts onward.

        Groups computed: momentum, trend, volatility, oscillators, volume,
        price_derivatives, trend_flags, nn_features.

        Iterates every (tf, timestamp) pair and calls Indicators.compute_group
        with a WideDataPoint wrapping the current row context.

        Args:
            df:       Wide DataFrame (mutated in place via WideDataPoint.get_df()).
            start_ts: First timestamp to compute for; earlier rows are warmup
                      input only. None = compute every row.
        """
        self._run_indicator_pass(df, BASE_GROUPS, CANDLES, start_ts)

    def _compute_base_attributes(self, df: pd.DataFrame) -> None:
        """Derive rsi_classification.json and diff_stats.pkl from the DataFrame.

        Instantiates DataAttributes and calls compute(df).  The write is
        idempotent — if the files already exist they are not overwritten.

        Args:
            df: Wide DataFrame that already has base indicator columns filled.
        """
        from indicators import DataAttributes  # lazy import
        data_attributes = DataAttributes()
        data_attributes.compute(df)

    def _compute_class_indicators(
        self, df: pd.DataFrame, start_ts: pd.Timestamp | None = None
    ) -> None:
        """Compute classification and target columns for TFs [15, 60, 240, 1440].

        Skips tf=1 and tf=5 which do not carry classification/target features.

        Args:
            df:       Wide DataFrame (mutated in place).
            start_ts: First timestamp to compute for; earlier rows are warmup
                      input only. None = compute every row.
        """
        self._run_indicator_pass(df, CLASS_GROUPS, CLASS_TFS, start_ts)

    def _compute_profit_labels(self, df: pd.DataFrame) -> None:
        """Append lookahead profit-label columns per the 'labels' config section.

        For each LabelSpecConfig and each of its tfs, calls
        add_profit_labels / add_profit_strict_labels from indicators.labels
        (vectorized over the whole frame — no per-row pass). No-op when the
        config has no labels section.

        Labels are lookahead by construction and are never indicator fields:
        they exist only in the saved wide frame, never in the registry or the
        live path.

        Args:
            df: Wide DataFrame (already trimmed to the simulation window;
                mutated in place).
        """
        from config_loader import load_labels_config
        specs = load_labels_config(self.config_path)
        if not specs:
            return

        from indicators.labels import add_profit_labels, add_profit_strict_labels

        for spec in specs:
            for tf in spec.tfs:
                if spec.type == "profit":
                    add_profit_labels(
                        df, tf, spec.n, spec.m, spec.x,
                        atr_period=spec.atr_period, ma_length=spec.ma_length,
                    )
                else:  # "profit_strict" — validated by load_labels_config
                    add_profit_strict_labels(
                        df, tf, spec.n, spec.m, spec.x, spec.l, spec.y,
                        atr_period=spec.atr_period, ma_length=spec.ma_length,
                    )

    def _merge_nn_output(self, df: pd.DataFrame) -> None:
        """Left-join columns from df_with_nn.pkl onto df (in place).

        If nn_output_path does not exist this method is a no-op.

        Args:
            df: Wide DataFrame.  New columns from df_with_nn.pkl are added;
                existing columns are NOT overwritten.
        """
        if not os.path.exists(self.nn_output_path):
            return

        nn_df: pd.DataFrame = pd.read_pickle(self.nn_output_path)

        # Add only columns that are not already in df
        new_cols = [c for c in nn_df.columns if c not in df.columns]
        if new_cols:
            joined = df.join(nn_df[new_cols], how="left")
            df[new_cols] = joined[new_cols]

    def _compute_nn_attributes(self, df: pd.DataFrame) -> "DataAttributes":
        """Compute NN normalisation stats and return a populated DataAttributes.

        Reads feature_cols from indicators_config.yaml (nn.feature_cols).
        Calls data_attributes.compute_nn_stats(df, feature_cols).

        Args:
            df: Wide DataFrame with all indicator columns populated.

        Returns:
            Populated DataAttributes instance (not yet saved to disk).
        """
        from indicators import DataAttributes  # lazy import
        nn_config = load_nn_config(self.config_path)
        feature_cols: list[str] = nn_config.get("feature_cols", [])

        data_attributes = DataAttributes()
        data_attributes.compute_nn_stats(df, feature_cols)
        return data_attributes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_indicator_pass(
        self,
        df: pd.DataFrame,
        groups: list[str],
        tfs: list[int],
        start_ts: "pd.Timestamp | None" = None,
    ) -> None:
        """Compute *groups* for every (tf, timestamp) pair via _compute_tf_rows.

        Each (ts, tf) computation reads only the lookback slice ending at ts
        (build_indicator_input) and never another row's computed values — the
        per-row results are bulk-assigned only after all rows are computed.
        That independence allows splitting compute_index into contiguous
        chunks and computing them in fork()ed worker processes when
        self.num_workers > 1; workers read the wide df through the
        copy-on-write _PARALLEL_DF module global.

        Args:
            df:       Wide DataFrame (mutated in place column-by-column).
            groups:   Indicator groups to compute.
            tfs:      Timeframes to compute indicators for.
            start_ts: First timestamp to compute for. Earlier rows still feed
                      build_indicator_input lookback slices but get no values
                      of their own (NaN after the bulk assign).
                      None = compute every row.
        """
        from indicators import Indicators  # lazy import

        compute_index = df.index if start_ts is None else df.index[df.index >= start_ts]

        tf_cols: dict[int, list[str]] = {}
        for tf in tfs:
            fields = Indicators._sorted_fields(tf, groups=groups)
            if fields:
                tf_cols[tf] = [f"{tf}_{field.name}" for field in fields]
        if not tf_cols:
            return

        n_workers = min(self.num_workers, len(compute_index))
        if n_workers > 1:
            tf_results = self._compute_parallel(df, tf_cols, compute_index, groups, n_workers)
        else:
            tf_results = {
                tf: _compute_tf_rows(df, tf, compute_index, groups) for tf in tf_cols
            }

        for tf, out_cols in tf_cols.items():
            # Single multi-column setitem — per-column inserts fragment the
            # frame (one block each) and trigger PerformanceWarning spam.
            # Index alignment leaves NaN in rows before start_ts.
            df[out_cols] = pd.DataFrame(tf_results[tf], index=compute_index)

    @staticmethod
    def _compute_parallel(
        df: pd.DataFrame,
        tf_cols: dict[int, list[str]],
        compute_index: pd.Index,
        groups: list[str],
        n_workers: int,
    ) -> dict[int, dict[str, list[float]]]:
        """Fan (tf, row-chunk) tasks out to a fork pool; return results per tf.

        Requires the fork start method: workers inherit the wide df through
        _PARALLEL_DF without pickling it (copy-on-write).
        """
        import multiprocessing

        global _PARALLEL_DF

        chunks = _split_index_chunks(compute_index, n_workers)
        tasks = [(tf, chunk, groups) for tf in tf_cols for chunk in chunks]

        ctx = multiprocessing.get_context("fork")
        _PARALLEL_DF = df
        try:
            with ctx.Pool(processes=n_workers) as pool:
                chunk_results = pool.map(_pool_compute_chunk, tasks)
        finally:
            _PARALLEL_DF = None

        # Reassemble: tasks are ordered tf-major, chunk-minor; concatenating
        # chunk lists in submission order restores compute_index order.
        tf_results: dict[int, dict[str, list[float]]] = {}
        for (tf, _, _), result in zip(tasks, chunk_results):
            merged = tf_results.setdefault(tf, {col: [] for col in tf_cols[tf]})
            for col in tf_cols[tf]:
                merged[col].extend(result[col])
        return tf_results
