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
        self.num_workers: int = int(os.environ.get("NUM_WORKERS", "4"))

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
          7. Merge df_with_nn.pkl columns (left-join on index; no-op if absent)
          8. Compute NN normalisation stats → data_attributes
          9. Save wide_df atomically to output_path
         10. Save data_attributes to attributes_output_path

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

        # Step 7 — optional NN output merge
        self._merge_nn_output(wide_df)

        # Step 8 — NN normalisation stats
        data_attributes = self._compute_nn_attributes(wide_df)

        # Step 9 — atomic save of wide_df
        tmp_out = self.output_path + ".tmp"
        wide_df.to_pickle(tmp_out)
        os.rename(tmp_out, self.output_path)

        # Step 10 — save DataAttributes
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
        """Iterate every (tf, timestamp) pair and call Indicators.compute_group.

        For each timestamp in df.index (from start_ts onward) and each tf in tfs:
          - Wraps df at ts into a WideDataPoint
          - Calls Indicators.compute_group(data_point, tf, groups=groups)

        The WideDataPoint.get_df(tf) call returns a per-tf slice that
        Indicators.compute_group mutates in place (via field.compute writing
        to df[f"{tf}_{field.name}"]).

        Because WideDataPoint.get_df delegates to build_indicator_input which
        returns a *view/copy* for indicator computation but writes results back
        via the WideDataPoint.get_df mechanism, we use a direct approach:
        for each ts we build a WideDataPoint and let Indicators.compute_group
        write directly into the slice.  The wide df is mutated row-by-row.

        Args:
            df:       Wide DataFrame (mutated in place column-by-column).
            groups:   Indicator groups to compute.
            tfs:      Timeframes to compute indicators for.
            start_ts: First timestamp to compute for. Earlier rows still feed
                      build_indicator_input lookback slices but get no values
                      of their own (NaN after the bulk assign). Safe because
                      per-row results are written back only after the loop, so
                      no row ever reads another row's computed values.
                      None = compute every row.
        """
        from indicators import Indicators, build_indicator_input  # lazy import

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

        compute_index = df.index if start_ts is None else df.index[df.index >= start_ts]

        for tf in tfs:
            fields = Indicators._sorted_fields(tf, groups=groups)
            if not fields:
                continue
            out_cols = [f"{tf}_{field.name}" for field in fields]

            # Collect per-row results in plain lists; bulk-assign once per tf
            # (per-cell .loc writes on a 20k×100+ frame are prohibitively slow).
            # Fragmentation of the throwaway slices is intentional — sequential
            # inserts beat pre-allocation there (see Indicators._run_fields) —
            # so silence pandas' PerformanceWarning for the loop.
            results: dict[str, list[float]] = {col: [] for col in out_cols}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                for ts in compute_index:
                    slice_df = build_indicator_input(df, ts, tf)
                    data_point = _SliceDataPoint(slice_df, ts)
                    Indicators.compute_group(data_point, tf, groups=groups)
                    last = slice_df.iloc[-1]
                    for col in out_cols:
                        results[col].append(last[col] if col in slice_df.columns else float("nan"))

            # Single multi-column setitem — per-column inserts fragment the
            # frame (one block each) and trigger PerformanceWarning spam.
            # Index alignment leaves NaN in rows before start_ts.
            df[out_cols] = pd.DataFrame(results, index=compute_index)
