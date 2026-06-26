# training/data_preparer.py — DataPreparer: orchestrates the full data-preparation
# pipeline from raw OHLCV → resampled wide DataFrame → indicators → attributes.
#
# Dependency order (must be respected):
#   raw OHLCV → wide multi-TF df
#       → base indicators (momentum/trend/volatility/oscillators/volume/price_derivatives/trend_flags/nn_features)
#       → base attributes (rsi_classification.json, diff_stats.pkl)
#       → class indicators (classification, targets — only TFs [15,60,240,1440])
#       → nn attributes (column_stats for normalization)
#       → save df_with_indicators.pkl + data_attributes.pkl
#
# df_with_indicators.pkl is single-writer: nn_res_* result columns are NEVER
# merged here. They reach consumers only via the load-time join of the additive
# df_with_nn.pkl (data.join_nn_results).

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from typing import Callable

import pandas as pd

# Heavy imports (talib-dependent) are deferred to method bodies to allow
# test-time patching without triggering talib at module import time.
# config_loader is safe to import eagerly.
from config_loader import CANDLES, chunk_config, load_nn_config, warmup_start_ms

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
# Chunked-preparation helpers (time-portion split, boundaries, progress)
# ---------------------------------------------------------------------------

_MS_PER_MIN = 60_000
_MS_PER_DAY = 86_400_000


def _window_row_count(start_ms: int, end_ms: int) -> int:
    """Number of 1-min rows in [start_ms, end_ms)."""
    if end_ms <= start_ms:
        return 0
    return (end_ms - start_ms) // _MS_PER_MIN


def _chunk_boundaries(
    start_ms: int, end_ms: int, span_days: int
) -> list[tuple[int, int]]:
    """Contiguous [start, end) time portions of *span_days* each.

    The last portion is clamped to *end_ms*. A window no larger than one span
    returns a single ``(start_ms, end_ms)``. Returns ``[]`` when the window is
    empty (``start_ms >= end_ms``). Pure function of its arguments — the single
    source of truth for portion identity (a part file at index *i* means that
    portion is done).
    """
    if end_ms <= start_ms:
        return []
    span_ms = span_days * _MS_PER_DAY
    boundaries: list[tuple[int, int]] = []
    cur = start_ms
    while cur < end_ms:
        boundaries.append((cur, min(cur + span_ms, end_ms)))
        cur += span_ms
    return boundaries


def _fmt_duration(seconds: float) -> str:
    """Format *seconds* as <h>h<m>m<s>s dropping leading zero units (e.g. 8m12s)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class _ChunkProgress:
    """Logs per-portion completion with elapsed time and a linear ETA.

    The clock is injected (``now``) so ETA math is unit-testable without
    sleeping and stays deterministic in tests.
    """

    def __init__(
        self, pass_name: str, total: int, *, now: Callable[[], float] = time.monotonic
    ) -> None:
        self.pass_name = pass_name
        self.total = total
        self._now = now
        self._start = now()

    def tick(self, done: int, rows: int) -> str:
        """Record *done*/total portions complete (*rows* processed). Log + return
        the formatted line. ETA = elapsed/done × (total − done); '—' while done==0."""
        if self.total <= 0:
            return ""
        from logs import log

        elapsed = self._now() - self._start
        pct = int(done * 100 / self.total)
        if done > 0:
            eta = elapsed / done * (self.total - done)
            eta_str = _fmt_duration(eta)
        else:
            eta_str = "—"
        line = (
            f"{self.pass_name}  portion {done}/{self.total}  {pct}%  "
            f"elapsed {_fmt_duration(elapsed)}  ETA {eta_str}  rows={rows}"
        )
        log(line)
        return line


# ---------------------------------------------------------------------------
# DataPreparer
# ---------------------------------------------------------------------------

class DataPreparer:
    """Orchestrates the full data-preparation pipeline.

    Args:
        config_path:            Path to indicators_config.yaml.
        output_path:            Path for df_with_indicators.pkl (plain DataFrame).
        attributes_output_path: Path for data_attributes.pkl.
    """

    def __init__(
        self,
        config_path: str,
        output_path: str,
        attributes_output_path: str,
    ) -> None:
        self.config_path = config_path
        self.output_path = output_path
        self.attributes_output_path = attributes_output_path
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
          8. Compute NN normalisation stats → data_attributes
          9. Save wide_df atomically to output_path
         10. Save data_attributes to attributes_output_path

        NN result columns (nn_res_*) are NOT merged here — df_with_indicators.pkl
        stays single-writer. nn_res_* reach consumers only via the load-time
        join (data.join_nn_results) of the additive, disposable df_with_nn.pkl.

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

        # Step 8 — NN normalisation stats
        data_attributes = self._compute_nn_attributes(wide_df)

        # Step 9 — atomic save of wide_df
        tmp_out = self.output_path + ".tmp"
        wide_df.to_pickle(tmp_out)
        os.rename(tmp_out, self.output_path)

        # Step 10 — save DataAttributes
        data_attributes.save(self.attributes_output_path)

    # ------------------------------------------------------------------
    # Chunked entry point (resumable, progress-logged)
    # ------------------------------------------------------------------

    def prepare_chunked(
        self, raw_data_path: str, data_start_ms: int, data_end_ms: int
    ) -> None:
        """Resumable, progress-logged preparation.

        Below CHUNK_MIN_ROWS the whole dataset runs through the unchanged
        single-pass ``prepare()``. Otherwise the window is split into
        CHUNK_SPAN_DAYS time portions, each computed and persisted to its own
        part file (skipped on rerun if present), then merged. The chunked output
        is identical to single-pass within float tolerance.

        Args:
            raw_data_path: Path to graber_data.pkl (shared by all portions).
            data_start_ms: DATA_START as Unix ms (inclusive).
            data_end_ms:   DATA_END as Unix ms (exclusive).
        """
        from logs import log

        span_days, min_rows = chunk_config()
        rows = _window_row_count(data_start_ms, data_end_ms)
        if rows < min_rows:
            log(
                f"[prepare] single-chunk path ({rows} rows < CHUNK_MIN_ROWS={min_rows})"
            )
            self.prepare(raw_data_path, data_start_ms)
            return

        boundaries = _chunk_boundaries(data_start_ms, data_end_ms, span_days)
        n = len(boundaries)
        log(
            f"[prepare] chunked path: {rows} rows → {n} portions "
            f"of {span_days}d (CHUNK_MIN_ROWS={min_rows})"
        )

        # Invalidate stale parts if the chunk config changed between runs.
        self._guard_chunk_manifest(data_start_ms, data_end_ms, span_days)

        # Shared raw frame — loaded once, read-only across every portion.
        raw_df = self._load_raw_data(raw_data_path)

        # Pass 1 — base indicators per portion.
        prog = _ChunkProgress("[prepare] pass1 base-ind", n)
        base_paths: list[str] = []
        done_rows = 0
        for i, window in enumerate(boundaries):
            base_paths.append(self._pass1_base_portion(raw_df, i, window))
            done_rows += _window_row_count(*window)
            prog.tick(i + 1, done_rows)

        # Global base-attribute stats (consistent thresholds for every portion).
        log("[prepare] computing global base-attribute stats")
        self._global_base_stats(base_paths)

        # Pass 2 — class indicators per portion.
        prog = _ChunkProgress("[prepare] pass2 class-ind", n)
        final_paths: list[str] = []
        done_rows = 0
        for i, window in enumerate(boundaries):
            prev = base_paths[i - 1] if i > 0 else None
            final_paths.append(self._pass2_class_portion(i, window, prev))
            done_rows += _window_row_count(*window)
            prog.tick(i + 1, done_rows)

        # Merge — labels + nn merge + nn-norm over the full frame, then save.
        log("[prepare] merging parts → final output")
        self._merge_parts(final_paths, base_paths)
        log("[prepare] done")

    # ------------------------------------------------------------------
    # Chunked-preparation part files + manifest
    # ------------------------------------------------------------------

    def _final_part_path(self, index: int) -> str:
        """output_path with .pkl → .part_NN.pkl (zero-padded 2-digit index)."""
        base, ext = os.path.splitext(self.output_path)
        return f"{base}.part_{index:02d}{ext}"

    def _base_part_path(self, index: int) -> str:
        """df_base.part_NN.pkl beside output_path."""
        return os.path.join(
            os.path.dirname(self.output_path), f"df_base.part_{index:02d}.pkl"
        )

    def _chunk_manifest_path(self) -> str:
        """chunk_manifest.json beside output_path."""
        return os.path.join(os.path.dirname(self.output_path), "chunk_manifest.json")

    @staticmethod
    def _chunk_config_hash(start_ms: int, end_ms: int, span_days: int) -> str:
        """Stable hash of the chunk config that determines portion identity."""
        payload = f"{start_ms}|{end_ms}|{span_days}".encode()
        return hashlib.sha256(payload).hexdigest()

    def _purge_parts(self) -> None:
        """Delete every base + final part file beside output_path."""
        import glob

        out_dir = os.path.dirname(self.output_path) or "."
        out_name = os.path.basename(os.path.splitext(self.output_path)[0])
        patterns = [
            os.path.join(out_dir, "df_base.part_*.pkl"),
            os.path.join(out_dir, f"{out_name}.part_*.pkl"),
        ]
        for pat in patterns:
            for path in glob.glob(pat):
                os.remove(path)

    def _guard_chunk_manifest(
        self, start_ms: int, end_ms: int, span_days: int
    ) -> None:
        """Purge stale parts when the chunk config changed, then record the hash."""
        cur = self._chunk_config_hash(start_ms, end_ms, span_days)
        manifest_path = self._chunk_manifest_path()
        if os.path.exists(manifest_path):
            try:
                stored = json.load(open(manifest_path)).get("config_hash")
            except (ValueError, OSError):
                stored = None
            if stored != cur:
                from logs import log

                log("[prepare] chunk config changed — purging stale parts")
                self._purge_parts()
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"config_hash": cur}, f)
        os.rename(tmp, manifest_path)

    # ------------------------------------------------------------------
    # Chunked-preparation portion workers
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_to_pickle(df: pd.DataFrame, path: str) -> None:
        """Pickle *df* to *path* via temp-file + os.rename (crash-safe)."""
        tmp = path + ".tmp"
        df.to_pickle(tmp)
        os.rename(tmp, path)

    def _pass1_base_portion(
        self, raw_df: pd.DataFrame, index: int, window: tuple[int, int]
    ) -> str:
        """Compute base indicators for owned window [w0, w1); save owned rows.

        Lookback (warmup) rows before w0 are taken from *raw_df* (the shared
        grabbed data) so per-row slices are full; only owned rows are saved.
        Skips (returns path) when the base part already exists.
        """
        part_path = self._base_part_path(index)
        if os.path.exists(part_path):
            return part_path

        w0, w1 = window
        w0_ts = pd.Timestamp(w0, unit="ms", tz="UTC")
        w1_ts = pd.Timestamp(w1, unit="ms", tz="UTC")
        lb_ts = pd.Timestamp(warmup_start_ms(w0), unit="ms", tz="UTC")

        raw_slice = raw_df.loc[(raw_df.index >= lb_ts) & (raw_df.index < w1_ts)]
        wide_df = self._build_base_dataframe(raw_slice)
        self._compute_base_indicators(wide_df, w0_ts)

        owned = wide_df.loc[(wide_df.index >= w0_ts) & (wide_df.index < w1_ts)]
        self._atomic_to_pickle(owned, part_path)
        return part_path

    def _global_base_stats(self, base_part_paths: list[str]) -> None:
        """Concatenate base parts and compute rsi_classification.json + diff_stats.pkl.

        Idempotent: DataAttributes.compute no-ops when the stats files exist.
        """
        from indicators import DataAttributes  # lazy import

        frames = [pd.read_pickle(p) for p in base_part_paths]
        full = pd.concat(frames)
        DataAttributes().compute(full)

    def _pass2_class_portion(
        self, index: int, window: tuple[int, int], prev_base_path: "str | None"
    ) -> str:
        """Compute class indicators for owned window [w0, w1); save the final part.

        Prepends a lookback margin from *prev_base_path* (when set) so per-row
        slices at the portion start are full (targets use shift(1)); the margin
        rows are dropped before save. Skips (returns path) when the final part
        already exists.
        """
        final_path = self._final_part_path(index)
        if os.path.exists(final_path):
            return final_path

        from constants import INDICATOR_WINDOW_ROWS  # lazy import

        base = pd.read_pickle(self._base_part_path(index))
        if prev_base_path is not None:
            margin = INDICATOR_WINDOW_ROWS * max(CANDLES)
            lead = pd.read_pickle(prev_base_path).iloc[-margin:]
            frame = pd.concat([lead, base])
        else:
            frame = base.copy()

        w0_ts = pd.Timestamp(window[0], unit="ms", tz="UTC")
        self._compute_class_indicators(frame, w0_ts)

        owned = frame.loc[frame.index >= w0_ts]
        self._atomic_to_pickle(owned, final_path)
        return final_path

    def _merge_parts(
        self, final_part_paths: list[str], base_part_paths: list[str]
    ) -> None:
        """Concat final parts → full frame; labels + nn-norm; save; cleanup.

        Labels and NN-normalisation run on the whole concatenated frame so chunk
        boundaries never affect them. Mirrors prepare() steps 7–10 exactly (the
        df_with_nn.pkl left-join lives in consumers, not prepare). Part files are
        deleted only after both outputs are saved (so a failed save resumes from
        merge, not Pass 1); cleanup is skipped when CHUNK_KEEP_PARTS is set.
        """
        frames = [pd.read_pickle(p) for p in final_part_paths]
        full = pd.concat(frames)

        # Steps mirror prepare() 7–10 on the concatenated frame.
        self._compute_profit_labels(full)
        data_attributes = self._compute_nn_attributes(full)

        self._atomic_to_pickle(full, self.output_path)
        data_attributes.save(self.attributes_output_path)

        if os.environ.get("CHUNK_KEEP_PARTS"):
            return
        for path in list(final_part_paths) + list(base_part_paths):
            if os.path.exists(path):
                os.remove(path)

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
