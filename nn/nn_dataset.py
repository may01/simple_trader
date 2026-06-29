"""nn/nn_dataset.py — NNDataset: modular, content-addressed tensor cache.

NNDataset materialises a prepared wide DataFrame into per-timeframe feature
tensors and a multi-target tensor ONCE per ``(feature-set, target-set, history,
split)``, caches them on disk under ``datasets/{dataset_hash}/``, and serves
them back with zero re-derivation. The agentic search loop reloads data
hundreds of times across trials; keying the cache by a content hash lets every
spec that resolves to the same config reuse the same directory.

This is a pure CONSUMER of ``df_with_indicators.pkl``:
  - NN features are engineered columns (``{tf}_{indicator}``) already present.
  - Targets are READ from profit-label columns (``{tf}_plong_*`` / ``_pshort_*``)
    or price-derived (regression); NNDataset never classifies them itself.

Lookback semantics replicate ``WideDataPoint.get(col, tf, shift)`` exactly:
  - shift=0    → the value at the timestamp row (current forming candle);
  - shift=N≥1  → the value at the Nth-last row where ``{tf}_is_closed == True``
                 at/up to that timestamp; NaN when history is insufficient.

Normalisation is leakage-free and shares the project's single layer with
``DataAttributes.compute_nn_stats``: robust winsorised stats
``{q01,q99,mean,std}`` are estimated on the RAW per-column source values,
closed-candle rows only, each timestamp once — restricted to the TRAIN split's
timestamp span (no val/holdout leak). The SAME stats are then applied to the
windowed feature tensors (``clip(x, q01, q99) → (x - mean)/std → clip(z, -4,
+4)``) and stored in the manifest for verbatim reuse by ``run_inference``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from indicators.labels import _fmt
from nn.device import nn_artefact_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    from indicators import DataAttributes
    from nn.nn_model_spec import GroupingSpec, NNModelSpec, TargetSpec


# ---------------------------------------------------------------------------
# Profit-label column naming (reuse labels.py helpers — never hand-roll floats)
# ---------------------------------------------------------------------------


def _profit_suffix(spec: "TargetSpec", horizon: int) -> str:
    """Suffix shared by add_profit_labels / add_profit_strict_labels.

    Non-strict:  n{n}_m{m}_x{x}
    Strict:      n{n}_m{m}_x{x}_l{l}_y{y}   (label_l, label_y read off spec)
    """
    base = f"n{_fmt(horizon)}_m{_fmt(spec.label_m)}_x{_fmt(spec.label_x)}"
    if spec.strict:
        label_l = getattr(spec, "label_l", None)
        label_y = getattr(spec, "label_y", None)
        if label_l is None or label_y is None:
            raise ValueError("strict target needs label_l/label_y on TargetSpec")
        base = f"{base}_l{_fmt(label_l)}_y{_fmt(label_y)}"
    return base


def _profit_long_col(spec: "TargetSpec", horizon: int) -> str:
    kind = "pslong" if spec.strict else "plong"
    return f"{spec.label_tf}_{kind}_{_profit_suffix(spec, horizon)}"


def _profit_short_col(spec: "TargetSpec", horizon: int) -> str:
    kind = "psshort" if spec.strict else "pshort"
    return f"{spec.label_tf}_{kind}_{_profit_suffix(spec, horizon)}"


# ---------------------------------------------------------------------------
# Lookback block builder (replicates WideDataPoint.get, vectorised)
# ---------------------------------------------------------------------------


def _build_tf_block(
    df: pd.DataFrame, tf: int, feature_cols: list[str], history_points: int
) -> np.ndarray:
    """Build the (rows, history_points, n_features) lookback block for one tf.

    History axis: [shift=0, shift=1, ..., shift=history_points-1].
      shift=0      → value at the row itself (forming candle).
      shift=k≥1    → value at the k-th-last closed row at/up to that row.

    Vectorisation:
      closed_pos      = positions (in row order) of closed-candle rows.
      n_le[i]         = number of closed rows at position <= i  (searchsorted).
      For shift k, the source position is closed_pos[n_le[i] - k] when that
      index is >= 0, else the value is NaN (insufficient history). Building a
      single (rows, history_points) gather-index matrix per tf lets us index
      each feature column in one vectorised take.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "NNDataset lookback requires a monotonically-ascending index; "
            "the vectorised searchsorted gather assumes ascending row order"
        )
    n_rows = len(df)
    closed_mask = df[f"{tf}_is_closed"].to_numpy().astype(bool)
    closed_pos = np.flatnonzero(closed_mask)  # ascending row positions

    # n_le[i] = count of closed positions <= i (1-indexed depth available)
    row_pos = np.arange(n_rows)
    n_le = np.searchsorted(closed_pos, row_pos, side="right")  # (n_rows,)

    # gather_idx[i, k] = source ROW position for (row i, shift k); -1 = invalid
    gather_idx = np.full((n_rows, history_points), -1, dtype=np.int64)
    # shift 0 → the row itself
    if history_points >= 1:
        gather_idx[:, 0] = row_pos
    # shift k≥1 → closed_pos[n_le - k] when n_le - k >= 0
    for k in range(1, history_points):
        depth = n_le - k  # index into closed_pos
        valid = depth >= 0
        idx = np.where(valid, depth, 0)
        src = closed_pos[idx] if len(closed_pos) else np.zeros(n_rows, dtype=np.int64)
        gather_idx[:, k] = np.where(valid, src, -1)

    valid_mask = gather_idx >= 0
    safe_idx = np.where(valid_mask, gather_idx, 0)

    block = np.empty((n_rows, history_points, len(feature_cols)), dtype=np.float64)
    for fi, col in enumerate(feature_cols):
        colvals = df[col].to_numpy(dtype=np.float64)
        gathered = colvals[safe_idx]  # (n_rows, history_points)
        gathered = np.where(valid_mask, gathered, np.nan)
        block[:, :, fi] = gathered
    return block


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------


def _target_block(
    df: pd.DataFrame, target: "TargetSpec"
) -> tuple[np.ndarray, dict]:
    """Build the (rows, width) target block and manifest entry for one target.

    Returns (block, manifest_entry). block has one column-group per horizon.
    """
    out_columns = target.out_columns()
    entry: dict = {
        "name": target.name,
        "kind": target.kind,
        "horizons": list(target.horizons),
        "out_columns": out_columns,
    }

    cols: list[np.ndarray] = []

    if target.kind == "direction":
        # 3-class per horizon: up=0, neutral=1, down=2 → one-hot width 3.
        srcs = []
        for h in target.horizons:
            long_col = _profit_long_col(target, h)
            short_col = _profit_short_col(target, h)
            srcs.append({"long": long_col, "short": short_col})
            for c in (long_col, short_col):
                if c not in df.columns:
                    raise ValueError(f"missing profit-label column {c!r}")
            lo = df[long_col].to_numpy(dtype=np.float64)
            sh = df[short_col].to_numpy(dtype=np.float64)
            cls = _derive_direction(lo, sh)  # float code in {0,1,2} or NaN
            onehot = _onehot3(cls)
            cols.append(onehot)
        entry["encoding"] = {"up": 0, "neutral": 1, "down": 2}
        entry["source"] = (
            {**srcs[0], "strict": target.strict}
            if len(srcs) == 1
            else {"per_horizon": srcs, "strict": target.strict}
        )

    elif target.kind == "direction_binary":
        if target.side not in ("long", "short"):
            raise ValueError(
                f"direction_binary target {target.name!r} needs "
                f"side='long'|'short', got {target.side!r}"
            )
        srcs = []
        for h in target.horizons:
            col_name = (
                _profit_long_col(target, h)
                if target.side == "long"
                else _profit_short_col(target, h)
            )
            srcs.append(col_name)
            if col_name not in df.columns:
                raise ValueError(f"missing profit-label column {col_name!r}")
            c = df[col_name].to_numpy(dtype=np.float64)
            cols.append(_binary_onehot(c))
        entry["side"] = target.side
        entry["encoding"] = {target.side: 0, "other": 1}
        entry["source"] = (
            {"column": srcs[0], "strict": target.strict}
            if len(srcs) == 1
            else {"per_horizon": srcs, "strict": target.strict}
        )

    elif target.kind == "label":
        srcs = []
        for h in target.horizons:
            long_col = _profit_long_col(target, h)
            srcs.append(long_col)
            if long_col not in df.columns:
                raise ValueError(f"missing profit-label column {long_col!r}")
            cols.append(df[long_col].to_numpy(dtype=np.float64).reshape(-1, 1))
        entry["source"] = (
            {"column": srcs[0], "strict": target.strict}
            if len(srcs) == 1
            else {"per_horizon": srcs, "strict": target.strict}
        )

    elif target.kind == "regression":
        close_col = (
            f"{target.label_tf}_close" if target.label_tf is not None else None
        )
        if close_col is None or close_col not in df.columns:
            raise ValueError(
                f"regression target {target.name!r} needs a close column "
                f"({close_col!r} not found)"
            )
        close = df[close_col].to_numpy(dtype=np.float64)
        for h in target.horizons:
            fwd = _forward_logret(close, h)
            cols.append(fwd.reshape(-1, 1))
        entry["transform"] = target.transform
        entry["source"] = {"close": close_col}

    else:
        raise ValueError(f"unknown target kind {target.kind!r}")

    block = np.concatenate(cols, axis=1)
    assert block.shape[1] == len(out_columns), (
        f"target {target.name!r} width {block.shape[1]} != "
        f"out_columns {len(out_columns)}"
    )
    return block, entry


def _derive_direction(lo: np.ndarray, sh: np.ndarray) -> np.ndarray:
    """3-class code: up=0 (long==1 & short==0), down=2 (short==1), else 1.

    NaN in either source → NaN class (row dropped at build).
    """
    out = np.full(lo.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(lo) & ~np.isnan(sh)
    up = valid & (lo == 1.0) & (sh == 0.0)
    down = valid & (sh == 1.0) & ~up
    neutral = valid & ~up & ~down
    out[up] = 0.0
    out[down] = 2.0
    out[neutral] = 1.0
    return out


def _onehot3(cls: np.ndarray) -> np.ndarray:
    """(rows,) class code in {0,1,2}/NaN → (rows, 3) one-hot; NaN row → all NaN."""
    out = np.full((cls.shape[0], 3), np.nan, dtype=np.float64)
    valid = ~np.isnan(cls)
    codes = cls[valid].astype(int)
    rows = np.flatnonzero(valid)
    out[valid] = 0.0
    out[rows, codes] = 1.0
    return out


def _binary_onehot(col: np.ndarray) -> np.ndarray:
    """(rows,) profit label {0,1}/NaN -> (rows, 2) one-hot [positive, other].

    positive (col == 1) -> [1, 0]; other (col == 0) -> [0, 1]; NaN row -> all NaN.
    """
    out = np.full((col.shape[0], 2), np.nan, dtype=np.float64)
    valid = ~np.isnan(col)
    out[valid] = 0.0
    pos = np.flatnonzero(valid & (col == 1.0))
    oth = np.flatnonzero(valid & (col == 0.0))
    out[pos, 0] = 1.0
    out[oth, 1] = 1.0
    return out


def _forward_logret(close: np.ndarray, horizon: int) -> np.ndarray:
    """Forward logret log(close[t+h]/close[t]); rows past end → NaN."""
    out = np.full(close.shape, np.nan, dtype=np.float64)
    n = len(close)
    if horizon < n:
        future = close[horizon:]
        cur = close[: n - horizon]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: n - horizon] = np.log(future / cur)
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def _source_content_hash(df: pd.DataFrame) -> str:
    """Stable content hash of the source frame (index + values).

    Hashes pandas' per-row hash of the frame plus the index, so any change to a
    used value or row count flips the hash. Cheap enough for build-time use.
    """
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df.index, index=False).values.tobytes())
    h.update(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    )
    return h.hexdigest()


def _dataset_hash(
    source_hash: str, spec: "NNModelSpec", feature_cols_by_tf: dict
) -> str:
    """Content hash over (source, feature set, target set, history, split)."""
    payload = {
        "source": source_hash,
        "timeframes": list(spec.timeframes),
        "feature_cols": feature_cols_by_tf,
        "history_points": spec.history_points,
        "targets": [t.out_columns() for t in spec.targets],
        "target_specs": [
            {
                "name": t.name,
                "kind": t.kind,
                "side": t.side,
                "horizons": list(t.horizons),
                "label_tf": t.label_tf,
                "label_m": t.label_m,
                "label_x": t.label_x,
                "strict": t.strict,
                "transform": t.transform,
            }
            for t in spec.targets
        ],
        "validation_split": spec.validation_split,
        "val_strategy": spec.val_strategy,
    }
    serialised = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()


# ---------------------------------------------------------------------------
# NNDataset
# ---------------------------------------------------------------------------


class _MmapWindowDataset:
    """Map-style dataset that lazily gathers one row's multi-TF window.

    Holds memory-mapped X_{tf}.npy / y.npy handles and a row range. __getitem__
    reads ONE row's ``(history_points, n_features_tf)`` block from each TF mmap
    and concatenates them along the feature axis (in the given timeframe order)
    into ``(history_points, sum_tf n_features)``. Only the requested row leaves
    the mmap, so a DataLoader over this keeps host RAM at O(batch + page cache),
    independent of total row count. Tensors are already normalised float32 on
    disk, so no transform happens here.
    """

    def __init__(self, x_mmaps, y_mmap, start: int, end: int) -> None:
        self._x = x_mmaps
        self._y = y_mmap
        self._start = int(start)
        self._n = int(end) - int(start)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        r = self._start + i
        # np.concatenate allocates a fresh writable array; y needs an explicit
        # copy (a same-dtype mmap slice would otherwise be non-writable).
        x = np.concatenate([m[r] for m in self._x], axis=-1)
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.array(self._y[r], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class NNDataset:
    """Content-addressed tensor cache for NN training/inference."""

    def __init__(
        self,
        dataset_dir_path: str,
        manifest: dict,
        cached: bool,
        row_slice: tuple[int, int] | None = None,
    ) -> None:
        self.dataset_dir_path = dataset_dir_path
        self.manifest = manifest
        self.dataset_hash = manifest["dataset_hash"]
        self.cached = cached
        # row_slice limits views (split / group); None = all rows.
        self._row_slice = row_slice

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _base_dir(dataset_dir: str | None) -> Path:
        if dataset_dir is not None:
            return Path(dataset_dir)
        return nn_artefact_root(os.environ["PAIR"]) / "datasets"

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        data_attributes: "DataAttributes",
        spec: "NNModelSpec",
        dataset_dir: str | None = None,
    ) -> "NNDataset":
        base = cls._base_dir(dataset_dir)

        # Feature columns per TF (bare indicator names → {tf}_{indicator}).
        feature_cols_by_tf: dict[str, list[str]] = {}
        for tf in spec.timeframes:
            feature_cols_by_tf[str(tf)] = [
                f"{tf}_{ind}" for ind in spec.indicators
            ]

        # Validate every feature column exists.
        missing = [
            c
            for cols in feature_cols_by_tf.values()
            for c in cols
            if c not in df.columns
        ]
        if missing:
            raise ValueError(
                f"feature columns missing from source frame: {missing}"
            )

        source_hash = _source_content_hash(df)
        dataset_hash = _dataset_hash(source_hash, spec, feature_cols_by_tf)
        dataset_dir_path = base / dataset_hash

        # Cache hit? Re-verify manifest source content-hash.
        if dataset_dir_path.exists():
            cached = cls._try_load_cached(dataset_dir_path, source_hash)
            if cached is not None:
                return cached
            # corrupt/stale → rebuild (overwrite)

        return cls._materialise(
            df,
            spec,
            feature_cols_by_tf,
            source_hash,
            dataset_hash,
            dataset_dir_path,
        )

    @classmethod
    def _try_load_cached(
        cls, dataset_dir_path: Path, source_hash: str
    ) -> "NNDataset | None":
        """Return a cached NNDataset if the dir is complete and source matches."""
        manifest_path = dataset_dir_path / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        # required artefacts present?
        for tf in manifest.get("timeframes", []):
            if not (dataset_dir_path / f"X_{tf}.npy").exists():
                return None
        if not (dataset_dir_path / "y.npy").exists():
            return None
        if not (dataset_dir_path / "index.npy").exists():
            return None
        if not (dataset_dir_path / "splits.json").exists():
            return None
        # source content-hash must still match (no silent stale serving).
        src = manifest.get("source", "")
        if "@" in src and src.split("@", 1)[1] != source_hash:
            return None
        return cls(str(dataset_dir_path), manifest, cached=True)

    @classmethod
    def _materialise(
        cls,
        df: pd.DataFrame,
        spec: "NNModelSpec",
        feature_cols_by_tf: dict,
        source_hash: str,
        dataset_hash: str,
        dataset_dir_path: Path,
    ) -> "NNDataset":
        n_rows = len(df)

        # --- Per-TF feature blocks (rows, history_points, n_features) ---
        tf_blocks: dict[str, np.ndarray] = {}
        for tf in spec.timeframes:
            tf_blocks[str(tf)] = _build_tf_block(
                df, tf, feature_cols_by_tf[str(tf)], spec.history_points
            )

        # --- Target blocks (rows, total_target_width) ---
        target_blocks: list[np.ndarray] = []
        target_entries: list[dict] = []
        for target in spec.targets:
            tb, entry = _target_block(df, target)
            target_blocks.append(tb)
            target_entries.append(entry)
        y_full = (
            np.concatenate(target_blocks, axis=1)
            if target_blocks
            else np.empty((n_rows, 0))
        )

        # --- Drop rows with any NaN feature or NaN target ---
        feat_nan = np.zeros(n_rows, dtype=bool)
        for block in tf_blocks.values():
            feat_nan |= np.isnan(block).any(axis=(1, 2))
        tgt_nan = (
            np.isnan(y_full).any(axis=1)
            if y_full.shape[1] > 0
            else np.zeros(n_rows, dtype=bool)
        )
        keep = ~(feat_nan | tgt_nan)
        dropped = int((~keep).sum())

        if keep.sum() == 0:
            raise ValueError("no usable rows after dropping NaN feature/target rows")

        kept_index = df.index[keep]
        for tf_key in tf_blocks:
            tf_blocks[tf_key] = tf_blocks[tf_key][keep]
        y_kept = y_full[keep]
        n_kept = int(keep.sum())

        # --- Time-ordered splits (no shuffle) ---
        val_frac = float(spec.validation_split)
        # train / val / holdout: holdout mirrors val fraction (time_holdout).
        holdout_frac = val_frac
        train_frac = max(0.0, 1.0 - val_frac - holdout_frac)
        tr_end = int(round(n_kept * train_frac))
        va_end = int(round(n_kept * (train_frac + val_frac)))
        # clamp to keep ranges valid & contiguous
        tr_end = min(max(tr_end, 0), n_kept)
        va_end = min(max(va_end, tr_end), n_kept)
        splits = {
            "train": [0, tr_end],
            "val": [tr_end, va_end],
            "holdout": [va_end, n_kept],
        }

        # --- Robust normalisation stats on the TRAIN split only ---
        # Match DataAttributes.compute_nn_stats semantics EXACTLY: stats are
        # estimated on the RAW per-column source values, closed-candle rows
        # only, each timestamp counted once — NOT the windowed lookback tensor
        # (which is forming-candle inclusive and duplicates each closed value
        # once per window it appears in). The train split is a row range over
        # the kept (post-drop) rows; its timestamp span [train_lo, train_hi]
        # bounds which closed source rows feed the stats (no val/holdout leak).
        normalization: dict[str, dict] = {}
        train_lo = kept_index[0] if tr_end > 0 else None
        train_hi = kept_index[tr_end - 1] if tr_end > 0 else None
        for tf in spec.timeframes:
            tf_key = str(tf)
            closed_mask = df[f"{tf}_is_closed"].to_numpy().astype(bool)
            if train_hi is not None:
                in_train = (df.index >= train_lo) & (df.index <= train_hi)
            else:
                in_train = np.zeros(len(df), dtype=bool)
            stat_mask = closed_mask & in_train
            for col in feature_cols_by_tf[tf_key]:
                vals = df[col].to_numpy(dtype=np.float64)[stat_mask]
                vals = vals[np.isfinite(vals)]
                normalization[col] = _robust_stats(vals)

        # --- Apply normalisation: clip→z→clamp, write pre-normalised ---
        for tf in spec.timeframes:
            tf_key = str(tf)
            block = tf_blocks[tf_key]
            for fi, col in enumerate(feature_cols_by_tf[tf_key]):
                st = normalization[col]
                x = block[:, :, fi]
                xc = np.clip(x, st["q01"], st["q99"])
                z = (xc - st["mean"]) / st["std"]
                block[:, :, fi] = np.clip(z, -4.0, 4.0)
            tf_blocks[tf_key] = block.astype(np.float32)

        y_kept = y_kept.astype(np.float32)

        # --- Manifest ---
        manifest = {
            "dataset_hash": dataset_hash,
            "source": f"df_with_indicators.pkl@{source_hash}",
            "timeframes": list(spec.timeframes),
            "history_points": spec.history_points,
            "feature_cols": feature_cols_by_tf,
            "normalization": normalization,
            "targets": target_entries,
            "rows": n_kept,
            "dropped": dropped,
            "split": {
                "strategy": spec.val_strategy,
                "train": train_frac,
                "val": val_frac,
                "holdout": holdout_frac,
            },
        }

        # --- Persist atomically-ish (write into the dir) ---
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
        for tf in spec.timeframes:
            np.save(dataset_dir_path / f"X_{tf}.npy", tf_blocks[str(tf)])
        np.save(dataset_dir_path / "y.npy", y_kept)
        # Real prepared frames carry a tz-aware UTC DatetimeIndex; .astype to a
        # tz-naive datetime64[ns] raises, so drop the tz first (timestamps stay UTC).
        _idx = kept_index
        if getattr(_idx, "tz", None) is not None:
            _idx = _idx.tz_convert("UTC").tz_localize(None)
        np.save(
            dataset_dir_path / "index.npy",
            np.array(_idx.astype("datetime64[ns]")),
        )
        (dataset_dir_path / "splits.json").write_text(json.dumps(splits, indent=2))
        (dataset_dir_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        return cls(str(dataset_dir_path), manifest, cached=False)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls, dataset_hash: str, dataset_dir: str | None = None
    ) -> "NNDataset":
        base = cls._base_dir(dataset_dir)
        dataset_dir_path = base / dataset_hash
        manifest_path = dataset_dir_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no cached dataset at {dataset_dir_path} (missing manifest)"
            )
        manifest = json.loads(manifest_path.read_text())
        # corrupt dir → treat as cache miss (missing .npy → caller rebuilds)
        for tf in manifest.get("timeframes", []):
            if not (dataset_dir_path / f"X_{tf}.npy").exists():
                raise FileNotFoundError(
                    f"corrupt dataset dir {dataset_dir_path}: missing X_{tf}.npy"
                )
        if not (dataset_dir_path / "y.npy").exists():
            raise FileNotFoundError(
                f"corrupt dataset dir {dataset_dir_path}: missing y.npy"
            )
        return cls(str(dataset_dir_path), manifest, cached=True)

    # ------------------------------------------------------------------
    # tensors
    # ------------------------------------------------------------------

    def tensors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y).

        X concatenates the configured timeframes' feature blocks into one
        multi-TF input (rows, history_points, sum_tf n_features); y is
        (rows, total_target_width). Both are already z-scored.
        """
        d = Path(self.dataset_dir_path)
        blocks = [
            np.load(d / f"X_{tf}.npy", mmap_mode="r")
            for tf in self.manifest["timeframes"]
        ]
        X = np.concatenate(blocks, axis=2) if blocks else np.empty((0, 0, 0))
        y = np.load(d / "y.npy", mmap_mode="r")
        if self._row_slice is not None:
            s, e = self._row_slice
            X = X[s:e]
            y = y[s:e]
        return np.asarray(X), np.asarray(y)

    # ------------------------------------------------------------------
    # split
    # ------------------------------------------------------------------

    def split(self, name: str) -> "NNDataset":
        """Return a time-ordered row-range view ("train"/"val"/"holdout")."""
        splits = json.loads(
            (Path(self.dataset_dir_path) / "splits.json").read_text()
        )
        if name not in splits:
            raise ValueError(f"unknown split {name!r}; have {list(splits)}")
        s, e = splits[name]
        return NNDataset(
            self.dataset_dir_path, self.manifest, self.cached, row_slice=(s, e)
        )

    # ------------------------------------------------------------------
    # grouping
    # ------------------------------------------------------------------

    def groups(self, grouping: "GroupingSpec") -> list[str]:
        """Return the group keys for a GroupingSpec.

        mode="single" → one group ("all") containing all rows.
        mode="by_indicator" → not supported in-dataset; the routing column is
            not retained post-build, so per-class routing is orchestrated
            upstream (task 08).
        """
        if grouping.mode == "single":
            return ["all"]
        if grouping.mode == "by_indicator":
            raise NotImplementedError(
                "by_indicator grouping is orchestrated upstream (task 08); "
                "NNDataset does not retain the routing column"
            )
        raise ValueError(f"unknown grouping mode {grouping.mode!r}")

    def group(self, group_key: str) -> "NNDataset":
        """Return a row-subset view for one class/regime.

        For the single-mode group ("all") this is the full dataset.
        """
        if group_key == "all":
            return NNDataset(
                self.dataset_dir_path,
                self.manifest,
                self.cached,
                row_slice=self._row_slice,
            )
        raise ValueError(f"unknown group key {group_key!r}")

    # ------------------------------------------------------------------
    # Lazy mmap loading (host RAM O(batch), not O(rows)) — Task 14
    # ------------------------------------------------------------------

    def _row_range(self) -> tuple[int, int]:
        """Absolute [start, end) rows for this view (full range if unsliced)."""
        if self._row_slice is not None:
            return self._row_slice
        return (0, int(self.manifest["rows"]))

    def _subview(self, start: int, end: int) -> "NNDataset":
        """A view over an explicit absolute row range [start, end)."""
        return NNDataset(
            self.dataset_dir_path,
            self.manifest,
            self.cached,
            row_slice=(int(start), int(end)),
        )

    def torch_dataset(self) -> _MmapWindowDataset:
        """Lazy, mmap-backed map-style dataset over this view's rows.

        Each X_{tf}.npy / y.npy is opened with ``mmap_mode='r'``; one row's
        multi-TF window is assembled on access in ``manifest['timeframes']``
        order. Host RAM stays O(batch + page cache), not O(rows). Consumed by a
        DataLoader in NNModel training.
        """
        d = Path(self.dataset_dir_path)
        x_mmaps = [
            np.load(d / f"X_{tf}.npy", mmap_mode="r")
            for tf in self.manifest["timeframes"]
        ]
        y_mmap = np.load(d / "y.npy", mmap_mode="r")
        s, e = self._row_range()
        return _MmapWindowDataset(x_mmaps, y_mmap, s, e)

    def labels(self) -> np.ndarray:
        """The y rows for this view (small: rows × target_width).

        Opens only y.npy (via mmap) and materialises the slice — used for global
        class weights without ever opening an X_{tf}.npy.
        """
        y_mmap = np.load(Path(self.dataset_dir_path) / "y.npy", mmap_mode="r")
        s, e = self._row_range()
        return np.asarray(y_mmap[s:e])

    # ------------------------------------------------------------------
    # Inference feature matrix (parity with training — D6)
    # ------------------------------------------------------------------

    @classmethod
    def build_inference_matrix(
        cls,
        df: pd.DataFrame,
        feature_cols_by_tf: dict[str, list[str]],
        history_points: int,
        normalization: dict[str, dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a normalised multi-TF lookback matrix from ANY ``df``.

        This is the inference twin of ``_materialise``: it reuses the SAME
        ``_build_tf_block`` window builder and the SAME clip→z→clamp formula so
        that the features fed to ``run_batch`` are byte-for-byte comparable to
        the training tensors — but it normalises with the GIVEN ``normalization``
        stats (the checkpoint's bundled TRAIN stats), never stats recomputed
        from ``df`` (leakage guard).

        Args:
            df: A wide frame carrying the ``{tf}_is_closed`` and feature columns.
            feature_cols_by_tf: ``{tf_str: [cols]}`` in the SAME order the
                checkpoint was trained with (manifest['feature_cols']).
            history_points: Lookback depth used at training time.
            normalization: ``{col: {q01, q99, mean, std}}`` training stats.

        Returns:
            ``(X, valid_mask)`` where ``X`` is
            ``(rows, history_points, n_features)`` float32 (timeframes
            concatenated along the feature axis in ``feature_cols_by_tf``
            declaration order, which must match ``manifest["timeframes"]``
            order — guaranteeing byte-parity with ``tensors()``) and
            ``valid_mask`` is a
            ``(rows,)`` bool array True where NO feature in the window is NaN.
            Rows that are NaN are left in place (caller masks their output);
            no rows are dropped, so ``X`` stays aligned to ``df.index``.
        """
        n_rows = len(df)
        # Iterate in manifest declaration order (same order tensors() uses when
        # concatenating per-TF blocks). Do NOT sort — for specs whose timeframes
        # are non-ascending (e.g. [60, 15]) sorting would swap the feature
        # channels and silently feed the model wrong inputs.
        tf_keys = list(feature_cols_by_tf.keys())

        blocks: list[np.ndarray] = []
        for tf_key in tf_keys:
            feature_cols = feature_cols_by_tf[tf_key]
            block = _build_tf_block(df, int(tf_key), feature_cols, history_points)
            # Apply the training normalisation (clip→z→clamp) per column.
            for fi, col in enumerate(feature_cols):
                st = normalization[col]
                x = block[:, :, fi]
                xc = np.clip(x, st["q01"], st["q99"])
                z = (xc - st["mean"]) / st["std"]
                block[:, :, fi] = np.clip(z, -4.0, 4.0)
            blocks.append(block)

        if blocks:
            X = np.concatenate(blocks, axis=2).astype(np.float32)
        else:
            X = np.empty((n_rows, history_points, 0), dtype=np.float32)

        valid_mask = ~np.isnan(X).any(axis=(1, 2))
        return X, valid_mask


# ---------------------------------------------------------------------------
# Robust winsorised stats (mirrors DataAttributes.compute_nn_stats formula)
# ---------------------------------------------------------------------------


def _robust_stats(vals: np.ndarray) -> dict:
    """{q01,q99,mean,std} via the winsorised formula (raw closed-row values).

    q01/q99 on raw → xw=clip(q01,q99) → mean=xw.mean(), std=max(xw.std(),1e-8).
    Pandas Series.quantile (linear) and Series.std (ddof=1) reproduce
    DataAttributes.compute_nn_stats's numbers; the caller feeds it the same
    inputs that layer uses (raw per-column closed-candle values, one per
    timestamp), so the dataset and the global layer agree.
    """
    if len(vals) == 0:
        return {"q01": 0.0, "q99": 0.0, "mean": 0.0, "std": 1e-8}
    series = pd.Series(vals)
    q01 = float(series.quantile(0.01))
    q99 = float(series.quantile(0.99))
    winsorised = series.clip(q01, q99)
    return {
        "q01": q01,
        "q99": q99,
        "mean": float(winsorised.mean()),
        "std": max(float(winsorised.std()), 1e-8),
    }
