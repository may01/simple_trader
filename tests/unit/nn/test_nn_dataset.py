"""Unit tests for nn/nn_dataset.py — NNDataset content-addressed tensor cache.

The tests build SMALL synthetic wide DataFrames with explicit
``{tf}_is_closed`` columns, feature columns, and profit-label columns so that
exact values can be asserted against the reference ``WideDataPoint.get``
lookback semantics and the labels.py naming helpers.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data import WideDataPoint
from indicators import DataAttributes
from indicators.labels import _fmt
from nn.nn_dataset import NNDataset
from nn.nn_model_spec import GroupingSpec, NNModelSpec, TargetSpec


# ---------------------------------------------------------------------------
# Synthetic wide-frame fixtures
# ---------------------------------------------------------------------------


def _label_suffix(n: int, m: float, x: float) -> str:
    return f"n{_fmt(n)}_m{_fmt(m)}_x{_fmt(x)}"


def make_wide_df(rows: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic wide DataFrame for a single TF (15) plus a 60 TF.

    Columns:
      - {tf}_is_closed     (15 closed every 15th row, 60 closed every 60th row)
      - 15_logret, 15_rsi_14, 60_logret  (feature columns; deterministic)
      - 15_close, 60_close (price, for regression)
      - 15_plong_n1_m1_x0p3 / 15_pshort_n1_m1_x0p3 (binary labels)
    Index is a 1-minute DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    df = pd.DataFrame(index=idx)

    # closed flags: 15-min candle closes on minutes where (i+1) % 15 == 0
    pos = np.arange(rows)
    df["15_is_closed"] = ((pos + 1) % 15 == 0)
    df["60_is_closed"] = ((pos + 1) % 60 == 0)

    # deterministic feature columns (distinct per row so lookback is checkable)
    df["15_logret"] = (pos.astype(float) * 0.001) - 0.01
    df["15_rsi_14"] = 50.0 + pos.astype(float)
    df["60_logret"] = (pos.astype(float) * 0.002) - 0.02

    # price for regression
    df["15_close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, rows))
    df["60_close"] = 100.0 + np.cumsum(rng.normal(0, 0.2, rows))

    # binary profit labels (deterministic): alternate long/short outcomes
    suf = _label_suffix(1, 1.0, 0.3)
    plong = np.where(pos % 3 == 0, 1.0, 0.0)
    pshort = np.where(pos % 3 == 1, 1.0, 0.0)
    df[f"15_plong_{suf}"] = plong
    df[f"15_pshort_{suf}"] = pshort
    return df


def small_spec(**overrides) -> NNModelSpec:
    spec = NNModelSpec(
        name="t",
        timeframes=[15],
        indicators=["logret", "rsi_14"],
        history_points=4,
        targets=[
            TargetSpec(
                name="dir15",
                kind="direction",
                horizons=[1],
                label_tf=15,
                label_m=1.0,
                label_x=0.3,
            )
        ],
        validation_split=0.2,
        val_strategy="time_holdout",
        seed=0,
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


# ---------------------------------------------------------------------------
# Lookback window correctness vs WideDataPoint.get
# ---------------------------------------------------------------------------


class TestLookbackMatchesWideDataPoint:
    def test_block_matches_reference_get(self, tmp_path):
        df = make_wide_df()
        spec = small_spec(history_points=4)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

        X, _ = ds.tensors()
        index = np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
        manifest = ds.manifest
        feat_cols = manifest["feature_cols"]["15"]
        # X is normalised; recover raw is not possible directly. Instead, build
        # a SECOND dataset with degenerate stats (wide band, mean 0, std 1) so
        # tensors() returns RAW values, and compare to WideDataPoint.get.
        raw = _raw_block(df, feat_cols, 15, spec.history_points, pd.DatetimeIndex(index))

        # Reference values from WideDataPoint.get for a few rows/shifts.
        kept = pd.DatetimeIndex(index)
        for ts in [kept[0], kept[len(kept) // 2], kept[-1]]:
            wdp = WideDataPoint(df, ts)
            for fi, col in enumerate(feat_cols):
                bare = col.split("_", 1)[1]
                ri = list(kept).index(ts)
                for shift in range(spec.history_points):
                    ref = wdp.get(bare, 15, shift)
                    got = raw[ri, shift, fi]
                    if math.isnan(ref):
                        assert math.isnan(got)
                    else:
                        assert got == pytest.approx(ref)

    def test_nan_on_insufficient_history_dropped(self, tmp_path):
        # First rows lack history_points closed candles -> NaN -> dropped.
        # Need >= 4 closed 15-candles before a row: 4th closes at row 59, so
        # only rows >= 59 can survive. Use ample rows so some remain.
        df = make_wide_df(rows=240)
        spec = small_spec(history_points=4)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        kept = np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
        assert len(kept) >= 1
        # A surviving row needs the deepest non-zero shift (history_points-1)
        # to resolve, i.e. >= history_points-1 closed candles at/before it
        # (shift=0 is the row's own value and always present).
        closed_pos = np.flatnonzero(df["15_is_closed"].to_numpy().astype(bool))
        all_idx = list(df.index)
        for ts in pd.DatetimeIndex(kept):
            ri = all_idx.index(ts)
            n_closed = int((closed_pos <= ri).sum())
            assert n_closed >= spec.history_points - 1


def _raw_block(df, feat_cols, tf, history_points, kept_index):
    """Reference raw lookback block for kept rows (mirrors the spec)."""
    closed_pos = np.flatnonzero(df[f"{tf}_is_closed"].to_numpy().astype(bool))
    all_idx = list(df.index)
    rows = len(kept_index)
    out = np.full((rows, history_points, len(feat_cols)), np.nan)
    for ri, ts in enumerate(kept_index):
        i = all_idx.index(ts)
        n_le = int((closed_pos <= i).sum())
        for fi, col in enumerate(feat_cols):
            colvals = df[col].to_numpy(dtype=float)
            out[ri, 0, fi] = colvals[i]
            for shift in range(1, history_points):
                k = n_le - shift
                if k >= 0:
                    out[ri, shift, fi] = colvals[closed_pos[k]]
    return out


# ---------------------------------------------------------------------------
# tensors() shapes
# ---------------------------------------------------------------------------


class TestTensorShapes:
    def test_X_y_shapes(self, tmp_path):
        df = make_wide_df()
        spec = small_spec(history_points=4)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        X, y = ds.tensors()
        n_feat = len(spec.timeframes) * len(spec.indicators)
        assert X.shape == (ds.manifest["rows"], spec.history_points, n_feat)
        # direction target -> width 3
        assert y.shape == (ds.manifest["rows"], 3)

    def test_multi_tf_concatenation(self, tmp_path):
        # 60 TF closes every 60 rows; 3 closed 60-candles need >= 180 rows.
        df = make_wide_df(rows=300)
        spec = small_spec(timeframes=[15, 60], indicators=["logret"], history_points=3)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        X, _ = ds.tensors()
        # 2 TFs * 1 indicator = 2 feature channels
        assert X.shape[2] == 2
        assert X.shape[1] == 3


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


class TestManifestSchema:
    def test_manifest_keys(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        m = ds.manifest
        for key in (
            "dataset_hash",
            "source",
            "timeframes",
            "history_points",
            "feature_cols",
            "normalization",
            "targets",
            "rows",
            "split",
            "dropped",
        ):
            assert key in m, f"missing manifest key {key}"
        assert m["timeframes"] == [15]
        assert m["history_points"] == spec.history_points
        assert m["feature_cols"]["15"] == ["15_logret", "15_rsi_14"]

    def test_normalization_stat_keys(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        for col, stats in ds.manifest["normalization"].items():
            assert set(stats.keys()) == {"q01", "q99", "mean", "std"}

    def test_target_out_columns_and_encoding(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        t = ds.manifest["targets"][0]
        assert t["out_columns"] == [
            "nn_res_dir15_prob_up",
            "nn_res_dir15_prob_neutral",
            "nn_res_dir15_prob_down",
        ]
        assert t["encoding"] == {"up": 0, "neutral": 1, "down": 2}
        assert t["source"]["long"] == f"15_plong_{_label_suffix(1, 1.0, 0.3)}"
        assert t["source"]["short"] == f"15_pshort_{_label_suffix(1, 1.0, 0.3)}"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSplits:
    def test_time_ordered_splits(self, tmp_path):
        df = make_wide_df(rows=120)
        spec = small_spec(history_points=2)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        splits = json.loads((Path(ds.dataset_dir_path) / "splits.json").read_text())
        assert set(splits.keys()) >= {"train", "val", "holdout"}
        # ranges are contiguous and ordered
        tr, va, ho = splits["train"], splits["val"], splits["holdout"]
        assert tr[0] == 0
        assert tr[1] == va[0]
        assert va[1] == ho[0]
        assert ho[1] == ds.manifest["rows"]

    def test_split_view_rows(self, tmp_path):
        df = make_wide_df(rows=120)
        spec = small_spec(history_points=2)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        train = ds.split("train")
        Xtr, ytr = train.tensors()
        splits = json.loads((Path(ds.dataset_dir_path) / "splits.json").read_text())
        tr = splits["train"]
        assert Xtr.shape[0] == tr[1] - tr[0]


# ---------------------------------------------------------------------------
# Content addressing / caching
# ---------------------------------------------------------------------------


class TestContentAddressing:
    def test_same_inputs_reuse_cache(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds1 = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        assert ds1.cached is False
        ds2 = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        assert ds2.cached is True
        assert ds2.dataset_hash == ds1.dataset_hash

    def test_changed_spec_field_changes_hash(self, tmp_path):
        df = make_wide_df()
        ds1 = NNDataset.build(df, DataAttributes(), small_spec(history_points=4),
                              dataset_dir=str(tmp_path))
        ds2 = NNDataset.build(df, DataAttributes(), small_spec(history_points=3),
                              dataset_dir=str(tmp_path))
        assert ds1.dataset_hash != ds2.dataset_hash

    def test_stale_source_rebuilds(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds1 = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        # mutate source content -> a cache hit must NOT serve old tensors
        df2 = df.copy()
        df2["15_logret"] = df2["15_logret"] + 1.0
        ds2 = NNDataset.build(df2, DataAttributes(), spec, dataset_dir=str(tmp_path))
        # different source content-hash -> different dataset_hash (rebuild)
        assert ds2.dataset_hash != ds1.dataset_hash
        assert ds2.cached is False

    def test_load_by_hash(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds1 = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        ds2 = NNDataset.load(ds1.dataset_hash, dataset_dir=str(tmp_path))
        X1, y1 = ds1.tensors()
        X2, y2 = ds2.tensors()
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# NaN drops
# ---------------------------------------------------------------------------


class TestNaNDrops:
    def test_nan_target_rows_dropped_and_counted(self, tmp_path):
        df = make_wide_df(rows=120)
        suf = _label_suffix(1, 1.0, 0.3)
        # inject NaN labels at a few rows that otherwise survive
        df.loc[df.index[-1], f"15_plong_{suf}"] = np.nan
        spec = small_spec(history_points=2)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        assert ds.manifest["dropped"] >= 1

    def test_empty_after_drops_raises(self, tmp_path):
        df = make_wide_df(rows=120)
        suf = _label_suffix(1, 1.0, 0.3)
        df[f"15_plong_{suf}"] = np.nan
        df[f"15_pshort_{suf}"] = np.nan
        spec = small_spec(history_points=2)
        with pytest.raises(ValueError, match="no usable rows"):
            NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

    def test_missing_feature_col_raises(self, tmp_path):
        df = make_wide_df()
        spec = small_spec(indicators=["logret", "does_not_exist"])
        with pytest.raises(ValueError, match="does_not_exist"):
            NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_emitted_tensors_within_clamp(self, tmp_path):
        df = make_wide_df(rows=200)
        spec = small_spec(history_points=2)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        X, _ = ds.tensors()
        finite = X[np.isfinite(X)]
        assert finite.min() >= -4.0 - 1e-6
        assert finite.max() <= 4.0 + 1e-6

    def test_stats_train_split_only(self, tmp_path):
        """Stats must be computed on the train split rows only (no leakage)."""
        df = make_wide_df(rows=200)
        spec = small_spec(history_points=2)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

        # Recompute expected stats from the TRAIN split raw values of one col.
        splits = json.loads((Path(ds.dataset_dir_path) / "splits.json").read_text())
        index = np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
        kept = pd.DatetimeIndex(index)
        feat_cols = ds.manifest["feature_cols"]["15"]
        col = feat_cols[0]
        bare = col.split("_", 1)[1]
        raw = _raw_block(df, feat_cols, 15, spec.history_points, kept)
        tr0, tr1 = splits["train"]
        # all raw values for this feature across the train rows' history window
        train_vals = raw[tr0:tr1, :, 0].ravel()
        train_vals = train_vals[np.isfinite(train_vals)]
        q01 = float(np.quantile(train_vals, 0.01))
        q99 = float(np.quantile(train_vals, 0.99))
        xw = np.clip(train_vals, q01, q99)
        exp_mean = float(xw.mean())
        got = ds.manifest["normalization"][col]
        # q01/q99 close; mean within tolerance (quantile interpolation differences)
        assert got["q01"] == pytest.approx(q01, abs=1e-6)
        assert got["q99"] == pytest.approx(q99, abs=1e-6)
        assert got["mean"] == pytest.approx(exp_mean, rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# Direction 3-class derivation
# ---------------------------------------------------------------------------


class TestDirectionDerivation:
    def test_three_class_encoding(self, tmp_path):
        # Construct an explicit df where long/short outcomes drive each class.
        rows = 90
        idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
        df = pd.DataFrame(index=idx)
        pos = np.arange(rows)
        df["15_is_closed"] = ((pos + 1) % 1 == 0)  # every row closed -> ample history
        df["15_logret"] = pos.astype(float) * 0.0
        df["15_rsi_14"] = 50.0
        df["15_close"] = 100.0
        suf = _label_suffix(1, 1.0, 0.3)
        plong = np.zeros(rows)
        pshort = np.zeros(rows)
        # row pattern: up (long=1, short=0), down (short=1), neutral (both 0)
        plong[pos % 3 == 0] = 1.0
        pshort[pos % 3 == 1] = 1.0
        df[f"15_plong_{suf}"] = plong
        df[f"15_pshort_{suf}"] = pshort

        spec = small_spec(history_points=1)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        _, y = ds.tensors()
        index = np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
        kept = pd.DatetimeIndex(index)
        all_idx = list(df.index)
        # y is one-hot/3-class probabilities? Spec stores class index encoding.
        # Derive expected class per kept row.
        for ri, ts in enumerate(kept):
            i = all_idx.index(ts)
            lo, sh = plong[i], pshort[i]
            if lo == 1.0 and sh == 0.0:
                exp = 0  # up
            elif sh == 1.0:
                exp = 2  # down
            else:
                exp = 1  # neutral
            # y row encodes class as a one-hot of width 3
            assert int(np.argmax(y[ri])) == exp
            assert y[ri].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regression target
# ---------------------------------------------------------------------------


class TestRegressionTarget:
    def test_regression_forward_logret(self, tmp_path):
        df = make_wide_df(rows=120)
        spec = small_spec(
            history_points=2,
            targets=[
                TargetSpec(name="ret15", kind="regression", horizons=[1],
                           label_tf=15, transform="logret")
            ],
        )
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        _, y = ds.tensors()
        assert y.shape[1] == 1
        index = np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
        kept = pd.DatetimeIndex(index)
        close = df["15_close"]
        all_idx = list(df.index)
        for ri, ts in enumerate(kept):
            i = all_idx.index(ts)
            if i + 1 < len(close):
                exp = math.log(close.iloc[i + 1] / close.iloc[i])
                assert y[ri, 0] == pytest.approx(exp, rel=1e-5, abs=1e-9)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_single_group_all_rows(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        keys = ds.groups(GroupingSpec(mode="single"))
        assert len(keys) == 1
        sub = ds.group(keys[0])
        X, _ = sub.tensors()
        assert X.shape[0] == ds.manifest["rows"]


# ---------------------------------------------------------------------------
# dataset_dir=None resolution
# ---------------------------------------------------------------------------


class TestDefaultDatasetDir:
    def test_none_resolves_to_artefact_root(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        env = {
            "NN_ARTEFACT_ROOT": str(tmp_path),
            "DATA_ROOT": "data",
            "PAIR": "BTCUSDT",
        }
        with patch.dict(os.environ, env, clear=False):
            ds = NNDataset.build(df, DataAttributes(), spec)
            expected = tmp_path / "data" / "BTCUSDT" / "nn" / "datasets"
            assert Path(ds.dataset_dir_path).parent == expected
