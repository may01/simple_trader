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
from nn.nn_dataset import NNDataset, _build_tf_block
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
        """The PRODUCTION un-normalised lookback block == WideDataPoint.get.

        This asserts the real production builder ``_build_tf_block`` (the
        highest-risk piece) directly, BEFORE any normalisation — no test-local
        re-implementation. Covers several rows × every shift: shift=0 (forming
        candle) AND shift>=1 (closed candles), including the
        NaN-on-insufficient-history warm-up boundary at the earliest rows.
        """
        df = make_wide_df(rows=90)
        feat_cols = ["15_logret", "15_rsi_14"]
        history_points = 4

        # REAL production output — un-normalised by construction.
        block = _build_tf_block(df, 15, feat_cols, history_points)
        assert block.shape == (len(df), history_points, len(feat_cols))

        all_idx = list(df.index)
        # Rows spanning the warm-up (early rows have <history_points closed
        # candles → NaN at deep shifts) through fully-resolved later rows.
        sample_positions = [0, 1, 14, 15, 29, 30, 44, len(df) // 2, len(df) - 1]
        saw_nan = False
        for ri in sample_positions:
            ts = all_idx[ri]
            wdp = WideDataPoint(df, ts)
            for fi, col in enumerate(feat_cols):
                bare = col.split("_", 1)[1]
                for shift in range(history_points):
                    ref = wdp.get(bare, 15, shift)
                    got = block[ri, shift, fi]
                    if math.isnan(ref):
                        saw_nan = True
                        assert math.isnan(got), (
                            f"row {ri} shift {shift} {col}: expected NaN, got {got}"
                        )
                    else:
                        assert got == pytest.approx(ref), (
                            f"row {ri} shift {shift} {col}: {got} != {ref}"
                        )
        # The warm-up boundary must actually be exercised (teeth on NaN path).
        assert saw_nan, "test did not exercise the NaN-insufficient-history boundary"

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

    def test_ragged_selection_drops_nonapplicable_tf(self, tmp_path):
        # 60_rsi_14 does not exist in make_wide_df → dropped at tf=60 only.
        # history_points=1 (shift=0 only) so the assertion isolates column
        # selection from the closed-candle lookback guard: make_wide_df's
        # 90-row fixture only closes tf=60 once, which would otherwise NaN
        # out every row regardless of feature selection.
        df = make_wide_df(rows=90)
        spec = small_spec(
            timeframes=[15, 60], indicators=["logret", "rsi_14"], history_points=1
        )
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        assert ds.manifest["feature_cols"]["15"] == ["15_logret", "15_rsi_14"]
        assert ds.manifest["feature_cols"]["60"] == ["60_logret"]

    def test_indicator_absent_at_all_tfs_raises(self, tmp_path):
        df = make_wide_df(rows=90)
        spec = small_spec(timeframes=[15, 60], indicators=["logret", "does_not_exist"])
        with pytest.raises(ValueError, match="does_not_exist"):
            NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

    def test_timeframe_with_zero_features_raises(self, tmp_path):
        # rsi_14 resolves at 15 (typo guard passes) but tf=60 has no features.
        df = make_wide_df(rows=90)
        spec = small_spec(timeframes=[15, 60], indicators=["rsi_14"])
        with pytest.raises(ValueError, match="zero features"):
            NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Strict target guard / non-monotonic index guard
# ---------------------------------------------------------------------------


class TestBuildGuards:
    def test_strict_target_without_l_y_raises(self, tmp_path):
        # TargetSpec carries no label_l/label_y; strict=True must fail loudly
        # rather than silently fabricating a "_lNone_yNone" column name.
        df = make_wide_df()
        spec = small_spec(
            targets=[
                TargetSpec(
                    name="dir15",
                    kind="direction",
                    horizons=[1],
                    label_tf=15,
                    label_m=1.0,
                    label_x=0.3,
                    strict=True,
                )
            ]
        )
        with pytest.raises(ValueError, match="strict target needs label_l/label_y"):
            NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

    def test_non_monotonic_index_raises(self, tmp_path):
        df = make_wide_df()
        df = df.iloc[::-1]  # descending index breaks the searchsorted gather
        spec = small_spec(history_points=2)
        with pytest.raises(ValueError, match="monotonically-ascending"):
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

    def test_stats_match_winsorised_semantics(self, tmp_path):
        """Manifest stats match the canonical winsorised formula: raw per-column
        values, closed-candle rows only, each timestamp once, restricted to the
        train split's timestamp span (no val/holdout leak)."""
        df = make_wide_df(rows=200)
        spec = small_spec(history_points=2)
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

        splits = json.loads((Path(ds.dataset_dir_path) / "splits.json").read_text())
        index = np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
        kept = pd.DatetimeIndex(index)
        feat_cols = ds.manifest["feature_cols"]["15"]
        col = feat_cols[0]

        tr0, tr1 = splits["train"]
        train_lo, train_hi = kept[tr0], kept[tr1 - 1]

        # Reference: the SAME inputs compute_nn_stats would use — raw source
        # column on closed rows, one per timestamp, within the train span.
        closed = df["15_is_closed"].to_numpy().astype(bool)
        in_train = (df.index >= train_lo) & (df.index <= train_hi)
        series = df.loc[closed & in_train, col].dropna()
        q01 = float(series.quantile(0.01))
        q99 = float(series.quantile(0.99))
        xw = series.clip(q01, q99)
        exp_mean = float(xw.mean())
        exp_std = max(float(xw.std()), 1e-8)

        got = ds.manifest["normalization"][col]
        assert got["q01"] == pytest.approx(q01, abs=1e-9)
        assert got["q99"] == pytest.approx(q99, abs=1e-9)
        assert got["mean"] == pytest.approx(exp_mean, abs=1e-9)
        assert got["std"] == pytest.approx(exp_std, abs=1e-9)


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

    def test_by_indicator_raises_not_implemented(self, tmp_path):
        df = make_wide_df()
        spec = small_spec()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        with pytest.raises(NotImplementedError, match="orchestrated upstream"):
            ds.groups(GroupingSpec(mode="by_indicator", column="15_rsi_14"))


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


# ---------------------------------------------------------------------------
# build_inference_matrix — TF-order parity with training tensors()
# ---------------------------------------------------------------------------


def make_wide_df_two_tf(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Wide DataFrame with 60 and 15 timeframes (non-ascending order).

    Columns:
      - 60_is_closed, 15_is_closed
      - 60_logret, 15_logret          (feature; distinct per row)
      - 15_plong_n1_m1_x0p3, 15_pshort_n1_m1_x0p3  (labels for target)
    The 60 TF closes every 60th row; 15 TF every 15th.
    """
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    df = pd.DataFrame(index=idx)
    pos = np.arange(rows)
    df["60_is_closed"] = ((pos + 1) % 60 == 0)
    df["15_is_closed"] = ((pos + 1) % 15 == 0)
    df["60_logret"] = pos.astype(float) * 0.003 - 0.05
    df["15_logret"] = pos.astype(float) * 0.001 - 0.02
    suf = _label_suffix(1, 1.0, 0.3)
    df[f"15_plong_{suf}"] = np.where(pos % 3 == 0, 1.0, 0.0)
    df[f"15_pshort_{suf}"] = np.where(pos % 3 == 1, 1.0, 0.0)
    return df


class TestInferenceMatrixTFOrderParity:
    """Regression: build_inference_matrix must use manifest TF order, not sorted."""

    def _build_spec_reversed(self):
        """Spec with timeframes=[60, 15] — non-ascending order."""
        return NNModelSpec(
            name="rev",
            timeframes=[60, 15],  # non-ascending: 60 first, then 15
            indicators=["logret"],
            history_points=3,
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

    def test_inference_matrix_column_blocks_follow_manifest_order(self, tmp_path):
        """Feature channel blocks in inference matrix must match manifest TF order.

        With timeframes=[60, 15], the first feature channel (axis-2 index 0)
        must come from the 60-TF block and the second from the 15-TF block —
        matching what tensors() produces for training.

        This test FAILS if build_inference_matrix sorts keys (ascending=15,60)
        and PASSES when it follows manifest["timeframes"] order (60,15).
        """
        df = make_wide_df_two_tf(rows=300)
        spec = self._build_spec_reversed()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

        # --- Training tensors (ground truth) ---
        X_train, _ = ds.tensors()

        # --- Inference matrix using the same full df ---
        manifest = ds.manifest
        assert manifest["timeframes"] == [60, 15], (
            "spec declares [60, 15]; manifest must preserve that order"
        )

        X_inf, valid_mask = NNDataset.build_inference_matrix(
            df,
            feature_cols_by_tf=manifest["feature_cols"],
            history_points=manifest["history_points"],
            normalization=manifest["normalization"],
        )

        # --- Align: inference keeps ALL rows (no drop); training drops NaN rows ---
        index_npy = np.load(
            Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True
        )
        kept_timestamps = pd.DatetimeIndex(index_npy)
        all_idx = list(df.index)
        kept_positions = [all_idx.index(ts) for ts in kept_timestamps]

        X_inf_kept = X_inf[kept_positions]  # rows present in training tensors
        # valid_mask must be True for all kept rows
        assert valid_mask[kept_positions].all(), (
            "some kept training rows are flagged invalid by build_inference_matrix"
        )

        # --- Byte-level parity: inference features == training features ---
        np.testing.assert_array_equal(
            X_inf_kept,
            X_train,
            err_msg=(
                "build_inference_matrix produces different features than tensors() "
                "for the same rows — TF concatenation order mismatch"
            ),
        )

    def test_inference_matrix_first_channel_is_first_manifest_tf(self, tmp_path):
        """The first feature channel in X must be from manifest['timeframes'][0].

        With timeframes=[60, 15], channel-0 must be 60_logret values;
        with timeframes=[15, 60], channel-0 must be 15_logret values.
        This directly catches a sort-by-int regression (sorted gives [15,60]
        regardless of manifest order).
        """
        df = make_wide_df_two_tf(rows=300)
        spec = self._build_spec_reversed()
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
        manifest = ds.manifest

        X_inf, _ = NNDataset.build_inference_matrix(
            df,
            feature_cols_by_tf=manifest["feature_cols"],
            history_points=manifest["history_points"],
            normalization=manifest["normalization"],
        )

        # The training tensors() block for TF 60 should match channel 0.
        # Load just the 60-TF block as written to disk (normalised).
        X_60 = np.load(Path(ds.dataset_dir_path) / "X_60.npy")  # (kept, h, 1)
        X_15 = np.load(Path(ds.dataset_dir_path) / "X_15.npy")  # (kept, h, 1)

        # The 60-TF block comes FIRST per manifest order [60, 15].
        # tensors() concatenates [X_60, X_15] → channel 0 = 60, channel 1 = 15.
        # For inference we use full df (no NaN-drop), but we can verify
        # that for rows that survive the training drop the channel ordering matches.
        index_npy = np.load(
            Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True
        )
        kept_timestamps = pd.DatetimeIndex(index_npy)
        all_idx = list(df.index)
        kept_positions = [all_idx.index(ts) for ts in kept_timestamps]

        # channel 0 of inference (kept rows) == channel 0 of X_60 (training)
        np.testing.assert_array_equal(
            X_inf[kept_positions, :, 0:1],
            X_60,
            err_msg="channel-0 should be TF-60 (manifest order); got TF-15 instead",
        )
        # channel 1 of inference (kept rows) == channel 0 of X_15 (training)
        np.testing.assert_array_equal(
            X_inf[kept_positions, :, 1:2],
            X_15,
            err_msg="channel-1 should be TF-15 (manifest order); got TF-60 instead",
        )


# ---------------------------------------------------------------------------
# Regression: tz-aware UTC DatetimeIndex must not raise during materialise
# ---------------------------------------------------------------------------


def test_binary_onehot_positive_other_nan():
    from nn.nn_dataset import _binary_onehot
    col = np.array([1.0, 0.0, np.nan, 1.0])
    out = _binary_onehot(col)
    assert out.shape == (4, 2)
    np.testing.assert_array_equal(out[0], [1.0, 0.0])   # positive
    np.testing.assert_array_equal(out[1], [0.0, 1.0])   # other
    assert np.isnan(out[2]).all()                        # NaN row dropped at build
    np.testing.assert_array_equal(out[3], [1.0, 0.0])


def test_target_block_direction_binary_long():
    from nn.nn_dataset import _target_block, _profit_long_col
    t = TargetSpec(
        name="long15", kind="direction_binary", side="long",
        horizons=[1], label_tf=15, label_m=1.0, label_x=0.3,
    )
    col = _profit_long_col(t, 1)
    df = pd.DataFrame({col: [1.0, 0.0, 1.0, np.nan]})
    block, entry = _target_block(df, t)
    assert block.shape == (4, 2)
    assert entry["out_columns"] == [
        "nn_res_long15_prob_long", "nn_res_long15_prob_other",
    ]
    assert entry["side"] == "long"
    assert entry["encoding"] == {"long": 0, "other": 1}
    assert entry["source"] == {"column": col, "strict": False}
    np.testing.assert_array_equal(block[0], [1.0, 0.0])
    np.testing.assert_array_equal(block[1], [0.0, 1.0])


def test_target_block_direction_binary_short_reads_short_col():
    from nn.nn_dataset import _target_block, _profit_short_col
    t = TargetSpec(
        name="short15", kind="direction_binary", side="short",
        horizons=[1], label_tf=15, label_m=1.0, label_x=0.3,
    )
    col = _profit_short_col(t, 1)
    df = pd.DataFrame({col: [1.0, 0.0]})
    block, entry = _target_block(df, t)
    assert entry["out_columns"] == [
        "nn_res_short15_prob_short", "nn_res_short15_prob_other",
    ]
    np.testing.assert_array_equal(block[0], [1.0, 0.0])
    np.testing.assert_array_equal(block[1], [0.0, 1.0])


def test_target_block_direction_binary_missing_column_raises():
    from nn.nn_dataset import _target_block
    t = TargetSpec(
        name="long15", kind="direction_binary", side="long",
        horizons=[1], label_tf=15, label_m=1.0, label_x=0.3,
    )
    with pytest.raises(ValueError, match="missing profit-label column"):
        _target_block(pd.DataFrame({"other": [1.0]}), t)


def test_target_block_direction_binary_multi_horizon():
    from nn.nn_dataset import _target_block, _profit_long_col
    t = TargetSpec(
        name="long15", kind="direction_binary", side="long",
        horizons=[1, 2], label_tf=15, label_m=1.0, label_x=0.3,
    )
    c1 = _profit_long_col(t, 1)
    c2 = _profit_long_col(t, 2)
    df = pd.DataFrame({c1: [1.0, 0.0, 1.0], c2: [0.0, 1.0, 0.0]})
    block, entry = _target_block(df, t)
    assert block.shape == (3, 4)
    assert entry["out_columns"] == [
        "nn_res_long15_h1_prob_long", "nn_res_long15_h1_prob_other",
        "nn_res_long15_h2_prob_long", "nn_res_long15_h2_prob_other",
    ]
    assert entry["source"] == {"per_horizon": [c1, c2], "strict": False}
    # horizon 1 row 0: long profitable -> [1,0]; horizon 2 row 0: not -> [0,1]
    np.testing.assert_array_equal(block[0], [1.0, 0.0, 0.0, 1.0])


class TestTzAwareDatetimeIndex:
    """Regression test for NNDataset._materialise tz-aware index handling.

    Real prepared frames carry a tz-aware UTC DatetimeIndex.  Before the fix,
    ``kept_index.astype("datetime64[ns]")`` raised TypeError when the index was
    tz-aware because pandas refuses to convert tz-aware datetimes to a tz-naive
    dtype that way.  The fix drops the tz via ``tz_localize(None)`` first.

    This test FAILS without the fix (the ``_idx.tz_localize(None)`` path in
    ``_materialise``) and passes with it.
    """

    def test_build_succeeds_with_tz_aware_utc_index(self, tmp_path):
        """NNDataset.build completes and writes index.npy with a UTC DatetimeIndex."""
        rows = 120
        # tz-aware UTC index — mirrors real prepared frames
        idx = pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC")
        df = pd.DataFrame(index=idx)

        pos = np.arange(rows)
        df["15_is_closed"] = ((pos + 1) % 15 == 0)
        df["15_logret"] = pos.astype(float) * 0.001 - 0.01
        df["15_rsi_14"] = 50.0 + pos.astype(float)
        df["15_close"] = 100.0 + np.cumsum(np.ones(rows) * 0.1)

        suf = _label_suffix(1, 1.0, 0.3)
        df[f"15_plong_{suf}"] = np.where(pos % 3 == 0, 1.0, 0.0)
        df[f"15_pshort_{suf}"] = np.where(pos % 3 == 1, 1.0, 0.0)

        spec = small_spec(history_points=2)

        # Must not raise TypeError from astype("datetime64[ns]") on tz-aware index.
        ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))

        # index.npy must exist and be loadable.
        index_path = Path(ds.dataset_dir_path) / "index.npy"
        assert index_path.exists(), "index.npy was not written"
        index_arr = np.load(index_path, allow_pickle=True)
        assert len(index_arr) == ds.manifest["rows"]

        # tensors() must return the expected shapes.
        X, y = ds.tensors()
        n_feat = len(spec.timeframes) * len(spec.indicators)
        assert X.shape == (ds.manifest["rows"], spec.history_points, n_feat)
        assert y.shape == (ds.manifest["rows"], 3)  # direction → 3-class one-hot
