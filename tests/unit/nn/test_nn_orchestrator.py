"""Tests for NNOrchestrator: grouping train + tf-agnostic batch inference.

Phase-11 surface (task 08):
  - __init__(checkpoint_dir, dataset_dir, base_spec)
  - from_trainer(pair, trainer) — pair-scoped factory
  - train(df, data_attributes, spec=None, epoch_callback=None) -> {group: metrics}
  - run_inference(df, data_attributes, spec=None) -> nn_res_* DataFrame | empty

The per-TF ``{tf}_nn_prob_*`` design and ``data_attributes.get_stats``
inference-normalisation are gone: a single multi-TF NNDataset is built ONCE,
rows are partitioned by ``spec.grouping`` (single → one group "all"), and
inference normalises via the checkpoint's BUNDLED manifest (leakage guard),
emitting timeframe-agnostic ``nn_res_*`` columns only.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from indicators import DataAttributes
from indicators.labels import _fmt
from nn.nn_model_spec import GroupingSpec, NNModelSpec, TargetSpec
from nn.nn_orchestrator import NNOrchestrator


# =====================================================================
# Fixtures
# =====================================================================


def _label_suffix(n: int, m: float, x: float) -> str:
    return f"n{_fmt(n)}_m{_fmt(m)}_x{_fmt(x)}"


def make_wide_df(rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """Small synthetic wide frame with 15/60 closed flags, features, labels.

    Mirrors the NNDataset test fixture so a REAL dataset can be built/trained.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    df = pd.DataFrame(index=idx)
    pos = np.arange(rows)

    df["15_is_closed"] = (pos + 1) % 15 == 0
    df["60_is_closed"] = (pos + 1) % 60 == 0

    df["15_logret"] = (pos.astype(float) * 0.001) - 0.01
    df["15_rsi_14"] = 50.0 + (pos.astype(float) % 30)
    df["60_logret"] = (pos.astype(float) * 0.002) - 0.02
    df["60_rsi_14"] = 40.0 + (pos.astype(float) % 20)

    df["15_close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, rows))
    df["60_close"] = 100.0 + np.cumsum(rng.normal(0, 0.2, rows))

    suf = _label_suffix(1, 1.0, 0.3)
    df[f"15_plong_{suf}"] = np.where(pos % 3 == 0, 1.0, 0.0)
    df[f"15_pshort_{suf}"] = np.where(pos % 3 == 1, 1.0, 0.0)
    return df


def small_spec(**overrides) -> NNModelSpec:
    """A minimal but real spec (grouping=single, one direction target)."""
    spec = NNModelSpec(
        name="t",
        timeframes=[15],
        indicators=["logret", "rsi_14"],
        history_points=4,
        layers=[],
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
        epochs=2,
        validation_split=0.2,
        val_strategy="time_holdout",
        seed=0,
    )
    # one tiny dense layer so the network can build
    from nn.nn_model_spec import LayerSpec

    spec.layers = [LayerSpec(kind="dense", units=8)]
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


@pytest.fixture
def base_spec():
    return small_spec()


@pytest.fixture
def data_attributes():
    return DataAttributes()


# =====================================================================
# Test: __init__
# =====================================================================


def test_init_stores_paths_and_spec(base_spec):
    """__init__ stores checkpoint_dir/dataset_dir/base_spec; empty trained_models."""
    env = os.environ.copy()
    env.pop("NUM_WORKERS", None)
    env.pop("AVAIABLE_THREADS", None)
    with patch.dict(os.environ, env, clear=True):
        orch = NNOrchestrator(
            checkpoint_dir="/ckpts",
            dataset_dir="/datasets",
            base_spec=base_spec,
        )
    assert orch.checkpoint_dir == "/ckpts"
    assert orch.dataset_dir == "/datasets"
    assert orch.base_spec is base_spec
    assert orch.trained_models == {}
    assert orch.num_workers == 4  # default


def test_init_reads_num_workers_from_env(base_spec):
    """NUM_WORKERS env overrides the default worker count."""
    with patch.dict(os.environ, {"NUM_WORKERS": "8"}):
        orch = NNOrchestrator("/c", "/d", base_spec)
        assert orch.num_workers == 8


def test_init_falls_back_to_available_threads(base_spec):
    """NUM_WORKERS absent → AVAIABLE_THREADS is used."""
    env = os.environ.copy()
    env.pop("NUM_WORKERS", None)
    env["AVAIABLE_THREADS"] = "6"
    with patch.dict(os.environ, env, clear=True):
        orch = NNOrchestrator("/c", "/d", base_spec)
        assert orch.num_workers == 6


def test_init_defaults_num_workers_if_env_missing(base_spec):
    """Neither env var set → default 4."""
    env = os.environ.copy()
    env.pop("NUM_WORKERS", None)
    env.pop("AVAIABLE_THREADS", None)
    with patch.dict(os.environ, env, clear=True):
        orch = NNOrchestrator("/c", "/d", base_spec)
        assert orch.num_workers == 4


# =====================================================================
# Test: from_trainer
# =====================================================================


def test_from_trainer_builds_paths_and_workers(base_spec, tmp_path):
    """from_trainer builds checkpoint_dir/dataset_dir from spec_hash + artefact
    root and reads num_workers from trainer.available_threads()."""
    trainer = MagicMock()
    trainer.available_threads.return_value = 11

    fake_root = tmp_path / "BTCUSDT" / "nn"

    with patch(
        "nn.nn_orchestrator.NNModelSpec.from_yaml", return_value=base_spec
    ) as mock_from_yaml, patch(
        "nn.device.nn_artefact_root", return_value=fake_root
    ) as mock_root:
        orch = NNOrchestrator.from_trainer("BTCUSDT", trainer)

    mock_from_yaml.assert_called_once_with("configs/nn_spec.yaml")
    mock_root.assert_called_once_with("BTCUSDT")
    assert orch.base_spec is base_spec
    assert orch.checkpoint_dir == f"{fake_root}/checkpoints/{base_spec.spec_hash}"
    assert orch.dataset_dir == f"{fake_root}/datasets"
    assert orch.num_workers == 11
    trainer.available_threads.assert_called_once()


# =====================================================================
# Test: train()  (mocked NNDataset / NNModel / CheckpointManager)
# =====================================================================


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
@patch("nn.nn_orchestrator.NNDataset")
def test_train_builds_dataset_once_and_keys_by_group(
    mock_ds_cls, mock_model_cls, mock_cm_cls, base_spec, data_attributes
):
    """train() builds the NNDataset ONCE and returns a dict keyed by group string."""
    dataset = MagicMock()
    dataset.groups.return_value = ["all"]
    group_view = MagicMock()
    group_view.manifest = {"normalization": {}, "feature_cols": {}}
    dataset.group.return_value = group_view
    mock_ds_cls.build.return_value = dataset

    model = MagicMock()
    model.train.return_value = {"val_accuracy": 0.7, "loss": 0.3}
    mock_model_cls.return_value = model

    cm = MagicMock()
    mock_cm_cls.return_value = cm

    orch = NNOrchestrator("/c", "/d", base_spec)
    df = make_wide_df()
    result = orch.train(df, data_attributes)

    # Dataset built exactly once, under dataset_dir, with the resolved spec.
    mock_ds_cls.build.assert_called_once()
    _, kwargs = mock_ds_cls.build.call_args
    assert kwargs.get("dataset_dir") == "/d"

    # Rows partitioned via spec.grouping; result keyed by group string.
    dataset.groups.assert_called_once_with(base_spec.grouping)
    assert set(result.keys()) == {"all"}
    assert result["all"] == {"val_accuracy": 0.7, "loss": 0.3}

    # One NNModel(spec).train(group_view) per group.
    mock_model_cls.assert_called_once_with(base_spec)
    model.train.assert_called_once()
    assert model.train.call_args[0][0] is group_view

    # CheckpointManager.save called with a manifest=.
    cm.save.assert_called_once()
    assert "manifest" in cm.save.call_args.kwargs
    assert cm.save.call_args.kwargs["manifest"] is group_view.manifest

    # Model stored under the group key.
    assert orch.trained_models["all"] is model


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
@patch("nn.nn_orchestrator.NNDataset")
def test_train_wraps_epoch_callback_with_group_key(
    mock_ds_cls, mock_model_cls, mock_cm_cls, base_spec, data_attributes
):
    """train() wraps the (group, epoch, metrics) callback into NNModel's
    (epoch, metrics) callback, injecting the group key."""
    dataset = MagicMock()
    dataset.groups.return_value = ["all"]
    gv = MagicMock()
    gv.manifest = {"normalization": {}}
    dataset.group.return_value = gv
    mock_ds_cls.build.return_value = dataset

    model = MagicMock()
    model.train.return_value = {"val_accuracy": 0.5}
    mock_model_cls.return_value = model
    mock_cm_cls.return_value = MagicMock()

    seen = []

    def cb(group_key, epoch, metrics):
        seen.append((group_key, epoch, metrics))

    orch = NNOrchestrator("/c", "/d", base_spec)
    orch.train(make_wide_df(), data_attributes, epoch_callback=cb)

    # The wrapped callback handed to NNModel.train is (epoch, metrics).
    wrapped = model.train.call_args.kwargs["epoch_callback"]
    wrapped(3, {"loss": 0.1})
    assert seen == [("all", 3, {"loss": 0.1})]


# =====================================================================
# Test: run_inference()  (mocked load_best for absence + bundled-manifest)
# =====================================================================


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_absence_safe_when_no_checkpoint(
    mock_model_cls, mock_cm_cls, base_spec, data_attributes
):
    """No best checkpoint for any group → empty df.index-aligned result."""
    mock_model_cls.return_value = MagicMock()
    cm = MagicMock()
    cm.load_best.return_value = None  # no checkpoint
    mock_cm_cls.return_value = cm

    orch = NNOrchestrator("/c", "/d", base_spec)
    df = make_wide_df(rows=30)
    result = orch.run_inference(df, data_attributes)

    assert list(result.columns) == []
    pd.testing.assert_index_equal(result.index, df.index)


@patch("nn.nn_orchestrator.NNDataset.build_inference_matrix")
@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_normalises_via_bundled_manifest(
    mock_model_cls, mock_cm_cls, mock_build_inf, base_spec, data_attributes
):
    """Inference builds its matrix from the checkpoint's BUNDLED manifest stats
    (feature_cols/history_points/normalization), not from data_attributes."""
    df = make_wide_df(rows=20)

    bundled_norm = {"15_logret": {"q01": 0, "q99": 1, "mean": 0, "std": 1}}
    bundled_fcols = {"15": ["15_logret", "15_rsi_14"]}
    manifest = {
        "normalization": bundled_norm,
        "feature_cols": bundled_fcols,
        "history_points": 4,
    }

    model = MagicMock()
    model.spec = base_spec
    model.manifest = manifest
    model.run_batch.return_value = np.tile([0.2, 0.5, 0.3], (len(df), 1))
    mock_model_cls.return_value = model

    cm = MagicMock()
    cm.load_best.return_value = {"manifest": manifest, "feature_cols": bundled_fcols}
    mock_cm_cls.return_value = cm

    X = np.zeros((len(df), 4, 2), dtype=np.float32)
    valid = np.ones(len(df), dtype=bool)
    mock_build_inf.return_value = (X, valid)

    # data_attributes must NOT be consulted for stats on the inference path.
    da = MagicMock()

    orch = NNOrchestrator("/c", "/d", base_spec)
    orch.run_inference(df, da)

    # build_inference_matrix called with the BUNDLED manifest pieces.
    args, _ = mock_build_inf.call_args
    assert args[1] == bundled_fcols           # feature_cols_by_tf
    assert args[2] == 4                        # history_points
    assert args[3] == bundled_norm             # normalization (bundled stats)
    da.get_stats.assert_not_called()


@patch("nn.nn_orchestrator.NNDataset.build_inference_matrix")
@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_emits_only_nn_res_columns(
    mock_model_cls, mock_cm_cls, mock_build_inf, base_spec, data_attributes
):
    """Output carries ONLY tf-agnostic nn_res_* columns, index==df.index,
    and the input df is not mutated."""
    df = make_wide_df(rows=20)
    df_before = df.copy(deep=True)

    manifest = {
        "normalization": {},
        "feature_cols": {"15": ["15_logret", "15_rsi_14"]},
        "history_points": 4,
    }
    model = MagicMock()
    model.spec = base_spec
    model.manifest = manifest
    model.run_batch.return_value = np.tile([0.1, 0.7, 0.2], (len(df), 1))
    mock_model_cls.return_value = model

    cm = MagicMock()
    cm.load_best.return_value = {"manifest": manifest, "feature_cols": None}
    mock_cm_cls.return_value = cm

    X = np.zeros((len(df), 4, 2), dtype=np.float32)
    mock_build_inf.return_value = (X, np.ones(len(df), dtype=bool))

    orch = NNOrchestrator("/c", "/d", base_spec)
    result = orch.run_inference(df, MagicMock())

    expected = {
        "nn_res_dir15_prob_up",
        "nn_res_dir15_prob_neutral",
        "nn_res_dir15_prob_down",
    }
    assert set(result.columns) == expected
    assert all(c.startswith("nn_res_") for c in result.columns)
    # No tf-prefixed legacy columns anywhere.
    assert not any("_nn_prob_" in c for c in result.columns)
    pd.testing.assert_index_equal(result.index, df.index)
    # Input df untouched.
    pd.testing.assert_frame_equal(df, df_before)


# =====================================================================
# Test: end-to-end on a small REAL df (train then infer)
# =====================================================================


def test_end_to_end_train_then_infer(tmp_path, data_attributes):
    """Train a single-group model on a tiny real df, then run_inference on the
    SAME df: assert ONLY nn_res_* columns, index equality, df not mutated."""
    spec = small_spec()
    ckpt_dir = str(tmp_path / "ckpts")
    ds_dir = str(tmp_path / "datasets")

    orch = NNOrchestrator(ckpt_dir, ds_dir, spec)

    df = make_wide_df(rows=240)
    df_before = df.copy(deep=True)

    train_result = orch.train(df, data_attributes)
    assert set(train_result.keys()) == {"all"}
    assert "val_accuracy" in train_result["all"]

    # Fresh orchestrator → forces a real load_best round-trip from disk.
    orch2 = NNOrchestrator(ckpt_dir, ds_dir, spec)
    result = orch2.run_inference(df, data_attributes)

    assert len(result.columns) > 0
    assert all(c.startswith("nn_res_") for c in result.columns), result.columns.tolist()
    assert set(result.columns) == {
        "nn_res_dir15_prob_up",
        "nn_res_dir15_prob_neutral",
        "nn_res_dir15_prob_down",
    }
    pd.testing.assert_index_equal(result.index, df.index)
    # Input df not mutated.
    pd.testing.assert_frame_equal(df, df_before)

    # At least the warmed-up (post-lookback) rows produced finite probabilities.
    finite_rows = result.dropna()
    assert len(finite_rows) >= 1
    # direction probs sum to ~1 on produced rows.
    probs = finite_rows[
        ["nn_res_dir15_prob_up", "nn_res_dir15_prob_neutral", "nn_res_dir15_prob_down"]
    ].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-4, atol=1e-4)


# =====================================================================
# Test: run_inference_dataset()  (D5 atomic write-wrapper, task 11)
# =====================================================================


def _write_dataset(dataset_dir, df, data_attributes):
    """Write df_with_indicators.pkl + data_attributes.pkl into dataset_dir."""
    os.makedirs(dataset_dir, exist_ok=True)
    df.to_pickle(os.path.join(dataset_dir, "df_with_indicators.pkl"))
    data_attributes.save(os.path.join(dataset_dir, "data_attributes.pkl"))


def test_run_inference_dataset_writes_df_with_nn(tmp_path, base_spec, data_attributes):
    """When run_inference yields nn_res_* columns, the wrapper writes
    {dataset_dir}/df_with_nn.pkl atomically and returns the result."""
    dataset_dir = str(tmp_path / "ds")
    df = make_wide_df(rows=20)
    _write_dataset(dataset_dir, df, data_attributes)

    nn_df = pd.DataFrame(
        {"nn_res_dir15_prob_up": np.full(len(df), 0.5)}, index=df.index
    )

    orch = NNOrchestrator("/c", "/d", base_spec)
    with patch.object(orch, "run_inference", return_value=nn_df) as mock_inf:
        result = orch.run_inference_dataset(dataset_dir=dataset_dir)

    # run_inference fed with the loaded df + data_attributes.
    mock_inf.assert_called_once()
    out_path = os.path.join(dataset_dir, "df_with_nn.pkl")
    assert os.path.exists(out_path)
    written = pd.read_pickle(out_path)
    assert "nn_res_dir15_prob_up" in written.columns
    pd.testing.assert_frame_equal(result, nn_df)


def test_run_inference_dataset_absence_safe_writes_nothing(
    tmp_path, base_spec, data_attributes
):
    """run_inference None or no nn_res_* columns → no df_with_nn.pkl, returns None."""
    df = make_wide_df(rows=20)

    # Case A: run_inference returns None
    ds_a = str(tmp_path / "a")
    _write_dataset(ds_a, df, data_attributes)
    orch = NNOrchestrator("/c", "/d", base_spec)
    with patch.object(orch, "run_inference", return_value=None):
        assert orch.run_inference_dataset(dataset_dir=ds_a) is None
    assert not os.path.exists(os.path.join(ds_a, "df_with_nn.pkl"))

    # Case B: run_inference returns a frame with NO nn_res_* columns (absence)
    ds_b = str(tmp_path / "b")
    _write_dataset(ds_b, df, data_attributes)
    empty = pd.DataFrame(index=df.index)  # no nn_res_* columns
    with patch.object(orch, "run_inference", return_value=empty):
        assert orch.run_inference_dataset(dataset_dir=ds_b) is None
    assert not os.path.exists(os.path.join(ds_b, "df_with_nn.pkl"))


def test_run_inference_dataset_never_mutates_indicators_pkl(
    tmp_path, base_spec, data_attributes
):
    """df_with_indicators.pkl mtime is unchanged after run_inference_dataset."""
    import time

    dataset_dir = str(tmp_path / "ds")
    df = make_wide_df(rows=20)
    _write_dataset(dataset_dir, df, data_attributes)

    indicators_path = os.path.join(dataset_dir, "df_with_indicators.pkl")
    mtime_before = os.stat(indicators_path).st_mtime_ns
    time.sleep(0.01)

    nn_df = pd.DataFrame(
        {"nn_res_dir15_prob_up": np.full(len(df), 0.5)}, index=df.index
    )
    orch = NNOrchestrator("/c", "/d", base_spec)
    with patch.object(orch, "run_inference", return_value=nn_df):
        orch.run_inference_dataset(dataset_dir=dataset_dir)

    assert os.stat(indicators_path).st_mtime_ns == mtime_before


def test_end_to_end_inference_parity_with_training_windows(tmp_path, data_attributes):
    """The inference feature matrix reuses the SAME window builder + bundled
    stats as training (D6): build_inference_matrix on the train df reproduces the
    normalised training tensors for the kept rows."""
    from pathlib import Path

    from nn.nn_dataset import NNDataset

    spec = small_spec()
    ds_dir = str(tmp_path / "datasets")
    df = make_wide_df(rows=240)

    ds = NNDataset.build(df, data_attributes, spec, dataset_dir=ds_dir)
    X_train, _ = ds.tensors()

    # Rebuild the matrix from the full df using the dataset's bundled manifest.
    X_inf, valid = NNDataset.build_inference_matrix(
        df,
        ds.manifest["feature_cols"],
        ds.manifest["history_points"],
        ds.manifest["normalization"],
    )

    # Restrict the inference matrix to the kept (post-drop) training rows.
    kept = pd.DatetimeIndex(
        np.load(Path(ds.dataset_dir_path) / "index.npy", allow_pickle=True)
    )
    pos = {ts: i for i, ts in enumerate(df.index)}
    kept_pos = [pos[ts] for ts in kept]
    X_inf_kept = X_inf[kept_pos]

    assert X_inf_kept.shape == X_train.shape
    np.testing.assert_allclose(X_inf_kept, X_train, rtol=1e-5, atol=1e-6)
    # Every kept training row is valid (no NaN) in the inference matrix.
    assert valid[kept_pos].all()
