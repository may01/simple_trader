"""Tests for CheckpointManager (Phase-11 Task 06).

Self-contained PyTorch bundle checkpoints: {state_dict, spec, manifest,
feature_cols, metrics}.  Models are built via NNModel(spec) and trained on
tiny in-memory NNDatasets — same pattern as test_nn_model.py.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from nn.checkpoint_manager import CheckpointManager
from nn.nn_dataset import NNDataset
from nn.nn_model import NNModel
from nn.nn_model_spec import LayerSpec, NNModelSpec, TargetSpec


# ---------------------------------------------------------------------------
# Helpers: tiny spec + tiny on-disk NNDataset (mirrors test_nn_model.py)
# ---------------------------------------------------------------------------


def _tiny_spec(**overrides) -> NNModelSpec:
    """Minimal spec: 1 tf, 2 indicators, hp=1, 1 dense layer, direction head.
    input_size = 2, output_size = 3.
    """
    kwargs = dict(
        name="ckpt_test",
        timeframes=[15],
        indicators=["rsi", "atr_ma"],
        history_points=1,
        layers=[LayerSpec(kind="dense", units=8)],
        targets=[TargetSpec(name="dir15", kind="direction", label_tf=15)],
        epochs=1,
        validation_split=0.2,
        val_strategy="time_holdout",
        device="cpu",
        seed=42,
    )
    kwargs.update(overrides)
    return NNModelSpec(**kwargs)


def _write_tiny_dataset(tmpdir: str, spec: NNModelSpec, n: int = 40, seed: int = 0) -> NNDataset:
    """Write a minimal NNDataset directory and return the NNDataset object."""
    rng = np.random.RandomState(seed)
    n_feat_per_tf = len(spec.indicators)
    hp = spec.history_points

    feature_cols = {}
    normalization = {}
    for tf in spec.timeframes:
        cols = [f"{tf}_{ind}" for ind in spec.indicators]
        feature_cols[str(tf)] = cols
        for c in cols:
            normalization[c] = {"q01": -3.0, "q99": 3.0, "mean": 0.0, "std": 1.0}
        X_tf = rng.randn(n, hp, n_feat_per_tf).astype(np.float32)
        np.save(os.path.join(tmpdir, f"X_{tf}.npy"), X_tf)

    # direction head: one-hot over 3 classes
    codes = rng.randint(0, 3, size=n)
    y = np.zeros((n, 3), dtype=np.float32)
    y[np.arange(n), codes] = 1.0
    np.save(os.path.join(tmpdir, "y.npy"), y)
    np.save(
        os.path.join(tmpdir, "index.npy"),
        np.arange(n).astype("datetime64[s]").astype("datetime64[ns]"),
    )

    val = spec.validation_split
    tr_end = int(round(n * (1.0 - 2 * val)))
    va_end = int(round(n * (1.0 - val)))
    splits = {"train": [0, tr_end], "val": [tr_end, va_end], "holdout": [va_end, n]}
    with open(os.path.join(tmpdir, "splits.json"), "w") as f:
        json.dump(splits, f)

    manifest = {
        "dataset_hash": "testhash",
        "timeframes": list(spec.timeframes),
        "history_points": hp,
        "feature_cols": feature_cols,
        "normalization": normalization,
        "rows": n,
    }
    with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    return NNDataset(tmpdir, manifest, cached=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MANIFEST = {
    "dataset_hash": "abc123",
    "timeframes": [15],
    "history_points": 1,
    "feature_cols": {"15": ["15_rsi", "15_atr_ma"]},
    "normalization": {
        "15_rsi": {"mean": 0.0, "std": 1.0},
        "15_atr_ma": {"mean": 0.5, "std": 0.3},
    },
}


@pytest.fixture
def temp_checkpoint_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def spec():
    return _tiny_spec()


@pytest.fixture
def dataset(tmp_path, spec):
    d = tmp_path / "ds"
    d.mkdir()
    return _write_tiny_dataset(str(d), spec)


@pytest.fixture
def small_model(spec):
    return NNModel(spec)


@pytest.fixture
def trained_model(small_model, dataset):
    small_model.train(dataset)
    return small_model


# ---------------------------------------------------------------------------
# 1. Constructor creates checkpoint_dir if it doesn't exist
# ---------------------------------------------------------------------------


def test_constructor_creates_checkpoint_dir(temp_checkpoint_dir):
    checkpoint_dir = os.path.join(temp_checkpoint_dir, "nested", "checkpoints")
    cm = CheckpointManager(checkpoint_dir)
    assert os.path.isdir(checkpoint_dir)


# ---------------------------------------------------------------------------
# 2. Constructor initializes best_metric to -inf for mode="max" (default)
# ---------------------------------------------------------------------------


def test_constructor_initializes_best_metric_max():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir)
        assert cm.best_metric == float("-inf")


# ---------------------------------------------------------------------------
# 3. Constructor initializes best_metric to +inf for mode="min"
# ---------------------------------------------------------------------------


def test_constructor_initializes_best_metric_min():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir, mode="min")
        assert cm.best_metric == float("+inf")


# ---------------------------------------------------------------------------
# 4. Constructor default model_name is "model"
# ---------------------------------------------------------------------------


def test_constructor_default_model_name(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    assert cm.model_name == "model"


# ---------------------------------------------------------------------------
# 5. Constructor accepts custom model_name
# ---------------------------------------------------------------------------


def test_constructor_custom_model_name(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir, model_name="my_model")
    assert cm.model_name == "my_model"


# ---------------------------------------------------------------------------
# 6. Constructor exposes expected attributes
# ---------------------------------------------------------------------------


def test_constructor_attributes(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir, model_name="m", mode="min")
    assert cm.checkpoint_dir == temp_checkpoint_dir
    assert cm.model_name == "m"
    assert cm.mode == "min"
    assert cm.best_metric == float("+inf")


# ---------------------------------------------------------------------------
# 7. save() creates epoch checkpoint file
# ---------------------------------------------------------------------------


def test_save_creates_epoch_checkpoint(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    returned_path = cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    expected_path = os.path.join(temp_checkpoint_dir, "model_epoch1.pt")
    assert os.path.isfile(expected_path)
    assert returned_path == expected_path


# ---------------------------------------------------------------------------
# 8. save() returns path of epoch checkpoint (not best)
# ---------------------------------------------------------------------------


def test_save_returns_epoch_path(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    returned_path = cm.save(trained_model, {"val_accuracy": 0.75}, epoch=2, manifest=SAMPLE_MANIFEST)
    assert "epoch2" in returned_path
    assert "best" not in returned_path


# ---------------------------------------------------------------------------
# 9. save() creates _best.pt when gate metric improves (mode="max")
# ---------------------------------------------------------------------------


def test_save_creates_best_when_improves(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    # First save: best_metric is -inf, so this improves → writes best
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)


# ---------------------------------------------------------------------------
# 10. save() updates best_metric when gate metric improves (mode="max")
# ---------------------------------------------------------------------------


def test_save_updates_best_metric(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    assert cm.best_metric == 0.5
    cm.save(trained_model, {"val_accuracy": 0.7}, epoch=2, manifest=SAMPLE_MANIFEST)
    assert cm.best_metric == 0.7


# ---------------------------------------------------------------------------
# 11. save() does NOT overwrite _best.pt when gate metric does not improve
# ---------------------------------------------------------------------------


def test_save_does_not_update_best_when_not_improved(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    # First save at 0.7 → becomes best
    cm.save(trained_model, {"val_accuracy": 0.7}, epoch=1, manifest=SAMPLE_MANIFEST)
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    stat1 = os.stat(best_path)

    time.sleep(0.01)
    # Second save at 0.5 — lower, should NOT overwrite best
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=2, manifest=SAMPLE_MANIFEST)
    stat2 = os.stat(best_path)

    assert stat1.st_mtime == stat2.st_mtime, "best.pt should not have been updated"
    assert cm.best_metric == 0.7


# ---------------------------------------------------------------------------
# 12. save() with promote=True always writes _best.pt even if metric is worse
# ---------------------------------------------------------------------------


def test_save_promote_writes_best_even_when_not_improved(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    # Establish a good baseline at 0.9
    cm.save(trained_model, {"val_accuracy": 0.9}, epoch=1, manifest=SAMPLE_MANIFEST)
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    stat1 = os.stat(best_path)

    time.sleep(0.01)
    # Forced promote at 0.3 — worse metric but promote=True must still write best
    cm.save(trained_model, {"val_accuracy": 0.3}, epoch=2, manifest=SAMPLE_MANIFEST, promote=True)
    stat2 = os.stat(best_path)

    assert stat2.st_mtime != stat1.st_mtime, "best.pt should have been overwritten by promote=True"


# ---------------------------------------------------------------------------
# 13. save() is atomic: no .tmp files left on success
# ---------------------------------------------------------------------------


def test_save_is_atomic(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    tmp_files = list(Path(temp_checkpoint_dir).glob("*.tmp"))
    assert len(tmp_files) == 0, "Atomic save should not leave .tmp files"


# ---------------------------------------------------------------------------
# 14. save() bundle contains expected keys including metrics
# ---------------------------------------------------------------------------


def test_save_bundle_contains_metrics(trained_model, temp_checkpoint_dir):
    import torch

    cm = CheckpointManager(temp_checkpoint_dir)
    path = cm.save(trained_model, {"val_accuracy": 0.6}, epoch=1, manifest=SAMPLE_MANIFEST)
    bundle = torch.load(path, weights_only=False)
    assert "state_dict" in bundle
    assert "spec" in bundle
    assert "manifest" in bundle
    assert "feature_cols" in bundle
    assert "metrics" in bundle
    assert bundle["metrics"]["val_accuracy"] == 0.6


# ---------------------------------------------------------------------------
# 15. load_best() returns {manifest, feature_cols} dict on success
# ---------------------------------------------------------------------------


def test_load_best_returns_dict_on_success(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1, manifest=SAMPLE_MANIFEST)

    new_model = NNModel(trained_model.spec)
    result = cm.load_best(new_model)
    assert result is not None
    assert isinstance(result, dict)
    assert "manifest" in result
    assert "feature_cols" in result


# ---------------------------------------------------------------------------
# 16. load_best() returns None when no best checkpoint exists
# ---------------------------------------------------------------------------


def test_load_best_returns_none_when_no_best(temp_checkpoint_dir, spec):
    cm = CheckpointManager(temp_checkpoint_dir)
    new_model = NNModel(spec)
    result = cm.load_best(new_model)
    assert result is None


# ---------------------------------------------------------------------------
# 17. load_best() round-trips manifest and feature_cols
# ---------------------------------------------------------------------------


def test_load_best_round_trips_manifest_and_feature_cols(trained_model, temp_checkpoint_dir):
    manifest = {
        "dataset_hash": "roundtrip",
        "timeframes": [15],
        "feature_cols": {"15": ["15_rsi", "15_atr_ma"]},
        "normalization": {"15_rsi": {"mean": 1.0, "std": 2.0}},
    }
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.8}, epoch=1, manifest=manifest, promote=True)

    new_model = NNModel(trained_model.spec)
    result = cm.load_best(new_model)
    assert result is not None
    assert result["manifest"] == manifest
    assert result["feature_cols"] == manifest["feature_cols"]


# ---------------------------------------------------------------------------
# 18. load_best() actually loads weights (round-trip prediction match)
# ---------------------------------------------------------------------------


def test_load_best_loads_model_weights(trained_model, temp_checkpoint_dir):
    x = np.random.randn(trained_model.input_size).astype("float32")
    pred_before = trained_model.run(x)

    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1, manifest=SAMPLE_MANIFEST)

    new_model = NNModel(trained_model.spec)
    cm.load_best(new_model)
    pred_after = new_model.run(x)
    assert np.allclose(pred_before, pred_after, atol=1e-6)


# ---------------------------------------------------------------------------
# 19. load_epoch() returns True on success
# ---------------------------------------------------------------------------


def test_load_epoch_returns_true_on_success(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=5, manifest=SAMPLE_MANIFEST)

    new_model = NNModel(trained_model.spec)
    result = cm.load_epoch(new_model, epoch=5)
    assert result is True


# ---------------------------------------------------------------------------
# 20. load_epoch() returns False when epoch not found
# ---------------------------------------------------------------------------


def test_load_epoch_returns_false_when_not_found(temp_checkpoint_dir, spec):
    cm = CheckpointManager(temp_checkpoint_dir)
    new_model = NNModel(spec)
    result = cm.load_epoch(new_model, epoch=99)
    assert result is False


# ---------------------------------------------------------------------------
# 21. load_epoch() actually loads the correct epoch weights
# ---------------------------------------------------------------------------


def test_load_epoch_loads_model(trained_model, temp_checkpoint_dir):
    x = np.random.randn(trained_model.input_size).astype("float32")
    pred_before = trained_model.run(x)

    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=3, manifest=SAMPLE_MANIFEST)

    new_model = NNModel(trained_model.spec)
    cm.load_epoch(new_model, epoch=3)
    pred_after = new_model.run(x)
    assert np.allclose(pred_before, pred_after, atol=1e-6)


# ---------------------------------------------------------------------------
# 22. list_checkpoints() returns empty list when no checkpoints
# ---------------------------------------------------------------------------


def test_list_checkpoints_empty(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    assert cm.list_checkpoints() == []


# ---------------------------------------------------------------------------
# 23. list_checkpoints() returns correct structure including metrics key
# ---------------------------------------------------------------------------


def test_list_checkpoints_structure(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    cm.save(trained_model, {"val_accuracy": 0.6}, epoch=2, manifest=SAMPLE_MANIFEST)

    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 2

    for cp in checkpoints:
        assert "epoch" in cp
        assert "path" in cp
        assert "size_bytes" in cp
        assert "metrics" in cp
        assert isinstance(cp["epoch"], int)
        assert isinstance(cp["path"], str)
        assert isinstance(cp["size_bytes"], int)
        assert isinstance(cp["metrics"], dict)


# ---------------------------------------------------------------------------
# 24. list_checkpoints() sorted by epoch ascending
# ---------------------------------------------------------------------------


def test_list_checkpoints_sorted(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=3, manifest=SAMPLE_MANIFEST)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=5, manifest=SAMPLE_MANIFEST)

    checkpoints = cm.list_checkpoints()
    epochs = [cp["epoch"] for cp in checkpoints]
    assert epochs == [1, 3, 5]


# ---------------------------------------------------------------------------
# 25. list_checkpoints() does NOT include _best.pt
# ---------------------------------------------------------------------------


def test_list_checkpoints_excludes_best(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1, manifest=SAMPLE_MANIFEST)

    checkpoints = cm.list_checkpoints()
    for cp in checkpoints:
        assert "best" not in cp["path"]


# ---------------------------------------------------------------------------
# 26. list_checkpoints() size_bytes is accurate
# ---------------------------------------------------------------------------


def test_list_checkpoints_size_bytes(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)

    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 1
    actual_size = os.path.getsize(checkpoints[0]["path"])
    assert checkpoints[0]["size_bytes"] == actual_size


# ---------------------------------------------------------------------------
# 27. list_checkpoints() metrics carries the saved metrics
# ---------------------------------------------------------------------------


def test_list_checkpoints_metrics_value(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    saved_metrics = {"val_accuracy": 0.77, "val_loss": 0.33}
    cm.save(trained_model, saved_metrics, epoch=1, manifest=SAMPLE_MANIFEST)

    checkpoints = cm.list_checkpoints()
    assert checkpoints[0]["metrics"]["val_accuracy"] == 0.77
    assert checkpoints[0]["metrics"]["val_loss"] == 0.33


# ---------------------------------------------------------------------------
# 28. cleanup() keeps only last N epoch files
# ---------------------------------------------------------------------------


def test_cleanup_keeps_last_n(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    for epoch in range(1, 6):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch, manifest=SAMPLE_MANIFEST)

    assert len(cm.list_checkpoints()) == 5
    cm.cleanup(keep_best=False, keep_last_n=2)

    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 2
    epochs = [cp["epoch"] for cp in checkpoints]
    assert epochs == [4, 5]


# ---------------------------------------------------------------------------
# 29. cleanup() never deletes _best.pt when keep_best=True
# ---------------------------------------------------------------------------


def test_cleanup_never_deletes_best_when_keep_best_true(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.9}, epoch=1, manifest=SAMPLE_MANIFEST)
    for epoch in range(2, 5):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch, manifest=SAMPLE_MANIFEST)

    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)

    cm.cleanup(keep_best=True, keep_last_n=1)
    assert os.path.isfile(best_path)


# ---------------------------------------------------------------------------
# 30. cleanup() deletes _best.pt when keep_best=False
# ---------------------------------------------------------------------------


def test_cleanup_deletes_best_when_keep_best_false(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1, manifest=SAMPLE_MANIFEST)

    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)

    cm.cleanup(keep_best=False, keep_last_n=0)
    assert not os.path.isfile(best_path)


# ---------------------------------------------------------------------------
# 31. cleanup() with keep_last_n=0 and keep_best=False deletes everything
# ---------------------------------------------------------------------------


def test_cleanup_deletes_all(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    for epoch in range(1, 4):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch, manifest=SAMPLE_MANIFEST)

    cm.cleanup(keep_best=False, keep_last_n=0)

    assert len(cm.list_checkpoints()) == 0
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert not os.path.isfile(best_path)


# ---------------------------------------------------------------------------
# 32. cleanup() with custom model_name
# ---------------------------------------------------------------------------


def test_cleanup_with_custom_model_name(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir, model_name="custom")
    for epoch in range(1, 4):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch, manifest=SAMPLE_MANIFEST)

    assert os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch1.pt"))

    cm.cleanup(keep_best=False, keep_last_n=1)

    assert not os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch1.pt"))
    assert not os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch2.pt"))
    assert os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch3.pt"))


# ---------------------------------------------------------------------------
# 33. mode="min": best_metric starts at +inf; lower value promotes
# ---------------------------------------------------------------------------


def test_mode_min_promotes_on_lower_value(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir, mode="min", metric="val_loss")
    assert cm.best_metric == float("+inf")

    cm.save(trained_model, {"val_loss": 0.5}, epoch=1, manifest=SAMPLE_MANIFEST)
    assert cm.best_metric == 0.5
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)

    # Higher loss → should NOT update best
    stat1 = os.stat(best_path)
    time.sleep(0.01)
    cm.save(trained_model, {"val_loss": 0.8}, epoch=2, manifest=SAMPLE_MANIFEST)
    stat2 = os.stat(best_path)
    assert stat1.st_mtime == stat2.st_mtime
    assert cm.best_metric == 0.5

    # Lower loss → SHOULD update best
    cm.save(trained_model, {"val_loss": 0.3}, epoch=3, manifest=SAMPLE_MANIFEST)
    assert cm.best_metric == 0.3


# ---------------------------------------------------------------------------
# 34. best_metric is reconstructable from _best.pt on manager restart
# ---------------------------------------------------------------------------


def test_best_metric_reconstructable_after_restart(trained_model, temp_checkpoint_dir):
    """A second CheckpointManager on the same dir reads best_metric from _best.pt.

    After restart, a worse model must NOT overwrite _best.pt.
    """
    # First manager: save a good model and promote it to _best.pt
    cm1 = CheckpointManager(temp_checkpoint_dir)
    cm1.save(trained_model, {"val_accuracy": 0.88}, epoch=1, manifest=SAMPLE_MANIFEST)
    assert cm1.best_metric == 0.88

    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    stat_before = os.stat(best_path)

    # Second manager on the same dir: must reconstruct best_metric = 0.88
    cm2 = CheckpointManager(temp_checkpoint_dir)
    assert cm2.best_metric == 0.88, (
        f"Restarted manager should read best_metric=0.88 from _best.pt, got {cm2.best_metric}"
    )

    # Save a WORSE model (no promote=True) — _best.pt must NOT be overwritten
    time.sleep(0.01)
    cm2.save(trained_model, {"val_accuracy": 0.50}, epoch=2, manifest=SAMPLE_MANIFEST)
    stat_after = os.stat(best_path)
    assert stat_after.st_mtime == stat_before.st_mtime, (
        "_best.pt must not be overwritten by a worse model after manager restart"
    )

    # Verify _best.pt still holds the original metric
    import torch
    bundle = torch.load(best_path, weights_only=False)
    assert bundle["metrics"]["val_accuracy"] == 0.88


# ---------------------------------------------------------------------------
# 35. Invalid mode raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_mode_raises(temp_checkpoint_dir):
    """CheckpointManager with mode other than 'max'/'min' must raise ValueError."""
    with pytest.raises(ValueError, match="mode"):
        CheckpointManager(temp_checkpoint_dir, mode="foo")
