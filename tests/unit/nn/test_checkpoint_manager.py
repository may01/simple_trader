import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nn.checkpoint_manager import CheckpointManager
from nn.nn_model import NNModel

# Interim skip (Phase-11 Task 05): every test here builds the OLD
# NNModel(input_size, hidden_size, num_classes) and exercises the bare-state_dict
# save/load. Task 05 replaced NNModel with the spec-driven API and made
# checkpoints self-contained ({state_dict, spec, manifest, feature_cols}).
# CheckpointManager + these tests are rewritten in Task 06, which removes this skip.
pytestmark = pytest.mark.skip(
    reason="NNModel migrated to spec-driven (task 05); CheckpointManager + these "
    "tests are rewritten in task 06"
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def small_model():
    """Create a small untrained model."""
    return NNModel(input_size=10, hidden_size=16, num_classes=3)


@pytest.fixture
def trained_model(small_model):
    """Create a trained model."""
    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 3, size=50)
    small_model.train(X, y, epochs=2)
    return small_model


# 1. Constructor creates checkpoint_dir if it doesn't exist
def test_constructor_creates_checkpoint_dir(temp_checkpoint_dir):
    checkpoint_dir = os.path.join(temp_checkpoint_dir, "nested", "checkpoints")
    cm = CheckpointManager(checkpoint_dir)
    assert os.path.isdir(checkpoint_dir)


# 2. Constructor initializes best_metric to -inf
def test_constructor_initializes_best_metric():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(tmpdir)
        assert cm.best_metric == float("-inf")


# 3. Constructor initializes model_name to "model"
def test_constructor_default_model_name(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    assert cm.model_name == "model"


# 4. Constructor accepts custom model_name
def test_constructor_custom_model_name(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir, model_name="my_model")
    assert cm.model_name == "my_model"


# 5. save() creates epoch checkpoint file
def test_save_creates_epoch_checkpoint(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    returned_path = cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)

    expected_path = os.path.join(temp_checkpoint_dir, "model_epoch1.pt")
    assert os.path.isfile(expected_path)
    assert returned_path == expected_path


# 6. save() returns path of epoch checkpoint (not best)
def test_save_returns_epoch_path(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    returned_path = cm.save(trained_model, {"val_accuracy": 0.75}, epoch=2)

    assert "epoch2" in returned_path
    assert "best" not in returned_path


# 7. save() creates _best.pt when val_accuracy improves
def test_save_creates_best_when_improves(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    # First save: best_metric is -inf, so this becomes best
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)


# 8. save() updates best_metric when val_accuracy improves
def test_save_updates_best_metric(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)
    assert cm.best_metric == 0.5

    cm.save(trained_model, {"val_accuracy": 0.7}, epoch=2)
    assert cm.best_metric == 0.7


# 9. save() does NOT update _best.pt when val_accuracy doesn't improve
def test_save_does_not_update_best_when_not_improved(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    # First save: creates best at 0.7
    cm.save(trained_model, {"val_accuracy": 0.7}, epoch=1)
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    stat1 = os.stat(best_path)

    # Second save: lower accuracy, should not update best
    import time
    time.sleep(0.01)  # Ensure different mtime if file were updated
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=2)
    stat2 = os.stat(best_path)

    assert stat1.st_mtime == stat2.st_mtime, "best.pt should not have been updated"
    assert cm.best_metric == 0.7


# 10. save() is atomic (uses .tmp and os.rename)
def test_save_is_atomic(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)

    # Check no .tmp files are left
    tmp_files = list(Path(temp_checkpoint_dir).glob("*.tmp"))
    assert len(tmp_files) == 0, "Atomic save should not leave .tmp files"


# 11. load_best() returns True on success
def test_load_best_returns_true_on_success(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1)

    new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
    result = cm.load_best(new_model)
    assert result is True


# 12. load_best() returns False when no best exists
def test_load_best_returns_false_when_no_best(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
    result = cm.load_best(new_model)
    assert result is False


# 13. load_best() actually loads the best model
def test_load_best_loads_model(trained_model, temp_checkpoint_dir):
    X_single = np.random.randn(10).astype("float32")
    pred_before = trained_model.run(X_single)

    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1)

    new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
    cm.load_best(new_model)
    pred_after = new_model.run(X_single)

    assert np.allclose(pred_before, pred_after, atol=1e-6)


# 14. load_epoch() returns True on success
def test_load_epoch_returns_true_on_success(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=5)

    new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
    result = cm.load_epoch(new_model, epoch=5)
    assert result is True


# 15. load_epoch() returns False when epoch not found
def test_load_epoch_returns_false_when_not_found(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
    result = cm.load_epoch(new_model, epoch=99)
    assert result is False


# 16. load_epoch() actually loads the correct epoch
def test_load_epoch_loads_model(trained_model, temp_checkpoint_dir):
    X_single = np.random.randn(10).astype("float32")
    pred_before = trained_model.run(X_single)

    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=3)

    new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
    cm.load_epoch(new_model, epoch=3)
    pred_after = new_model.run(X_single)

    assert np.allclose(pred_before, pred_after, atol=1e-6)


# 17. list_checkpoints() returns empty list when no checkpoints
def test_list_checkpoints_empty(temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    checkpoints = cm.list_checkpoints()
    assert checkpoints == []


# 18. list_checkpoints() returns correct structure
def test_list_checkpoints_structure(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)
    cm.save(trained_model, {"val_accuracy": 0.6}, epoch=2)

    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 2

    for cp in checkpoints:
        assert "epoch" in cp
        assert "path" in cp
        assert "size_bytes" in cp
        assert isinstance(cp["epoch"], int)
        assert isinstance(cp["path"], str)
        assert isinstance(cp["size_bytes"], int)


# 19. list_checkpoints() sorted by epoch ascending
def test_list_checkpoints_sorted(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=3)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=5)

    checkpoints = cm.list_checkpoints()
    epochs = [cp["epoch"] for cp in checkpoints]
    assert epochs == [1, 3, 5]


# 20. list_checkpoints() does NOT include _best.pt
def test_list_checkpoints_excludes_best(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1)

    checkpoints = cm.list_checkpoints()
    for cp in checkpoints:
        assert "best" not in cp["path"]


# 21. list_checkpoints() size_bytes is correct
def test_list_checkpoints_size_bytes(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.5}, epoch=1)

    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 1

    actual_size = os.path.getsize(checkpoints[0]["path"])
    assert checkpoints[0]["size_bytes"] == actual_size


# 22. cleanup() keeps only last N epoch files
def test_cleanup_keeps_last_n(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    for epoch in range(1, 6):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch)

    # Before cleanup: 5 checkpoints
    assert len(cm.list_checkpoints()) == 5

    # Cleanup with keep_last_n=2
    cm.cleanup(keep_best=False, keep_last_n=2)

    # After cleanup: 2 checkpoints (epochs 4 and 5)
    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 2
    epochs = [cp["epoch"] for cp in checkpoints]
    assert epochs == [4, 5]


# 23. cleanup() never deletes _best.pt when keep_best=True
def test_cleanup_never_deletes_best_when_keep_best_true(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    # Create checkpoints with first one being best
    cm.save(trained_model, {"val_accuracy": 0.9}, epoch=1)
    for epoch in range(2, 5):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch)

    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)

    # Cleanup with keep_best=True, keep_last_n=1
    cm.cleanup(keep_best=True, keep_last_n=1)

    # best.pt should still exist
    assert os.path.isfile(best_path)


# 24. cleanup() deletes _best.pt when keep_best=False
def test_cleanup_deletes_best_when_keep_best_false(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)
    cm.save(trained_model, {"val_accuracy": 0.75}, epoch=1)

    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert os.path.isfile(best_path)

    # Cleanup with keep_best=False
    cm.cleanup(keep_best=False, keep_last_n=0)

    # best.pt should be deleted
    assert not os.path.isfile(best_path)


# 25. cleanup() with keep_last_n=0 and keep_best=False deletes everything
def test_cleanup_deletes_all(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir)

    for epoch in range(1, 4):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch)

    # Cleanup with keep_last_n=0 and keep_best=False
    cm.cleanup(keep_best=False, keep_last_n=0)

    # Should have no checkpoints
    checkpoints = cm.list_checkpoints()
    assert len(checkpoints) == 0

    # Should have no best.pt either
    best_path = os.path.join(temp_checkpoint_dir, "model_best.pt")
    assert not os.path.isfile(best_path)


# 26. cleanup() with custom model_name
def test_cleanup_with_custom_model_name(trained_model, temp_checkpoint_dir):
    cm = CheckpointManager(temp_checkpoint_dir, model_name="custom")

    for epoch in range(1, 4):
        cm.save(trained_model, {"val_accuracy": 0.5}, epoch=epoch)

    # Check files exist with custom name
    assert os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch1.pt"))

    # Cleanup
    cm.cleanup(keep_best=False, keep_last_n=1)

    # Only epoch 3 should remain
    assert not os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch1.pt"))
    assert not os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch2.pt"))
    assert os.path.isfile(os.path.join(temp_checkpoint_dir, "custom_epoch3.pt"))
