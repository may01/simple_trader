import os
import tempfile

import numpy as np
import pytest

from nn.nn_model import NNModel


@pytest.fixture
def small_model():
    return NNModel(input_size=10, hidden_size=16, num_classes=3)


@pytest.fixture
def trained_model(small_model):
    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 3, size=50)
    small_model.train(X, y, epochs=2)
    return small_model


# 1. build() is no-op on second call
def test_build_is_noop_on_second_call(small_model):
    small_model.build()
    model_ref = small_model.model
    small_model.build()
    assert small_model.model is model_ref, "build() should not replace model on second call"


# 2. run() raises RuntimeError when not trained
def test_run_raises_when_not_trained(small_model):
    small_model.build()
    X_single = np.random.randn(10).astype("float32")
    with pytest.raises(RuntimeError, match="not trained"):
        small_model.run(X_single)


# 3. train() returns correct metric keys
def test_train_returns_correct_metric_keys(small_model):
    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 3, size=50)
    metrics = small_model.train(X, y, epochs=2)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics


# 4. run() output shape (3,)
def test_run_output_shape(trained_model):
    X_single = np.random.randn(10).astype("float32")
    pred = trained_model.run(X_single)
    assert pred.shape == (3,), f"Expected (3,) got {pred.shape}"


# 5. run_batch() output shape (N, 3)
def test_run_batch_output_shape(trained_model):
    X_batch = np.random.randn(7, 10).astype("float32")
    preds = trained_model.run_batch(X_batch)
    assert preds.shape == (7, 3), f"Expected (7, 3) got {preds.shape}"


# 6. run() output sums to ~1 (softmax probabilities)
def test_run_output_is_probabilities(trained_model):
    X_single = np.random.randn(10).astype("float32")
    pred = trained_model.run(X_single)
    assert abs(pred.sum() - 1.0) < 1e-5, f"Probabilities should sum to 1, got {pred.sum()}"


# 7. run_batch() each row sums to ~1
def test_run_batch_output_is_probabilities(trained_model):
    X_batch = np.random.randn(5, 10).astype("float32")
    preds = trained_model.run_batch(X_batch)
    row_sums = preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Each row should sum to 1"


# 8. save_model() / load_model() round-trip
def test_save_load_round_trip(trained_model):
    X_single = np.random.randn(10).astype("float32")
    pred_before = trained_model.run(X_single)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        trained_model.save_model(path)

        new_model = NNModel(input_size=10, hidden_size=16, num_classes=3)
        assert not new_model.is_trained
        new_model.load_model(path)
        assert new_model.is_trained

        pred_after = new_model.run(X_single)
        assert np.allclose(pred_before, pred_after, atol=1e-6), "Predictions differ after load"
    finally:
        os.unlink(path)


# 9. is_trained is False before training and True after
def test_is_trained_flag(small_model):
    assert not small_model.is_trained
    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 3, size=50)
    small_model.train(X, y, epochs=1)
    assert small_model.is_trained


# 10. epoch_callback is called with correct keys
def test_epoch_callback_called(small_model):
    calls = []

    def cb(epoch, metrics):
        calls.append((epoch, metrics))

    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 3, size=50)
    small_model.train(X, y, epochs=3, epoch_callback=cb)

    assert len(calls) == 3
    for epoch, m in calls:
        assert "loss" in m
        assert "accuracy" in m
        assert "val_loss" in m
        assert "val_accuracy" in m


# 11. binary classification (num_classes=2) works
def test_binary_classification():
    model = NNModel(input_size=10, hidden_size=16, num_classes=2)
    X = np.random.randn(50, 10).astype("float32")
    y = np.random.randint(0, 2, size=50)
    metrics = model.train(X, y, epochs=2)
    pred = model.run(X[0])
    assert pred.shape == (2,)
    assert abs(pred.sum() - 1.0) < 1e-5
