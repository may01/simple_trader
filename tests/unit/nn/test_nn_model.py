"""Tests for the spec-driven NNModel (Phase-11 Task 05).

The network is built ENTIRELY from an NNModelSpec: input size from
indicators x timeframes x history_points, output size from the per-target
heads. train() consumes an NNDataset (not raw arrays); checkpoints embed the
spec + normalisation manifest + feature_cols so inference is self-contained.

NNDataset is exercised here through small, hand-written on-disk dataset dirs
that satisfy the contract tensors()/split() read (X_{tf}.npy, y.npy, splits.json,
manifest.json) — no pandas / indicator pipeline needed.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from nn.nn_dataset import NNDataset
from nn.nn_model import NNModel
from nn.nn_model_spec import LayerSpec, NNModelSpec, TargetSpec


# ---------------------------------------------------------------------------
# Helpers: build a tiny in-memory NNModelSpec + a tiny on-disk NNDataset
# ---------------------------------------------------------------------------


def _direction_spec(**overrides) -> NNModelSpec:
    """Tiny spec: 1 tf, 2 indicators, history_points=1, one dense layer,
    one direction head. input_size = 2*1*1 = 2, output_size = 3."""
    kwargs = dict(
        name="tiny",
        timeframes=[15],
        indicators=["rsi", "atr_ma"],
        history_points=1,
        layers=[LayerSpec(kind="dense", units=8)],
        targets=[TargetSpec(name="dir15", kind="direction", label_tf=15)],
        epochs=2,
        validation_split=0.2,
        val_strategy="time_holdout",
        device="cpu",
        seed=0,
    )
    kwargs.update(overrides)
    return NNModelSpec(**kwargs)


def _label_spec(**overrides) -> NNModelSpec:
    """Single-head label (binary) spec: head width 1 (sigmoid)."""
    kwargs = dict(
        name="tiny_label",
        timeframes=[15],
        indicators=["rsi", "atr_ma"],
        history_points=1,
        layers=[LayerSpec(kind="dense", units=8)],
        targets=[TargetSpec(name="lab15", kind="label", label_tf=15)],
        epochs=2,
        validation_split=0.2,
        val_strategy="time_holdout",
        device="cpu",
        seed=0,
    )
    kwargs.update(overrides)
    return NNModelSpec(**kwargs)


def _multi_target_spec(**overrides) -> NNModelSpec:
    """Multi-head: direction (3) + label (1) + regression (1) = output_size 5."""
    kwargs = dict(
        name="tiny_multi",
        timeframes=[15],
        indicators=["rsi", "atr_ma"],
        history_points=1,
        layers=[LayerSpec(kind="dense", units=8)],
        targets=[
            TargetSpec(name="dir15", kind="direction", label_tf=15),
            TargetSpec(name="lab15", kind="label", label_tf=15),
            TargetSpec(name="reg15", kind="regression", label_tf=15),
        ],
        epochs=2,
        validation_split=0.2,
        val_strategy="time_holdout",
        device="cpu",
        seed=0,
    )
    kwargs.update(overrides)
    return NNModelSpec(**kwargs)


def _binary_spec(**overrides) -> NNModelSpec:
    """Single-head direction_binary (long) spec: head width 2 (softmax)."""
    kwargs = dict(
        name="tiny_binary",
        timeframes=[15],
        indicators=["rsi", "atr_ma"],
        history_points=1,
        layers=[LayerSpec(kind="dense", units=8)],
        targets=[
            TargetSpec(
                name="long15", kind="direction_binary", side="long",
                label_tf=15, label_m=1.0, label_x=0.3,
            )
        ],
        epochs=2,
        validation_split=0.2,
        val_strategy="time_holdout",
        device="cpu",
        seed=0,
    )
    kwargs.update(overrides)
    return NNModelSpec(**kwargs)


def test_direction_binary_spec_to_softmax_head():
    """Integration: a direction_binary TargetSpec -> one width-2 softmax head."""
    spec = _binary_spec()
    model = NNModel(spec)
    model.build()

    assert model.output_size == 2
    assert [m["width"] for m in model.model.head_meta] == [2]
    assert model.model.head_meta[0]["kind"] == "direction_binary"

    x = torch.zeros(3, spec.history_points, model._n_features)
    logits = model.model.head_logits(x)[0]
    probs = NNModel._apply_head_activation(logits, model.model.head_meta[0])
    assert probs.shape == (3, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)


def _make_y(spec: NNModelSpec, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Build a (n, output_size) y matrix matching the spec's heads.

    direction → one-hot over 3 classes; label → {0,1}; regression → float.
    """
    cols = []
    for t in spec.targets:
        for _h in t.horizons:
            if t.kind == "direction":
                codes = rng.randint(0, 3, size=n)
                oh = np.zeros((n, 3), dtype=np.float32)
                oh[np.arange(n), codes] = 1.0
                cols.append(oh)
            elif t.kind == "direction_binary":
                codes = rng.randint(0, 2, size=n)
                oh = np.zeros((n, 2), dtype=np.float32)
                oh[np.arange(n), codes] = 1.0
                cols.append(oh)
            elif t.kind == "label":
                cols.append(rng.randint(0, 2, size=(n, 1)).astype(np.float32))
            elif t.kind == "regression":
                cols.append(rng.randn(n, 1).astype(np.float32))
    return np.concatenate(cols, axis=1).astype(np.float32)


def _write_dataset(tmpdir: str, spec: NNModelSpec, n: int = 60, seed: int = 0) -> NNDataset:
    """Write a minimal on-disk NNDataset dir for `spec` and return it.

    X is (n, history_points, sum_tf n_features); y matches the heads. Splits
    are time-ordered train/val/holdout per validation_split. Manifest carries
    normalization stats + feature_cols so checkpoints are self-contained.
    """
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

    y = _make_y(spec, n, rng)
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

    dataset_hash = "testhash"
    manifest = {
        "dataset_hash": dataset_hash,
        "timeframes": list(spec.timeframes),
        "history_points": hp,
        "feature_cols": feature_cols,
        "normalization": normalization,
        "rows": n,
    }
    with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    return NNDataset(tmpdir, manifest, cached=False)


@pytest.fixture
def small_model():
    return NNModel(_direction_spec())


@pytest.fixture
def small_dataset(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    return _write_dataset(str(d), _direction_spec())


@pytest.fixture
def trained_model(small_model, small_dataset):
    small_model.train(small_dataset, epoch_callback=lambda e, m: None)
    return small_model


# ---------------------------------------------------------------------------
# Construction / sizing from spec
# ---------------------------------------------------------------------------


def test_input_size_from_spec():
    spec = _direction_spec(timeframes=[15, 60], indicators=["a", "b", "c"], history_points=4)
    m = NNModel(spec)
    # indicators(3) * timeframes(2) * history_points(4)
    assert m.input_size == 3 * 2 * 4


def test_n_features_uses_resolved_feature_cols():
    m = NNModel(_direction_spec(timeframes=[15, 60], indicators=["a", "b"], history_points=4))
    # Simulate a ragged resolved selection: 2 cols @15, 1 col @60 → 3 (NOT 2*2)
    m.feature_cols = {"15": ["15_a", "15_b"], "60": ["60_a"]}
    assert m._n_features == 3
    assert m.input_size == 3 * 4


def test_n_features_falls_back_to_spec_product_when_unresolved():
    m = NNModel(_direction_spec(timeframes=[15, 60], indicators=["a", "b", "c"], history_points=4))
    assert m.feature_cols is None
    assert m._n_features == 3 * 2          # spec product fallback
    assert m.input_size == 3 * 2 * 4       # keeps test_input_size_from_spec semantics


def test_output_size_single_direction_head():
    m = NNModel(_direction_spec())
    assert m.output_size == 3  # direction head width


def test_output_size_label_head():
    m = NNModel(_label_spec())
    assert m.output_size == 1  # label head width (sigmoid)


def test_output_size_multi_target_concatenation():
    m = NNModel(_multi_target_spec())
    # direction(3) + label(1) + regression(1)
    assert m.output_size == 5


def test_output_size_multi_horizon():
    spec = _direction_spec(
        targets=[TargetSpec(name="dir15", kind="direction", label_tf=15, horizons=[1, 3])]
    )
    m = NNModel(spec)
    assert m.output_size == 6  # two horizons x 3-wide direction head


def test_does_not_build_on_construction():
    m = NNModel(_direction_spec())
    assert m.model is None
    assert not m.is_trained


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def test_build_is_noop_on_second_call(small_model):
    small_model.build()
    model_ref = small_model.model
    small_model.build()
    assert small_model.model is model_ref


def test_build_empty_layers_raises():
    spec = _direction_spec(layers=[])
    with pytest.raises(ValueError):
        NNModel(spec).build()


def test_build_empty_targets_raises():
    spec = _direction_spec(targets=[])
    with pytest.raises(ValueError):
        NNModel(spec).build()


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_returns_correct_metric_keys(small_model, small_dataset):
    metrics = small_model.train(small_dataset)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    assert "per_target" in metrics


def test_train_per_target_keyed_by_target_name(small_model, small_dataset):
    metrics = small_model.train(small_dataset)
    assert "dir15" in metrics["per_target"]


def test_is_trained_flag(small_model, small_dataset):
    assert not small_model.is_trained
    small_model.train(small_dataset)
    assert small_model.is_trained


def test_epoch_callback_called(small_dataset):
    spec = _direction_spec(epochs=3, early_stopping_patience=None)
    model = NNModel(spec)
    calls = []

    def cb(epoch, metrics):
        calls.append((epoch, metrics))

    model.train(small_dataset, epoch_callback=cb)
    assert len(calls) == 3
    for _epoch, m in calls:
        assert "loss" in m
        assert "accuracy" in m
        assert "val_loss" in m
        assert "val_accuracy" in m
        assert "per_target" in m


def test_epoch_callback_stop_signal_prunes(small_dataset):
    spec = _direction_spec(epochs=10, early_stopping_patience=None)
    model = NNModel(spec)
    calls = []

    def cb(epoch, metrics):
        calls.append(epoch)
        return True  # Optuna-style prune / stop signal

    model.train(small_dataset, epoch_callback=cb)
    assert len(calls) == 1  # stopped after the first epoch


def test_train_captures_manifest_and_feature_cols(small_model, small_dataset):
    small_model.train(small_dataset)
    assert small_model.manifest is not None
    assert "normalization" in small_model.manifest
    assert small_model.feature_cols is not None


# ---------------------------------------------------------------------------
# Inference: shapes and probability semantics
# ---------------------------------------------------------------------------


def test_run_raises_when_not_trained(small_model):
    small_model.build()
    x = np.random.randn(small_model.input_size).astype("float32")
    with pytest.raises(RuntimeError, match="not trained"):
        small_model.run(x)


def test_run_batch_raises_when_not_trained(small_model):
    small_model.build()
    x = np.random.randn(5, small_model.input_size).astype("float32")
    with pytest.raises(RuntimeError, match="not trained"):
        small_model.run_batch(x)


def test_run_output_shape(trained_model):
    x = np.random.randn(trained_model.input_size).astype("float32")
    pred = trained_model.run(x)
    assert pred.shape == (trained_model.output_size,)


def test_run_batch_output_shape(trained_model):
    x = np.random.randn(7, trained_model.input_size).astype("float32")
    preds = trained_model.run_batch(x)
    assert preds.shape == (7, trained_model.output_size)


def test_run_direction_head_is_softmax(trained_model):
    # single direction head → all 3 outputs are a softmax → sum ~1.
    x = np.random.randn(trained_model.input_size).astype("float32")
    pred = trained_model.run(x)
    assert abs(pred.sum() - 1.0) < 1e-5


def test_run_batch_direction_head_is_softmax(trained_model):
    x = np.random.randn(5, trained_model.input_size).astype("float32")
    preds = trained_model.run_batch(x)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)


def test_multi_target_head_semantics(tmp_path):
    spec = _multi_target_spec()
    d = tmp_path / "ds_multi"
    d.mkdir()
    ds = _write_dataset(str(d), spec)
    model = NNModel(spec)
    model.train(ds)

    x = np.random.randn(4, model.input_size).astype("float32")
    preds = model.run_batch(x)
    assert preds.shape == (4, 5)

    # cols 0..2 = direction softmax (sum ~1); col 3 = label sigmoid in (0,1);
    # col 4 = regression (unconstrained).
    direction = preds[:, 0:3]
    assert np.allclose(direction.sum(axis=1), 1.0, atol=1e-5)
    label = preds[:, 3]
    assert np.all((label >= 0.0) & (label <= 1.0))


def test_label_head_is_sigmoid(tmp_path):
    spec = _label_spec()
    d = tmp_path / "ds_label"
    d.mkdir()
    ds = _write_dataset(str(d), spec)
    model = NNModel(spec)
    model.train(ds)
    x = np.random.randn(6, model.input_size).astype("float32")
    preds = model.run_batch(x)
    assert preds.shape == (6, 1)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


# ---------------------------------------------------------------------------
# Width-mismatch guard
# ---------------------------------------------------------------------------


def test_run_width_mismatch_raises(trained_model):
    bad = np.random.randn(trained_model.input_size + 3).astype("float32")
    with pytest.raises(ValueError, match=str(trained_model.input_size)):
        trained_model.run(bad)


def test_run_batch_width_mismatch_raises(trained_model):
    bad = np.random.randn(4, trained_model.input_size + 1).astype("float32")
    with pytest.raises(ValueError, match=str(trained_model.input_size)):
        trained_model.run_batch(bad)


# ---------------------------------------------------------------------------
# Persistence: self-contained checkpoint
# ---------------------------------------------------------------------------


def test_save_load_round_trip(trained_model):
    x = np.random.randn(trained_model.input_size).astype("float32")
    pred_before = trained_model.run(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trained_model.save_model(path)

        # New model rebuilt purely from the embedded spec — no sizes passed.
        new_model = NNModel(trained_model.spec)
        assert not new_model.is_trained
        new_model.load_model(path)
        assert new_model.is_trained

        pred_after = new_model.run(x)
        assert np.allclose(pred_before, pred_after, atol=1e-6)
    finally:
        os.unlink(path)


def test_checkpoint_embeds_spec_manifest_feature_cols(trained_model):
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trained_model.save_model(path)
        bundle = torch.load(path, weights_only=False)
        assert "state_dict" in bundle
        assert "spec" in bundle
        assert "manifest" in bundle
        assert "feature_cols" in bundle
        assert bundle["manifest"] is not None
        assert bundle["feature_cols"] is not None
    finally:
        os.unlink(path)


def test_load_restores_manifest_and_feature_cols(trained_model):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trained_model.save_model(path)
        new_model = NNModel(trained_model.spec)
        new_model.load_model(path)
        assert new_model.manifest == trained_model.manifest
        assert new_model.feature_cols == trained_model.feature_cols
    finally:
        os.unlink(path)


def test_load_rebuilds_from_embedded_spec_into_fresh_model(trained_model):
    """A fresh NNModel built from the SAME spec content can load the bundle and
    the embedded spec drives the rebuild (round-trip hash matches)."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        trained_model.save_model(path)
        new_model = NNModel(trained_model.spec)
        new_model.load_model(path)
        assert new_model.spec.spec_hash == trained_model.spec.spec_hash
        assert new_model.output_size == trained_model.output_size
    finally:
        os.unlink(path)


def test_checkpoint_roundtrip_preserves_ragged_width(tmp_path):
    from indicators import DataAttributes
    from nn.nn_dataset import NNDataset
    from tests.unit.nn.test_nn_dataset import make_wide_df, small_spec

    # rows=300 (not the brief's 120): with the small_spec default
    # history_points=4, tf=60 needs >=4 closed 60-candles before any row has
    # full lookback depth; 120 rows only yields 2 closes, so every row is
    # NaN-dropped and NNDataset.build raises "no usable rows". 300 rows give
    # enough 60-closes (5) to leave a usable slice after the ragged 15/60 NaN
    # drop while still exercising the same ragged feature_cols_by_tf shape.
    df = make_wide_df(rows=300)
    spec = small_spec(
        timeframes=[15, 60], indicators=["logret", "rsi_14"],
        layers=[LayerSpec(kind="dense", units=8)], epochs=1, device="cpu",
    )
    ds = NNDataset.build(df, DataAttributes(), spec, dataset_dir=str(tmp_path))
    trained = NNModel(spec)
    trained.train(ds)
    assert trained.input_size == 3 * spec.history_points

    path = str(tmp_path / "ckpt.pt")
    trained.save_model(path)

    loaded = NNModel(spec)
    loaded.load_model(path)
    assert loaded.feature_cols == trained.feature_cols
    assert loaded.input_size == trained.input_size          # ragged width, NOT 2*2*hp
    x = np.random.randn(loaded.input_size).astype("float32")
    assert loaded.run(x).shape[0] == loaded.output_size


def test_save_model_raises_when_not_built(small_model):
    with pytest.raises(RuntimeError):
        small_model.save_model("/tmp/should_not_exist_task05.pt")


# ---------------------------------------------------------------------------
# _split_dataset: narrow except — unexpected errors must propagate
# ---------------------------------------------------------------------------


def test_split_dataset_propagates_unexpected_exception(tmp_path):
    """An unexpected exception from dataset.split() must NOT be swallowed.

    The broad ``except Exception: pass`` was narrowed to only catch
    FileNotFoundError and ValueError (the genuine "no splits.json" and
    "unknown split name" cases).  Any other exception — e.g. RuntimeError
    from corrupt data — must bubble up so callers see the real failure.
    """
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), _direction_spec())

    # Monkeypatch dataset.split to raise an unexpected RuntimeError.
    def _boom(name):
        raise RuntimeError("boom — unexpected error from split path")

    dataset.split = _boom

    model = NNModel(_direction_spec())
    with pytest.raises(RuntimeError, match="boom"):
        model.train(dataset)


def test_split_dataset_fallback_when_splits_json_absent(tmp_path):
    """A dataset with no splits.json must still train via the time-holdout
    fallback (FileNotFoundError from split() is the expected "no splits" signal
    and must NOT prevent training).
    """
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), _direction_spec())

    # Remove splits.json so dataset.split() raises FileNotFoundError.
    import os as _os
    _os.remove(os.path.join(str(d), "splits.json"))

    model = NNModel(_direction_spec())
    # Should complete without error — fallback to time-holdout.
    model.train(dataset, epoch_callback=lambda e, m: None)


# ---------------------------------------------------------------------------
# Minibatched training (Task 13): GPU memory scales with batch, not dataset
# ---------------------------------------------------------------------------


def _count_optimizer_steps(monkeypatch) -> dict:
    """Wrap NNModel._make_optimizer so the returned optimizer counts step()."""
    counter = {"steps": 0}
    real_make = NNModel._make_optimizer

    def counting_make(self, model):
        opt = real_make(self, model)
        real_step = opt.step

        def step(*args, **kwargs):
            counter["steps"] += 1
            return real_step(*args, **kwargs)

        opt.step = step
        return opt

    monkeypatch.setattr(NNModel, "_make_optimizer", counting_make)
    return counter


def test_steps_per_epoch_equals_ceil_rows_over_batch(tmp_path, monkeypatch):
    """One optimizer step per minibatch: ceil(n_train / batch_size) per epoch."""
    import math

    spec = _direction_spec(batch_size=8, epochs=1, early_stopping_patience=None)
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)  # train rows = round(60*0.6) = 36

    counter = _count_optimizer_steps(monkeypatch)
    NNModel(spec).train(dataset, epoch_callback=lambda e, m: None)

    assert counter["steps"] == math.ceil(36 / 8)  # 5, not 1 (full-batch)


def test_batch_ge_rows_is_single_step(tmp_path, monkeypatch):
    """batch_size >= n_train → one step/epoch (parity with old full-batch)."""
    spec = _direction_spec(batch_size=1000, epochs=1, early_stopping_patience=None)
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)

    counter = _count_optimizer_steps(monkeypatch)
    NNModel(spec).train(dataset, epoch_callback=lambda e, m: None)

    assert counter["steps"] == 1


def test_minibatch_training_reduces_train_loss(tmp_path):
    """Over many epochs the minibatched loop drives train loss down (the model
    learns/memorises the train split) — proves gradients actually flow per batch."""
    spec = _direction_spec(batch_size=8, epochs=40, early_stopping_patience=None)
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)

    losses: list = []
    NNModel(spec).train(dataset, epoch_callback=lambda e, m: losses.append(m["loss"]))

    assert losses[-1] < losses[0]


def test_training_deterministic_same_seed(tmp_path):
    """Same seed + shuffle_train=True → identical per-epoch loss across two runs
    (seeded weight init AND seeded batch shuffle)."""
    spec = _direction_spec(
        batch_size=8, epochs=3, seed=123, shuffle_train=True,
        early_stopping_patience=None,
    )
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)

    def run() -> list:
        losses: list = []
        NNModel(spec).train(dataset, epoch_callback=lambda e, m: losses.append(m["loss"]))
        return losses

    assert run() == run()


def test_class_weights_computed_once_over_full_train(tmp_path, monkeypatch):
    """Balanced class weights are derived ONCE from the full train labels, not
    per batch and not per epoch."""
    spec = _direction_spec(
        batch_size=8, epochs=2, class_weight="balanced",
        early_stopping_patience=None,
    )
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)  # n_train = 36

    seen_rows: list = []
    real = NNModel._class_weights

    def spy(self, y):
        seen_rows.append(int(y.shape[0]))
        return real(self, y)

    monkeypatch.setattr(NNModel, "_class_weights", spy)
    NNModel(spec).train(dataset, epoch_callback=lambda e, m: None)

    assert seen_rows == [36]  # once, full train rows — not per-batch(8) / per-epoch


# ---------------------------------------------------------------------------
# Lazy mmap dataset (Task 14): host RAM O(batch), not O(rows)
# ---------------------------------------------------------------------------


def test_torch_dataset_byte_parity_with_tensors(tmp_path):
    """torch_dataset() stacked over all rows == tensors()[0], incl. non-ascending
    timeframe order (feature channels must follow manifest order, not sorted)."""
    import torch

    spec = _direction_spec(timeframes=[60, 15])  # non-ascending: order matters
    d = tmp_path / "ds"
    d.mkdir()
    ds = _write_dataset(str(d), spec, n=40)

    X_full, y_full = ds.tensors()
    td = ds.torch_dataset()

    assert len(td) == X_full.shape[0]
    xs = torch.stack([td[i][0] for i in range(len(td))]).numpy()
    ys = torch.stack([td[i][1] for i in range(len(td))]).numpy()
    np.testing.assert_array_equal(xs, X_full)
    np.testing.assert_array_equal(ys, y_full)


def _spy_np_load(monkeypatch) -> list:
    """Record (path, mmap_mode) for every np.load call (numpy module-level)."""
    import numpy

    calls: list = []
    real = numpy.load

    def spy(path, *args, **kwargs):
        calls.append((str(path), kwargs.get("mmap_mode")))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(numpy, "load", spy)
    return calls


def test_X_opened_with_mmap(tmp_path, monkeypatch):
    """torch_dataset() opens every X_{tf}.npy with mmap_mode='r' (never a full
    resident load)."""
    spec = _direction_spec(timeframes=[15, 60])
    d = tmp_path / "ds"
    d.mkdir()
    ds = _write_dataset(str(d), spec, n=30)

    calls = _spy_np_load(monkeypatch)
    ds.torch_dataset()

    x_calls = [(p, m) for (p, m) in calls if "X_" in p]
    assert x_calls, "no X_{tf}.npy opened"
    assert all(m == "r" for (_p, m) in x_calls)


def test_labels_loads_y_only_not_X(tmp_path, monkeypatch):
    """labels() reads y.npy only — never opens an X_{tf}.npy."""
    spec = _direction_spec(timeframes=[15, 60])
    d = tmp_path / "ds"
    d.mkdir()
    ds = _write_dataset(str(d), spec, n=30)

    calls = _spy_np_load(monkeypatch)
    y = ds.labels()

    assert y.shape[0] == 30
    assert not any("X_" in p for (p, _m) in calls)
    assert any(p.endswith("y.npy") for (p, _m) in calls)


def test_training_consumes_lazy_dataset_not_tensors(tmp_path, monkeypatch):
    """Training routes through the lazy mmap dataset (train + val views) and
    never materialises the full X via tensors()."""
    from nn.nn_dataset import NNDataset

    spec = _direction_spec(batch_size=8, epochs=1, early_stopping_patience=None)
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)

    td_calls = {"n": 0}
    tn_calls = {"n": 0}
    real_td = NNDataset.torch_dataset
    real_tn = NNDataset.tensors

    def spy_td(self):
        td_calls["n"] += 1
        return real_td(self)

    def spy_tn(self):
        tn_calls["n"] += 1
        return real_tn(self)

    monkeypatch.setattr(NNDataset, "torch_dataset", spy_td)
    monkeypatch.setattr(NNDataset, "tensors", spy_tn)

    model = NNModel(spec)
    model.train(dataset, epoch_callback=lambda e, m: None)

    assert model.is_trained
    assert td_calls["n"] >= 2  # train view + val view
    assert tn_calls["n"] == 0  # full-X materialisation gone from the train path


def test_head_width_direction_binary():
    from nn.nn_model import _HEAD_WIDTH
    assert _HEAD_WIDTH["direction_binary"] == 2


def test_compute_output_size_direction_binary():
    spec = _binary_spec()
    assert NNModel._compute_output_size(spec) == 2


def test_apply_activation_direction_binary_is_softmax():
    logits = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    out = NNModel._apply_head_activation(logits, {"kind": "direction_binary"})
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-6)
    assert out[0, 0] > out[0, 1]


def test_splits_json_absent_lazy_fallback(tmp_path):
    """No splits.json → lazy time-holdout fallback still trains (parity with the
    Task-13 fallback, now over the lazy dataset)."""
    spec = _direction_spec(batch_size=8, epochs=2, early_stopping_patience=None)
    d = tmp_path / "ds"
    d.mkdir()
    dataset = _write_dataset(str(d), spec, n=60)
    os.remove(os.path.join(str(d), "splits.json"))

    model = NNModel(spec)
    metrics = model.train(dataset, epoch_callback=lambda e, m: None)

    assert model.is_trained
    assert "loss" in metrics and "val_loss" in metrics
