"""nn/nn_model.py — spec-driven neural network for the NN subsystem.

The network is constructed ENTIRELY from an ``NNModelSpec`` (Task 02): nothing
about the architecture, inputs, or targets is hardcoded. Input is multi-
timeframe (``indicators x timeframes x history_points``); output is the
concatenation of one or more jointly-trained target heads. Two models differ
only by their specs.

Checkpoints are self-contained: they embed the serialised spec AND the
normalisation manifest (per-feature training stats + ordered ``feature_cols``)
so a model can be rebuilt and run on ANY dataset without the training folder
present.

Normalisation is NEVER done inside ``NNModel`` (leakage guard). Callers
normalise features with the bundled training manifest before ``run`` /
``run_batch``; this class never recomputes stats from inference data.

Device policy is delegated to ``nn/device.py`` (``resolve_device``), the single
source of device policy. A CUDA OOM during training triggers ONE CPU retry
before failing (per the training-coordinator device policy).
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nn.device import resolve_device
from nn.nn_model_spec import (
    GroupingSpec,
    LayerSpec,
    NNModelSpec,
    TargetSpec,
)

# Per-target-kind head widths (per horizon).
_HEAD_WIDTH = {"direction": 3, "direction_binary": 2, "label": 1, "regression": 1}


class _EpochStats:
    """Row-weighted accumulator for one epoch over minibatches.

    Loss and per-target loss are weighted by batch row count, so the epoch
    aggregate equals the full-batch mean when there is a single step. Accuracy
    accumulates raw correct/count across batches.
    """

    def __init__(self) -> None:
        self._loss_sum = 0.0
        self._rows = 0
        self._correct = 0.0
        self._count = 0.0
        self._per_target_sum: dict = {}

    def add(self, loss, per_target, correct, count, rows) -> None:
        self._loss_sum += loss * rows
        self._rows += rows
        self._correct += correct
        self._count += count
        for key, value in per_target.items():
            self._per_target_sum[key] = self._per_target_sum.get(key, 0.0) + value * rows

    def loss(self) -> float:
        return self._loss_sum / self._rows if self._rows else 0.0

    def accuracy(self) -> float:
        return self._correct / self._count if self._count else 0.0

    def per_target(self) -> dict:
        if not self._rows:
            return {}
        return {key: value / self._rows for key, value in self._per_target_sum.items()}


# ---------------------------------------------------------------------------
# Spec (de)serialisation for self-contained checkpoints
# ---------------------------------------------------------------------------


def spec_to_dict(spec: NNModelSpec) -> dict:
    """Plain-dict form of a spec (nested dataclasses → dicts)."""
    return asdict(spec)


def _spec_from_dict(d: dict) -> NNModelSpec:
    """Rebuild an NNModelSpec from its ``asdict`` form (reconstruct nested)."""
    d = dict(d)
    grouping = GroupingSpec(**d.pop("grouping", {}) or {})
    layers = [LayerSpec(**ls) for ls in d.pop("layers", [])]
    targets = [TargetSpec(**ts) for ts in d.pop("targets", [])]
    return NNModelSpec(grouping=grouping, layers=layers, targets=targets, **d)


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------


def _activation(name: str) -> nn.Module:
    table = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    cls = table.get(name.lower())
    if cls is None:
        raise ValueError(f"unknown activation {name!r}")
    return cls()


# ---------------------------------------------------------------------------
# Sequence backbones (lstm / gru / conv1d treat history as the sequence dim)
# ---------------------------------------------------------------------------


class _SqueezeFlatten(nn.Module):
    """Flatten a (B, T, F) window into (B, T*F)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x.reshape(x.shape[0], -1)


class _RecurrentLast(nn.Module):
    """Wrap nn.LSTM/nn.GRU, return the last timestep's hidden output."""

    def __init__(self, rnn: nn.Module) -> None:
        super().__init__()
        self.rnn = rnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (B, T, H)
        return out[:, -1, :]  # (B, H)


class _Conv1dBlock(nn.Module):
    """Conv1d over the sequence dim, then global-average-pool to (B, C)."""

    def __init__(self, conv: nn.Conv1d, act: nn.Module) -> None:
        super().__init__()
        self.conv = conv
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T) for conv → (B, C, T') → pool → (B, C)
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        return x.mean(dim=2)


# ---------------------------------------------------------------------------
# The spec-built network
# ---------------------------------------------------------------------------


class _SpecNet(nn.Module):
    """Backbone (from spec.layers) + one Linear head per (target, horizon).

    The forward returns RAW logits for every head, concatenated in spec order.
    Activation (softmax/sigmoid/linear) is applied at inference time in
    ``run`` / ``run_batch`` so that training can use numerically-stable losses
    (cross-entropy / BCE-with-logits) on the logits.
    """

    def __init__(self, spec: NNModelSpec, history_points: int, n_features: int) -> None:
        super().__init__()
        self.history_points = history_points
        self.n_features = n_features

        layers = spec.layers
        kinds = {layer.kind for layer in layers}
        sequence_mode = bool(kinds & {"lstm", "gru", "conv1d"})

        backbone: list[nn.Module] = []
        # When any sequence layer is present, the window stays (B, T, F) until
        # the recurrent/conv block collapses it. With only dense layers we flatten
        # the (T, F) window up front.
        if not sequence_mode:
            backbone.append(_SqueezeFlatten())
            in_dim = history_points * n_features
        else:
            in_dim = n_features  # per-timestep feature width

        seq_collapsed = False
        for layer in layers:
            in_dim, seq_collapsed = self._add_layer(
                backbone, layer, in_dim, n_features, seq_collapsed, spec
            )

        # If a sequence backbone never collapsed the time dim (e.g. only an
        # lstm with return-sequence semantics — not used here), guard with a
        # flatten so heads receive a 2-D tensor.
        if sequence_mode and not seq_collapsed:
            backbone.append(_SqueezeFlatten())
            in_dim = in_dim * history_points

        self.backbone = nn.Sequential(*backbone)
        self.backbone_out = in_dim

        # --- Heads: one Linear per (target, horizon) ---
        self.heads = nn.ModuleList()
        self.head_meta: list[dict] = []
        for target in spec.targets:
            width = _HEAD_WIDTH[target.kind]
            for hk in target.horizons:
                self.heads.append(nn.Linear(in_dim, width))
                self.head_meta.append(
                    {"name": target.name, "kind": target.kind, "horizon": hk, "width": width}
                )

    def _add_layer(
        self,
        backbone: list[nn.Module],
        layer: LayerSpec,
        in_dim: int,
        n_features: int,
        seq_collapsed: bool,
        spec: NNModelSpec,
    ) -> tuple[int, bool]:
        kind = layer.kind
        if kind == "dense":
            backbone.append(nn.Linear(in_dim, layer.units))
            backbone.append(_activation(spec.activation))
            if spec.dropout > 0.0:
                backbone.append(nn.Dropout(spec.dropout))
            return layer.units, seq_collapsed

        if kind in ("lstm", "gru"):
            rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
            bidir = bool(layer.params.get("bidirectional", False))
            num_layers = int(layer.params.get("num_layers", 1))
            rnn = rnn_cls(
                input_size=in_dim,
                hidden_size=layer.units,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidir,
            )
            backbone.append(_RecurrentLast(rnn))
            out = layer.units * (2 if bidir else 1)
            return out, True

        if kind == "conv1d":
            kernel = int(layer.params.get("kernel_size", 3))
            padding = int(layer.params.get("padding", kernel // 2))
            conv = nn.Conv1d(
                in_channels=in_dim,
                out_channels=layer.units,
                kernel_size=kernel,
                padding=padding,
            )
            backbone.append(_Conv1dBlock(conv, _activation(spec.activation)))
            return layer.units, True

        raise ValueError(f"unknown layer kind {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return torch.cat([head(feats) for head in self.heads], dim=1)

    def head_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Per-head logits (list, spec order)."""
        feats = self.backbone(x)
        return [head(feats) for head in self.heads]


# ---------------------------------------------------------------------------
# NNModel
# ---------------------------------------------------------------------------


class NNModel:
    """Spec-driven multi-timeframe, multi-target neural network."""

    def __init__(self, spec: NNModelSpec) -> None:
        self.spec = spec
        self.device = resolve_device(spec.device)
        self.output_size = self._compute_output_size(spec)

        self.model: Optional[nn.Module] = None
        self.is_trained: bool = False

        # Captured at train()/load_model() so checkpoints are self-contained.
        self.manifest: Optional[dict] = None
        self.feature_cols: Optional[dict] = None

    @staticmethod
    def _compute_output_size(spec: NNModelSpec) -> int:
        total = 0
        for target in spec.targets:
            total += _HEAD_WIDTH[target.kind] * len(target.horizons)
        return total

    @property
    def _n_features(self) -> int:
        """Per-timestep feature width.

        Uses the RESOLVED feature_cols (from the dataset at train time or the
        bundled manifest at load time) when available; before either is known,
        falls back to the spec's uniform indicators×timeframes product.
        """
        if self.feature_cols:
            return sum(len(cols) for cols in self.feature_cols.values())
        return len(self.spec.indicators) * len(self.spec.timeframes)

    @property
    def input_size(self) -> int:
        return self._n_features * self.spec.history_points

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct ``self.model`` from the spec. Idempotent."""
        if self.model is not None:
            return
        if not self.spec.layers:
            raise ValueError("spec.layers is empty: cannot build a network")
        if not self.spec.targets:
            raise ValueError("spec.targets is empty: cannot build output heads")

        net = _SpecNet(self.spec, self.spec.history_points, self._n_features)
        net.to(self.device)
        self.model = net

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        epoch_callback: Optional[Callable[[int, dict], object]] = None,
    ) -> dict:
        """Train on an ``NNDataset``; return the final metrics dict.

        Splits train/val per ``spec.val_strategy`` / ``spec.validation_split``
        (time-holdout: a row-ordered view, no shuffle, no look-ahead). Per-target
        losses are combined as a weighted sum; classification heads honour
        ``spec.class_weight``. ``epoch_callback(epoch, metrics)`` may return a
        truthy stop signal (Optuna pruning) to end training early.
        """
        # Seed BEFORE build() so weight initialisation is reproducible too —
        # not only the DataLoader shuffle. (Was seeded after build, leaving init
        # dependent on ambient RNG state.)
        if self.spec.seed is not None:
            torch.manual_seed(self.spec.seed)

        # Capture the manifest + resolved feature_cols BEFORE build so the net
        # is sized to the dataset's (possibly ragged) feature width, not the
        # spec's uniform product. Stats are the TRAIN stats (leakage guard).
        self._capture_manifest(dataset)
        self.build()

        train_view, val_view = self._split_dataset(dataset)

        try:
            return self._run_training(
                train_view, val_view, epoch_callback, self.device
            )
        except RuntimeError as exc:
            if self._is_cuda_oom(exc) and self.device.type == "cuda":
                # One CPU retry before failing (training-coordinator device policy).
                torch.cuda.empty_cache()
                self.device = resolve_device("cpu")
                self.model = None
                self.build()
                return self._run_training(
                    train_view, val_view, epoch_callback, self.device
                )
            raise

    # -- training internals --------------------------------------------

    def _split_dataset(self, dataset):
        """Return ``(train_view, val_view)`` lazy NNDataset views.

        Uses the dataset's own time-ordered splits when present; otherwise a
        time-holdout over the full row range (no shuffle). Views are mmap-backed
        — no X tensor is materialised here.
        """
        try:
            train_view = dataset.split("train")
            val_view = dataset.split("val")
            tr_s, tr_e = train_view._row_range()
            va_s, va_e = val_view._row_range()
            if (tr_e - tr_s) > 0 and (va_e - va_s) > 0:
                return train_view, val_view
        except (FileNotFoundError, ValueError):
            # FileNotFoundError: splits.json is absent (in-memory / no-splits dataset).
            # ValueError: split name not found in splits.json.
            # Any other exception (corrupt data, shape mismatch, etc.) propagates.
            pass

        # Fallback: time-holdout over the full row range.
        s, e = dataset._row_range()
        n = e - s
        split = int(round(n * (1.0 - self.spec.validation_split)))
        split = min(max(split, 1), n - 1) if n > 1 else n
        return dataset._subview(s, s + split), dataset._subview(s + split, e)

    def _run_training(
        self, train_view, val_view, epoch_callback, device
    ) -> dict:
        """Minibatched training over lazy, mmap-backed dataset views.

        Each batch's windows are gathered from the on-disk mmaps and moved to
        ``device`` per step, so BOTH host RAM and GPU memory scale with
        ``spec.batch_size``, not dataset size. Loss/accuracy/per-target are
        row-weighted means over batches, so a single-step epoch
        (``batch_size >= rows``) reproduces the old full-batch numbers exactly.
        """
        model = self.model

        train_loader = self._loader_for_view(
            train_view, shuffle=bool(getattr(self.spec, "shuffle_train", True))
        )
        # Validation is NEVER shuffled (order-invariant metrics, deterministic).
        val_loader = self._loader_for_view(val_view, shuffle=False)

        optimizer = self._make_optimizer(model)

        # Class weights computed ONCE over the full train labels (global), not
        # per batch — keeps the 'balanced' weighting stable across batch sizes.
        # labels() loads only y (small); no X_{tf}.npy is materialised.
        full_ytr = torch.tensor(
            train_view.labels(), dtype=torch.float32, device=device
        )
        class_weights = self._class_weights(full_ytr)
        del full_ytr

        best_val = float("inf")
        epochs_no_improve = 0
        patience = self.spec.early_stopping_patience
        metrics: dict = {}

        for epoch in range(self.spec.epochs):
            model.train()
            tr = _EpochStats()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model.head_logits(xb)
                loss, per_target = self._combined_loss(logits, yb, class_weights)
                loss.backward()
                optimizer.step()
                tr.add(float(loss.item()), per_target, *self._accuracy_counts(logits, yb), xb.shape[0])

            model.eval()
            va = _EpochStats()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model.head_logits(xb)
                    loss, per_target = self._combined_loss(logits, yb, class_weights)
                    va.add(float(loss.item()), per_target, *self._accuracy_counts(logits, yb), xb.shape[0])

            val_loss = va.loss()
            metrics = {
                "loss": tr.loss(),
                "accuracy": tr.accuracy(),
                "val_loss": val_loss,
                "val_accuracy": va.accuracy(),
                "per_target": self._merge_per_target(tr.per_target(), va.per_target()),
            }

            if epoch_callback is not None:
                stop = epoch_callback(epoch, metrics)
                if stop:
                    break

            # Early stopping on val_loss.
            if patience is not None:
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

        self.is_trained = True
        return metrics

    def _loader_for_view(self, view, *, shuffle: bool) -> DataLoader:
        """Build a DataLoader over a view's lazy, mmap-backed dataset.

        num_workers=0: each item is a cheap per-row mmap gather; worker
        processes would only add fork overhead (memmaps pickle by path, so
        NUM_WORKERS>0 is a valid later opt-in to overlap I/O). When shuffling, a
        generator seeded by ``spec.seed`` makes the per-epoch order reproducible.
        Shuffle random-accesses mmap rows — fine on SSD/NVMe.
        """
        generator = None
        if shuffle and self.spec.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.spec.seed))
        return DataLoader(
            view.torch_dataset(),
            batch_size=max(int(self.spec.batch_size), 1),
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )

    def _make_optimizer(self, model: nn.Module):
        name = self.spec.optimizer.lower()
        lr = self.spec.learning_rate
        wd = self.spec.weight_decay
        if name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        if name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        raise ValueError(f"unknown optimizer {self.spec.optimizer!r}")

    def _class_weights(self, y: torch.Tensor) -> dict:
        """Per-direction-head class weights when spec.class_weight == 'balanced'.

        Keyed by head index. Only computed for direction heads (3-class). Label
        heads use a scalar pos_weight; regression heads have no class weight.
        """
        weights: dict = {}
        if self.spec.class_weight != "balanced":
            return weights
        offset = 0
        for meta in self.model.head_meta:
            width = meta["width"]
            block = y[:, offset : offset + width]
            if meta["kind"] in ("direction", "direction_binary"):
                counts = block.sum(dim=0)  # one-hot → per-class counts
                counts = torch.clamp(counts, min=1.0)
                w = counts.sum() / (width * counts)
                weights[meta["name"] + f"#{offset}"] = w.to(block.device)
            elif meta["kind"] == "label":
                pos = torch.clamp(block.sum(), min=1.0)
                neg = torch.clamp(block.numel() - block.sum(), min=1.0)
                weights[meta["name"] + f"#{offset}"] = (neg / pos).to(block.device)
            offset += width
        return weights

    def _combined_loss(self, logits_list, y, class_weights):
        """Weighted sum of per-(target,horizon) losses; also return per-target."""
        total = None
        per_target: dict = {}
        offset = 0
        for logits, meta in zip(logits_list, self.model.head_meta):
            width = meta["width"]
            target_y = y[:, offset : offset + width]
            key = meta["name"] + f"#{offset}"
            cw = class_weights.get(key)

            if meta["kind"] in ("direction", "direction_binary"):
                tgt = target_y.argmax(dim=1)
                loss = nn.functional.cross_entropy(logits, tgt, weight=cw)
            elif meta["kind"] == "label":
                pos_weight = cw if cw is not None else None
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, target_y, pos_weight=pos_weight
                )
            else:  # regression
                loss = nn.functional.smooth_l1_loss(logits, target_y)

            total = loss if total is None else total + loss
            per_target.setdefault(meta["name"], 0.0)
            per_target[meta["name"]] += float(loss.item())
            offset += width

        return total, per_target

    def _accuracy_counts(self, logits_list, y) -> tuple[float, float]:
        """Return (correct, count) over classification heads (direction/label).

        Direction → argmax match; label → (sigmoid>0.5) match. Regression heads
        do not contribute. Returned as raw counts so batches accumulate before
        the ratio is taken.
        """
        correct = 0.0
        count = 0.0
        offset = 0
        for logits, meta in zip(logits_list, self.model.head_meta):
            width = meta["width"]
            target_y = y[:, offset : offset + width]
            if meta["kind"] in ("direction", "direction_binary"):
                pred = logits.argmax(dim=1)
                tgt = target_y.argmax(dim=1)
                correct += float((pred == tgt).float().sum().item())
                count += float(target_y.shape[0])
            elif meta["kind"] == "label":
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += float((pred == target_y).float().sum().item())
                count += float(target_y.numel())
            offset += width
        return correct, count

    def _overall_accuracy(self, logits_list, y) -> float:
        """Mean accuracy over classification heads. Returns 0.0 when none."""
        correct, count = self._accuracy_counts(logits_list, y)
        return correct / count if count > 0 else 0.0

    @staticmethod
    def _merge_per_target(tr: dict, va: dict) -> dict:
        out: dict = {}
        for name in tr:
            out[name] = {"loss": tr[name], "val_loss": va.get(name)}
        return out

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or ("cuda" in msg and "memory" in msg)

    def _capture_manifest(self, dataset) -> None:
        manifest = getattr(dataset, "manifest", None)
        if manifest is not None:
            self.manifest = copy.deepcopy(manifest)
            self.feature_cols = copy.deepcopy(manifest.get("feature_cols"))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, features: np.ndarray) -> np.ndarray:
        """Single-sample inference. Returns the concatenated head outputs."""
        if not self.is_trained:
            raise RuntimeError("model not trained")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 1 or features.shape[0] != self.input_size:
            actual = features.shape[-1] if features.ndim >= 1 else features.shape
            raise ValueError(
                f"feature width mismatch: expected {self.input_size}, got {actual}"
            )
        return self.run_batch(features.reshape(1, -1))[0]

    def run_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch inference: (M, input_size) → (M, output_size)."""
        if not self.is_trained:
            raise RuntimeError("model not trained")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.input_size:
            actual = features.shape[1] if features.ndim == 2 else features.shape
            raise ValueError(
                f"feature width mismatch: expected {self.input_size}, got {actual}"
            )

        X = features.reshape(
            features.shape[0], self.spec.history_points, self._n_features
        )
        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits_list = self.model.head_logits(x)
            outs = [
                self._apply_head_activation(logits, meta)
                for logits, meta in zip(logits_list, self.model.head_meta)
            ]
            out = torch.cat(outs, dim=1)
        return out.cpu().numpy()

    @staticmethod
    def _apply_head_activation(logits: torch.Tensor, meta: dict) -> torch.Tensor:
        if meta["kind"] in ("direction", "direction_binary"):
            return torch.softmax(logits, dim=1)
        if meta["kind"] == "label":
            return torch.sigmoid(logits)
        return logits  # regression: linear

    # ------------------------------------------------------------------
    # Persistence (self-contained)
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save a self-contained checkpoint bundle.

        Bundle: ``{state_dict, spec, manifest, feature_cols}`` — the embedded
        spec rebuilds the architecture; the manifest + feature_cols carry the
        training normalisation stats for leakage-free inference on any dataset.
        """
        if self.model is None:
            raise RuntimeError("model has not been built yet")
        bundle = {
            "state_dict": self.model.state_dict(),
            "spec": spec_to_dict(self.spec),
            "manifest": self.manifest,
            "feature_cols": self.feature_cols,
        }
        torch.save(bundle, path)

    def load_model(self, path: str) -> None:
        """Rebuild from the embedded spec, load weights, restore manifest."""
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.spec = _spec_from_dict(bundle["spec"])
        self.output_size = self._compute_output_size(self.spec)
        self.device = resolve_device(self.spec.device)

        self.model = None
        self.build()
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device)

        self.manifest = bundle.get("manifest")
        self.feature_cols = bundle.get("feature_cols")
        self.is_trained = True
