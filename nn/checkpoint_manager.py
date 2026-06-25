"""nn/checkpoint_manager.py — self-contained PyTorch checkpoint management.

Each checkpoint is a bundle: {state_dict, spec, manifest, feature_cols,
metrics}.  The bundle is self-contained so inference can run on any dataset
using the training normalisation stats (leakage guard).

Promotion to _best.pt fires when:
  - ``promote=True`` (ExperimentTracker holdout decision), OR
  - the gate metric in ``metrics[self.metric]`` improves on ``best_metric``
    in the direction given by ``mode`` ("max" or "min").

Writes are atomic: temp .tmp file + os.rename; on failure the .tmp is removed
and the exception re-raised.
"""

import os
import re
from pathlib import Path

import torch

from nn.nn_model import NNModel, spec_to_dict


class CheckpointManager:
    """Save/load self-contained PyTorch NNModel checkpoints with best-model tracking."""

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str = "model",
        mode: str = "max",
        metric: str = "val_accuracy",
    ) -> None:
        """Create checkpoint_dir if absent; init best_metric per mode.

        Args:
            checkpoint_dir: Directory where checkpoints are stored.
            model_name: Base name for checkpoint files (default: "model").
            mode: "max" (higher is better) or "min" (lower is better).
            metric: Key in the metrics dict used for auto-promote comparison.
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.mode = mode
        self.metric = metric
        self.best_metric = float("-inf") if mode == "max" else float("+inf")

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Reconstruct best_metric from an existing _best.pt so that a restarted
        # manager never overwrites a good checkpoint on the first save.
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        if os.path.isfile(best_path):
            try:
                bundle = torch.load(best_path, map_location="cpu", weights_only=False)
                self.best_metric = bundle["metrics"][self.metric]
            except Exception:
                pass  # corrupt file or missing key → keep the -inf/+inf default

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        """Return True if ``value`` improves on ``best_metric`` per ``mode``."""
        if self.mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def _atomic_save(self, bundle: dict, dest_path: str) -> None:
        """Write bundle to a .tmp file then rename atomically to dest_path."""
        tmp_path = dest_path + ".tmp"
        try:
            torch.save(bundle, tmp_path)
            os.rename(tmp_path, dest_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _build_bundle(self, model: NNModel, metrics: dict, manifest: dict) -> dict:
        """Collect the self-contained checkpoint bundle from the model."""
        if model.model is None:
            raise RuntimeError("model has not been built yet")

        return {
            "state_dict": model.model.state_dict(),
            "spec": spec_to_dict(model.spec),
            "manifest": manifest,
            "feature_cols": model.feature_cols,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: NNModel,
        metrics: dict,
        epoch: int,
        manifest: dict,
        promote: bool = False,
    ) -> str:
        """Save {model_name}_epoch{epoch}.pt as a self-contained bundle.

        Writes _best.pt if ``promote`` is True OR if the gate metric improves
        on ``best_metric`` per ``mode``.  Writes are atomic.

        Args:
            model: Trained NNModel instance.
            metrics: Validation metrics dict (must contain the gate metric key
                unless ``promote=True`` and you don't need auto-gate).
            epoch: Epoch number for the checkpoint filename.
            manifest: Normalisation manifest dict (training stats).
            promote: If True, always write _best.pt regardless of the gate.

        Returns:
            Path to the saved epoch checkpoint file.
        """
        epoch_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch{epoch}.pt")
        bundle = self._build_bundle(model, metrics, manifest)
        self._atomic_save(bundle, epoch_path)

        # Determine whether to promote to _best.pt
        gate_value = metrics.get(self.metric)
        auto_improve = gate_value is not None and self._is_improvement(gate_value)
        should_promote = promote or auto_improve

        if should_promote:
            # Update best_metric only when the gate metric actually improved;
            # promote=True that is forced (worse metric) does NOT regress best_metric.
            if auto_improve:
                self.best_metric = gate_value

            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            self._atomic_save(bundle, best_path)

        return epoch_path

    def load_best(self, model: NNModel) -> "dict | None":
        """Load _best.pt into model; return {manifest, feature_cols} or None.

        Rebuilds the architecture from the embedded spec via model.load_model().
        Returns None if no best checkpoint exists (caller treats as "no model").
        """
        best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
        if not os.path.isfile(best_path):
            return None

        model.load_model(best_path)
        # load_model restores manifest and feature_cols onto the model;
        # also return them to the caller.
        return {
            "manifest": model.manifest,
            "feature_cols": model.feature_cols,
        }

    def load_epoch(self, model: NNModel, epoch: int) -> bool:
        """Load a specific epoch checkpoint into model. Returns True/False."""
        epoch_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_epoch{epoch}.pt")
        if not os.path.isfile(epoch_path):
            return False
        model.load_model(epoch_path)
        return True

    def list_checkpoints(self) -> list[dict]:
        """Return [{epoch, path, size_bytes, metrics}, ...] sorted by epoch asc.

        Excludes _best.pt.
        """
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if not (filename.startswith(self.model_name) and filename.endswith(".pt")):
                continue
            if "_best.pt" in filename:
                continue
            if "_epoch" not in filename:
                continue

            m = re.match(r"^.+_epoch(\d+)\.pt$", filename)
            if m is None:
                continue

            try:
                epoch = int(m.group(1))
            except ValueError:
                continue

            full_path = os.path.join(self.checkpoint_dir, filename)
            size_bytes = os.path.getsize(full_path)

            # Load metrics from bundle (cheap: torch.load is fast for small bundles)
            try:
                bundle = torch.load(full_path, weights_only=False)
                metrics = bundle.get("metrics", {})
            except Exception:
                metrics = {}

            checkpoints.append(
                {
                    "epoch": epoch,
                    "path": full_path,
                    "size_bytes": size_bytes,
                    "metrics": metrics,
                }
            )

        checkpoints.sort(key=lambda x: x["epoch"])
        return checkpoints

    def cleanup(self, keep_best: bool = True, keep_last_n: int = 3) -> None:
        """Delete old epoch checkpoints, retaining the last N and optionally _best.pt."""
        checkpoints = self.list_checkpoints()

        if keep_last_n == 0:
            to_delete = checkpoints
        else:
            to_delete = checkpoints[:-keep_last_n]

        for cp in to_delete:
            os.remove(cp["path"])

        if not keep_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            if os.path.isfile(best_path):
                os.remove(best_path)
