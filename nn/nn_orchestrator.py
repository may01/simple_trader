"""nn/nn_orchestrator.py — execution-level coordinator for NN train + inference.

Reworked for Phase-11: the per-TF ``{tf}_nn_prob_*`` design is replaced by a
**dataset-built, grouping-partitioned** coordinator. Each model ingests
**multi-timeframe** input (per ``spec.timeframes``) and emits **timeframe-
agnostic** ``nn_res_*`` output. Rows are partitioned into groups per
``spec.grouping`` (one group when ``mode='single'``); one ``NNModel`` per group,
keyed by group string, with inference routing each row to its group's model.

Two modes, both called from ``Trainer``:
  - ``train()``        — build an NNDataset ONCE over all rows, partition by
                          ``spec.grouping``, train one NNModel per group, persist
                          each via CheckpointManager. Usable standalone (single-
                          shot, no TrainingLoop).
  - ``run_inference()`` — load the best checkpoint per group, build a multi-TF
                          feature matrix from ANY target ``df``, normalise via the
                          checkpoint's BUNDLED training manifest (leakage guard),
                          route each row to its group's model, and return a
                          NN-columns-only DataFrame indexed by ``df.index``.

``run_inference`` is PURE: it returns the ``nn_res_*`` DataFrame and writes no
files. ``run_inference_dataset`` (task 11) is the thin write-wrapper: it loads a
dataset folder, calls ``run_inference``, and atomically writes
``{dataset}/df_with_nn.pkl`` (absence-safe; never mutates
``df_with_indicators.pkl``).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from indicators import DataAttributes
from nn.checkpoint_manager import CheckpointManager
from nn.nn_dataset import NNDataset
from nn.nn_model import NNModel
from nn.nn_model_spec import NNModelSpec


class NNOrchestrator:
    """Execution-level coordinator for NN training and batch inference.

    Each model ingests multi-timeframe features (per spec.timeframes) and emits
    timeframe-agnostic nn_res_* output. Rows are partitioned into groups per
    spec.grouping; one NNModel per group (a single group if grouping=single),
    with inference routing each row to its group's model.

    Attributes:
        checkpoint_dir: Where CheckpointManager stores weights per group.
        dataset_dir: Where NNDataset materialised tensors live.
        base_spec: Default NNModelSpec (the TrainingLoop may override fields).
        num_workers: From NUM_WORKERS env (fallback AVAIABLE_THREADS, default 4).
        trained_models: dict[str, NNModel] keyed by group string.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        dataset_dir: str,
        base_spec: NNModelSpec,
    ) -> None:
        """Init paths + base_spec; read num_workers from NUM_WORKERS env
        (fallback AVAIABLE_THREADS, default 4); start trained_models empty."""
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.base_spec = base_spec
        self.num_workers = int(
            os.environ.get("NUM_WORKERS", os.environ.get("AVAIABLE_THREADS", "4"))
        )
        self.trained_models: dict[str, NNModel] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_trainer(cls, pair: str, trainer: "object") -> "NNOrchestrator":
        """Pair-scoped factory.

        Loads ``base_spec = NNModelSpec.from_yaml('configs/nn_spec.yaml')``;
        ``artefact_root = nn_artefact_root(pair)``; builds
            checkpoint_dir = f"{artefact_root}/checkpoints/{base_spec.spec_hash}"
            dataset_dir    = f"{artefact_root}/datasets"
        and passes ``num_workers = trainer.available_threads()``
        (duck-typed; the real method lands on Trainer in task 11).
        """
        # Imported lazily so device-env reads happen at call time, not import.
        from nn.device import nn_artefact_root

        base_spec = NNModelSpec.from_yaml("configs/nn_spec.yaml")
        artefact_root = nn_artefact_root(pair)
        checkpoint_dir = f"{artefact_root}/checkpoints/{base_spec.spec_hash}"
        dataset_dir = f"{artefact_root}/datasets"

        orch = cls(
            checkpoint_dir=checkpoint_dir,
            dataset_dir=dataset_dir,
            base_spec=base_spec,
        )
        orch.num_workers = trainer.available_threads()
        return orch

    # ------------------------------------------------------------------
    # Train Mode
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        data_attributes: DataAttributes,
        spec: Optional[NNModelSpec] = None,
        epoch_callback: Optional[Callable[[str, int, dict], object]] = None,
        promote: bool = True,
    ) -> dict:
        """Train one NNModel per group; return {group_key: final_metrics}.

        Builds the NNDataset ONCE over all rows (multi-TF input per
        spec.timeframes), cached under ``dataset_dir``. Partitions rows into
        groups via ``dataset.groups(spec.grouping)`` (mode='single' → the one
        group "all"). For each group: train an NNModel on that group's view,
        persist via CheckpointManager (manifest = the dataset manifest, which
        bundles the TRAIN normalisation stats + ordered feature_cols), and store
        the model under ``trained_models[group_key]``.

        ``epoch_callback`` is ``(group_key, epoch, metrics)``; it is WRAPPED to
        adapt NNModel's ``(epoch, metrics)`` callback (injecting group_key).

        ``promote`` is threaded into ``CheckpointManager.save``. The default
        ``True`` preserves the STANDALONE single-shot behaviour: a one-off
        ``train()`` (no TrainingLoop) force-promotes each group's ``_best.pt``.
        The TrainingLoop passes ``promote=False`` so per-trial training saves
        an epoch checkpoint WITHOUT force-promoting ``_best.pt``; the loop's
        holdout gate (``_maybe_promote``) is then the SOLE promoter, keeping
        saved weights and ``best.json`` from diverging across trials.

        Standalone single-shot: no TrainingLoop. ``mode='by_indicator'`` defers
        to ``dataset.groups`` (raises NotImplementedError) per the brief.
        """
        spec = spec or self.base_spec

        # Build the dataset ONCE over all rows (multi-TF), content-addressed.
        dataset = NNDataset.build(
            df, data_attributes, spec, dataset_dir=self.dataset_dir
        )

        results: dict = {}
        for group_key in dataset.groups(spec.grouping):
            group_ds = dataset.group(group_key)

            model = NNModel(spec)

            wrapped_cb = self._wrap_epoch_callback(epoch_callback, group_key)
            metrics = model.train(group_ds, epoch_callback=wrapped_cb)

            cm = CheckpointManager(
                self.checkpoint_dir,
                model_name=group_key,
                mode="max",
            )
            if not promote:
                # Loop-driven trial: the TrainingLoop's holdout gate is the SOLE
                # promoter (it owns ``_best.pt``). Neutralise this manager's
                # auto-promote gate so a per-trial ``val_accuracy`` improvement
                # cannot silently overwrite ``_best.pt`` behind the loop's back
                # (which would diverge from ``best.json``). We still write the
                # epoch checkpoint below.
                cm.best_metric = float("inf")
            cm.save(
                model,
                metrics,
                epoch=spec.epochs,
                manifest=group_ds.manifest,
                promote=promote,
            )

            self.trained_models[group_key] = model
            results[group_key] = metrics

        return results

    @staticmethod
    def _wrap_epoch_callback(
        epoch_callback: Optional[Callable[[str, int, dict], object]],
        group_key: str,
    ) -> Optional[Callable[[int, dict], object]]:
        """Adapt a (group_key, epoch, metrics) callback to NNModel's
        (epoch, metrics) signature by injecting ``group_key``.

        Returns the model-side callback (passing through any truthy stop signal),
        or None when no callback was supplied.
        """
        if epoch_callback is None:
            return None

        def model_cb(epoch: int, metrics: dict) -> object:
            return epoch_callback(group_key, epoch, metrics)

        return model_cb

    # ------------------------------------------------------------------
    # Inference Mode
    # ------------------------------------------------------------------

    def run_inference(
        self,
        df: pd.DataFrame,
        data_attributes: DataAttributes,
        spec: Optional[NNModelSpec] = None,
    ) -> Optional[pd.DataFrame]:
        """Run batch inference and return a NN-columns-only DataFrame.

        Loads the best checkpoint per group via ``CheckpointManager.load_best``
        (each checkpoint embeds its own spec + normalisation manifest + ordered
        feature_cols). Builds the multi-TF feature matrix from ``df`` with the
        SAME closed-candle lookback as training (NNDataset.build_inference_matrix)
        and normalises via the checkpoint's BUNDLED manifest stats — NEVER stats
        recomputed from the inference data (leakage guard; ``data_attributes`` is
        unused on this path). Routes each row to its group's model (single group
        → all rows), ``run_batch`` → output sized to the checkpoint spec's
        targets. Appends ONLY the timeframe-agnostic ``nn_res_*`` columns (from
        ``TargetSpec.out_columns()``) to a FRESH DataFrame indexed by ``df.index``.

        Input ``df`` is NOT mutated. Absence-safe: a group with no best
        checkpoint contributes nothing; NO checkpoints at all → an empty
        DataFrame (caller writes nothing).
        """
        spec = spec or self.base_spec

        result_data: dict[str, np.ndarray] = {}
        any_checkpoint = False

        for group_key in self._inference_group_keys(spec):
            model = NNModel(spec)
            cm = CheckpointManager(
                self.checkpoint_dir,
                model_name=group_key,
                mode="max",
            )
            loaded = cm.load_best(model)
            if loaded is None:
                # No best checkpoint for this group → contributes nothing.
                continue
            any_checkpoint = True

            # The checkpoint's embedded spec wins (it produced the weights).
            ckpt_spec = model.spec
            manifest = loaded.get("manifest") or model.manifest or {}
            feature_cols_by_tf = (
                loaded.get("feature_cols")
                or manifest.get("feature_cols")
                or {}
            )
            normalization = manifest.get("normalization", {})
            history_points = manifest.get(
                "history_points", ckpt_spec.history_points
            )

            # Build the normalised multi-TF window matrix using the SAME builder
            # as training, with the checkpoint's BUNDLED stats (no recompute).
            X, valid_mask = NNDataset.build_inference_matrix(
                df,
                feature_cols_by_tf,
                history_points,
                normalization,
            )

            out_columns = self._spec_out_columns(ckpt_spec)
            self._ensure_result_columns(result_data, out_columns, len(df))

            if valid_mask.any():
                X_valid = X[valid_mask].reshape(int(valid_mask.sum()), -1)
                preds = model.run_batch(X_valid)  # (M, output_size)
                self._scatter_predictions(
                    result_data, out_columns, preds, valid_mask
                )

        if not any_checkpoint:
            # Absence-safe: nothing to write. Empty, df.index-aligned frame.
            return pd.DataFrame(index=df.index)

        return pd.DataFrame(result_data, index=df.index)

    # ------------------------------------------------------------------
    # Dataset write-wrapper (D5)
    # ------------------------------------------------------------------

    def run_inference_dataset(
        self, dataset_dir: str, checkpoint_id: str = "best"
    ) -> Optional[pd.DataFrame]:
        """Run inference over a dataset folder and atomically persist results.

        Loads ``df_with_indicators.pkl`` (+ ``data_attributes.pkl`` if present)
        from ``dataset_dir``, runs the PURE :meth:`run_inference`, and — only when
        the result carries at least one ``nn_res_*`` column — atomically writes
        ``{dataset_dir}/df_with_nn.pkl`` via a ``.tmp`` + ``os.replace`` swap.
        ``df_with_indicators.pkl`` is NEVER mutated.

        Absence-safe: a ``None`` result, or a result with no ``nn_res_*`` columns
        (no promoted checkpoint), writes nothing and returns ``None``.

        ``checkpoint_id`` selects which checkpoint set to score with; ``"best"``
        (the only value wired today) routes through ``CheckpointManager.load_best``
        inside :meth:`run_inference`. Non-``"best"`` ids are a reserved future hook
        (named checkpoints) and currently fall through to the same ``best`` path.
        """
        df = pd.read_pickle(os.path.join(dataset_dir, "df_with_indicators.pkl"))

        attrs_path = os.path.join(dataset_dir, "data_attributes.pkl")
        data_attributes = (
            DataAttributes.load(attrs_path)
            if os.path.exists(attrs_path)
            else DataAttributes()
        )

        result = self.run_inference(df, data_attributes)

        if result is None or not any(
            str(col).startswith("nn_res_") for col in result.columns
        ):
            # Absence-safe: no checkpoint / nothing to write.
            return None

        out_path = os.path.join(dataset_dir, "df_with_nn.pkl")
        tmp_path = out_path + ".tmp"
        result.to_pickle(tmp_path)
        os.replace(tmp_path, out_path)
        return result

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _inference_group_keys(self, spec: NNModelSpec) -> list[str]:
        """Group keys to attempt at inference time.

        For ``mode='single'`` this is the single ``"all"`` group. (Per the brief,
        ``by_indicator`` routing is deferred; ``NNDataset.groups`` raises for it,
        so we mirror the single-group surface here.)
        """
        if spec.grouping.mode == "single":
            return ["all"]
        raise NotImplementedError(
            f"grouping mode {spec.grouping.mode!r} routing is deferred (task 08 "
            "ships grouping=single)"
        )

    @staticmethod
    def _spec_out_columns(spec: NNModelSpec) -> list[str]:
        """Ordered nn_res_* output column names across all of the spec's targets."""
        cols: list[str] = []
        for target in spec.targets:
            cols.extend(target.out_columns())
        return cols

    @staticmethod
    def _ensure_result_columns(
        result_data: dict[str, np.ndarray], out_columns: list[str], n: int
    ) -> None:
        """Pre-create each nn_res_* column as an all-NaN (n,) array (idempotent).

        All groups write the SAME nn_res_* columns; only the producing model
        differs per row. A group with no valid rows still contributes the columns
        (all-NaN), keeping the output schema stable across groups.
        """
        for col in out_columns:
            if col not in result_data:
                result_data[col] = np.full(n, np.nan, dtype=np.float64)

    @staticmethod
    def _scatter_predictions(
        result_data: dict[str, np.ndarray],
        out_columns: list[str],
        preds: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        """Write a group's (M, output_size) predictions into the valid rows.

        ``out_columns`` is the model's output layout in spec order, so column j
        maps to ``preds[:, j]``.
        """
        for j, col in enumerate(out_columns):
            result_data[col][valid_mask] = preds[:, j]
