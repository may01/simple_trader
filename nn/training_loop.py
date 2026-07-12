"""nn/training_loop.py — hybrid Optuna + NNStrategist round loop (task 10).

Turns NN training from a single run into a *search for a better model*. Each
round is ``propose → Optuna study → train+evaluate trials → record → review``:

  - **NNStrategist** (LLM, optional) decides *which* indicators / timeframes /
    targets to explore and *when to stop* — the search *scope*.
  - **Optuna** samples + prunes the numeric knobs (lr / units / dropout) *within*
    that scope (architecture depth is reasoned, not searched — ADR-0001).
  - **NNOrchestrator** trains each concrete spec (one NNModel per group).
  - **ExperimentTracker** records every trial and gates promotion on a
    time-ordered holdout.

The loop owns ORCHESTRATION only: no model weights, no metric definitions, no
LLM prompt — those belong to the orchestrator/CheckpointManager, the tracker,
and the strategist respectively. The terminal artefact is the promoted best
checkpoint per group plus the tracker's full history, returned as a RunResult.

Optuna isolation
----------------
``optuna`` is installed ONLY in the nn-train image, never the always-on
``simple_trader`` base image. It is therefore imported LAZILY inside ``run()``
(and the helpers it reaches) so that ``import nn.training_loop`` succeeds in the
base image — the Trainer (task 11) imports this module unconditionally.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from logs import log, log_warning
from nn.checkpoint_manager import CheckpointManager
from nn.nn_dataset import NNDataset
from nn.nn_model_spec import LayerSpec, NNModelSpec

if TYPE_CHECKING:  # pragma: no cover — typing only
    from indicators import DataAttributes
    from nn.experiment_tracker import ExperimentTracker
    from nn.nn_orchestrator import NNOrchestrator
    from nn.nn_strategist import NNStrategist, Proposal


# ---------------------------------------------------------------------------
# Holdout chunking
# ---------------------------------------------------------------------------

#: Maximum rows per ``run_batch`` call during holdout evaluation.
#: Keeps GPU memory bounded regardless of holdout size (fix for OOM on large
#: holdouts with LSTM models on constrained GPUs).
HOLDOUT_EVAL_CHUNK: int = 4096

#: Top-fraction of bars (ranked by long score) used for the precision@k gate.
#: Single edit point to retune k.
PRECISION_AT_K_FRAC: float = 0.05

#: Promotion/search gate metric selector. "accuracy" preserves the legacy argmax
#: gate; "precision_at_k" optimises long-class precision on rare-positive targets.
GATE_METRIC_ACCURACY: str = "accuracy"
GATE_METRIC_PRECISION_AT_K: str = "precision_at_k"
VALID_GATE_METRICS: tuple[str, ...] = (GATE_METRIC_ACCURACY, GATE_METRIC_PRECISION_AT_K)


def _chunked_predict(model, X_flat: np.ndarray, chunk_size: int) -> np.ndarray:
    """Run ``model.run_batch`` in fixed-size chunks and concatenate results.

    Args:
        model:      Any object with a ``run_batch(X: np.ndarray) -> np.ndarray``
                    method.
        X_flat:     2-D float32 array of shape ``(n_rows, features)``.
        chunk_size: Maximum number of rows per ``run_batch`` call.

    Returns:
        Predictions array of shape ``(n_rows, out_width)``, identical to a
        single ``model.run_batch(X_flat)`` call but GPU-memory-bounded.
    """
    n_rows = X_flat.shape[0]
    return np.concatenate(
        [
            model.run_batch(X_flat[i : i + chunk_size])
            for i in range(0, n_rows, chunk_size)
        ],
        axis=0,
    )


def precision_at_k(
    p_long: np.ndarray, y_long: np.ndarray, k_frac: float = PRECISION_AT_K_FRAC
) -> float:
    """Precision within the top ``k_frac`` of rows ranked by long score.

    ``p_long`` = per-row long score (higher = more long, e.g. the logit margin
    ``preds[:,0] - preds[:,1]``). ``y_long`` = 0/1 truth (1 = truly long). Rows
    with NaN in either array are dropped. Returns ``TP / k`` over the top
    ``k = ceil(k_frac * N_valid)`` rows (stable sort for ties). Degenerate input
    (no valid rows) → ``0.0``.
    """
    p = np.asarray(p_long, dtype=np.float64).ravel()
    y = np.asarray(y_long, dtype=np.float64).ravel()
    valid = ~np.isnan(p) & ~np.isnan(y)
    p = p[valid]
    y = y[valid]
    n = p.shape[0]
    if n == 0:
        return 0.0
    k = int(np.ceil(k_frac * n))
    k = max(1, min(k, n))
    order = np.argsort(-p, kind="stable")
    top = order[:k]
    tp = float((y[top] == 1.0).sum())
    return tp / k


def long_base_rate(y_long: np.ndarray) -> float:
    """Fraction of valid (non-NaN) rows that are truly long. Empty → ``0.0``."""
    y = np.asarray(y_long, dtype=np.float64).ravel()
    valid = ~np.isnan(y)
    if not valid.any():
        return 0.0
    return float((y[valid] == 1.0).mean())


def lift(precision: float, base_rate: float) -> float:
    """``precision / base_rate``; ``0.0`` when ``base_rate <= 0`` (no div-by-zero)."""
    if base_rate <= 0.0:
        return 0.0
    return precision / base_rate


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Terminal artefact of a TrainingLoop run.

    best          — ``tracker.best()`` snapshot: {group_key: incumbent record}.
    study_name    — the tracker's study name (history pointer; not derived here).
    rounds_run    — number of rounds executed.
    trials_run    — number of trials executed across all rounds.
    stopped_by    — why the loop ended: "max_rounds" | "cap" | "strategist_stop".
    """

    best: dict
    study_name: str
    rounds_run: int = 0
    trials_run: int = 0
    stopped_by: str = "max_rounds"


# ---------------------------------------------------------------------------
# Caps reached sentinel (raised internally to unwind the round loop)
# ---------------------------------------------------------------------------


class _CapReached(Exception):
    """Internal control-flow signal: a per-run cap has been hit."""


# ---------------------------------------------------------------------------
# TrainingLoop
# ---------------------------------------------------------------------------


class TrainingLoop:
    """Hybrid Optuna + NNStrategist round loop.

    Parameters
    ----------
    orchestrator:
        Trains one concrete spec per trial (Task 08); exposes ``base_spec`` and
        ``trained_models`` plus ``train(df, data_attributes, spec,
        epoch_callback)`` and ``dataset_dir``.
    tracker:
        Trial store + holdout promotion gate (Task 07). Already constructed AND
        NAMED by ``Trainer._run_train_nn()``; the loop NEVER derives or mutates
        ``study_name``.
    strategist:
        LLM steering (Task 09), or ``None`` → pure-Optuna degrade mode (no
        propose/review; search ``base_spec``'s scope; terminate only on caps).
    search_config:
        Read (not invented): ``max_rounds``, ``trials_per_round``,
        ``sampler`` ("tpe"), ``pruner`` ("median"), ``search_space`` bounds +
        clamps (lr/units/dropout), ``max_wall_clock_s``, ``max_compute``,
        ``seed``.
    """

    def __init__(
        self,
        orchestrator: "NNOrchestrator",
        tracker: "ExperimentTracker",
        strategist: "Optional[NNStrategist]",
        search_config: dict,
    ) -> None:
        self.orchestrator = orchestrator
        self.tracker = tracker
        self.strategist = strategist
        self.search_config = search_config

        self.base_spec: NNModelSpec = orchestrator.base_spec
        # Per-run counters (set up at the start of run()).
        self._trials_run = 0
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, df, data_attributes: "DataAttributes") -> RunResult:
        """Execute the round loop; return ``tracker.best()`` as a RunResult.

        Ordered steps per round (task-10 brief):
          1. propose scope (strategist → clamped proposal; degrade → base scope)
          2. create the per-round Optuna study (direction from tracker mode)
          3. run ``trials_per_round`` trials: build_spec → train → holdout →
             record → promotion gate → caps check (stop at first cap)
          4. review (strategist verb; "stop" breaks; degrade → no review)
          5. return tracker.best() wrapped as RunResult
        """
        # Lazy optuna import — keeps this module importable in the base image.
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._trials_run = 0
        self._start_time = time.monotonic()

        max_rounds = int(self.search_config.get("max_rounds", 1))
        trials_per_round = int(self.search_config.get("trials_per_round", 1))
        seed = self.search_config.get("seed", 0)
        direction = "minimize" if self._tracker_mode() == "min" else "maximize"

        rounds_run = 0
        stopped_by = "max_rounds"

        try:
            for round_idx in range(max_rounds):
                rounds_run = round_idx + 1

                # 1. Propose scope (or degrade to base_spec scope).
                proposal = self._propose(round_idx)

                # 2. Per-round Optuna study (incumbent lives in the tracker).
                sampler = self._make_sampler(seed)
                pruner = optuna.pruners.MedianPruner()
                study = optuna.create_study(
                    direction=direction, sampler=sampler, pruner=pruner
                )

                # 3. Run trials.
                for trial_idx in range(trials_per_round):
                    trial = study.ask()
                    self._run_one_trial(
                        trial, proposal, df, data_attributes, round_idx, seed
                    )
                    # Optuna bookkeeping: tell the study the trial completed so
                    # the TPE sampler can learn from it on the next ask().
                    self._tell(study, trial)
                    # Caps check AFTER each trial — stop at the first cap.
                    self._check_caps()

                # 4. Review (strategist verb; degrade → no review).
                if self._review(round_idx) == "stop":
                    stopped_by = "strategist_stop"
                    break
        except _CapReached:
            stopped_by = "cap"

        # 5. Return.
        best = self.tracker.best()
        return RunResult(
            best=best,
            study_name=self._study_name(),
            rounds_run=rounds_run,
            trials_run=self._trials_run,
            stopped_by=stopped_by,
        )

    # ------------------------------------------------------------------
    # Step 1 — propose scope
    # ------------------------------------------------------------------

    def _propose(self, round_idx: int) -> "Optional[Proposal]":
        """Strategist-proposed (clamped) scope, or None in degrade mode.

        The strategist clamps its own search_space to ``search_config`` limits
        (task 09); we re-clamp defensively here so the loop never lets unbounded
        bounds reach Optuna, per the brief's "bounds clamped before Optuna sees
        them" constraint.
        """
        if self.strategist is None:
            return None
        proposal = self.strategist.propose(self.tracker.summary())
        return self._clamp_proposal(proposal)

    def _clamp_proposal(self, proposal: "Proposal") -> "Proposal":
        """Clamp a proposal's search_space bounds to the search_config limits."""
        cfg_space = self.search_config.get("search_space", {})
        clamped: dict = {}
        for key, bounds in (proposal.search_space or {}).items():
            cfg_bounds = cfg_space.get(key)
            if cfg_bounds is None or bounds is None:
                clamped[key] = bounds
                continue
            cfg_lo, cfg_hi = float(cfg_bounds[0]), float(cfg_bounds[1])
            lo = max(cfg_lo, min(float(bounds[0]), cfg_hi))
            hi = max(cfg_lo, min(float(bounds[1]), cfg_hi))
            if lo > hi:
                lo, hi = hi, lo
            clamped[key] = (lo, hi)
        return dataclasses.replace(proposal, search_space=clamped)

    # ------------------------------------------------------------------
    # Step 3 — one trial
    # ------------------------------------------------------------------

    def _run_one_trial(
        self, trial, proposal, df, data_attributes, round_idx: int, seed
    ) -> None:
        """Build → train → holdout → record → promotion gate for one trial.

        Trial failure is ISOLATED: a raise → recorded ``status="failed"`` (so the
        strategist sees the dead end) and the loop continues. A CUDA OOM is
        retried ONCE on CPU before being marked failed.
        """
        self._trials_run += 1
        spec = self.build_spec(self.base_spec, proposal, trial, seed)

        prune_cb = self._make_prune_callback(trial)

        try:
            metrics = self._train_with_oom_retry(
                spec, df, data_attributes, prune_cb
            )
        except Exception as exc:  # noqa: BLE001 — isolate ANY trial failure
            self._record_failed(spec, round_idx, exc, proposal)
            trial.set_user_attr("status", "failed")
            return

        # Holdout score on the time-ordered holdout split (untouched by training
        # / Optuna pruning), computing the tracker's metric.
        holdout = self.evaluate_on_holdout(spec, df, data_attributes)
        trial.set_user_attr("holdout_score", holdout.get("holdout_score"))

        rationale = proposal.rationale if proposal is not None else None
        trial_id = self.tracker.record(
            spec, metrics, holdout, round_idx, status="ok", rationale=rationale
        )

        # Promotion gate per group: promote only when the tracker says the
        # holdout beats the incumbent; then save weights + best.json together.
        self._maybe_promote(spec, holdout, trial_id)

    def _train_with_oom_retry(self, spec, df, data_attributes, prune_cb) -> dict:
        """Train; on CUDA OOM retry ONCE on CPU before letting it propagate.

        ``promote=False`` so per-trial training saves an epoch checkpoint but
        does NOT force-promote ``_best.pt``. The loop's holdout gate
        (``_maybe_promote``) is the SOLE promoter, so a worse later trial can
        never overwrite the genuinely-best ``_best.pt`` and diverge from
        ``best.json``.
        """
        try:
            return self.orchestrator.train(
                df, data_attributes, spec, epoch_callback=prune_cb, promote=False
            )
        except RuntimeError as exc:
            if not self._is_cuda_oom(exc):
                raise
            log_warning(
                "[TrainingLoop] CUDA OOM on trial; retrying once on CPU "
                f"(spec_hash={spec.spec_hash[:8]})"
            )
            cpu_spec = dataclasses.replace(spec, device="cpu")
            return self.orchestrator.train(
                df, data_attributes, cpu_spec, epoch_callback=prune_cb, promote=False
            )

    def _maybe_promote(self, spec: NNModelSpec, holdout: dict, trial_id: str) -> None:
        """Promote the just-trained model(s) per group when holdout improves.

        The tracker owns the gate; we never re-implement it. On improvement we
        save the checkpoint with ``promote=True`` AND call ``tracker.promote``
        together so saved weights and ``best.json`` never diverge.
        """
        trained = getattr(self.orchestrator, "trained_models", {}) or {}
        for group_key, model in trained.items():
            group_holdout = self._group_holdout(holdout, group_key)
            if not self.tracker.is_improvement(group_key, group_holdout):
                continue
            try:
                cm = CheckpointManager(
                    self.orchestrator.checkpoint_dir,
                    model_name=group_key,
                    mode="max",
                )
                cm.save(
                    model,
                    group_holdout,
                    epoch=spec.epochs,
                    manifest=getattr(model, "manifest", None) or {},
                    promote=True,
                )
            except Exception as exc:  # noqa: BLE001 — promotion is best-effort
                log_warning(
                    f"[TrainingLoop] checkpoint save failed for group "
                    f"{group_key!r}: {exc}"
                )
                continue
            self.tracker.promote(group_key, trial_id)

    @staticmethod
    def _group_holdout(holdout: dict, group_key: str) -> dict:
        """Return the holdout dict for one group.

        Single-group (grouping=single → "all") training scores the whole holdout
        once, so every group shares it. If a future multi-group holdout nests
        per-group dicts under ``holdout["groups"][group_key]`` we prefer that.
        """
        groups = holdout.get("groups") if isinstance(holdout, dict) else None
        if isinstance(groups, dict) and group_key in groups:
            return groups[group_key]
        return holdout

    # ------------------------------------------------------------------
    # build_spec
    # ------------------------------------------------------------------

    def build_spec(
        self,
        base_spec: NNModelSpec,
        proposal: "Optional[Proposal]",
        trial,
        seed,
    ) -> NNModelSpec:
        """Concrete spec for one trial.

        The proposal sets the *structural* scope (indicators / timeframes /
        targets); Optuna suggests the *numeric* knobs (lr, units, dropout)
        within the CLAMPED bounds (depth is a declared architectural choice, not
        searched — ADR-0001); ``spec.seed`` is set for
        reproducibility. In degrade mode (proposal is None) the base_spec scope
        is kept unchanged and only the ``search_config`` bounds apply.
        """
        spec = copy.deepcopy(base_spec)

        # Structural scope from the proposal (degrade → keep base scope).
        if proposal is not None:
            spec.indicators = list(proposal.indicators)
            spec.timeframes = list(proposal.timeframes)
            spec.targets = list(proposal.targets)
            space = proposal.search_space or {}
        else:
            space = self.search_config.get("search_space", {})

        # Resolve the effective numeric bounds (proposal/clamped or config).
        lr_lo, lr_hi = self._bounds(space, "lr", (1e-5, 1e-2))
        units_lo, units_hi = self._bounds(space, "units", (16, 128))
        drop_lo, drop_hi = self._bounds(space, "dropout", (0.0, 0.5))

        # Optuna suggestions within the clamped bounds.
        lr = trial.suggest_float("lr", lr_lo, lr_hi, log=True)
        units = trial.suggest_int("units", int(round(units_lo)), int(round(units_hi)))
        dropout = trial.suggest_float("dropout", drop_lo, drop_hi)

        spec.learning_rate = lr
        spec.dropout = dropout
        # ADR-0001: keep declared architecture (kind/params/count); tune width only.
        spec.layers = [dataclasses.replace(layer, units=int(units)) for layer in spec.layers]
        spec.seed = int(seed) if seed is not None else 0

        return spec

    @staticmethod
    def _bounds(space: dict, key: str, default: tuple) -> tuple:
        """Return (lo, hi) bounds for ``key`` from ``space`` (fallback default)."""
        bounds = space.get(key)
        if bounds is None:
            return default
        return float(bounds[0]), float(bounds[1])

    # ------------------------------------------------------------------
    # Optuna prune callback
    # ------------------------------------------------------------------

    def _make_prune_callback(self, trial):
        """Adapt Optuna pruning to the orchestrator's epoch callback.

        The orchestrator calls ``(group_key, epoch, metrics)``. We report the
        epoch's ``val_accuracy`` (a maximise signal; the loss for min-mode) to
        Optuna at the epoch step, then ``trial.should_prune()`` decides whether
        training should stop early. NNModel honours a truthy return as a stop
        signal, so weak trials are pruned mid-training.

        Pruning reports must be MONOTONIC in step; multiple groups would collide
        on the same step, so we only report for the first group seen each epoch
        (single-group training → exactly one group).
        """
        import optuna

        minimize = self._tracker_mode() == "min"

        def prune_cb(group_key: str, epoch: int, metrics: dict) -> bool:
            # Choose an intermediate value consistent with the study direction.
            if minimize:
                value = metrics.get("val_loss")
            else:
                value = metrics.get("val_accuracy")
            if value is None:
                return False
            try:
                trial.report(float(value), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            except optuna.TrialPruned:
                # Surface pruning to the caller as a stop signal; the loop's
                # _tell() will mark the Optuna trial pruned.
                trial.set_user_attr("pruned", True)
                return True
            return False

        return prune_cb

    def _tell(self, study, trial) -> None:
        """Finalise an asked trial in Optuna based on its user attrs.

        Pruned → TrialState.PRUNED; failed → TrialState.FAIL; otherwise COMPLETE
        with the recorded holdout score so the TPE sampler learns from it.
        """
        import optuna

        if trial.user_attrs.get("pruned"):
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            return
        if trial.user_attrs.get("status") == "failed":
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return
        score = trial.user_attrs.get("holdout_score")
        if score is None:
            # No usable objective value → mark FAIL so Optuna doesn't choke.
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return
        study.tell(trial, float(score))

    # ------------------------------------------------------------------
    # evaluate_on_holdout
    # ------------------------------------------------------------------

    def evaluate_on_holdout(self, spec, df, data_attributes) -> dict:
        """Score the just-trained model(s) on the time-ordered HOLDOUT split.

        The orchestrator built (and cached) the NNDataset for this spec during
        ``train``; rebuilding here is a cache HIT (same content hash) so it is
        cheap. We take the ``holdout`` split — untouched by training and Optuna
        pruning — route its pre-normalised tensors through each trained model,
        and compute the tracker's metric (direction accuracy for the smoke /
        default direction targets).

        Returns ``{holdout_score, score, per_target, n_rows}``.
        """
        dataset = NNDataset.build(
            df, data_attributes, spec, dataset_dir=self.orchestrator.dataset_dir
        )
        holdout = dataset.split("holdout")

        trained = getattr(self.orchestrator, "trained_models", {}) or {}
        scores: list[float] = []
        per_target: dict = {}
        n_rows_total = 0

        for group_key, model in trained.items():
            try:
                group_view = holdout.group(group_key)
            except ValueError:
                group_view = holdout
            X, y = group_view.tensors()
            n_rows = int(len(X))
            if n_rows == 0:
                continue
            n_rows_total += n_rows
            # Flatten (rows, T, F) → (rows, T*F) for run_batch.
            X_flat = np.asarray(X, dtype=np.float32).reshape(n_rows, -1)
            preds = _chunked_predict(model, X_flat, HOLDOUT_EVAL_CHUNK)
            score, target_scores = self._score_predictions(spec, preds, y)
            if score is not None:
                scores.append(score)
            for name, val in target_scores.items():
                per_target[name] = val

        overall = float(np.mean(scores)) if scores else 0.0
        return {
            "holdout_score": overall,
            "score": overall,
            "per_target": per_target,
            "n_rows": n_rows_total,
        }

    @staticmethod
    def _score_predictions(spec, preds: np.ndarray, y: np.ndarray):
        """Per-target holdout score from model outputs vs the holdout targets.

        Direction → argmax accuracy; label → (prob>0.5) accuracy; regression →
        a bounded R-like score ``1/(1+MSE)``. Returns ``(overall_mean,
        {target_name: score})`` over the classification heads (the tracker's
        direction-accuracy metric); regression heads contribute their score too.
        """
        preds = np.asarray(preds, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        per_target: dict = {}
        offset = 0
        head_scores: list[float] = []

        for target in spec.targets:
            for _hk in target.horizons:
                if target.kind in ("direction", "direction_binary"):
                    width = 3 if target.kind == "direction" else 2
                    p = preds[:, offset : offset + width]
                    t = y[:, offset : offset + width]
                    acc = float((p.argmax(axis=1) == t.argmax(axis=1)).mean())
                    per_target[target.name] = acc
                    head_scores.append(acc)
                elif target.kind == "label":
                    width = 1
                    p = preds[:, offset : offset + width]
                    t = y[:, offset : offset + width]
                    acc = float(((p > 0.5).astype(float) == t).mean())
                    per_target[target.name] = acc
                    head_scores.append(acc)
                else:  # regression
                    width = 1
                    p = preds[:, offset : offset + width]
                    t = y[:, offset : offset + width]
                    mse = float(np.mean((p - t) ** 2))
                    r = 1.0 / (1.0 + mse)
                    per_target[target.name] = r
                    head_scores.append(r)
                offset += width

        overall = float(np.mean(head_scores)) if head_scores else None
        return overall, per_target

    # ------------------------------------------------------------------
    # Step 4 — review
    # ------------------------------------------------------------------

    def _review(self, round_idx: int) -> str:
        """Strategist review verb, or "continue" in degrade mode.

        Only ``"stop"`` is acted on here (breaks the loop); ``narrow`` /
        ``broaden`` / ``change_targets`` shape the NEXT round's proposal via the
        strategist's own state, so they fall through to ``continue`` from the
        loop's perspective.
        """
        if self.strategist is None:
            return "continue"
        verb = self.strategist.review(self.tracker.round_summary(round_idx))
        return verb

    # ------------------------------------------------------------------
    # Caps
    # ------------------------------------------------------------------

    def _check_caps(self) -> None:
        """Raise ``_CapReached`` if any per-run cap is hit (first cap wins).

        ``max_rounds`` / ``trials_per_round`` are enforced by the loop ranges;
        here we check the run-wide budgets: ``max_compute`` (total trials) and
        ``max_wall_clock_s`` (elapsed wall-clock).
        """
        max_compute = self.search_config.get("max_compute")
        if max_compute is not None and self._trials_run >= int(max_compute):
            log("[TrainingLoop] max_compute cap reached; stopping loop")
            raise _CapReached()

        max_wall = self.search_config.get("max_wall_clock_s")
        if max_wall is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= float(max_wall):
                log("[TrainingLoop] max_wall_clock_s cap reached; stopping loop")
                raise _CapReached()

    # ------------------------------------------------------------------
    # Failure recording
    # ------------------------------------------------------------------

    def _record_failed(
        self, spec, round_idx: int, exc: Exception, proposal
    ) -> None:
        """Record a failed trial so the strategist sees the dead end."""
        log_warning(
            f"[TrainingLoop] trial failed (round={round_idx}, "
            f"spec_hash={spec.spec_hash[:8]}): {exc}"
        )
        rationale = proposal.rationale if proposal is not None else None
        try:
            self.tracker.record(
                spec, {}, {}, round_idx, status="failed", rationale=rationale
            )
        except Exception as rec_exc:  # noqa: BLE001 — never let recording abort
            log_warning(f"[TrainingLoop] failed to record failed trial: {rec_exc}")

    # ------------------------------------------------------------------
    # Tracker introspection (mode / study_name are OWNED by the tracker)
    # ------------------------------------------------------------------

    def _tracker_mode(self) -> str:
        """Read the tracker's optimisation mode ("max"/"min")."""
        return getattr(self.tracker, "_mode", getattr(self.tracker, "mode", "max"))

    def _study_name(self) -> str:
        """Read the tracker's study name (the loop never derives it)."""
        return getattr(
            self.tracker, "_study_name", getattr(self.tracker, "study_name", "")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_sampler(self, seed: int) -> "optuna.samplers.TPESampler":
        """Build the Optuna TPESampler with n_startup_trials from search_config.

        Reads ``search_config["n_startup_trials"]`` (default 10 — matches the
        optuna library default so an absent key is a no-op). Phase-17 locks this
        to 4 (half of ``trials_per_round: 8``) so TPE switches from random
        sampling to its surrogate model within a single short round.

        Extracted from ``run()`` so the config-driven kwarg is directly testable
        without invoking the heavy training path.
        """
        import optuna  # lazy — keeps module importable in the base image

        n_startup = int(self.search_config.get("n_startup_trials", 10))
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or ("cuda" in msg and "memory" in msg)
