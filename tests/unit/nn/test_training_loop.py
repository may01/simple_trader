"""Tests for TrainingLoop — hybrid Optuna + NNStrategist round loop (task 10).

The loop orchestrates ONLY: propose scope (strategist) → per-round Optuna study
→ for each trial build_spec + orchestrator.train + evaluate_on_holdout + record
+ promotion gate → review → return tracker.best() as a RunResult.

optuna is installed ONLY in the nn-train image, so the whole module is SKIPPED
where optuna is absent (base simple_trader image) via importorskip at the top.

The pure-Optuna smoke exercises the REAL loop control flow against the REAL
ExperimentTracker, a REAL CheckpointManager (via the orchestrator) and a tiny
synthetic dataset; targeted tests use fakes to isolate caps / failure / degrade
/ seeding behaviour.
"""

from __future__ import annotations

import copy
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")  # skip whole file in the base image

from indicators import DataAttributes
from indicators.labels import _fmt
from nn.checkpoint_manager import CheckpointManager
from nn.experiment_tracker import ExperimentTracker
from nn.nn_model import NNModel
from nn.nn_model_spec import LayerSpec, NNModelSpec, TargetSpec
from nn.nn_orchestrator import NNOrchestrator
from nn import training_loop as tl_mod
from nn.training_loop import (
    RunResult,
    TrainingLoop,
    precision_at_k,
    long_base_rate,
    lift,
    GATE_METRIC_PRECISION_AT_K,
)


# =====================================================================
# Fixtures — real, tiny, trainable wide frame + spec
# =====================================================================


def _label_suffix(n: int, m: float, x: float) -> str:
    return f"n{_fmt(n)}_m{_fmt(m)}_x{_fmt(x)}"


def make_wide_df(rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """Small synthetic wide frame with 15-min closed flags, features, labels.

    Mirrors the orchestrator/dataset test fixtures so a REAL NNDataset can be
    built and trained over it.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    df = pd.DataFrame(index=idx)
    pos = np.arange(rows)

    df["15_is_closed"] = (pos + 1) % 15 == 0

    df["15_logret"] = (pos.astype(float) * 0.001) - 0.01
    df["15_rsi_14"] = 50.0 + (pos.astype(float) % 30)
    df["15_close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, rows))

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
        layers=[LayerSpec(kind="dense", units=8)],
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
        epochs=3,
        validation_split=0.2,
        val_strategy="time_holdout",
        seed=0,
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


SEARCH_CONFIG = {
    "max_rounds": 1,
    "trials_per_round": 2,
    "sampler": "tpe",
    "pruner": "median",
    "max_wall_clock_s": 120,
    "max_compute": None,
    "seed": 0,
    "search_space": {
        "lr": [1e-4, 1e-2],
        "depth": [1, 2],
        "units": [8, 16],
        "dropout": [0.0, 0.2],
    },
}


@pytest.fixture
def data_attributes():
    return DataAttributes()


@pytest.fixture
def base_spec():
    return small_spec()


@pytest.fixture
def real_setup(tmp_path, base_spec):
    """Build a real orchestrator + tracker over isolated temp dirs."""
    ckpt = str(tmp_path / "ckpt")
    dsdir = str(tmp_path / "ds")
    track = str(tmp_path / "track")
    orch = NNOrchestrator(checkpoint_dir=ckpt, dataset_dir=dsdir, base_spec=base_spec)
    tracker = ExperimentTracker(
        track, "study_smoke", metric="holdout_score", mode="max"
    )
    return orch, tracker, track, ckpt


# =====================================================================
# Pure-Optuna smoke — exercises the REAL loop control flow
# =====================================================================


def test_pure_optuna_smoke_returns_runresult_and_populates_tracker(
    real_setup, data_attributes
):
    """strategist=None, 1 round, 2 trials over a tiny real dataset.

    Asserts run() returns a RunResult, tracker.best('all') is recorded, and the
    tracking dir is populated (index.sqlite + trials/).
    """
    orch, tracker, track, _ = real_setup

    loop = TrainingLoop(
        orchestrator=orch,
        tracker=tracker,
        strategist=None,
        search_config=SEARCH_CONFIG,
    )

    df = make_wide_df()
    result = loop.run(df, data_attributes)

    assert isinstance(result, RunResult)
    assert result is not None

    best = tracker.best("all")
    assert best is not None, "no incumbent recorded for group 'all'"

    # tracking dir populated
    assert os.path.exists(os.path.join(track, "study_smoke", "index.sqlite"))
    assert os.path.isdir(os.path.join(track, "study_smoke", "trials"))

    # Exactly trials_per_round trials recorded, all in round 0.
    summary = tracker.summary()
    assert summary["total_trials"] == 2
    assert summary["rounds"] == [0]


def test_pure_optuna_records_holdout_scores(real_setup, data_attributes):
    """Each ok trial carries a holdout score (direction accuracy in [0,1])."""
    orch, tracker, _, _ = real_setup
    loop = TrainingLoop(orch, tracker, None, SEARCH_CONFIG)

    loop.run(make_wide_df(), data_attributes)

    best = tracker.best("all")
    holdout = best["holdout"]
    score = holdout.get("holdout_score", holdout.get("score"))
    assert score is not None
    assert 0.0 <= float(score) <= 1.0
    assert holdout.get("n_rows", 0) > 0


# =====================================================================
# Caps stop the loop
# =====================================================================


def _fake_orch_tracker(monkeypatch_trained=True):
    """A fake orchestrator + tracker that count trials without real training."""
    orch = MagicMock()
    orch.base_spec = small_spec()
    orch.train.return_value = {"all": {"val_accuracy": 0.5, "loss": 0.2}}
    orch.trained_models = {"all": MagicMock()}

    tracker = MagicMock()
    tracker._mode = "max"
    tracker.mode = "max"
    tracker.summary.return_value = {}
    tracker.round_summary.return_value = {"round": 0}
    tracker.is_improvement.return_value = False
    tracker.best.return_value = {"all": {"holdout": {"holdout_score": 0.5}}}
    tracker.record.return_value = "trial-id"
    return orch, tracker


def test_trials_per_round_cap_stops_loop():
    """trials_per_round bounds the number of trials in a round."""
    orch, tracker = _fake_orch_tracker()

    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 1
    cfg["trials_per_round"] = 3

    loop = TrainingLoop(orch, tracker, None, cfg)
    # Stub holdout eval to avoid needing a real dataset.
    loop.evaluate_on_holdout = MagicMock(
        return_value={"holdout_score": 0.5, "score": 0.5, "n_rows": 10, "per_target": {}}
    )

    loop.run(MagicMock(), {})

    assert orch.train.call_count == 3
    assert tracker.record.call_count == 3


def test_max_rounds_cap_stops_loop():
    """max_rounds bounds the number of rounds (pure-Optuna → no review)."""
    orch, tracker = _fake_orch_tracker()
    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 2
    cfg["trials_per_round"] = 1

    loop = TrainingLoop(orch, tracker, None, cfg)
    loop.evaluate_on_holdout = MagicMock(
        return_value={"holdout_score": 0.5, "score": 0.5, "n_rows": 10, "per_target": {}}
    )
    loop.run(MagicMock(), {})

    # 2 rounds x 1 trial each.
    assert orch.train.call_count == 2


def test_max_compute_cap_stops_loop_after_first_trial():
    """A compute cap of 1 trial stops the loop at the first trial."""
    orch, tracker = _fake_orch_tracker()
    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 5
    cfg["trials_per_round"] = 5
    cfg["max_compute"] = 1  # one trial total

    loop = TrainingLoop(orch, tracker, None, cfg)
    loop.evaluate_on_holdout = MagicMock(
        return_value={"holdout_score": 0.5, "score": 0.5, "n_rows": 10, "per_target": {}}
    )
    loop.run(MagicMock(), {})

    assert orch.train.call_count == 1


# =====================================================================
# Trial failure is isolated → status="failed", loop continues
# =====================================================================


def test_trial_failure_recorded_failed_and_loop_continues():
    """A trial whose training raises is recorded status='failed'; loop continues."""
    orch, tracker = _fake_orch_tracker()

    # First train() raises, second succeeds.
    orch.train.side_effect = [
        RuntimeError("bad spec"),
        {"all": {"val_accuracy": 0.6, "loss": 0.1}},
    ]

    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 1
    cfg["trials_per_round"] = 2

    loop = TrainingLoop(orch, tracker, None, cfg)
    loop.evaluate_on_holdout = MagicMock(
        return_value={"holdout_score": 0.6, "score": 0.6, "n_rows": 10, "per_target": {}}
    )
    loop.run(MagicMock(), {})

    # Both trials recorded; the first as failed.
    assert tracker.record.call_count == 2
    statuses = [c.kwargs.get("status", "ok") for c in tracker.record.call_args_list]
    assert "failed" in statuses
    assert "ok" in statuses


def test_gpu_oom_retries_once_on_cpu_before_failing():
    """A CUDA-OOM trial is retried once on CPU before being marked failed."""
    orch, tracker = _fake_orch_tracker()

    oom = RuntimeError("CUDA out of memory")
    # First call OOMs, the CPU retry succeeds.
    orch.train.side_effect = [oom, {"all": {"val_accuracy": 0.55, "loss": 0.2}}]

    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 1
    cfg["trials_per_round"] = 1

    loop = TrainingLoop(orch, tracker, None, cfg)
    loop.evaluate_on_holdout = MagicMock(
        return_value={"holdout_score": 0.55, "score": 0.55, "n_rows": 10, "per_target": {}}
    )
    loop.run(MagicMock(), {})

    # train called twice (OOM + CPU retry), recorded ok (retry succeeded).
    assert orch.train.call_count == 2
    # The retry spec was forced onto CPU.
    retry_spec = orch.train.call_args_list[1].args[2] if orch.train.call_args_list[1].args else orch.train.call_args_list[1].kwargs["spec"]
    assert retry_spec.device == "cpu"
    statuses = [c.kwargs.get("status", "ok") for c in tracker.record.call_args_list]
    assert statuses == ["ok"]


# =====================================================================
# Pure-Optuna degrade → no strategist calls
# =====================================================================


def test_pure_optuna_degrade_makes_no_strategist_calls(real_setup, data_attributes):
    """With strategist=None the loop never calls propose/review (no strategist)."""
    orch, tracker, _, _ = real_setup
    # A strategist that, if called, would explode the test.
    strategist = MagicMock()
    # Pass None — the degrade path. The MagicMock is here only as a tripwire if
    # the loop ever tried to call a strategist it was not given.
    loop = TrainingLoop(orch, tracker, strategist=None, search_config=SEARCH_CONFIG)
    loop.run(make_wide_df(), data_attributes)

    strategist.propose.assert_not_called()
    strategist.review.assert_not_called()


# =====================================================================
# Strategist drives propose/review when present
# =====================================================================


def test_strategist_present_calls_propose_and_review_and_stop_breaks():
    """A strategist returning 'stop' on review breaks the loop after one round."""
    from nn.nn_strategist import Proposal

    orch, tracker = _fake_orch_tracker()

    strategist = MagicMock()
    strategist.propose.return_value = Proposal(
        indicators=["logret", "rsi_14"],
        timeframes=[15],
        targets=orch.base_spec.targets,
        search_space={"lr": (1e-4, 1e-2), "depth": (1, 2)},
        rationale="explore",
    )
    strategist.review.return_value = "stop"

    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 5
    cfg["trials_per_round"] = 1

    loop = TrainingLoop(orch, tracker, strategist, cfg)
    loop.evaluate_on_holdout = MagicMock(
        return_value={"holdout_score": 0.5, "score": 0.5, "n_rows": 10, "per_target": {}}
    )
    loop.run(MagicMock(), {})

    # propose called for round 0, review returned stop → only one round ran.
    assert strategist.propose.call_count == 1
    assert strategist.review.call_count == 1
    assert orch.train.call_count == 1  # one round, one trial


# =====================================================================
# Seeding reproducibility — same seed → same suggested params
# =====================================================================


def test_same_seed_yields_same_suggested_params():
    """Two runs with the same seed suggest identical Optuna params."""
    captured_a: list[NNModelSpec] = []
    captured_b: list[NNModelSpec] = []

    def make_loop(sink):
        orch, tracker = _fake_orch_tracker()

        def _record_spec(df, da, spec, epoch_callback=None, promote=True):
            sink.append(spec)
            return {"all": {"val_accuracy": 0.5, "loss": 0.2}}

        orch.train.side_effect = _record_spec
        cfg = dict(SEARCH_CONFIG)
        cfg["max_rounds"] = 1
        cfg["trials_per_round"] = 3
        cfg["seed"] = 1234
        loop = TrainingLoop(orch, tracker, None, cfg)
        loop.evaluate_on_holdout = MagicMock(
            return_value={"holdout_score": 0.5, "score": 0.5, "n_rows": 10, "per_target": {}}
        )
        return loop

    make_loop(captured_a).run(MagicMock(), {})
    make_loop(captured_b).run(MagicMock(), {})

    assert len(captured_a) == len(captured_b) == 3
    lr_a = [s.learning_rate for s in captured_a]
    lr_b = [s.learning_rate for s in captured_b]
    assert lr_a == lr_b
    depth_a = [len(s.layers) for s in captured_a]
    depth_b = [len(s.layers) for s in captured_b]
    assert depth_a == depth_b
    # spec.seed is set deterministically too.
    assert all(s.seed == captured_b[0].seed for s in captured_a)


# =====================================================================
# direction "minimize" mode flips the Optuna study direction
# =====================================================================


def test_min_mode_creates_minimize_study(real_setup, data_attributes, tmp_path):
    """A tracker in mode='min' produces a minimise study (no crash, runs)."""
    orch, _, _, _ = real_setup
    tracker_min = ExperimentTracker(
        str(tmp_path / "track_min"), "study_min", metric="holdout_score", mode="min"
    )
    loop = TrainingLoop(orch, tracker_min, None, SEARCH_CONFIG)
    result = loop.run(make_wide_df(), data_attributes)
    assert isinstance(result, RunResult)


# =====================================================================
# Promotion gate — saved _best.pt weights must NEVER diverge from best.json
# =====================================================================


def _flat_state(state_dict) -> np.ndarray:
    """Flatten a model state_dict into one 1-D float array for comparison."""
    parts = [np.asarray(v.detach().cpu().numpy()).ravel() for v in state_dict.values()]
    return np.concatenate(parts) if parts else np.array([])


def test_worse_last_trial_does_not_overwrite_best_pt(real_setup, data_attributes):
    """The promotion gate must gate the SAVED weights, not just best.json.

    Runs two trials where the LAST trial scores WORSE on holdout than the first.
    The incumbent is therefore trial 0. After the run we RELOAD ``all_best.pt``
    from disk and assert its weights + metric match the INCUMBENT (trial 0), NOT
    the last trial.

    FAILS on the old force-promote code: ``orchestrator.train(promote=True)``
    overwrites ``all_best.pt`` with the LAST trial's weights every call, so disk
    diverges from ``best.json`` (which the holdout gate keeps on trial 0).
    PASSES after the fix: the loop trains with ``promote=False`` and the holdout
    gate (``_maybe_promote``) is the sole promoter.
    """
    orch, tracker, _, ckpt = real_setup

    # Capture each trial's trained-model weights right after training, before the
    # next trial overwrites ``trained_models['all']``.
    trial_states: list[np.ndarray] = []
    real_train = orch.train

    def capturing_train(*args, **kwargs):
        result = real_train(*args, **kwargs)
        model = orch.trained_models["all"]
        trial_states.append(_flat_state(model.model.state_dict()))
        return result

    orch.train = capturing_train

    cfg = dict(SEARCH_CONFIG)
    cfg["max_rounds"] = 1
    cfg["trials_per_round"] = 2

    loop = TrainingLoop(orch, tracker, None, cfg)

    # Force a STRICTLY DECREASING holdout sequence: trial 0 best, trial 1 worse.
    holdout_scores = iter([0.90, 0.10])

    def fake_holdout(spec, df, da):
        s = next(holdout_scores)
        return {"holdout_score": s, "score": s, "n_rows": 10, "per_target": {}}

    loop.evaluate_on_holdout = fake_holdout

    loop.run(make_wide_df(), data_attributes)

    # Sanity: two distinct trials were trained with distinguishable weights
    # (different architecture and/or different parameter values).
    assert len(trial_states) == 2
    distinguishable = (
        trial_states[0].shape != trial_states[1].shape
        or not np.array_equal(trial_states[0], trial_states[1])
    )
    assert distinguishable, "trials produced identical weights; cannot distinguish incumbent"

    # best.json points at the incumbent = trial 0 (holdout 0.90).
    incumbent = tracker.best("all")
    assert incumbent is not None
    inc_holdout = incumbent["holdout"]
    inc_score = inc_holdout.get("holdout_score", inc_holdout.get("score"))
    assert inc_score == pytest.approx(0.90)

    # RELOAD all_best.pt from disk and compare against the captured weights.
    reloaded = NNModel(loop.base_spec)
    cm = CheckpointManager(ckpt, model_name="all", mode="max")
    loaded = cm.load_best(reloaded)
    assert loaded is not None, "no all_best.pt on disk"
    best_state = _flat_state(reloaded.model.state_dict())

    # The promoted weights on disk must equal the INCUMBENT (trial 0), not the
    # worse LAST trial — and best.json must agree with the saved weights.
    assert best_state.shape == trial_states[0].shape, (
        "all_best.pt holds the LAST trial's architecture, not the incumbent's"
    )
    np.testing.assert_array_equal(
        best_state,
        trial_states[0],
        err_msg="all_best.pt diverged from best.json (holds the worse last trial)",
    )

    # And the saved bundle's metric is the incumbent's holdout, not the last's.
    best_metric = reloaded.metrics.get("holdout_score") if hasattr(reloaded, "metrics") else None
    if best_metric is None:
        # Fall back to reading the bundle directly.
        import torch

        bundle = torch.load(
            os.path.join(ckpt, "all_best.pt"), map_location="cpu", weights_only=False
        )
        best_metric = bundle["metrics"].get("holdout_score")
    assert best_metric == pytest.approx(0.90)


class TestScorePredictionsDirectionBinary:
    def test_mixed_direction_and_binary_offset_and_accuracy(self):
        spec = NNModelSpec(
            name="m",
            timeframes=[15],
            indicators=["15_close"],
            layers=[LayerSpec(kind="dense", units=8)],
            targets=[
                TargetSpec(name="dir15", kind="direction",
                           label_tf=15, label_m=1.0, label_x=0.3),
                TargetSpec(name="long15", kind="direction_binary", side="long",
                           label_tf=15, label_m=1.0, label_x=0.3),
            ],
        )
        # columns: [dir up, neutral, down | long prob_long, prob_other]
        y = np.array([
            [1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1],
        ], dtype=float)
        preds = np.array([
            [0.7, 0.2, 0.1, 0.9, 0.1],
            [0.1, 0.2, 0.7, 0.2, 0.8],
        ], dtype=float)
        overall, per_target = TrainingLoop._score_predictions(spec, preds, y)
        assert per_target["dir15"] == 1.0
        assert per_target["long15"] == 1.0   # FAILS before the fix (~0.976, MSE-scored on 1 col)
        assert overall == 1.0


class TestPrecisionAtK:
    def test_perfect_ranking_top_all_long(self):
        p = np.array([0.9, 0.8, 0.1, 0.2])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        # k = ceil(0.5*4) = 2; top-2 by score are both long → 1.0
        assert precision_at_k(p, y, k_frac=0.5) == 1.0

    def test_all_negative_zero(self):
        p = np.array([0.9, 0.8, 0.1, 0.2])
        y = np.zeros(4)
        assert precision_at_k(p, y, k_frac=0.5) == 0.0

    def test_k_ceil_rounding(self):
        p = np.arange(10, 0, -1).astype(float)  # 10..1 descending
        y = np.zeros(10); y[0] = 1.0
        # k = ceil(0.05*10) = 1 → top-1 is the single long → 1.0
        assert precision_at_k(p, y, k_frac=0.05) == 1.0

    def test_k_ge_n_uses_all_rows(self):
        p = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 0.0, 1.0])
        assert precision_at_k(p, y, k_frac=1.0) == pytest.approx(2 / 3)

    def test_empty_returns_zero(self):
        assert precision_at_k(np.array([]), np.array([])) == 0.0

    def test_ties_stable_sort(self):
        p = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.array([1.0, 0.0, 0.0, 0.0])
        # all tied; k=1; stable sort keeps index 0 first → the long → 1.0
        assert precision_at_k(p, y, k_frac=0.05) == 1.0

    def test_nan_labels_dropped(self):
        p = np.array([0.9, 0.8, 0.1])
        y = np.array([1.0, np.nan, 0.0])
        # valid rows {0:long, 2:other}; k=ceil(0.5*2)=1 → top-1 idx0 long → 1.0
        assert precision_at_k(p, y, k_frac=0.5) == 1.0


class TestLongBaseRate:
    def test_fraction_long_ignoring_nan(self):
        y = np.array([1.0, 0.0, np.nan, 1.0])
        assert long_base_rate(y) == pytest.approx(2 / 3)

    def test_all_nan_returns_zero(self):
        assert long_base_rate(np.array([np.nan, np.nan])) == 0.0


class TestLift:
    def test_normal(self):
        assert lift(0.5, 0.1) == pytest.approx(5.0)

    def test_zero_base_rate_returns_zero(self):
        assert lift(0.5, 0.0) == 0.0


def _binary_spec():
    return NNModelSpec(
        name="t",
        targets=[
            TargetSpec(name="long15", kind="direction_binary", horizons=[15], side="long")
        ],
    )


class TestScorePredictionsPrecisionGate:
    # 4 rows: cols = [prob_long, prob_other]; y col0 = long truth.
    Y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    PREDS = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]], dtype=float
    )  # long-margin ranking: row0 > row1 > row2 > row3

    def test_precision_gate_scores_top_k(self):
        overall, per_target = TrainingLoop._score_predictions(
            _binary_spec(), self.PREDS, self.Y, GATE_METRIC_PRECISION_AT_K
        )
        # k = ceil(0.05*4) = 1; top-margin row0 has y_long=1 → precision@5% = 1.0
        assert per_target["long15"] == 1.0
        assert per_target["long15__p@5"] == 1.0
        assert per_target["long15__lift@5"] == pytest.approx(1.0 / 0.5)  # base_rate 0.5
        assert "long15__p@1" in per_target and "long15__p@10" in per_target
        assert overall == 1.0

    def test_accuracy_gate_is_default_and_unchanged(self):
        overall, per_target = TrainingLoop._score_predictions(
            _binary_spec(), self.PREDS, self.Y
        )
        # argmax: row0 pred0==y0 ✓, row1 pred0 vs y1 ✗, row2 pred1 vs y0 ✗, row3 pred1==y1 ✓
        assert per_target["long15"] == 0.5
        assert overall == 0.5
        assert "long15__p@5" not in per_target  # no report extras in accuracy mode


class _FakeGroupView:
    def __init__(self, X, y):
        self._X, self._y = X, y

    def tensors(self):
        return self._X, self._y


class _FakeSplit:
    def __init__(self, X, y):
        self._gv = _FakeGroupView(X, y)

    def group(self, key):
        return self._gv


class _FakeDataset:
    def __init__(self, X, y):
        self._split = _FakeSplit(X, y)

    def split(self, name):
        return self._split


class _FakeModel:
    def __init__(self, preds):
        self._preds = preds

    def run_batch(self, X):
        return self._preds[: len(X)]


class _FakeOrch:
    dataset_dir = "/tmp"

    def __init__(self, preds):
        self.trained_models = {"long": _FakeModel(preds)}


class TestEvaluateForwardsGateMetric:
    X = np.zeros((4, 1, 2), dtype=np.float32)  # (rows, T, F); F must equal head width 2
    Y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float64)
    PREDS = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]], dtype=np.float64)

    def _loop(self, gate_metric, monkeypatch):
        monkeypatch.setattr(
            tl_mod.NNDataset, "build",
            staticmethod(lambda *a, **k: _FakeDataset(self.X, self.Y)),
        )
        tl = TrainingLoop.__new__(TrainingLoop)
        tl.orchestrator = _FakeOrch(self.PREDS)
        tl.search_config = {"gate_metric": gate_metric}
        return tl

    def test_precision_gate_flips_holdout_score(self, monkeypatch):
        tl = self._loop(GATE_METRIC_PRECISION_AT_K, monkeypatch)
        out = tl.evaluate_on_holdout(_binary_spec(), df=None, data_attributes=None)
        assert out["holdout_score"] == 1.0            # precision@5% of top-margin row
        assert out["per_target"]["long15__p@5"] == 1.0

    def test_accuracy_gate_holdout_score(self, monkeypatch):
        tl = self._loop("accuracy", monkeypatch)
        out = tl.evaluate_on_holdout(_binary_spec(), df=None, data_attributes=None)
        # argmax match rate: row0 ✓, row1 ✗, row2 ✗, row3 ✓ = 2/4 = 0.5
        assert out["holdout_score"] == 0.5
