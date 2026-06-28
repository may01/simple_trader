"""Tests for TrainingLoop.build_spec — ADR-0001: architecture declared, not searched.

build_spec must keep the layer KIND, PARAMS, and COUNT declared in base_spec,
and let Optuna tune width (units), learning_rate and dropout only — no depth.

These build real Optuna Trial objects (suggest_* must actually run), so this
module runs in the nn-train image:
    docker compose run --rm nn-train python -m pytest tests/nn/test_build_spec_layers.py -q
"""

import dataclasses

import optuna

from nn.nn_model_spec import LayerSpec, NNModelSpec
from nn.training_loop import TrainingLoop


class _FakeOrchestrator:
    """Minimal stub — build_spec only needs orchestrator.base_spec at init time."""
    base_spec = NNModelSpec.default()


def _loop():
    # Empty search_config → build_spec uses its default bounds.
    # TrainingLoop.__init__ requires (orchestrator, tracker, strategist, search_config).
    # In degrade mode (proposal=None) only orchestrator.base_spec is accessed (at init).
    return TrainingLoop(
        orchestrator=_FakeOrchestrator(),
        tracker=None,
        strategist=None,
        search_config={},
    )


def _ask():
    study = optuna.create_study()
    return study, study.ask()


def test_lstm_base_preserves_kind_and_params():
    base = dataclasses.replace(
        NNModelSpec.default(),
        layers=[LayerSpec(kind="lstm", units=32, params={"num_layers": 2})],
    )
    study, trial = _ask()
    spec = _loop().build_spec(base, proposal=None, trial=trial, seed=7)
    study.tell(trial, 0.0)

    assert len(spec.layers) == 1
    layer = spec.layers[0]
    assert layer.kind == "lstm"
    assert layer.params == {"num_layers": 2}
    # width is whatever Optuna suggested, and it must equal trial's "units"
    assert layer.units == trial.params["units"]
    # architecture untouched in the base object (deepcopy, not mutation)
    assert base.layers[0].units == 32
