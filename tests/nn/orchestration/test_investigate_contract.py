"""Contract test for the nn-investigate skill's documented output.

Phase 17 Task 09. The skill's contract (README Layer C): any emitted spec.yaml
MUST load via NNModelSpec.from_yaml and be well-formed. We pin that here against
a representative Tier-0b v1 spec fixture — proving the documented output shape is
actually loadable and buildable, not just plausible markdown.
"""

from pathlib import Path

from nn.nn_model_spec import NNModelSpec

FIXTURE = Path(__file__).parent / "fixtures" / "sample_v1_spec.yaml"

BUILDABLE_LAYER_KINDS = {"dense", "lstm", "gru", "conv1d"}
TARGET_KINDS = {"direction", "label", "regression"}


def _load():
    return NNModelSpec.from_yaml(str(FIXTURE))


def test_sample_v1_spec_loads_via_from_yaml():
    spec = _load()  # must not raise
    assert spec is not None
    assert spec.spec_hash  # hashable => fully constructed


def test_layers_are_nonempty_and_buildable():
    spec = _load()
    assert spec.layers, "v1 spec must declare at least one layer"
    for layer in spec.layers:
        assert layer.kind in BUILDABLE_LAYER_KINDS, (
            f"layer kind {layer.kind!r} not buildable by _SpecNet today; "
            f"needs the engine extension recipe first"
        )


def test_targets_are_nonempty_and_well_formed():
    spec = _load()
    assert spec.targets, "v1 spec must declare at least one target"
    for target in spec.targets:
        assert target.kind in TARGET_KINDS, (
            f"target kind {target.kind!r} not in {sorted(TARGET_KINDS)}"
        )


def test_locked_starting_values_are_set():
    spec = _load()
    # recurrent archetype (lstm present) => batch_size 64; others 128. Fixture is cnn_lstm => 64.
    assert spec.batch_size == 64
    assert spec.epochs == 50
    assert spec.early_stopping_patience == 8
