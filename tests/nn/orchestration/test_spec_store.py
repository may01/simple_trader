from pathlib import Path

from nn.nn_model_spec import LayerSpec, NNModelSpec
from nn.orchestration.spec_store import materialize_spec


def test_materialize_returns_hash_keyed_path_and_writes_file(tmp_path):
    spec = NNModelSpec.default()

    returned = materialize_spec(spec, tmp_path)

    expected = tmp_path / "specs" / spec.spec_hash / "spec.yaml"
    assert returned == expected
    assert returned.exists()
    assert returned.is_file()


def test_materialized_file_reloads_with_same_hash(tmp_path):
    spec = NNModelSpec.default()

    returned = materialize_spec(spec, tmp_path)

    reloaded = NNModelSpec.from_yaml(str(returned))
    assert reloaded.spec_hash == spec.spec_hash


def test_materialize_is_idempotent(tmp_path):
    spec = NNModelSpec.default()

    first = materialize_spec(spec, tmp_path)
    second = materialize_spec(spec, tmp_path)  # must not raise

    assert first == second
    assert second.exists()
    # second write reproduces identical bytes (content-addressed, deterministic dump)
    assert first.read_text() == second.read_text()


def test_distinct_specs_are_isolated_by_hash(tmp_path):
    import dataclasses

    spec_a = NNModelSpec.default()
    spec_b = dataclasses.replace(
        spec_a,
        layers=[LayerSpec("lstm", 128, {"num_layers": 2})],  # different architecture
    )
    assert spec_a.spec_hash != spec_b.spec_hash  # precondition: genuinely different

    path_a = materialize_spec(spec_a, tmp_path)
    path_b = materialize_spec(spec_b, tmp_path)

    assert path_a != path_b
    assert path_a.parent != path_b.parent  # different specs/{hash}/ dirs
    assert path_a.exists() and path_b.exists()

    # each file reloads to its own spec, no cross-contamination
    assert NNModelSpec.from_yaml(str(path_a)).spec_hash == spec_a.spec_hash
    assert NNModelSpec.from_yaml(str(path_b)).spec_hash == spec_b.spec_hash
