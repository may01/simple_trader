import yaml

from nn.device import nn_spec_path
from nn.nn_model_spec import NNModelSpec


def test_nn_spec_path_default_when_unset(monkeypatch):
    monkeypatch.delenv("NN_SPEC_PATH", raising=False)
    assert nn_spec_path() == "configs/nn_spec.yaml"


def test_nn_spec_path_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("NN_SPEC_PATH", "/trader_data_long/x/spec.yaml")
    assert nn_spec_path() == "/trader_data_long/x/spec.yaml"


def test_from_yaml_via_nn_spec_path_loads_env_spec(monkeypatch, tmp_path):
    import dataclasses

    spec = NNModelSpec.default()
    spec.name = "phase17_override_spec"
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(dataclasses.asdict(spec)))

    monkeypatch.setenv("NN_SPEC_PATH", str(spec_file))
    loaded = NNModelSpec.from_yaml(nn_spec_path())
    assert loaded.name == "phase17_override_spec"
