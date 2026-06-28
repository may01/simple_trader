import yaml

from nn.device import nn_spec_path
from nn.nn_model_spec import NNModelSpec


def test_nn_spec_path_default_when_unset(monkeypatch):
    monkeypatch.delenv("NN_SPEC_PATH", raising=False)
    assert nn_spec_path() == "configs/nn_spec.yaml"


def test_nn_spec_path_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("NN_SPEC_PATH", "/trader_data_long/x/spec.yaml")
    assert nn_spec_path() == "/trader_data_long/x/spec.yaml"
