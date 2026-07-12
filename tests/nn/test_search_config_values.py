from pathlib import Path
import yaml
import pytest
from training.trainer import Trainer

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "nn_search.yaml"

def _load():
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)

def _loader(tmp_path, yaml_text):
    (tmp_path / "nn_search.yaml").write_text(yaml_text)
    t = Trainer.__new__(Trainer)  # bypass full init; loader only uses self.config_path
    t.config_path = str(tmp_path)
    return t._load_nn_search_config

def test_locked_scalar_values():
    cfg = _load()
    assert cfg["max_rounds"] == 1
    assert cfg["trials_per_round"] == 8
    assert cfg["n_startup_trials"] == 4
    assert cfg["max_wall_clock_s"] == 3600
    assert cfg["margin"] == 0.01

def test_kept_scalar_values_unchanged():
    cfg = _load()
    assert cfg["sampler"] == "tpe"
    assert cfg["pruner"] == "median"
    assert cfg["max_compute"] is None
    assert cfg["seed"] == 0

def test_depth_removed_from_search_space():
    space = _load()["search_space"]
    assert "depth" not in space          # ADR-0001: architecture not searched

def test_searched_params_present():
    space = _load()["search_space"]
    assert "lr" in space
    assert "units" in space
    assert "dropout" in space
    assert space["units"] == [16, 64]
    assert space["dropout"] == [0.0, 0.3]

def test_gate_metric_defaults_to_accuracy_when_absent(tmp_path):
    cfg = _loader(tmp_path, "margin: 0.01\n")()
    assert cfg["gate_metric"] == "accuracy"

def test_gate_metric_from_yaml(tmp_path):
    cfg = _loader(tmp_path, "gate_metric: precision_at_k\n")()
    assert cfg["gate_metric"] == "precision_at_k"

def test_gate_metric_env_overrides_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("NN_GATE_METRIC", "precision_at_k")
    cfg = _loader(tmp_path, "gate_metric: accuracy\n")()
    assert cfg["gate_metric"] == "precision_at_k"

def test_gate_metric_invalid_raises(tmp_path):
    with pytest.raises(ValueError, match="gate_metric"):
        _loader(tmp_path, "gate_metric: bogus\n")()
