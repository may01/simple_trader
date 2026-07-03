from pathlib import Path
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "nn_search.yaml"

def _load():
    with CONFIG.open() as fh:
        return yaml.safe_load(fh)

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
