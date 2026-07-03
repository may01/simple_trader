import optuna

from nn.training_loop import TrainingLoop


def _loop(search_config):
    # Construct TrainingLoop without running it; only _make_sampler is exercised.
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.search_config = search_config
    return loop


def test_optuna_param_name_is_real():
    # Guards the kwarg name against an optuna API drift.
    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=4)
    assert sampler._n_startup_trials == 4


def test_make_sampler_reads_config():
    loop = _loop({"n_startup_trials": 4})
    sampler = loop._make_sampler(seed=0)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._n_startup_trials == 4


def test_make_sampler_default_when_absent():
    loop = _loop({})  # no n_startup_trials key
    sampler = loop._make_sampler(seed=0)
    assert sampler._n_startup_trials == 10


def test_make_sampler_captures_kwargs(monkeypatch):
    captured = {}

    class FakeSampler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(optuna.samplers, "TPESampler", FakeSampler)
    loop = _loop({"n_startup_trials": 4})
    loop._make_sampler(seed=7)
    assert captured == {"seed": 7, "n_startup_trials": 4}
