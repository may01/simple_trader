import pytest
from simple_trader.infrastructure.config import Config


def test_config_reads_pair_from_env(monkeypatch):
    monkeypatch.setenv("PAIR", "btc_usdt")
    config = Config()
    assert config.pair == "btc_usdt"


def test_config_defaults_pair_to_link_usdt(monkeypatch):
    monkeypatch.delenv("PAIR", raising=False)
    config = Config()
    assert config.pair == "link_usdt"


def test_config_reads_available_threads_as_int(monkeypatch):
    monkeypatch.setenv("AVAILABLE_THREADS", "4")
    config = Config()
    assert config.available_threads == 4


def test_config_use_nn_simulation_true(monkeypatch):
    monkeypatch.setenv("USE_NN_SIMULATION", "True")
    config = Config()
    assert config.use_nn_simulation is True


def test_config_use_nn_simulation_false_by_default(monkeypatch):
    monkeypatch.delenv("USE_NN_SIMULATION", raising=False)
    config = Config()
    assert config.use_nn_simulation is False


def test_config_is_trader_test_when_env_is_one(monkeypatch):
    monkeypatch.setenv("IS_TRAIDER_TEST", "1")
    config = Config()
    assert config.is_trader_test is True


def test_config_data_start_as_int(monkeypatch):
    monkeypatch.setenv("DATA_START", "1700000000000")
    config = Config()
    assert config.data_start == 1700000000000
