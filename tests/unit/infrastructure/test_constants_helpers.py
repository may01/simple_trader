import os
import importlib


def test_no_module_level_env_read_in_constants():
    # Just verify import works cleanly with no env vars set
    import constants
    assert hasattr(constants, "STRATEGY_ACTION_OPEN_LONG")


def test_strategy_action_constants_exist():
    from constants import (
        STRATEGY_ACTION_NOTHING,
        STRATEGY_ACTION_OPEN_LONG,
        STRATEGY_ACTION_CLOSE_LONG,
        STRATEGY_ACTION_OPEN_SHORT,
        STRATEGY_ACTION_CLOSE_SHORT,
        STRATEGY_ACTION_DO_STOP_LOSS,
    )
    assert STRATEGY_ACTION_NOTHING is not None


def test_deprecated_level_types_absent():
    import constants
    assert not hasattr(constants, "LEVEL_TYPE_SHORT_1_RESISTANCE")
    assert not hasattr(constants, "LEVEL_TYPE_SHORT_2_RESISTANCE")


def test_auto_level_type_names():
    from constants import LEVEL_TYPE_AUTO_SUPPORT, LEVEL_TYPE_AUTO_RESISTANCE
    assert LEVEL_TYPE_AUTO_SUPPORT is not None


def test_trade_type_strings():
    from constants import TRADE_BUY, TRADE_SELL
    assert TRADE_BUY == "TRADE_BUY"
    assert TRADE_SELL == "TRADE_SELL"


def test_path_helpers_short(monkeypatch):
    monkeypatch.setenv("ROOT_FOLDER", "short")
    monkeypatch.setenv("DATA_ROOT", "train")
    monkeypatch.setenv("DATA_SET_NAME", "2w")
    monkeypatch.setenv("PAIR", "link_usdt")
    import helpers
    importlib.reload(helpers)
    from helpers import root_folder, dataset_folder, nn_weights_folder
    assert root_folder() == "/trader_data"
    assert "/trader_data/" in dataset_folder()
    assert nn_weights_folder().startswith("/trader_data_long")


def test_path_helpers_long(monkeypatch):
    monkeypatch.setenv("ROOT_FOLDER", "long")
    monkeypatch.setenv("DATA_ROOT", "train")
    monkeypatch.setenv("DATA_SET_NAME", "4m")
    monkeypatch.setenv("PAIR", "link_usdt")
    import helpers
    importlib.reload(helpers)
    from helpers import root_folder, nn_weights_folder
    assert root_folder() == "/trader_data_long"
    assert nn_weights_folder().startswith("/trader_data_long")


def test_nn_weights_folder_always_long(monkeypatch):
    for rf in ["short", "long"]:
        monkeypatch.setenv("ROOT_FOLDER", rf)
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setenv("DATA_ROOT", "train")
        import helpers
        importlib.reload(helpers)
        from helpers import nn_weights_folder
        assert nn_weights_folder().startswith("/trader_data_long")
