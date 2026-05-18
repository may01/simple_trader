from pathlib import Path
import pytest
from simple_trader.infrastructure.config import Config
from simple_trader.infrastructure.paths import (
    root_folder, data_folder, shared_folder,
    nn_folder, action_folder, stats_folder,
)


def make_config(**kwargs):
    import os
    defaults = {
        "PAIR": "link_usdt",
        "DATA_SET_NAME": "train",
        "ROOT_FOLDER": "local",
        "DATA_ROOT": "trader_data",
    }
    defaults.update(kwargs)
    for k, v in defaults.items():
        os.environ[k] = v
    return Config()


def test_root_folder_local_uses_trader_data_local():
    config = make_config(ROOT_FOLDER="local", DATA_ROOT="mydata",
                         DATA_SET_NAME="train", PAIR="link_usdt")
    assert root_folder(config) == Path("/trader_data_local/mydata/train_link_usdt")


def test_root_folder_remote_uses_trader_data():
    config = make_config(ROOT_FOLDER="remote", DATA_ROOT="mydata",
                         DATA_SET_NAME="train", PAIR="link_usdt")
    assert root_folder(config) == Path("/trader_data/mydata/train_link_usdt")


def test_data_folder_is_root_slash_data():
    config = make_config()
    assert data_folder(config) == root_folder(config) / "data"


def test_shared_folder_is_root_slash_shared():
    config = make_config()
    assert shared_folder(config) == root_folder(config) / "shared"


def test_nn_folder_is_shared_slash_nn_data():
    config = make_config()
    assert nn_folder(config) == shared_folder(config) / "nn_data"


def test_action_folder_is_shared_slash_actions():
    config = make_config()
    assert action_folder(config) == shared_folder(config) / "actions"


def test_stats_folder_is_repo_local():
    config = make_config(DATA_ROOT="trader_data", PAIR="link_usdt")
    assert stats_folder(config) == Path("stats/trader_data/link_usdt")
