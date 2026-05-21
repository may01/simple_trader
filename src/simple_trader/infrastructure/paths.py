from pathlib import Path
from .config import Config


def root_folder(config: Config) -> Path:
    if config.root_folder == "remote":
        base = Path("/trader_data")
    else:
        base = Path("/trader_data_local")
    return base / config.data_root / f"{config.data_set_name}_{config.pair}"


def data_folder(config: Config) -> Path:
    return root_folder(config) / "data"


def shared_folder(config: Config) -> Path:
    return root_folder(config) / "shared"


def nn_folder(config: Config) -> Path:
    return shared_folder(config) / "nn_data"


def action_folder(config: Config) -> Path:
    return shared_folder(config) / "actions"


def stats_folder(config: Config) -> Path:
    return Path("stats") / config.data_root / config.pair
