import os
from dataclasses import dataclass, field


@dataclass
class Config:
    pair: str = field(
        default_factory=lambda: os.environ.get("PAIR", "link_usdt")
    )
    data_set_name: str = field(
        default_factory=lambda: os.environ.get("DATA_SET_NAME", "default")
    )
    root_folder: str = field(
        default_factory=lambda: os.environ.get("ROOT_FOLDER", "local")
    )
    data_root: str = field(
        default_factory=lambda: os.environ.get("DATA_ROOT", "trader_data")
    )
    data_start: int = field(
        default_factory=lambda: int(os.environ.get("DATA_START", "0"))
    )
    data_end: int = field(
        default_factory=lambda: int(os.environ.get("DATA_END", "9999999999999"))
    )
    trainer_time_step: int = field(
        default_factory=lambda: int(os.environ.get("TRAINER_TIME_STEP", "1"))
    )
    available_threads: int = field(
        default_factory=lambda: int(os.environ.get("AVAILABLE_THREADS", "1"))
    )
    use_nn_simulation: bool = field(
        default_factory=lambda: os.environ.get("USE_NN_SIMULATION", "False") == "True"
    )
    is_trader_test: bool = field(
        default_factory=lambda: os.environ.get("IS_TRAIDER_TEST", "0") == "1"
    )
    binance_api_key: str = field(
        default_factory=lambda: os.environ.get("BINANCE_API_KEY", "")
    )
    binance_api_secret: str = field(
        default_factory=lambda: os.environ.get("BINANCE_API_SECRET", "")
    )
