# helpers.py — path utilities
# ZERO business logic — pure path construction only.
# All os.environ reads happen at CALL TIME, never at import time.

import os


def root_folder() -> str:
    """Return the root data folder based on ROOT_FOLDER env var.

    "short"  -> /trader_data
    "long"   -> /trader_data_long
    Raises KeyError if ROOT_FOLDER is not set.
    """
    value = os.environ["ROOT_FOLDER"]
    if value == "long":
        return "/trader_data_long"
    return "/trader_data"


def dataset_folder() -> str:
    """Return {root_folder()}/{DATA_ROOT}/{DATA_SET_NAME}_{PAIR}/"""
    data_root = os.environ["DATA_ROOT"]
    data_set_name = os.environ["DATA_SET_NAME"]
    pair = os.environ["PAIR"]
    return f"{root_folder()}/{data_root}/{data_set_name}_{pair}/"


def data_folder() -> str:
    """Return {dataset_folder()}/data/"""
    return f"{dataset_folder()}data/"


def shared_folder() -> str:
    """Return {dataset_folder()}/shared/"""
    return f"{dataset_folder()}shared/"


def stats_folder() -> str:
    """Return stats/{DATA_ROOT}/{PAIR}/  (in-repo, NOT on Docker volume)"""
    data_root = os.environ["DATA_ROOT"]
    pair = os.environ["PAIR"]
    return f"stats/{data_root}/{pair}/"


def nn_folder() -> str:
    """Return {shared_folder()}/nn_data/"""
    return f"{shared_folder()}nn_data/"


def nn_weights_folder() -> str:
    """Return /trader_data_long/{DATA_ROOT}/{PAIR}/nn_weights/

    ALWAYS on simple_trader_vol_long — hardcoded, NOT affected by ROOT_FOLDER.
    """
    data_root = os.environ["DATA_ROOT"]
    pair = os.environ["PAIR"]
    return f"/trader_data_long/{data_root}/{pair}/nn_weights/"


def action_folder() -> str:
    """Return {shared_folder()}/actions/"""
    return f"{shared_folder()}actions/"


def graber_data_path() -> str:
    """Return {dataset_folder()}/graber_data.pkl"""
    return f"{dataset_folder()}graber_data.pkl"


def wide_df_path() -> str:
    """Return {dataset_folder()}/df_with_indicators.pkl"""
    return f"{dataset_folder()}df_with_indicators.pkl"


def data_attributes_path() -> str:
    """Return {dataset_folder()}/data_attributes.pkl"""
    return f"{dataset_folder()}data_attributes.pkl"


def position_json_path() -> str:
    """Return manual_setup/{PAIR}/position.json"""
    pair = os.environ["PAIR"]
    return f"manual_setup/{pair}/position.json"
