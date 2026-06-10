"""Unit tests for dataset folder initialization."""

import os
import pytest
from unittest.mock import patch


def test_all_folders_created(tmp_path, monkeypatch):
    """Test that all required folders are created."""
    monkeypatch.setenv("ROOT_FOLDER", "short")
    monkeypatch.setenv("DATA_ROOT", "train")
    monkeypatch.setenv("DATA_SET_NAME", "2w")
    monkeypatch.setenv("PAIR", "link_usdt")
    base = str(tmp_path)
    dataset = f"{base}/train/2w_link_usdt"

    with patch("helpers.root_folder", return_value=base), \
         patch("helpers.nn_weights_folder", return_value=f"{base}_long/train/link_usdt/nn_weights"), \
         patch("helpers.stats_folder", return_value=f"{base}/stats/train/link_usdt"):
        from grabers.init_folders import init_dataset_folders
        init_dataset_folders()

    assert os.path.isdir(f"{dataset}/data")
    assert os.path.isdir(f"{dataset}/shared")
    assert os.path.isdir(f"{dataset}/shared/nn_data")
    assert os.path.isdir(f"{dataset}/shared/actions")
    assert os.path.isdir(f"{base}_long/train/link_usdt/nn_weights")
    assert os.path.isdir(f"{base}/stats/train/link_usdt")


def test_idempotent(tmp_path, monkeypatch):
    """Test that calling init_dataset_folders multiple times is safe."""
    monkeypatch.setenv("ROOT_FOLDER", "short")
    monkeypatch.setenv("DATA_ROOT", "train")
    monkeypatch.setenv("DATA_SET_NAME", "2w")
    monkeypatch.setenv("PAIR", "link_usdt")
    base = str(tmp_path)

    with patch("helpers.root_folder", return_value=base), \
         patch("helpers.nn_weights_folder", return_value=f"{base}_long/train/link_usdt/nn_weights"), \
         patch("helpers.stats_folder", return_value=f"{base}/stats/train/link_usdt"):
        from grabers.init_folders import init_dataset_folders
        init_dataset_folders()
        init_dataset_folders()  # must not raise


def test_missing_root_folder_raises(monkeypatch):
    """Test that missing ROOT_FOLDER env var raises KeyError."""
    monkeypatch.delenv("ROOT_FOLDER", raising=False)
    monkeypatch.setenv("DATA_ROOT", "train")
    monkeypatch.setenv("DATA_SET_NAME", "2w")
    monkeypatch.setenv("PAIR", "link_usdt")

    from grabers.init_folders import init_dataset_folders
    with pytest.raises(KeyError):
        init_dataset_folders()
