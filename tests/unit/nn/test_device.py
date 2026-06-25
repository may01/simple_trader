"""Unit tests for nn/device.py — resolve_device() and nn_artefact_root().

All tests mock torch.cuda.is_available and env vars so no GPU is required.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nn.device import nn_artefact_root, resolve_device


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


class TestResolveDeviceCpu:
    """pref="cpu" must return CPU without ever probing CUDA."""

    def test_returns_cpu_device(self):
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_does_not_probe_cuda(self):
        with patch("torch.cuda.is_available") as mock_cuda:
            resolve_device("cpu")
        mock_cuda.assert_not_called()


class TestResolveDeviceAuto:
    """pref="auto" (default) falls back to CPU silently when CUDA absent."""

    def test_auto_returns_cpu_when_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = resolve_device("auto")
        assert device == torch.device("cpu")

    def test_auto_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("auto")
        assert device == torch.device("cuda")

    def test_default_arg_is_auto(self):
        """resolve_device() with no args behaves like pref='auto'."""
        with patch("torch.cuda.is_available", return_value=False):
            device = resolve_device()
        assert device == torch.device("cpu")


class TestResolveDeviceCudaRequested:
    """pref="cuda" warns and falls back to CPU when CUDA is unavailable."""

    def test_cuda_unavailable_returns_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = resolve_device("cuda")
        assert device == torch.device("cpu")

    def test_cuda_unavailable_logs_warning(self, caplog):
        import logging

        with patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.WARNING, logger="nn.device"):
                resolve_device("cuda")
        assert any(
            "cuda requested but unavailable" in record.message.lower()
            for record in caplog.records
        ), f"Expected warning not found in: {[r.message for r in caplog.records]}"

    def test_cuda_available_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("cuda")
        assert device == torch.device("cuda")


# ---------------------------------------------------------------------------
# nn_artefact_root
# ---------------------------------------------------------------------------


class TestNnArtefactRoot:
    """nn_artefact_root(pair) returns {NN_ARTEFACT_ROOT}/{DATA_ROOT}/{pair}/nn"""

    def test_basic_path_construction(self):
        env = {
            "NN_ARTEFACT_ROOT": "/trader_data_long",
            "DATA_ROOT": "train_2024",
        }
        with patch.dict(os.environ, env, clear=False):
            result = nn_artefact_root("BTCUSDT")
        assert result == Path("/trader_data_long/train_2024/BTCUSDT/nn")

    def test_returns_path_object(self):
        env = {
            "NN_ARTEFACT_ROOT": "/artefacts",
            "DATA_ROOT": "data",
        }
        with patch.dict(os.environ, env, clear=False):
            result = nn_artefact_root("ETHUSDT")
        assert isinstance(result, Path)

    def test_different_pair(self):
        env = {
            "NN_ARTEFACT_ROOT": "/mnt/storage",
            "DATA_ROOT": "backtest_set",
        }
        with patch.dict(os.environ, env, clear=False):
            result = nn_artefact_root("SOLUSDT")
        assert result == Path("/mnt/storage/backtest_set/SOLUSDT/nn")

    def test_missing_nn_artefact_root_raises(self):
        env = {"DATA_ROOT": "train_2024"}
        # Ensure NN_ARTEFACT_ROOT is absent
        clean_env = {k: v for k, v in os.environ.items() if k != "NN_ARTEFACT_ROOT"}
        clean_env["DATA_ROOT"] = "train_2024"
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(KeyError):
                nn_artefact_root("BTCUSDT")

    def test_missing_data_root_raises(self):
        clean_env = {k: v for k, v in os.environ.items() if k != "DATA_ROOT"}
        clean_env["NN_ARTEFACT_ROOT"] = "/trader_data_long"
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(KeyError):
                nn_artefact_root("BTCUSDT")
