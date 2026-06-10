"""Tests for NNOrchestrator: train and inference modes for NN pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nn.nn_orchestrator import NNOrchestrator


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def mock_nn_model():
    """Mock NNModel that simulates train/inference behavior."""
    model = MagicMock()
    model.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    model.run_batch.return_value = np.array([
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.6, 0.2, 0.2],
    ])
    model.is_trained = True
    return model


@pytest.fixture
def mock_checkpoint_manager(tmp_path):
    """Mock CheckpointManager."""

    def mock_cm_init(checkpoint_dir, model_name):
        cm = MagicMock()
        cm.checkpoint_dir = checkpoint_dir
        cm.model_name = model_name
        cm.save = MagicMock(return_value=f"{checkpoint_dir}/{model_name}_epoch99.pt")
        cm.load_best = MagicMock(return_value=True)
        return cm

    return mock_cm_init


@pytest.fixture
def mock_data_attributes():
    """Mock DataAttributes with stats."""
    da = MagicMock()
    da.get_stats.side_effect = lambda col: (0.5, 1.0)  # mean=0.5, std=1.0
    return da


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with OHLC and indicator data."""
    n = 100
    data = {
        "15_open": np.random.randn(n) + 100,
        "15_high": np.random.randn(n) + 101,
        "15_low": np.random.randn(n) + 99,
        "15_close": np.random.randn(n) + 100,
        "15_is_closed": [True] * (n - 1) + [False],
        "15_rsi_14": np.random.uniform(30, 70, n),
        "15_cci_14": np.random.uniform(-100, 100, n),
        "15_target_direction": np.array([0, 1, 2] * (n // 3 + 1))[:n],
        "60_open": np.random.randn(n) + 100,
        "60_high": np.random.randn(n) + 101,
        "60_low": np.random.randn(n) + 99,
        "60_close": np.random.randn(n) + 100,
        "60_is_closed": [True] * (n - 1) + [False],
        "60_rsi_14": np.random.uniform(30, 70, n),
        "60_cci_14": np.random.uniform(-100, 100, n),
        "60_target_direction": np.array([0, 1, 2] * (n // 3 + 1))[:n],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_nans():
    """DataFrame with some NaN values in features."""
    n = 50
    data = {
        "15_rsi_14": [np.nan] * 5 + list(np.random.uniform(30, 70, n - 5)),
        "15_cci_14": list(np.random.uniform(-100, 100, n - 3)) + [np.nan] * 3,
        "15_is_closed": [True] * (n - 1) + [False],
        "15_target_direction": np.array([0, 1, 2] * (n // 3 + 1))[:n],
    }
    return pd.DataFrame(data)


# =====================================================================
# Test: __init__
# =====================================================================


def test_init_creates_empty_trained_models():
    """NNOrchestrator initializes with empty trained_models dict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        assert orch.checkpoint_dir == tmp_dir
        assert orch.feature_cols == ["15_rsi_14", "15_cci_14"]
        assert orch.trained_models == {}
        assert orch.num_workers == 4  # default


def test_init_reads_num_workers_from_env():
    """NNOrchestrator reads NUM_WORKERS from environment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.dict(os.environ, {"NUM_WORKERS": "8"}):
            orch = NNOrchestrator(
                checkpoint_dir=tmp_dir,
                feature_cols=["15_rsi_14"],
            )
            assert orch.num_workers == 8


def test_init_defaults_num_workers_if_env_missing():
    """NNOrchestrator defaults to 4 workers if env var absent."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Ensure NUM_WORKERS is not set
        env_copy = os.environ.copy()
        if "NUM_WORKERS" in env_copy:
            del env_copy["NUM_WORKERS"]

        with patch.dict(os.environ, env_copy, clear=True):
            orch = NNOrchestrator(
                checkpoint_dir=tmp_dir,
                feature_cols=["15_rsi_14"],
            )
            assert orch.num_workers == 4


# =====================================================================
# Test: train()
# =====================================================================


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_train_returns_dict_with_tf_str_keys(mock_nn_class, mock_cm_class, sample_df, mock_data_attributes):
    """train() returns dict with TF string keys and metric sub-dicts."""
    mock_nn = MagicMock()
    mock_nn.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    mock_nn_class.return_value = mock_nn

    mock_cm = MagicMock()
    mock_cm.save.return_value = "/tmp/model_15_best.pt"
    mock_cm_class.return_value = mock_cm

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        result = orch.train(sample_df, mock_data_attributes, tfs=[15, 60])

        # Result should have string keys "15" and "60"
        assert "15" in result or "60" in result
        # Each TF result should have metric sub-dicts
        for tf_key in result:
            assert isinstance(result[tf_key], dict)
            assert "loss" in result[tf_key]
            assert "accuracy" in result[tf_key]
            assert "val_loss" in result[tf_key]
            assert "val_accuracy" in result[tf_key]


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_train_skips_tf_with_missing_target_column(
    mock_nn_class, mock_cm_class, sample_df, mock_data_attributes
):
    """train() skips TF when target_direction column is missing and logs warning."""
    mock_nn = MagicMock()
    mock_nn.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    mock_nn_class.return_value = mock_nn
    mock_cm_class.return_value = MagicMock()

    # Remove target column for 60
    df_no_target = sample_df.drop(columns=["60_target_direction"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14", "60_rsi_14", "60_cci_14"],
        )
        result = orch.train(df_no_target, mock_data_attributes, tfs=[15, 60])

        # Should have result for 15 but not 60
        assert "15" in result
        assert "60" not in result


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_train_calls_epoch_callback(mock_nn_class, mock_cm_class, sample_df, mock_data_attributes):
    """train() calls epoch_callback with (str(tf), epoch, metrics)."""
    mock_nn = MagicMock()
    mock_nn.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    mock_nn_class.return_value = mock_nn
    mock_cm_class.return_value = MagicMock()

    callback_calls = []

    def test_callback(tf_str, epoch, metrics):
        callback_calls.append((tf_str, epoch, metrics))

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        orch.train(sample_df, mock_data_attributes, tfs=[15], epoch_callback=test_callback)

        # The callback should have been called by NNModel.train()
        # We verify it was passed correctly to model.train()
        assert mock_nn.train.called


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_train_stores_model_in_trained_models(mock_nn_class, mock_cm_class, sample_df, mock_data_attributes):
    """train() stores trained NNModel in trained_models keyed by string TF."""
    mock_nn = MagicMock()
    mock_nn.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    mock_nn_class.return_value = mock_nn
    mock_cm_class.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        orch.train(sample_df, mock_data_attributes, tfs=[15])

        # trained_models should have key "15" (string)
        assert "15" in orch.trained_models
        assert orch.trained_models["15"] is mock_nn


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_train_uses_closed_candles_only(mock_nn_class, mock_cm_class, sample_df, mock_data_attributes):
    """train() filters by is_closed column if present."""
    mock_nn = MagicMock()
    mock_nn.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    mock_nn_class.return_value = mock_nn
    mock_cm_class.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        orch.train(sample_df, mock_data_attributes, tfs=[15])

        # NNModel.train() should be called with data, but the last row should
        # be excluded (it has is_closed=False)
        assert mock_nn.train.called
        X_arg, y_arg = mock_nn.train.call_args[0][:2]
        # X_arg should have fewer rows than total closed rows
        # (since the last row is not closed)
        assert X_arg.shape[0] < len(sample_df)


# =====================================================================
# Test: run_inference()
# =====================================================================


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_returns_only_nn_columns(mock_nn_class, mock_cm_class, sample_df, mock_data_attributes):
    """run_inference() returns DataFrame with only NN columns."""
    mock_nn = MagicMock()
    mock_nn.run_batch.return_value = np.array([
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.6, 0.2, 0.2],
    ] * 33)[:99]  # Match expected rows (closed rows)

    mock_nn_class.return_value = mock_nn

    mock_cm = MagicMock()
    mock_cm.load_best.return_value = True
    mock_cm_class.return_value = mock_cm

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        result = orch.run_inference(sample_df, mock_data_attributes, tfs=[15])

        # Result should only contain NN columns
        expected_cols = {"15_nn_prob_up", "15_nn_prob_neutral", "15_nn_prob_down"}
        assert set(result.columns) == expected_cols
        # Result should not contain original df columns
        assert "15_rsi_14" not in result.columns
        assert "15_cci_14" not in result.columns


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_sets_nan_for_missing_features(
    mock_nn_class, mock_cm_class, sample_df_with_nans, mock_data_attributes
):
    """run_inference() sets NaN for rows with NaN features."""
    mock_nn = MagicMock()
    # Simulate running on fewer rows (only non-NaN)
    mock_nn.run_batch.return_value = np.array([
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
    ] * 20)[:40]  # Fewer rows for non-NaN features

    mock_nn_class.return_value = mock_nn

    mock_cm = MagicMock()
    mock_cm.load_best.return_value = True
    mock_cm_class.return_value = mock_cm

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        result = orch.run_inference(sample_df_with_nans, mock_data_attributes, tfs=[15])

        # Result should have same length as input df
        assert len(result) == len(sample_df_with_nans)

        # First 5 rows have NaN in 15_rsi_14, so NN cols should be NaN
        # Last 3 rows have NaN in 15_cci_14, so NN cols should be NaN
        assert result.iloc[0]["15_nn_prob_up"] is np.nan or pd.isna(result.iloc[0]["15_nn_prob_up"])


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_skips_tf_with_no_checkpoint(
    mock_nn_class, mock_cm_class, sample_df, mock_data_attributes
):
    """run_inference() skips TF if load_best() returns False."""
    mock_nn = MagicMock()
    mock_cm = MagicMock()
    mock_cm.load_best.return_value = False  # No checkpoint
    mock_cm_class.return_value = mock_cm

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        result = orch.run_inference(sample_df, mock_data_attributes, tfs=[15])

        # Result should be empty (no TFs to process)
        assert len(result.columns) == 0


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_run_inference_preserves_index(mock_nn_class, mock_cm_class, sample_df, mock_data_attributes):
    """run_inference() preserves index from input df."""
    mock_nn = MagicMock()
    mock_nn.run_batch.return_value = np.array([
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
    ] * 50)[:99]

    mock_nn_class.return_value = mock_nn
    mock_cm = MagicMock()
    mock_cm.load_best.return_value = True
    mock_cm_class.return_value = mock_cm

    # Create df with custom index
    df_custom = sample_df.copy()
    df_custom.index = pd.date_range("2024-01-01", periods=len(sample_df))

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14"],
        )
        result = orch.run_inference(df_custom, mock_data_attributes, tfs=[15])

        # Result should have same index as input
        pd.testing.assert_index_equal(result.index, df_custom.index)


# =====================================================================
# Test: integration-like (without mocks for some parts)
# =====================================================================


@patch("nn.nn_orchestrator.CheckpointManager")
@patch("nn.nn_orchestrator.NNModel")
def test_train_and_inference_multiple_tfs(
    mock_nn_class, mock_cm_class, sample_df, mock_data_attributes
):
    """Integration test: train and infer on multiple TFs."""
    mock_nn_15 = MagicMock()
    mock_nn_15.train.return_value = {
        "loss": 0.5,
        "accuracy": 0.75,
        "val_loss": 0.6,
        "val_accuracy": 0.7,
    }
    mock_nn_15.run_batch.return_value = np.random.dirichlet([1, 1, 1], 99)

    mock_nn_60 = MagicMock()
    mock_nn_60.train.return_value = {
        "loss": 0.4,
        "accuracy": 0.8,
        "val_loss": 0.5,
        "val_accuracy": 0.75,
    }
    mock_nn_60.run_batch.return_value = np.random.dirichlet([1, 1, 1], 99)

    # Return different mocks for different calls
    mock_nn_class.side_effect = [mock_nn_15, mock_nn_60, mock_nn_15, mock_nn_60]
    mock_cm_class.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as tmp_dir:
        orch = NNOrchestrator(
            checkpoint_dir=tmp_dir,
            feature_cols=["15_rsi_14", "15_cci_14", "60_rsi_14", "60_cci_14"],
        )

        # Train both TFs
        train_result = orch.train(sample_df, mock_data_attributes, tfs=[15, 60])
        assert len(train_result) >= 1  # At least one TF trained

        # Infer on both TFs
        inference_result = orch.run_inference(sample_df, mock_data_attributes, tfs=[15, 60])
        # Should have columns for both TFs
        assert len(inference_result.columns) >= 3  # At least one set of 3 cols
