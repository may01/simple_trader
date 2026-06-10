import logging
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

import nn.nn_predictor as _nn_predictor_module
from nn.nn_predictor import NNPredictor
from nn.nn_model import NNModel


@pytest.fixture(autouse=True)
def reset_warned_missing_columns():
    """Reset the module-level warning set before each test to prevent cross-test pollution."""
    _nn_predictor_module._warned_missing_columns.clear()
    yield
    _nn_predictor_module._warned_missing_columns.clear()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_nn_model():
    """Create a mock NNModel that returns dummy probabilities."""
    model = MagicMock(spec=NNModel)
    # Return probabilities for 3 classes
    model.run.return_value = np.array([0.5, 0.3, 0.2])
    return model


@pytest.fixture
def mock_data_attributes():
    """Create a mock DataAttributes with predefined stats."""
    attr = MagicMock()

    # Mock get_stats to return (mean, std) for feature columns
    def get_stats_impl(col):
        stats_map = {
            "15_col1": (100.0, 10.0),
            "15_col2": (50.0, 5.0),
            "60_col1": (100.0, 10.0),
            "60_col2": (50.0, 5.0),
            "15_col_zero_std": (100.0, 0.0),  # std=0 case
        }
        return stats_map.get(col, (0.0, 1.0))

    attr.get_stats = get_stats_impl
    return attr


@pytest.fixture
def mock_data_point():
    """Create a mock DataPoint with test data and mutable DataFrame."""
    dp = MagicMock()

    # Store data for get() calls
    data = {
        ("15_col1", 15, 0): 105.0,
        ("15_col2", 15, 0): 52.0,
        ("60_col1", 60, 0): 110.0,
        ("60_col2", 60, 0): 48.0,
    }

    def get_impl(col, tf, shift=0):
        key = (col, tf, shift)
        return data.get(key)

    dp.get = get_impl

    # Create mutable DataFrames for each TF
    df_15 = pd.DataFrame({
        "15_col1": [100.0, 101.0, 102.0],
        "15_col2": [50.0, 51.0, 52.0],
        "15_nn_prob_up": [np.nan, np.nan, np.nan],
        "15_nn_prob_neutral": [np.nan, np.nan, np.nan],
        "15_nn_prob_down": [np.nan, np.nan, np.nan],
    })

    df_60 = pd.DataFrame({
        "60_col1": [100.0, 105.0, 110.0],
        "60_col2": [50.0, 49.0, 48.0],
        "60_nn_prob_up": [np.nan, np.nan, np.nan],
        "60_nn_prob_neutral": [np.nan, np.nan, np.nan],
        "60_nn_prob_down": [np.nan, np.nan, np.nan],
    })

    dfs = {15: df_15, 60: df_60}

    def get_df_impl(tf):
        return dfs.get(tf)

    dp.get_df = get_df_impl
    return dp, dfs


# ============================================================================
# Tests: compute() writes correct column names to last row
# ============================================================================


def test_compute_writes_prob_columns_to_last_row_15min(
    mock_data_attributes, mock_data_point, mock_nn_model
):
    """Test that compute() writes nn_prob_* columns to last row for 15min TF."""
    dp, dfs = mock_data_point

    models = {"15": mock_nn_model}
    predictor = NNPredictor(models, mock_data_attributes, ["15_col1", "15_col2"])

    predictor.compute(dp, 15)

    df = dfs[15]
    assert df.loc[2, "15_nn_prob_up"] == 0.5
    assert df.loc[2, "15_nn_prob_neutral"] == 0.3
    assert df.loc[2, "15_nn_prob_down"] == 0.2


def test_compute_writes_prob_columns_to_last_row_60min(
    mock_data_attributes, mock_data_point, mock_nn_model
):
    """Test that compute() writes nn_prob_* columns to last row for 60min TF."""
    dp, dfs = mock_data_point

    models = {"60": mock_nn_model}
    predictor = NNPredictor(models, mock_data_attributes, ["60_col1", "60_col2"])

    predictor.compute(dp, 60)

    df = dfs[60]
    assert df.loc[2, "60_nn_prob_up"] == 0.5
    assert df.loc[2, "60_nn_prob_neutral"] == 0.3
    assert df.loc[2, "60_nn_prob_down"] == 0.2


# ============================================================================
# Tests: normalization (mean/std)
# ============================================================================


def test_compute_normalizes_features_correctly(
    mock_data_attributes, mock_data_point, caplog
):
    """Test that compute() normalizes features: (value - mean) / std."""
    caplog.set_level(logging.DEBUG)
    dp, dfs = mock_data_point

    model = MagicMock(spec=NNModel)
    model.run = MagicMock()

    def run_side_effect(features):
        # Verify normalization: (105 - 100) / 10 = 0.5, (52 - 50) / 5 = 0.4
        expected_norm_col1 = (105.0 - 100.0) / 10.0  # 0.5
        expected_norm_col2 = (52.0 - 50.0) / 5.0     # 0.4

        np.testing.assert_array_almost_equal(
            features, [expected_norm_col1, expected_norm_col2]
        )
        return np.array([0.5, 0.3, 0.2])

    model.run.side_effect = run_side_effect

    models = {"15": model}
    predictor = NNPredictor(models, mock_data_attributes, ["15_col1", "15_col2"])

    predictor.compute(dp, 15)

    # Verify run() was called once
    assert model.run.call_count == 1


def test_compute_uses_zero_when_std_is_zero(
    mock_data_attributes, mock_data_point, caplog
):
    """Test that compute() uses raw value when std=0."""
    caplog.set_level(logging.DEBUG)
    dp, dfs = mock_data_point

    # Modify data to include col_zero_std
    dp.get = lambda col, tf, shift=0: {
        ("15_col_zero_std", 15, 0): 102.0,
    }.get((col, tf, shift))

    model = MagicMock(spec=NNModel)
    model.run = MagicMock()

    def run_side_effect(features):
        # When std=0, use raw value: 102.0
        np.testing.assert_array_almost_equal(features, [102.0])
        return np.array([0.5, 0.3, 0.2])

    model.run.side_effect = run_side_effect

    models = {"15": model}
    predictor = NNPredictor(
        models, mock_data_attributes, ["15_col_zero_std"]
    )

    predictor.compute(dp, 15)

    assert model.run.call_count == 1


# ============================================================================
# Tests: NaN when TF not in models
# ============================================================================


def test_compute_writes_nan_when_tf_not_in_models(
    mock_data_attributes, mock_data_point
):
    """Test that compute() writes NaN when str(tf) not in models."""
    dp, dfs = mock_data_point

    # Only model for 60min, not 15min
    models = {"60": MagicMock(spec=NNModel)}
    predictor = NNPredictor(models, mock_data_attributes, ["15_col1", "15_col2"])

    predictor.compute(dp, 15)

    df = dfs[15]
    assert np.isnan(df.loc[2, "15_nn_prob_up"])
    assert np.isnan(df.loc[2, "15_nn_prob_neutral"])
    assert np.isnan(df.loc[2, "15_nn_prob_down"])


# ============================================================================
# Tests: missing feature column uses 0.0 and logs warning
# ============================================================================


def test_compute_uses_zero_for_missing_feature_column(
    mock_data_attributes, mock_data_point, caplog
):
    """Test that compute() uses 0.0 for missing feature column."""
    caplog.set_level(logging.WARNING)
    dp, dfs = mock_data_point

    # data_point.get returns None for non-existent column
    original_get = dp.get
    def get_with_missing(col, tf, shift=0):
        if col == "15_missing":
            return None
        return original_get(col, tf, shift)

    dp.get = get_with_missing

    model = MagicMock(spec=NNModel)
    model.run = MagicMock()

    def run_side_effect(features):
        # First feature is missing (0.0), second is (52 - 50) / 5 = 0.4
        np.testing.assert_array_almost_equal(features, [0.0, 0.4])
        return np.array([0.5, 0.3, 0.2])

    model.run.side_effect = run_side_effect

    models = {"15": model}
    predictor = NNPredictor(models, mock_data_attributes, ["15_missing", "15_col2"])

    predictor.compute(dp, 15)

    assert model.run.call_count == 1
    assert "15_missing" in caplog.text or "missing" in caplog.text.lower()


def test_compute_logs_warning_only_once_per_column_per_session(
    mock_data_attributes, mock_data_point, caplog
):
    """Test that missing column warning is logged only ONCE per session."""
    caplog.set_level(logging.WARNING)
    dp, dfs = mock_data_point

    # data_point.get returns None for missing column
    original_get = dp.get
    def get_with_missing(col, tf, shift=0):
        if col == "15_missing":
            return None
        return original_get(col, tf, shift)

    dp.get = get_with_missing

    model = MagicMock(spec=NNModel)
    model.run.return_value = np.array([0.5, 0.3, 0.2])

    models = {"15": model}
    predictor = NNPredictor(models, mock_data_attributes, ["15_missing", "15_col2"])

    # Call compute twice
    predictor.compute(dp, 15)
    caplog.clear()
    predictor.compute(dp, 15)

    # Should not have warning on second call
    warning_count = len(
        [r for r in caplog.records if r.levelname == "WARNING" and "15_missing" in r.message]
    )
    # The first compute should have logged, the second should not
    assert warning_count == 0


# ============================================================================
# Tests: classmethod load()
# ============================================================================


def test_classmethod_load_skips_tfs_with_no_checkpoint(
    mock_data_attributes, caplog
):
    """Test that load() skips TFs with no checkpoint."""
    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch CheckpointManager to skip 15min, load 60min
        with patch("nn.nn_predictor.CheckpointManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm

            # Track which models' load_best is called
            call_count = [0]

            def load_best_side_effect(model):
                # First call (15min): return False (no checkpoint)
                # Second call (60min): return True (has checkpoint)
                if call_count[0] == 0:
                    call_count[0] += 1
                    return False  # 15min has no checkpoint
                else:
                    # Just mark as trained without actually loading
                    model.is_trained = True
                    return True  # 60min has checkpoint

            mock_cm.load_best.side_effect = load_best_side_effect

            predictor = NNPredictor.load(
                tmpdir,
                feature_cols=["15_col1", "15_col2", "60_col1", "60_col2"],
                data_attributes=mock_data_attributes,
                tfs=[15, 60],
            )

            # Only 60min should be in models
            assert "60" in predictor.models
            assert "15" not in predictor.models
            assert len(predictor.models) == 1
            # Verify warning was logged for 15min
            assert "no best checkpoint for TF 15" in caplog.text


def test_classmethod_load_counts_features_per_tf(mock_data_attributes):
    """Test that load() counts feature_cols starting with '{tf}_'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy checkpoints
        import os
        for tf_str in ["15", "60"]:
            checkpoint = os.path.join(tmpdir, "model_best.pt")
            model = NNModel(input_size=2, hidden_size=8, num_classes=3)
            X = np.random.randn(20, 2).astype("float32")
            y = np.random.randint(0, 3, size=20)
            model.train(X, y, epochs=1)
            model.save_model(checkpoint)

        with patch("nn.nn_predictor.CheckpointManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_best.return_value = True

            predictor = NNPredictor.load(
                tmpdir,
                feature_cols=["15_col1", "15_col2", "60_col1"],
                data_attributes=mock_data_attributes,
                tfs=[15, 60],
            )

            # 15min should have input_size=2 (col1, col2)
            assert predictor.models["15"].input_size == 2
            # 60min should have input_size=1 (col1)
            assert predictor.models["60"].input_size == 1


def test_classmethod_load_returns_predictor_with_feature_cols(
    mock_data_attributes,
):
    """Test that load() returns NNPredictor with feature_cols set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import os
        checkpoint = os.path.join(tmpdir, "model_best.pt")
        model = NNModel(input_size=2, hidden_size=8, num_classes=3)
        X = np.random.randn(20, 2).astype("float32")
        y = np.random.randint(0, 3, size=20)
        model.train(X, y, epochs=1)
        model.save_model(checkpoint)

        with patch("nn.nn_predictor.CheckpointManager") as mock_cm_class:
            mock_cm = MagicMock()
            mock_cm_class.return_value = mock_cm
            mock_cm.load_best.return_value = True

            feature_cols = ["15_col1", "15_col2", "60_col1"]
            predictor = NNPredictor.load(
                tmpdir,
                feature_cols=feature_cols,
                data_attributes=mock_data_attributes,
                tfs=[15, 60],
            )

            assert predictor.feature_cols == feature_cols
            assert predictor.data_attributes is mock_data_attributes


# ============================================================================
# Tests: Integration — multiple TFs
# ============================================================================


def test_compute_handles_multiple_tfs(mock_data_attributes, mock_data_point):
    """Test that compute() works for different TFs independently."""
    dp, dfs = mock_data_point

    model_15 = MagicMock(spec=NNModel)
    model_15.run.return_value = np.array([0.5, 0.3, 0.2])

    model_60 = MagicMock(spec=NNModel)
    model_60.run.return_value = np.array([0.2, 0.3, 0.5])

    models = {"15": model_15, "60": model_60}
    predictor = NNPredictor(
        models,
        mock_data_attributes,
        ["15_col1", "15_col2", "60_col1", "60_col2"],
    )

    # Compute for both TFs
    predictor.compute(dp, 15)
    predictor.compute(dp, 60)

    # Check 15min results
    df_15 = dfs[15]
    assert df_15.loc[2, "15_nn_prob_up"] == 0.5
    assert df_15.loc[2, "15_nn_prob_down"] == 0.2

    # Check 60min results
    df_60 = dfs[60]
    assert df_60.loc[2, "60_nn_prob_up"] == 0.2
    assert df_60.loc[2, "60_nn_prob_down"] == 0.5

    # Each model should be called once
    assert model_15.run.call_count == 1
    assert model_60.run.call_count == 1


# ============================================================================
# Tests: Edge cases
# ============================================================================


def test_compute_with_no_matching_features(mock_data_attributes, mock_data_point):
    """Test compute() when no feature_cols match the TF."""
    dp, dfs = mock_data_point

    model = MagicMock(spec=NNModel)
    model.run = MagicMock()

    def run_side_effect(features):
        # Empty feature array
        assert len(features) == 0
        return np.array([0.5, 0.3, 0.2])

    model.run.side_effect = run_side_effect

    models = {"15": model}
    # Only 60min features, but computing for 15min
    predictor = NNPredictor(models, mock_data_attributes, ["60_col1", "60_col2"])

    predictor.compute(dp, 15)

    # Should still write NaN because model expects input_size=0 (edge case)
    assert model.run.call_count == 1


def test_predictor_attributes(mock_data_attributes):
    """Test that NNPredictor stores attributes correctly."""
    model = MagicMock(spec=NNModel)
    feature_cols = ["15_col1", "60_col1"]
    models = {"15": model}

    predictor = NNPredictor(models, mock_data_attributes, feature_cols)

    assert predictor.models == models
    assert predictor.data_attributes is mock_data_attributes
    assert predictor.feature_cols == feature_cols
