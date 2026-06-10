"""NNOrchestrator: Coordinates train and inference modes for the NN pipeline.

The NN pipeline has two modes:
1. Train mode: called by Trainer._run_train_nn(), trains one NNModel per TF
2. Inference mode: called by Trainer._run_simulate_nn(), loads best checkpoints
   and runs batch inference, returns NN-columns-only DataFrame
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from indicators import DataAttributes
from logs import log_warning
from nn.checkpoint_manager import CheckpointManager
from nn.nn_model import NNModel


class NNOrchestrator:
    """Orchestrates training and inference for NN models across multiple timeframes.

    Attributes:
        checkpoint_dir: Directory where model checkpoints are stored.
        feature_cols: List of feature column names available for training.
        num_workers: Number of worker threads (from NUM_WORKERS env var, default 4).
        trained_models: Dict keyed by TF string ("15", "60") of trained NNModel instances.
    """

    def __init__(self, checkpoint_dir: str, feature_cols: list[str]) -> None:
        """Initialize NNOrchestrator.

        Args:
            checkpoint_dir: Directory for saving/loading model checkpoints.
            feature_cols: List of all feature column names.
        """
        self.checkpoint_dir = checkpoint_dir
        self.feature_cols = feature_cols
        self.num_workers = int(os.environ.get("NUM_WORKERS", "4"))
        self.trained_models: dict[str, NNModel] = {}

    # ------------------------------------------------------------------
    # Train Mode
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        data_attributes: DataAttributes,
        tfs: list[int],
        epoch_callback: Optional[Callable[[str, int, dict], None]] = None,
    ) -> dict:
        """Train NN models for each timeframe.

        For each TF:
        1. Determine feature columns for this TF (those starting with f"{tf}_")
        2. Extract feature matrix X and target labels y
        3. Filter to closed-candle rows (using is_closed column if present)
        4. Build and train NNModel
        5. Save via CheckpointManager
        6. Store in trained_models dict

        Args:
            df: Wide DataFrame containing features and targets.
            data_attributes: DataAttributes instance for stats.
            tfs: List of timeframes (integers) to train.
            epoch_callback: Optional callback(tf_str, epoch, metrics) for training progress.

        Returns:
            Dict mapping TF string (e.g., "15") to metrics dict with keys:
            loss, accuracy, val_loss, val_accuracy
        """
        result = {}

        for tf in tfs:
            tf_str = str(tf)

            # Check if target column exists
            target_col = f"{tf}_target_direction"
            if target_col not in df.columns:
                log_warning(f"Target column '{target_col}' not found in df. Skipping TF {tf}.")
                continue

            # Determine feature columns for this TF
            feature_cols_for_tf = [col for col in self.feature_cols if col.startswith(f"{tf}_")]
            if not feature_cols_for_tf:
                log_warning(f"No feature columns found for TF {tf}. Skipping.")
                continue

            # Start with all rows
            df_work = df.copy()

            # Filter to closed-candle rows if is_closed column exists
            is_closed_col = f"{tf}_is_closed"
            if is_closed_col in df_work.columns:
                df_work = df_work[df_work[is_closed_col].astype(bool)]

            # Extract features: only rows where all feature cols are non-NaN
            df_work = df_work[feature_cols_for_tf + [target_col]].dropna()

            if len(df_work) == 0:
                log_warning(f"No valid data for TF {tf} after filtering NaNs. Skipping.")
                continue

            # Extract X and y
            X = df_work[feature_cols_for_tf].values.astype("float32")
            y = df_work[target_col].values.astype("int64")

            # Build and train model
            model = NNModel(input_size=len(feature_cols_for_tf))

            # Create epoch callback wrapper
            def epoch_cb_wrapper(epoch: int, metrics: dict) -> None:
                if epoch_callback is not None:
                    epoch_callback(tf_str, epoch, metrics)

            metrics = model.train(X, y, epoch_callback=epoch_cb_wrapper)

            # Save via CheckpointManager
            cm = CheckpointManager(self.checkpoint_dir, model_name=f"model_{tf}")
            # Use a reasonable default epoch count (model.train returns metrics from last epoch)
            epochs = 100  # Default, could be parameterized
            cm.save(model, metrics, epoch=epochs)

            # Store model
            self.trained_models[tf_str] = model

            # Add to result
            result[tf_str] = metrics

        return result

    # ------------------------------------------------------------------
    # Inference Mode
    # ------------------------------------------------------------------

    def run_inference(
        self,
        df: pd.DataFrame,
        data_attributes: DataAttributes,
        tfs: list[int],
    ) -> pd.DataFrame:
        """Run inference on trained models and return NN output columns.

        For each TF:
        1. Load best checkpoint
        2. Determine feature columns for this TF
        3. Normalize features using data_attributes stats
        4. Run batch inference where all features are non-NaN
        5. Set NaN for rows with NaN features
        6. Create output columns: {tf}_nn_prob_up, {tf}_nn_prob_neutral, {tf}_nn_prob_down

        Args:
            df: Wide DataFrame with features (same structure as training).
            data_attributes: DataAttributes instance with feature stats.
            tfs: List of timeframes to run inference for.

        Returns:
            DataFrame containing ONLY the NN output columns (no original df columns).
            Indexed by df.index. All TF columns combined.
        """
        result_data = {}

        for tf in tfs:
            tf_str = str(tf)

            # Load best checkpoint
            model = NNModel(input_size=1)  # Placeholder, will be set during load
            cm = CheckpointManager(self.checkpoint_dir, model_name=f"model_{tf}")
            if not cm.load_best(model):
                # No checkpoint available for this TF
                continue

            # Determine feature columns for this TF
            feature_cols_for_tf = [col for col in self.feature_cols if col.startswith(f"{tf}_")]
            if not feature_cols_for_tf:
                continue

            # For each row, check if all features are non-NaN
            # If yes, normalize and run inference
            # If no, set NaN
            probs_up = []
            probs_neutral = []
            probs_down = []

            for idx in range(len(df)):
                row = df.iloc[idx]
                feature_values = row[feature_cols_for_tf]

                # Check if any feature is NaN
                if feature_values.isna().any():
                    probs_up.append(np.nan)
                    probs_neutral.append(np.nan)
                    probs_down.append(np.nan)
                else:
                    # Normalize features
                    normalized = []
                    for col in feature_cols_for_tf:
                        value = row[col]
                        mean, std = data_attributes.get_stats(col)
                        if std == 0:
                            # std=0 → use raw value
                            normalized.append(value)
                        else:
                            normalized.append((value - mean) / std)

                    # Run inference on this single row
                    X_single = np.array(normalized, dtype="float32")
                    probs = model.run(X_single)  # Returns (3,) array
                    probs_up.append(probs[0])
                    probs_neutral.append(probs[1])
                    probs_down.append(probs[2])

            # Create output columns
            result_data[f"{tf}_nn_prob_up"] = probs_up
            result_data[f"{tf}_nn_prob_neutral"] = probs_neutral
            result_data[f"{tf}_nn_prob_down"] = probs_down

        # Create result DataFrame with same index as input
        result_df = pd.DataFrame(result_data, index=df.index)
        return result_df
