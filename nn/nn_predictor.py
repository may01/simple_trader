"""NNPredictor — per-tick neural network inference for live trading.

NNPredictor is NOT used during simulation. It's a live-path-only indicator that
writes NN probability predictions directly to indicator DataFrames.

Stateless between ticks — no feature caching. Does NOT gate signals; strategies
read nn_prob_* columns as regular indicator columns.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from nn.checkpoint_manager import CheckpointManager
from nn.nn_model import NNModel

logger = logging.getLogger(__name__)

# Track which columns have already been warned about as missing (once per session)
_warned_missing_columns: set[str] = set()


class DataPoint(Protocol):
    """Protocol: an object providing OHLC data access."""

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        """Get a single data point value.

        Args:
            col: Column name (e.g., "15_rsi_14").
            tf: Timeframe in minutes.
            shift: Shift back in candles (default 0).

        Returns:
            Float value or None if not found.
        """
        ...

    def get_df(self, tf: int):
        """Get mutable DataFrame for the given timeframe.

        Args:
            tf: Timeframe in minutes.

        Returns:
            pd.DataFrame with mutable write access.
        """
        ...


class NNPredictor:
    """Per-tick NN inference for live trading.

    Extracts feature vectors from data_point, normalizes them, and writes
    probability predictions to the DataFrame's last row.
    """

    def __init__(
        self,
        models: dict[str, NNModel],
        data_attributes,
        feature_cols: list[str],
    ) -> None:
        """Initialize NNPredictor.

        Args:
            models: dict[str, NNModel] keyed by TF string ("15", "60", etc.)
            data_attributes: DataAttributes instance with get_stats(col) method.
            feature_cols: Ordered list of feature column names (e.g., "15_col1", "60_col1").
        """
        self.models = models
        self.data_attributes = data_attributes
        self.feature_cols = feature_cols

    def compute(self, data_point: DataPoint, tf: int) -> None:
        """Extract features, normalize, run model, write probabilities to last row.

        Args:
            data_point: DataPoint providing get() and get_df() access.
            tf: Timeframe in minutes (e.g., 15, 60).

        Side effects:
            Writes three columns to data_point.get_df(tf) at the last row:
            - {tf}_nn_prob_up
            - {tf}_nn_prob_neutral
            - {tf}_nn_prob_down
        """
        global _warned_missing_columns

        tf_str = str(tf)
        df = data_point.get_df(tf)

        # If TF not in models, write NaN for all columns
        if tf_str not in self.models:
            last_idx = len(df) - 1
            df.loc[last_idx, f"{tf}_nn_prob_up"] = np.nan
            df.loc[last_idx, f"{tf}_nn_prob_neutral"] = np.nan
            df.loc[last_idx, f"{tf}_nn_prob_down"] = np.nan
            return

        # Extract feature vector for this TF
        feature_vector = []
        for col in self.feature_cols:
            if not col.startswith(f"{tf}_"):
                continue

            # Get value from data_point
            value = data_point.get(col, tf, 0)

            # Handle missing feature: use 0.0 and warn once
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if col not in _warned_missing_columns:
                    _warned_missing_columns.add(col)
                    logger.warning(
                        f"NNPredictor: missing feature column '{col}' in data_point; "
                        "using 0.0"
                    )
                value = 0.0

            # Normalize: (value - mean) / std
            try:
                mean, std = self.data_attributes.get_stats(col)
            except (KeyError, AttributeError):
                # If stats not available, use raw value
                logger.warning(
                    f"NNPredictor: no stats for column '{col}'; using raw value"
                )
                feature_vector.append(float(value))
                continue

            if std == 0.0:
                # std=0: use raw value
                feature_vector.append(float(value))
            else:
                # Normal normalization
                normalized = (value - mean) / std
                feature_vector.append(normalized)

        # Run model
        features_array = np.array(feature_vector, dtype=np.float32)
        probs = self.models[tf_str].run(features_array)

        # Write probabilities to last row
        last_idx = len(df) - 1
        df.loc[last_idx, f"{tf}_nn_prob_up"] = probs[0]
        df.loc[last_idx, f"{tf}_nn_prob_neutral"] = probs[1]
        df.loc[last_idx, f"{tf}_nn_prob_down"] = probs[2]

    @classmethod
    def load(
        cls,
        checkpoint_dir: str,
        feature_cols: list[str],
        data_attributes,
        tfs: list[int],
    ) -> NNPredictor:
        """Load models from checkpoints, one per TF.

        For each TF:
          1. Count feature_cols starting with "{tf}_" → input_size
          2. Instantiate NNModel(input_size=input_size)
          3. Call CheckpointManager(checkpoint_dir).load_best(model)
          4. If load_best returns False, log warning and skip TF

        Args:
            checkpoint_dir: Directory containing checkpoint files.
            feature_cols: Ordered list of feature column names.
            data_attributes: DataAttributes instance.
            tfs: List of timeframes to load models for.

        Returns:
            NNPredictor with only successfully-loaded models.
        """
        models = {}

        for tf in tfs:
            tf_str = str(tf)

            # Count features for this TF
            input_size = sum(
                1 for col in feature_cols
                if col.startswith(f"{tf}_")
            )

            if input_size == 0:
                logger.warning(
                    f"NNPredictor.load: no features for TF {tf}; skipping"
                )
                continue

            # Create model
            model = NNModel(input_size=input_size)

            # Try to load checkpoint
            cm = CheckpointManager(checkpoint_dir)
            if not cm.load_best(model):
                logger.warning(
                    f"NNPredictor.load: no best checkpoint for TF {tf}; skipping"
                )
                continue

            models[tf_str] = model

        return cls(models, data_attributes, feature_cols)
