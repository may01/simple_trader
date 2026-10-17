"""Frozen Ridge bound models (StandardScaler + Ridge) predicted with numpy.

The models are fitted once, offline, by ``scripts/cb_export_models.py`` and
stored as ``models_{pair}.json`` next to this module; runtime never needs
scikit-learn or refits.
"""

import functools
import json
import os
from dataclasses import dataclass

import numpy as np

from indicators.candle_bounds.params import FEATURES, SIDES, TFS

_ARRAYS = ("mean", "scale", "coef")


@dataclass(frozen=True)
class BoundModel:
    tf: int
    side: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    band_pct: float
    n_train: int

    def predict_pct(self, X: np.ndarray) -> np.ndarray:
        """Predicted extension in % for rows of ``X`` (shape (n, 15), FEATURES order).

        A row containing NaN predicts NaN (no filling)."""
        X = np.asarray(X, dtype=float)
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept


ModelSet = dict[tuple[int, str], BoundModel]


def models_path(pair: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"models_{pair}.json")


@functools.lru_cache(maxsize=None)
def load_models(pair: str) -> "ModelSet | None":
    """Load the six (tf, side) models for ``pair``; None when no file exists.

    A file that exists but does not match the frozen contract raises ValueError."""
    path = models_path(pair)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if list(payload.get("features", [])) != list(FEATURES):
        raise ValueError(f"{path}: features {payload.get('features')} != {FEATURES}")
    raw = payload.get("models", {})
    out: ModelSet = {}
    for tf in TFS:
        for side in SIDES:
            key = f"{tf}_{side}"
            if key not in raw:
                raise ValueError(f"{path}: missing model {key}")
            m = raw[key]
            arrays = {}
            for name in _ARRAYS:
                arr = np.asarray(m[name], dtype=float)
                if arr.shape != (len(FEATURES),):
                    raise ValueError(
                        f"{path}: {key}.{name} has shape {arr.shape}, expected ({len(FEATURES)},)")
                arrays[name] = arr
            out[(tf, side)] = BoundModel(
                tf=tf, side=side, intercept=float(m["intercept"]),
                band_pct=float(m["band_pct"]), n_train=int(m["n_train"]), **arrays)
    return out
