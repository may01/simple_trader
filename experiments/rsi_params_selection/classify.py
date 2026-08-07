"""Cut fitting + class assignment for rsi_maN_diff values.

Techniques (spec §4):
  quantile — percentile cuts of the fit distribution.
  sym0     — symmetric around ZERO: 0 ± k·std (neutral = flat slope).
  zscore   — symmetric around the MEAN: mean ± k·std (legacy _five_tiers family).

Class ids are centered ints: 5-class → −2..2, 7-class → −3..3; 0 = neutral,
negative = falling RSI ma, positive = rising.
"""
import numpy as np

from .config import QUANTILE_CUTS, SYM0_K, ZSCORE_K


def fit_cuts(x: np.ndarray, technique: str, n_classes: int) -> np.ndarray:
    """Ascending cut array (len n_classes−1) fit on NaN-free values."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("fit_cuts: NaN in fit values — drop them first")
    if technique == "quantile":
        return np.percentile(x, QUANTILE_CUTS[n_classes])
    if technique == "sym0":
        k = np.asarray(SYM0_K[n_classes], dtype=float)
        s = float(x.std())
        return np.concatenate([-k[::-1] * s, k * s])
    if technique == "zscore":
        k = np.asarray(ZSCORE_K[n_classes], dtype=float)
        mu, s = float(x.mean()), float(x.std())
        return np.concatenate([mu - k[::-1] * s, mu + k * s])
    raise ValueError(f"unknown technique {technique!r}")


def apply_cuts(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Centered class ids via np.digitize. NaN input is a caller bug."""
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        raise ValueError("apply_cuts: NaN in values — drop them first")
    n_classes = len(cuts) + 1
    return np.digitize(x, cuts) - n_classes // 2
