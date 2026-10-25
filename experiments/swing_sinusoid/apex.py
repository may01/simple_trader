"""Ground-truth apex labels + forward targets for one series.

Everything here looks into the FUTURE on purpose: it produces label columns
(next apex time / value / sign, forward direction and return). Nothing in this
module may be used as a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def find_apexes(y: np.ndarray, prominence: float, distance: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Alternating highs (+1) and lows (-1) of y.

    `prominence` is in y units (caller passes frac × series std). Consecutive
    same-sign extrema keep only the more extreme one so the result alternates.
    Returns (indices, signs) sorted by index.
    """
    y = np.asarray(y, dtype=float)
    hi, _ = find_peaks(np.nan_to_num(y, nan=-np.inf), prominence=prominence, distance=distance)
    lo, _ = find_peaks(np.nan_to_num(-y, nan=-np.inf), prominence=prominence, distance=distance)
    idx = np.concatenate([hi, lo])
    sgn = np.concatenate([np.ones(len(hi), int), -np.ones(len(lo), int)])
    order = np.argsort(idx)
    idx, sgn = idx[order], sgn[order]
    keep_idx, keep_sgn = [], []
    for i, s in zip(idx, sgn):
        if keep_sgn and keep_sgn[-1] == s:
            prev = keep_idx[-1]
            better = (y[i] > y[prev]) if s > 0 else (y[i] < y[prev])
            if better:
                keep_idx[-1] = i
            continue
        keep_idx.append(int(i))
        keep_sgn.append(int(s))
    return np.asarray(keep_idx, int), np.asarray(keep_sgn, int)


def label_rows(y: np.ndarray, close: np.ndarray, apex_idx: np.ndarray, apex_sgn: np.ndarray,
               horizons=(1, 2, 3, 5)) -> pd.DataFrame:
    """Per-row future labels. Rows after the last apex / near the end get NaN."""
    n = len(y)
    y = np.asarray(y, dtype=float)
    close = np.asarray(close, dtype=float)
    out = {
        "next_apex_dt": np.full(n, np.nan), "next_apex_val": np.full(n, np.nan),
        "next_apex_sign": np.full(n, np.nan),
        "prev_apex_dt": np.full(n, np.nan), "prev_apex_sign": np.full(n, np.nan),
    }
    # next apex strictly after t
    pos = np.searchsorted(apex_idx, np.arange(n), side="right")
    has_next = pos < len(apex_idx)
    nxt = apex_idx[np.minimum(pos, len(apex_idx) - 1)]
    out["next_apex_dt"][has_next] = (nxt - np.arange(n))[has_next]
    out["next_apex_val"][has_next] = y[nxt][has_next]
    out["next_apex_sign"][has_next] = apex_sgn[np.minimum(pos, len(apex_idx) - 1)][has_next]
    # previous apex at or before t (known at t only if t is past it — still a label here)
    ppos = pos - 1
    has_prev = ppos >= 0
    prv = apex_idx[np.maximum(ppos, 0)]
    out["prev_apex_dt"][has_prev] = (np.arange(n) - prv)[has_prev]
    out["prev_apex_sign"][has_prev] = apex_sgn[np.maximum(ppos, 0)][has_prev]
    for k in horizons:
        d = np.full(n, np.nan)
        r = np.full(n, np.nan)
        d[: n - k] = np.sign(y[k:] - y[: n - k])
        r[: n - k] = close[k:] / close[: n - k] - 1.0
        out[f"fwd_dir_{k}"] = d
        out[f"fwd_ret_{k}"] = r
    return pd.DataFrame(out)


def label_series(frame: pd.DataFrame, series: str, prominence_frac: float = 0.5,
                 distance: int = 2, horizons=(1, 2, 3, 5)) -> tuple[pd.DataFrame, dict]:
    """Labels for one series of a closed-row frame; returns (labels, meta)."""
    y = frame[series].to_numpy(dtype=float)
    std = float(np.nanstd(y))
    idx, sgn = find_apexes(y, prominence=prominence_frac * std, distance=distance)
    labels = label_rows(y, frame["close"].to_numpy(dtype=float), idx, sgn, horizons)
    labels.index = frame.index
    half = np.diff(idx)
    meta = {
        "series": series, "n_rows": int(len(y)), "n_apex": int(len(idx)),
        "prominence": prominence_frac * std, "prominence_frac": prominence_frac,
        "std": std, "half_swing_median": float(np.median(half)) if len(half) else float("nan"),
        "half_swing_mean": float(np.mean(half)) if len(half) else float("nan"),
    }
    return labels, meta
