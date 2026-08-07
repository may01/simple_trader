"""Separation metrics for one (dataset, grid-cell): how profit labels
distribute over slope classes. numpy/pandas only — no scipy/sklearn.

Flip test is direction-agnostic (spec §6.4): dominance = rate_long − rate_short
per class; "flip" = dominance sign change between the strong (|id| = center−1)
and extra (|id| = center) class of the same sign; "shrink" = |dominance| falls.
"""
import math

import numpy as np
import pandas as pd

from .config import LABEL_PAIRS


def _is_finite(val):
    """Check if a value is finite (not NaN or inf)."""
    return val is not None and np.isfinite(float(val))


def mutual_information(cls: np.ndarray, y: np.ndarray) -> float:
    """MI(class; binary label) in bits, from the contingency table."""
    ct = pd.crosstab(pd.Series(cls), pd.Series(y)).to_numpy(dtype=float)
    p = ct / ct.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def eta_squared(cls: np.ndarray, y: np.ndarray) -> float:
    """Between-class variance share of a numeric target (0..1)."""
    y = np.asarray(y, dtype=float)
    total = y.var()
    if total == 0:
        return 0.0
    s = pd.Series(y)
    g = s.groupby(pd.Series(cls))
    between = (g.size() * (g.mean() - y.mean()) ** 2).sum() / len(y)
    return float(between / total)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, numpy-only."""
    ra = pd.Series(a).rank().to_numpy().copy()
    rb = pd.Series(b).rank().to_numpy().copy()
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _class_ids(n_classes: int) -> list[int]:
    c = n_classes // 2
    return list(range(-c, c + 1))


def _rates(cls, y, ids) -> tuple[dict, float, int]:
    """Per-class positive rate over non-NaN label rows + base rate + n."""
    ok = ~np.isnan(y)
    cls, y = cls[ok], y[ok]
    rate = {}
    for i in ids:
        sel = cls == i
        rate[str(i)] = float(y[sel].mean()) if sel.any() else None
    base = float(y.mean()) if len(y) else float("nan")
    return rate, base, int(ok.sum())


def evaluate(classes: np.ndarray, labels: pd.DataFrame, n_classes: int) -> dict:
    """Compute per-label and per-pair separation metrics.

    Returns {"n_rows", "population", "labels", "pairs"}, where each
    labels[key] dict carries "n", "base_rate", "rate", "lift", "mi",
    "eta2", and "mi_null_floor" (the analytic small-sample MI bias floor
    for that label's n; None when n == 0).
    """
    ids = _class_ids(n_classes)
    center = n_classes // 2
    population = {str(i): int((classes == i).sum()) for i in ids}

    out_labels = {}
    for key in labels.columns:
        y = labels[key].to_numpy(dtype=float)
        rate, base, n = _rates(classes, y, ids)
        ok = ~np.isnan(y)
        out_labels[key] = {
            "n": n,
            "base_rate": base,
            "rate": rate,
            "lift": {k: (None if r is None or base == 0 else r / base)
                     for k, r in rate.items()},
            "mi": mutual_information(classes[ok], y[ok]),
            "eta2": eta_squared(classes[ok], y[ok]),
            # Analytic plug-in bias floor for MI on a K x 2 contingency table
            # (Kx = n_classes, L = 2 binary label), in bits:
            #   (K-1)(L-1) / (2 N ln 2)
            # This is the expected MI under independence from small-sample
            # bias alone; "mi" above should be read against this floor, not
            # against zero. None when n == 0 (no non-NaN rows for this label).
            "mi_null_floor": ((n_classes - 1) / (2 * n * math.log(2))
                               if n else None),
        }

    out_pairs = {}
    for kind, hn in LABEL_PAIRS:
        rl = out_labels[f"{kind}_n{hn}_long"]["rate"]
        rs = out_labels[f"{kind}_n{hn}_short"]["rate"]
        spread = {str(i): (None if rl[str(i)] is None or rs[str(i)] is None
                           else rl[str(i)] - rs[str(i)]) for i in ids}
        # Spec §6.3 calls for Spearman of spread vs class-index *magnitude*;
        # this uses the *signed* class index instead (deliberate deviation:
        # stronger check on the odd-symmetric ladder — an inversion across
        # the zero class reads as rho = -1 rather than being masked by abs()).
        known = [(i, spread[str(i)]) for i in ids if spread[str(i)] is not None]
        if len(known) >= 3:
            rho = spearman(np.array([i for i, _ in known], dtype=float),
                           np.array([s for _, s in known], dtype=float))
            rho = rho if _is_finite(rho) else None
        else:
            rho = None

        flip = None
        if n_classes == 7:
            flip = {}
            for name, sign in (("neg", -1), ("pos", 1)):
                strong = spread[str(sign * (center - 1))]
                extra = spread[str(sign * center)]
                have = strong is not None and extra is not None
                flip[name] = {
                    "strong": strong,
                    "extra": extra,
                    "sign_flip": bool(np.sign(strong) != np.sign(extra)) if have else None,
                    "shrink": (abs(extra) < abs(strong)) if have else None,
                }
        out_pairs[f"{kind}_n{hn}"] = {
            "spread": spread, "monotonicity_rho": rho, "flip": flip,
        }

    return {
        "n_rows": int(len(classes)),
        "population": population,
        "labels": out_labels,
        "pairs": out_pairs,
    }
