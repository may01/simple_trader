"""tdlib/screen.py -- Task 4: univariate feature screening (Layer 4, part 1).

Before spending model-fitting effort on the full feature set a
``feature_matrix`` call produces, this module ranks EVERY candidate column
on its own, one-feature-at-a-time separation of long (``y == 1``) from
short (``y == 0``) -- a cheap, model-free first pass an experimenter can
skim to sanity-check the feature set and pick a manageable top-K to chart.

Two functions:

1. ``univariate_screen(X, y)`` -- one row per ``X`` column: an AUC (the
   Mann-Whitney-U-derived probability that a random long-side value beats a
   random short-side value), its Hanley-McNeil 95% confidence interval, a
   KS statistic, and the two group means. Pairwise-complete throughout: a
   column's own NaN rows are dropped before it is scored, independently of
   every other column's own NaN pattern.

2. ``screen_charts(scr, X, y, out_dir, top_k)`` -- for the strongest
   ``top_k`` features (by ``univariate_screen``'s own ranking), saves a
   class-conditional histogram and a long-rate-by-decile bar chart as PNGs.

A non-interactive Matplotlib backend (``Agg``) is forced at import time --
matching ``azlib.models``'s own precedent (zone-selection-experiment
worktree) -- so chart code never touches a display.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp, mannwhitneyu  # noqa: E402

# A feature with fewer than this many non-NaN rows on EITHER side cannot
# support a trustworthy AUC/CI estimate -- task-4-brief.md's exact floor.
_MIN_GROUP_N = 10

_RESULT_COLUMNS = [
    "n", "n_long", "n_short", "auc", "abs_auc_dev", "auc_se",
    "auc_low95", "auc_high95", "ks_stat", "mean_long", "mean_short", "note",
]


def _mannwhitney_auc(x1: np.ndarray, x0: np.ndarray) -> tuple[float, float]:
    """(U1, auc) for x1 (the ``y == 1`` / long group) vs x0 (the ``y == 0``
    / short group).

    ``U1`` is ``scipy.stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic``
    -- EMPIRICALLY CONFIRMED against scipy 1.18 (task-4-report.md) that
    scipy reports the statistic for the FIRST sample passed in: U1 counts
    ``(x1_i, x0_j)`` pairs where ``x1_i > x0_j`` (plus half a count per
    exact tie). ``x1`` must therefore be passed first, ``x0`` second, or
    the resulting ``auc`` silently flips to mean "P(x0 > x1)" instead of
    the intended "P(x1 > x0)".

    ``auc = U1 / (n1 * n0)`` -- the U statistic normalized to ``[0, 1]``:
    1.0 = every x1 value ranks above every x0 value (perfect separation in
    the long direction), 0.0 = the exact reverse, 0.5 = no separation
    (coin flip).
    """
    result = mannwhitneyu(x1, x0, alternative="two-sided")
    n1, n0 = len(x1), len(x0)
    u1 = float(result.statistic)
    return u1, u1 / (n1 * n0)


def _hanley_mcneil_se(auc: float, n1: int, n0: int) -> float:
    """Hanley-McNeil (1982) standard error of an AUC estimate, task-4-
    brief's exact formula: ``sqrt((A(1-A) + (n1-1)(Q1-A^2) + (n0-1)(Q2-A^2))
    / (n1*n0))`` with ``Q1 = A/(2-A)``, ``Q2 = 2*A^2/(1+A)``.

    Both ``Q1``/``Q2`` denominators (``2-A``, ``1+A``) stay firmly away
    from 0 for every ``A`` in ``[0, 1]`` (worst case ``A=1`` -> denominators
    1 and 2), so this never needs a divide-by-zero guard. At the ``A=1``
    (or ``A=0``) extreme the formula's own variance term collapses to
    exactly 0 -- a real property of this estimator at a perfect separator,
    not a bug: ``auc_low95 == auc_high95 == auc`` there.
    """
    a = auc
    q1 = a / (2.0 - a)
    q2 = (2.0 * a * a) / (1.0 + a)
    variance = (a * (1.0 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return math.sqrt(variance)


def univariate_screen(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """One row per ``X`` column: how well that single feature, alone,
    separates ``y == 1`` (long) from ``y == 0`` (short).

    Pairwise-complete: each feature drops only ITS OWN NaN rows, entirely
    independently of every other feature's NaN pattern -- two different
    columns can therefore be scored over two different row subsets, and
    ``n``/``n_long``/``n_short`` report exactly what survived for THAT
    column.

    ``auc``/``abs_auc_dev`` come from ``_mannwhitney_auc`` (1.0 / 0.0 / 0.5
    = perfect long-separator / perfect short(anti)-separator / no
    separation; ``abs_auc_dev = |auc - 0.5|`` is the direction-agnostic
    STRENGTH of separation, and the column this result is sorted by).
    ``auc_se``/``auc_low95``/``auc_high95`` are the Hanley-McNeil 95% CI
    around ``auc`` (``_hanley_mcneil_se``). ``ks_stat`` is
    ``scipy.stats.ks_2samp``'s D statistic. ``mean_long``/``mean_short``
    are the two groups' plain means.

    A feature with fewer than ``_MIN_GROUP_N`` (10) non-NaN long OR short
    rows ("too_few"), or zero variance across its own non-NaN values
    ("degenerate", e.g. a constant column), is KEPT in the result (never
    silently dropped -- an experimenter screening candidate features needs
    to see every candidate, including the useless ones) with every stat
    column NaN and ``note`` naming why. The too_few check runs FIRST: a
    column that is both too-few and constant (not enough rows to even ask
    the variance question) reports "too_few". Checking variance BEFORE
    calling ``_mannwhitney_auc`` also sidesteps a real scipy edge case: a
    fully constant column does not raise there, but its p-value comes back
    NaN and its "auc" is a meaningless tie-adjusted 0.5 -- not an
    informative "no separation found", just no information to separate AT
    ALL.

    Sorted by ``abs_auc_dev`` descending, NaN last (``too_few``/
    ``degenerate`` rows -- the least informative candidates -- sink to the
    bottom). Indexed by ``feature`` (the ``X`` column name).
    """
    rows: dict[str, dict] = {}

    for col in X.columns:
        feature = X[col]
        present = feature.notna()
        vals = feature[present]
        y_sub = y.loc[vals.index]

        n = int(present.sum())
        n_long = int((y_sub == 1).sum())
        n_short = int((y_sub == 0).sum())

        row = {c: np.nan for c in _RESULT_COLUMNS}
        row["n"], row["n_long"], row["n_short"] = n, n_long, n_short

        if n_long < _MIN_GROUP_N or n_short < _MIN_GROUP_N:
            row["note"] = "too_few"
            rows[col] = row
            continue

        if vals.nunique() <= 1:
            row["note"] = "degenerate"
            rows[col] = row
            continue

        x1 = vals[y_sub == 1].to_numpy(dtype=float)
        x0 = vals[y_sub == 0].to_numpy(dtype=float)

        _u1, auc = _mannwhitney_auc(x1, x0)
        auc_se = _hanley_mcneil_se(auc, n_long, n_short)
        ks_stat, _ks_p = ks_2samp(x1, x0)

        row.update({
            "auc": auc,
            "abs_auc_dev": abs(auc - 0.5),
            "auc_se": auc_se,
            "auc_low95": auc - 1.96 * auc_se,
            "auc_high95": auc + 1.96 * auc_se,
            "ks_stat": float(ks_stat),
            "mean_long": float(x1.mean()),
            "mean_short": float(x0.mean()),
        })
        rows[col] = row

    result = pd.DataFrame.from_dict(rows, orient="index", columns=_RESULT_COLUMNS)
    result.index.name = "feature"
    return result.sort_values("abs_auc_dev", ascending=False, na_position="last")


def screen_charts(scr: pd.DataFrame, X: pd.DataFrame, y: pd.Series, out_dir, top_k: int = 15) -> list[str]:
    """Two PNGs per screened feature, for the ``top_k`` features (in
    ``scr``'s own sort order) that actually HAVE stats -- ``too_few``/
    ``degenerate`` rows (NaN ``auc``) are skipped outright, there is
    nothing to plot for them.

    - ``<feature>__hist.png``: class-conditional density histogram, long vs
      short overlaid (alpha, legend), title names the feature and its auc.
    - ``<feature>__decile.png``: long-rate (``P(y == 1)``) per feature-
      decile bar chart. Deciles via ``pd.qcut(..., duplicates="drop")`` --
      a feature with many repeated values at a decile boundary may
      legitimately collapse to fewer than 10 bins; this is expected, not an
      error. A horizontal line marks the overall base rate for reference.

    ``out_dir`` is created (parents included) if it does not already exist.
    Every figure is closed (``plt.close(fig)``) immediately after its own
    ``savefig`` -- charting many features must never accumulate open
    figures. Returns the saved file paths (str), in (feature, then
    hist-before-decile) order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = scr.loc[scr["auc"].notna()].head(top_k)

    saved: list[str] = []
    for feature, stat_row in candidates.iterrows():
        col = X[feature]
        present = col.notna()
        vals = col[present]
        y_sub = y.loc[vals.index]
        x1 = vals[y_sub == 1]
        x0 = vals[y_sub == 0]

        hist_path = out_dir / f"{feature}__hist.png"
        fig, ax = plt.subplots()
        ax.hist(x0.to_numpy(), bins=30, density=True, alpha=0.5, label="short (y=0)")
        ax.hist(x1.to_numpy(), bins=30, density=True, alpha=0.5, label="long (y=1)")
        ax.set_title(f"{feature} (auc={stat_row['auc']:.3f})")
        ax.set_xlabel(str(feature))
        ax.set_ylabel("density")
        ax.legend(loc="best")
        fig.savefig(hist_path, dpi=110)
        plt.close(fig)
        saved.append(str(hist_path))

        decile_path = out_dir / f"{feature}__decile.png"
        deciles = pd.qcut(vals, 10, duplicates="drop")
        base_rate = float(y_sub.mean())
        long_rate = y_sub.groupby(deciles, observed=True).mean()

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(long_rate)), long_rate.to_numpy())
        ax.axhline(base_rate, color="black", linestyle="--", label="base rate")
        ax.set_title(f"{feature} long-rate by decile")
        ax.set_xlabel("feature decile (low -> high)")
        ax.set_ylabel("P(y=long)")
        ax.legend(loc="best")
        fig.savefig(decile_path, dpi=110)
        plt.close(fig)
        saved.append(str(decile_path))

    return saved
