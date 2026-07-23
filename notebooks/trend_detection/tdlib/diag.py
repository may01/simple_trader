"""tdlib/diag.py -- Task 7: geometry-excluded diagnostic layer.

The real 2y loop's best iteration scored a 0.96 mean test AUC -- but
screening on that run shows the classes are dominated by candle-k's own
observable geometry (``15_logret`` auc 0.03-0.05, i.e. near-PERFECT
separation the wrong direction of 0.5; ``wick_up``/``wick_dn`` 0.85-0.89; KS
0.76-0.90). The profit_strict labels' clean-entry gate + pessimistic entry
fill are functions of candle k's own interior (open/high/low/close shape) --
so a classifier that sees those columns can substantially RECONSTRUCT the
label mechanics from observable geometry, rather than learning a forward-
looking trend signal. This module answers one diagnostic question: what
separation remains when the ENTIRE per-candle geometry family is excluded?

Four things live here:

1. ``SHAPE_SUFFIXES`` / ``is_shape_excluded`` -- the excluded family and its
   membership test. A column is shape-excluded iff it is ``{tf}_{suffix}``
   for one of the 8 named suffixes, EXACTLY (regex ``fullmatch``, anchored
   both ends) -- not a substring match, and not anything with extra
   trailing tokens.

   The ``_rm_6*`` rolling-smoothed family (``{tf}_close_diff_prc_rm_6``,
   ``..._rm_6_std_above``, ``..._rm_6_std_below``, and the ``high``/``low``
   analogs) is DELIBERATELY KEPT IN -- a judgment call, not an oversight.
   Those are 6-bar backward-looking rolling statistics (a regime/momentum
   signal built from the last 6 candles' diffs), not candle k's own raw
   interior shape. Candle k's own diff DOES enter that rolling window (1 of
   6 terms), so a sliver of the same information leaks through, diluted
   1/6th -- accepted as a reasonable cost, since a rolling statistic's whole
   point is that no single bar dominates it, which is exactly what
   distinguishes it from the bare per-candle columns this diagnostic exists
   to strip out.

2. ``diag_feature_cols(slim_cols, tf)`` -- ``features.default_feature_cols``
   minus every shape-excluded column, the raw-column candidate set
   ``run_diag`` restricts ``feature_matrix`` to.

3. ``run_diag(slim, base_dir)`` -- for each combo in ``DIAG_TFS`` (15, 60 --
   NOT 240: see its own note) x ``tdlib.loop.SIDES``: assemble X/y with the
   geometry-excluded feature set, chrono-split (``tdlib.loop.chrono_split``),
   fit both classifiers, eval train+test, and TRAIN-side screen/importance
   (never test-side -- same leak discipline ``tdlib.loop.run_iteration``
   applies post-fix-pass-2, for the identical reason: nothing that could
   feed back into a test-set-judged number is ever computed from the test
   split). Skips a combo with fewer than ``tdlib.loop.MIN_COMBO_POINTS``
   usable points, exactly like the loop. Persists ``metrics.json``/
   ``screen.csv``/``importance.csv`` per non-skipped combo under
   ``{base_dir}/diag/{combo}/`` and returns a tidy DataFrame (one row per
   combo x model x split).

4. ``write_diag_report(table, base_dir, loop_summary_path)`` -- renders
   ``{base_dir}/diag/report.md``: a headline table comparing each combo's
   diag gbc test AUC against the ORIGINAL loop run's iter_01 (baseline,
   full feature set, untransformed) gbc test AUC -- deliberately iter_01,
   not whatever iteration ``best.json`` happened to settle on, since the
   diagnostic question is "what does geometry buy the plain baseline," not
   "what does it buy whichever transform the loop's own search process
   picked" -- plus per-combo top-10 screen/importance tables and a one-line
   "strong"/"moderate"/"near-chance" interpretation of the diag run's OWN
   residual separation.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

from tdlib.features import default_feature_cols, feature_matrix
from tdlib.loop import MIN_COMBO_POINTS, SIDES, chrono_split, combo_name
from tdlib.models import eval_classifier, fit_classifiers, importance_table
from tdlib.screen import univariate_screen

logger = logging.getLogger(__name__)

# tf in {15, 60} ONLY -- task-7-brief.md's exact scope, deliberately
# narrower than tdlib.config.ANALYSIS_TFS's full [15, 60, 240]. tf=240
# essentially never clears MIN_COMBO_POINTS in this experiment's own slim
# (very few closed-240 strong-move points per window -- see tdlib.loop's own
# combo-count notes), so including it here would just be a guaranteed-skip
# combo adding runtime for zero diagnostic value.
#
# Public (no leading underscore): tdlib.alt_truth (T8) reuses this SAME tf
# scope ("same combos (15_up/15_dn/60_up/60_dn)") rather than a second
# hardcoded (15, 60) that could silently drift from this one -- same
# rationale as tdlib.loop.combo_name/MIN_COMBO_POINTS's own promotion.
DIAG_TFS = (15, 60)

# The on-disk subdirectory (under base_dir) this module's outputs live in.
_DIAG_SUBDIR = "diag"

# How many rows of screen/importance to keep, both in the persisted CSVs AND
# (further trimmed to 10, see write_diag_report) in the report -- this
# diagnostic is deliberately lightweight (no charts, no full-table archival
# like tdlib.loop's own screen.csv/importance.csv), so only the strongest
# slice is kept at all, unlike Layer 5's "persist full table, truncate only
# in-memory" convention.
_PERSIST_TOP_N = 15
_REPORT_TOP_N = 10

_TABLE_COLUMNS = ["combo", "model", "split", "roc_auc", "acc", "base_rate", "lift_long", "lift_short", "n"]

SHAPE_SUFFIXES = (
    "logret", "close_diff_prc", "high_diff_prc", "low_diff_prc",
    "body_ratio", "wick_up", "wick_dn", "range_atr",
)

_SHAPE_RE = re.compile(r"^\d+_(?:" + "|".join(SHAPE_SUFFIXES) + r")$")


# ---------------------------------------------------------------------------
# shape exclusion
# ---------------------------------------------------------------------------


def is_shape_excluded(col: str) -> bool:
    """True iff ``col`` is exactly ``{tf}_{suffix}`` for one of
    ``SHAPE_SUFFIXES`` (any digit-string tf) -- a full-string match, not a
    substring one, so e.g. ``60_close_diff_prc_rm_6`` (extra trailing
    tokens) is NOT excluded even though it shares the ``close_diff_prc``
    root -- see the module docstring's ``_rm_6*`` judgment call."""
    return _SHAPE_RE.fullmatch(col) is not None


def diag_feature_cols(slim_cols: list, tf: int) -> list:
    """``features.default_feature_cols(slim_cols, tf)`` minus every
    shape-excluded column (``is_shape_excluded``) -- the raw-column
    candidate set ``run_diag`` restricts ``feature_matrix`` to. Order
    matches ``default_feature_cols``'s own (which in turn matches
    ``slim_cols``'s order) -- deterministic given a deterministic input.
    """
    base = default_feature_cols(slim_cols, tf)
    return [c for c in base if not is_shape_excluded(c)]


# ---------------------------------------------------------------------------
# run_diag
# ---------------------------------------------------------------------------


def run_diag(slim: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    """One diagnostic pass over every (tf, side) combo in ``DIAG_TFS`` x
    ``tdlib.loop.SIDES``. See the module docstring point 3 for the full
    per-combo pipeline and persistence.

    Belt-and-braces: after ``feature_matrix`` returns X (raw columns
    already restricted to ``diag_feature_cols``, PLUS whatever
    ``engineered_features`` adds -- which ``feature_matrix`` always adds
    regardless of ``feature_cols``), every final X column is re-checked
    against ``is_shape_excluded`` and a match raises ``ValueError`` naming
    the offending column(s) -- mirrors ``tdlib.features.feature_matrix``'s
    own leak-blocklist belt-and-braces check exactly (same rationale: this
    should never actually trigger given how ``engineered_features`` names
    its own output, but a diagnostic whose entire PURPOSE is "prove
    geometry is excluded" must fail loudly, not silently paper over it, if
    that invariant is ever violated by a future change).

    Returns a tidy DataFrame, columns exactly ``_TABLE_COLUMNS`` -- one row
    per (non-skipped combo) x ("logistic", "gbc") x ("train", "test").
    """
    rows: list = []

    for tf in DIAG_TFS:
        for side in SIDES:
            name = combo_name(tf, side)
            cols = diag_feature_cols(list(slim.columns), tf)
            X, y = feature_matrix(slim, tf, side, feature_cols=cols, horizon="n1")

            leaked = [c for c in X.columns if is_shape_excluded(c)]
            if leaked:
                raise ValueError(f"run_diag: shape-suffix column(s) slipped into X for combo {name!r}: {leaked!r}")

            if len(X) < MIN_COMBO_POINTS:
                logger.info("diag: combo=%s skipped (n=%d < %d)", name, len(X), MIN_COMBO_POINTS)
                continue

            X_tr, X_te, y_tr, y_te = chrono_split(X, y)
            bundle = fit_classifiers(X_tr, y_tr)

            metrics = {
                m: {
                    "train": eval_classifier(bundle, m, X_tr, y_tr),
                    "test": eval_classifier(bundle, m, X_te, y_te),
                }
                for m in ("logistic", "gbc")
            }

            # TRAIN-side, deliberately -- same leak discipline as
            # tdlib.loop.run_iteration post-fix-pass-2 (see module
            # docstring point 3): nothing here is allowed to have peeked at
            # the test split this run's own metrics are judged on.
            scr = univariate_screen(X_tr, y_tr).head(_PERSIST_TOP_N)
            imp = importance_table(bundle, X_tr, y_tr).head(_PERSIST_TOP_N)

            combo_dir = Path(base_dir) / _DIAG_SUBDIR / name
            combo_dir.mkdir(parents=True, exist_ok=True)
            with open(combo_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, sort_keys=True, indent=2)
            scr.to_csv(combo_dir / "screen.csv")
            imp.to_csv(combo_dir / "importance.csv", index=False)

            gbc_test_auc = metrics["gbc"]["test"]["roc_auc"]
            logger.info("diag: combo=%s n=%d gbc_test_auc=%.4f", name, len(X), gbc_test_auc)

            for m in ("logistic", "gbc"):
                for split, md in metrics[m].items():
                    rows.append({
                        "combo": name, "model": m, "split": split,
                        "roc_auc": md["roc_auc"], "acc": md["acc"], "base_rate": md["base_rate"],
                        "lift_long": md["lift_long"], "lift_short": md["lift_short"], "n": md["n"],
                    })

    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


# ---------------------------------------------------------------------------
# write_diag_report
# ---------------------------------------------------------------------------

_EXPECTED_DIAG_COMBOS = [(tf, side) for tf in DIAG_TFS for side in SIDES]


def _fmt(value, spec: str = ".4f") -> str:
    """Format a possibly-missing/NaN numeric value for a markdown cell --
    never raises on None/NaN/non-numeric (mirrors tdlib.loop's/tdlib.oos's
    own private ``_fmt`` helper; duplicated rather than imported since it is
    private to those modules -- the same judgment call tdlib.oos already
    made rather than reach into tdlib.loop's internals for a one-liner)."""
    if value is None:
        return "nan"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _combo_metric(table: pd.DataFrame, combo: str, model: str, split: str, col: str):
    row = table[(table["combo"] == combo) & (table["model"] == model) & (table["split"] == split)]
    if row.empty:
        return None
    return row.iloc[0][col]


def _read_baseline_gbc_test_auc(loop_base: Path, combo: str):
    """Best-effort read of the ORIGINAL loop run's own iter_01 (baseline,
    untransformed, horizon=n1) gbc test roc_auc for ``combo`` -- see the
    module docstring point 4 for why iter_01 specifically, not
    ``best.json``'s own best_iter. A missing/unparsable file returns None
    (mirrors ``tdlib.oos``'s own ``_load_test_metrics`` best-effort
    convention) rather than raising -- an absent baseline (e.g. that combo
    was itself skipped in the original loop run, or the loop hasn't been
    run at all yet) is real, reportable information, not a diagnostic-
    report failure.
    """
    path = loop_base / "iter_01" / combo / "metrics.json"
    try:
        with open(path) as f:
            metrics = json.load(f)
        return metrics["gbc"]["test"]["roc_auc"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def _interpretation(test_auc) -> str:
    """"strong"/"moderate"/"near-chance" by the diag run's OWN gbc test auc
    -- task-7-brief.md's exact thresholds (>=0.75 / >=0.6 / else). NaN/None
    -> "n/a" (nothing to interpret, not a silent "near-chance")."""
    if test_auc is None or (isinstance(test_auc, float) and math.isnan(test_auc)):
        return "n/a"
    if test_auc >= 0.75:
        return "strong"
    if test_auc >= 0.6:
        return "moderate"
    return "near-chance"


def write_diag_report(table: pd.DataFrame, base_dir: str, loop_summary_path: str) -> str:
    """Write ``{base_dir}/diag/report.md`` and return its path.

    ``loop_summary_path`` is the ORIGINAL loop run's own ``summary.md`` path
    (``tdlib.loop.write_summary``'s return value / what ``run_loop.py``
    prints) -- its PARENT directory is that loop run's own base_dir, from
    which ``iter_01/{combo}/metrics.json`` is read for the baseline
    comparison. Deliberately NOT assumed to be the same directory as this
    function's own ``base_dir`` (diag's outputs and the original loop's
    artifacts could in principle live under different roots) even though in
    normal usage (``run_diag.py``, both env-driven off the same
    ``config.artifacts_dir()``) they are the same directory.

    A combo entirely absent from ``table`` (skipped in THIS diag run, per
    ``run_diag``'s own ``MIN_COMBO_POINTS`` floor) gets a "skipped" headline
    row and no per-combo detail section -- there is nothing to report for
    it beyond that.
    """
    diag_dir = Path(base_dir) / _DIAG_SUBDIR
    diag_dir.mkdir(parents=True, exist_ok=True)
    report_path = diag_dir / "report.md"

    loop_base = Path(loop_summary_path).parent
    present_combos = set(table["combo"].unique()) if not table.empty else set()

    lines = [
        "# Diagnostic report -- per-candle geometry excluded",
        "",
        "Candle-k's OWN shape/geometry (logret, close/high/low_diff_prc, "
        "body_ratio, wick_up/dn, range_atr -- see diag.SHAPE_SUFFIXES) is "
        "excluded from the feature set entirely; the _rm_6* backward-smoothed "
        "rolling family stays IN (see the module docstring's judgment call). "
        "Compares this run's gbc test AUC against the ORIGINAL loop's iter_01 "
        "(baseline, full feature set, horizon=n1) gbc test AUC, per combo -- "
        "how much residual separation survives once the profit_strict labels' "
        "candle-interior fingerprint is no longer directly observable.",
        "",
        "## Headline: baseline (iter_01) vs diag, gbc test AUC",
        "",
        "| combo | baseline_test_auc | diag_test_auc | delta | interpretation |",
        "|---|---|---|---|---|",
    ]

    for tf, side in _EXPECTED_DIAG_COMBOS:
        name = combo_name(tf, side)
        if name not in present_combos:
            lines.append(f"| {name} | n/a | n/a | n/a | skipped (n<{MIN_COMBO_POINTS}) |")
            continue

        baseline_auc = _read_baseline_gbc_test_auc(loop_base, name)
        diag_auc = _combo_metric(table, name, "gbc", "test", "roc_auc")
        diag_auc_is_nan = diag_auc is None or (isinstance(diag_auc, float) and math.isnan(diag_auc))
        delta = (diag_auc - baseline_auc) if (baseline_auc is not None and not diag_auc_is_nan) else None

        baseline_str = _fmt(baseline_auc) if baseline_auc is not None else "n/a"
        delta_str = _fmt(delta, "+.4f") if delta is not None else "n/a"
        lines.append(f"| {name} | {baseline_str} | {_fmt(diag_auc)} | {delta_str} | {_interpretation(diag_auc)} |")
    lines.append("")

    for tf, side in _EXPECTED_DIAG_COMBOS:
        name = combo_name(tf, side)
        if name not in present_combos:
            continue
        combo_dir = diag_dir / name

        lines.append(f"## {name}")
        lines.append("")
        lines.append("### Top screened features (diag, train-side)")
        lines.append("")
        lines.append("| feature | auc | auc_low95 | ks |")
        lines.append("|---|---|---|---|")
        scr_path = combo_dir / "screen.csv"
        if scr_path.exists():
            scr = pd.read_csv(scr_path).head(_REPORT_TOP_N)
            for _, row in scr.iterrows():
                lines.append(
                    f"| {row.get('feature')} | {_fmt(row.get('auc'))} | "
                    f"{_fmt(row.get('auc_low95'))} | {_fmt(row.get('ks_stat'))} |"
                )
        lines.append("")

        lines.append("### Top importance (diag, train-side)")
        lines.append("")
        lines.append("| feature | imp_mean | imp_std |")
        lines.append("|---|---|---|")
        imp_path = combo_dir / "importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path).head(_REPORT_TOP_N)
            for _, row in imp.iterrows():
                lines.append(f"| {row.get('feature')} | {_fmt(row.get('imp_mean'))} | {_fmt(row.get('imp_std'))} |")
        lines.append("")

    report_path.write_text("\n".join(lines))
    return str(report_path)
