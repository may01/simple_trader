"""tdlib/alt_truth.py -- Task 8: truth decomposition.

``tdlib.diag`` (Task 7) asked "how much of the strict label's separation is
per-candle geometry vs something else." This module asks the orthogonal
question: "how much of the strict label's separation is the label
MECHANICS (the clean-entry gate profit_strict adds on top of a plain
race+fill) vs genuine future information?" Same points (``move_class_sym0``
+-2), same combos, same models/split -- only the TRUTH DEFINITION varies,
three ways, on top of the SAME two feature sets (full / shape-excluded)
Task 7 already established:

- "strict" = the existing ``truth.mark_truth`` (profit_strict pslong/psshort
  n1, exclusive long/short) -- NOT recomputed here. Read back from
  ``{base_dir}/iter_01/{combo}/metrics.json`` (``tdlib.loop``'s own baseline
  iteration) and ``{base_dir}/diag/{combo}/metrics.json`` (``tdlib.diag``'s
  own run) -- the exact same "read a prior run's persisted metrics.json"
  pattern ``tdlib.diag.write_diag_report`` already uses for ITS OWN baseline
  column.
- "plain" = the plain long/short pair (``plong_n1``/``pshort_n1``): the SAME
  race + pessimistic-entry-fill mechanics profit_strict uses, but WITHOUT
  its additional clean-entry gate. ``strict - plain`` isolates how much AUC
  that gate alone is worth.
- "fwd" = pure realized forward direction (``fwd_log_return(slim, tf, 1)``
  sign) -- zero label mechanics at all, not even a fill/race concept.
  ``plain - fwd`` isolates how much AUC the race+fill mechanics (minus the
  clean-entry gate) are worth; ``fwd - 0.5`` is whatever's left over --
  genuine forward-looking separation.

Five things live here:

1. ``mark_truth_plain(pts, tf)`` -- ``truth.mark_truth``'s EXACT 5-category
   (long/short/both/neither/nan) logic, reading ``LABEL_COLS[tf]
   ["plong_n1"]``/``["pshort_n1"]`` instead of the profit_strict pair. No
   ``horizon`` parameter -- the plain pair is n1-only (see
   ``tdlib.config.LABEL_COLS``'s own module docstring: "the PLAIN long/short
   pair ... is n1-only").

2. ``mark_truth_fwd(slim, tf)`` -- on ``slim``'s FULL index (like
   ``truth.fwd_log_return`` itself, NOT like ``mark_truth_plain``/
   ``mark_truth``, which both take an already-selected ``pts``): "long" if
   ``fwd_log_return(slim, tf, 1) > 0``, "short" if ``< 0``, "nan" otherwise
   (``== 0`` or missing -- both collapse into "nan", there is no meaningful
   separate "neither"/"both" concept for a single continuous return value).
   Computed on the full frame for the same reason ``engineered_features``
   is: a caller restricts to strong points AFTER, not before.

3. ``feature_matrix_alt(slim, tf, side, truth_kind, feature_cols=None)`` --
   ``features.feature_matrix``'s exact X-assembly shape (strong_points
   internally, ``engineered_features`` on the FULL slim, then both
   restricted to kept rows, same leak-blocklist belt-and-braces raise) but
   marking via ``mark_truth_plain``/``mark_truth_fwd`` instead of
   ``truth.mark_truth``. Duplicated rather than calling into
   ``features.feature_matrix`` -- that function hardcodes ``mark_truth``
   internally with no pluggable marking hook, and ``features.py`` is an
   existing file outside this task's edit scope (mirrors ``tdlib.oos``'s own
   precedent of duplicating ``tdlib.loop``'s combo-naming logic rather than
   reaching into a module not designed for it, back when that helper was
   still private). The reused ``LEAK_BLOCKLIST_RE`` already includes
   ``.*fwd_.*`` (see ``features.py``'s own docstring: "future/truth columns
   (e.g. a caller-added fwd_log_return diagnostic)") -- so the belt-and-
   braces check IS the "assert no fwd-derived column ever enters X"
   requirement, not a second, separate mechanism.

4. ``run_alt(slim, base_dir)`` -- for each combo in ``tdlib.diag.DIAG_TFS``
   x ``tdlib.loop.SIDES``, x ``truth_kind`` in ``{"plain", "fwd"}``, x
   ``feature_set`` in ``{"full", "noshape"}`` (``features.default_feature_cols``
   / ``diag.diag_feature_cols`` respectively): chrono-split 70/30, fit both
   classifiers, eval train+test, TRAIN-side screen/importance top-15 (same
   leak discipline as ``tdlib.loop``/``tdlib.diag`` post-fix-pass-2 -- never
   test-side). Skipped when ``len(X) < MIN_COMBO_POINTS`` -- a combo x
   truth_kind's usable-point count depends ONLY on the truth marking, never
   on which feature_set is selected, so (per the brief's own "combo x truth"
   framing, not "combo x truth x feature_set") a skip is identical across
   both feature_set values for the same (combo, truth_kind). Persists
   ``metrics.json``/``screen.csv``/``importance.csv`` per non-skipped
   (combo, truth_kind, feature_set) under
   ``{base_dir}/alt_truth/{combo}/{truth_kind}_{feature_set}/``. Returns a
   tidy DataFrame (columns per ``_TABLE_COLUMNS`` -- the brief does not spec
   ``run_alt``'s exact return columns the way it did for ``tdlib.diag.
   run_diag``, so this extends that module's own tidy-table shape with the
   two extra dimension columns this task adds: ``truth_kind``,
   ``feature_set``).

5. ``write_alt_report(table, base_dir)`` -- ``{base_dir}/alt_truth/report.md``:
   a headline decomposition table (one row per combo, one column per
   truth_kind x feature_set cell -- ``strict_full``/``strict_noshape`` read
   back from the OTHER modules' own prior runs, ``plain_*``/``fwd_*`` read
   directly off ``table``), each cell gbc test AUC + n (or "n/a" if that
   source is missing/skipped); an interpretation section (one line per
   combo: label share = strict-plain, entry-fill share = plain-fwd, future
   share = fwd-0.5, computed off the "full" feature-set column -- see the
   function's own docstring for why "full" specifically); then, per combo x
   truth_kind x feature_set, the top-10 screen/importance tables read back
   from their own persisted CSVs.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tdlib.config import LABEL_COLS
from tdlib.diag import DIAG_TFS, diag_feature_cols
from tdlib.features import LEAK_BLOCKLIST_RE, default_feature_cols, engineered_features
from tdlib.loop import MIN_COMBO_POINTS, SIDES, chrono_split, combo_name
from tdlib.models import eval_classifier, fit_classifiers, importance_table
from tdlib.points import strong_points
from tdlib.screen import univariate_screen
from tdlib.truth import fwd_log_return

logger = logging.getLogger(__name__)

_LEAK_RE = re.compile(LEAK_BLOCKLIST_RE)

_TRUTH_KINDS = ("plain", "fwd")
_FEATURE_SETS = ("full", "noshape")

# Same lightweight-diagnostic persistence/report truncation as tdlib.diag.
_PERSIST_TOP_N = 15
_REPORT_TOP_N = 10

_ALT_SUBDIR = "alt_truth"

_TABLE_COLUMNS = [
    "combo", "truth_kind", "feature_set", "model", "split",
    "roc_auc", "acc", "base_rate", "lift_long", "lift_short", "n",
]


# ---------------------------------------------------------------------------
# mark_truth_plain / mark_truth_fwd
# ---------------------------------------------------------------------------


def mark_truth_plain(pts: pd.DataFrame, tf: int) -> pd.Series:
    """``truth.mark_truth``'s exact 5-category (long/short/both/neither/nan)
    logic, reading the PLAIN long/short pair (``LABEL_COLS[tf]["plong_n1"]``/
    ``["pshort_n1"]``) instead of the profit_strict pair -- no ``horizon``
    argument, the plain pair is n1-only. See ``truth.mark_truth``'s own
    docstring for the exact per-row rule this mirrors:

      - plong == 1, pshort == 1 -> "both"
      - plong == 1, pshort == 0 -> "long"
      - plong == 0, pshort == 1 -> "short"
      - plong == 0, pshort == 0 -> "neither"
      - either value NaN         -> "nan" (overrides all of the above)

    Returns a Series of python ``str``, indexed exactly like ``pts``.
    """
    long_col = LABEL_COLS[tf]["plong_n1"]
    short_col = LABEL_COLS[tf]["pshort_n1"]

    long_val = pts[long_col]
    short_val = pts[short_col]

    is_long = long_val == 1
    is_short = short_val == 1
    is_nan = long_val.isna() | short_val.isna()

    marked = pd.Series("neither", index=pts.index, dtype=object)
    marked[is_long & is_short] = "both"
    marked[is_long & ~is_short] = "long"
    marked[~is_long & is_short] = "short"
    marked[is_nan] = "nan"  # overrides the above, same precedence as mark_truth

    return marked


def mark_truth_fwd(slim: pd.DataFrame, tf: int) -> pd.Series:
    """Pure realized forward direction: "long" where ``fwd_log_return(slim,
    tf, 1) > 0``, "short" where ``< 0``, "nan" everywhere else (``== 0``
    exactly, or NaN/not-yet-knowable -- both collapse into "nan", there is
    no "both"/"neither" concept for a single continuous return value, only
    3 categories total).

    Unlike ``mark_truth_plain``/``truth.mark_truth`` (which take an
    already-selected ``pts``), this takes the FULL ``slim`` and returns a
    Series on ``slim``'s own full index -- ``fwd_log_return`` itself needs
    the complete chronological closed-candle sequence to mean "N bars
    ahead" correctly (see its own docstring), so marking happens on the
    full frame first, exactly like ``features.engineered_features``'s own
    "full frame first, subset after" discipline. Callers restrict to a
    specific combo's strong points via ``.loc[pts.index]`` afterward.
    """
    fwd1 = fwd_log_return(slim, tf, 1)

    marked = pd.Series("nan", index=slim.index, dtype=object)
    marked[fwd1 > 0] = "long"
    marked[fwd1 < 0] = "short"
    return marked


# ---------------------------------------------------------------------------
# feature_matrix_alt
# ---------------------------------------------------------------------------


def feature_matrix_alt(
    slim: pd.DataFrame,
    tf: int,
    side: int,
    truth_kind: str,
    feature_cols: list | None = None,
) -> tuple:
    """``features.feature_matrix``'s exact X-assembly shape, marked via
    ``truth_kind`` in ``{"plain", "fwd"}`` instead of ``truth.mark_truth``.
    See the module docstring point 3 for why this duplicates rather than
    calls ``features.feature_matrix``.

    - ``pts = strong_points(slim, tf, side)``.
    - "plain": ``marked = mark_truth_plain(pts, tf)``. "fwd": ``marked =
      mark_truth_fwd(slim, tf).loc[pts.index]`` (computed on the full slim,
      see ``mark_truth_fwd``'s own docstring, restricted to this combo's
      points afterward). Any other ``truth_kind`` raises ``ValueError``.
    - Rows kept only where ``marked`` is "long"/"short"; ``y`` = 1 for
      "long", 0 for "short" (int8).
    - ``X`` = raw columns (``feature_cols`` or ``default_feature_cols``,
      booleans cast to int8, non-numeric dropped) joined with
      ``engineered_features(slim, tf, side)`` (computed ONCE on the full
      slim), restricted to kept rows, all-NaN columns dropped.
    - Belt-and-braces: every final X column checked against
      ``features.LEAK_BLOCKLIST_RE`` (which already includes ``.*fwd_.*``),
      raising ``ValueError`` naming any offender -- this is what guarantees
      no fwd-derived column (or any other blocklisted one) ever enters X,
      for either truth_kind.

    Returns ``(X, y)``, same row order as ``pts``'s kept rows. ``slim`` is
    never mutated.
    """
    pts = strong_points(slim, tf, side)

    if truth_kind == "plain":
        marked = mark_truth_plain(pts, tf)
    elif truth_kind == "fwd":
        marked = mark_truth_fwd(slim, tf).loc[pts.index]
    else:
        raise ValueError(f"feature_matrix_alt: unknown truth_kind {truth_kind!r} (expected 'plain' or 'fwd')")

    keep_mask = marked.isin(["long", "short"])
    kept_index = marked.index[keep_mask.to_numpy()]

    y = (marked.loc[kept_index] == "long").astype(np.int8)

    cols = list(feature_cols) if feature_cols is not None else default_feature_cols(list(slim.columns), tf)

    raw = slim.loc[kept_index, cols].copy()
    for col in raw.columns:
        if raw[col].dtype == bool:
            raw[col] = raw[col].astype(np.int8)
    numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    raw = raw[numeric_cols]

    engineered = engineered_features(slim, tf, side).loc[kept_index]

    X = pd.concat([raw, engineered], axis=1)
    X = X.dropna(axis=1, how="all")

    leaked = [c for c in X.columns if _LEAK_RE.fullmatch(c)]
    if leaked:
        raise ValueError(f"feature_matrix_alt: leak-blocklisted columns present in X: {leaked!r}")

    return X, y


# ---------------------------------------------------------------------------
# run_alt
# ---------------------------------------------------------------------------


def _feature_cols_for(feature_set: str, slim_cols: list, tf: int) -> list:
    if feature_set == "full":
        return default_feature_cols(slim_cols, tf)
    if feature_set == "noshape":
        return diag_feature_cols(slim_cols, tf)
    raise ValueError(f"run_alt: unknown feature_set {feature_set!r} (expected 'full' or 'noshape')")


def run_alt(slim: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    """One decomposition pass over every combo in ``DIAG_TFS`` x ``SIDES``,
    x ``truth_kind`` in ``{"plain", "fwd"}``, x ``feature_set`` in
    ``{"full", "noshape"}``. See the module docstring point 4.
    """
    rows: list = []

    for tf in DIAG_TFS:
        for side in SIDES:
            name = combo_name(tf, side)
            for truth_kind in _TRUTH_KINDS:
                for feature_set in _FEATURE_SETS:
                    cols = _feature_cols_for(feature_set, list(slim.columns), tf)
                    X, y = feature_matrix_alt(slim, tf, side, truth_kind, feature_cols=cols)

                    sub_name = f"{truth_kind}_{feature_set}"
                    if len(X) < MIN_COMBO_POINTS:
                        logger.info(
                            "alt_truth: combo=%s %s skipped (n=%d < %d)",
                            name, sub_name, len(X), MIN_COMBO_POINTS,
                        )
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
                    # tdlib.loop/tdlib.diag post-fix-pass-2.
                    scr = univariate_screen(X_tr, y_tr).head(_PERSIST_TOP_N)
                    imp = importance_table(bundle, X_tr, y_tr).head(_PERSIST_TOP_N)

                    combo_dir = Path(base_dir) / _ALT_SUBDIR / name / sub_name
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    with open(combo_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, sort_keys=True, indent=2)
                    scr.to_csv(combo_dir / "screen.csv")
                    imp.to_csv(combo_dir / "importance.csv", index=False)

                    gbc_test_auc = metrics["gbc"]["test"]["roc_auc"]
                    logger.info(
                        "alt_truth: combo=%s %s n=%d gbc_test_auc=%.4f",
                        name, sub_name, len(X), gbc_test_auc,
                    )

                    for m in ("logistic", "gbc"):
                        for split, md in metrics[m].items():
                            rows.append({
                                "combo": name, "truth_kind": truth_kind, "feature_set": feature_set,
                                "model": m, "split": split,
                                "roc_auc": md["roc_auc"], "acc": md["acc"], "base_rate": md["base_rate"],
                                "lift_long": md["lift_long"], "lift_short": md["lift_short"], "n": md["n"],
                            })

    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


# ---------------------------------------------------------------------------
# write_alt_report
# ---------------------------------------------------------------------------

_EXPECTED_ALT_COMBOS = [(tf, side) for tf in DIAG_TFS for side in SIDES]


def _fmt(value, spec: str = ".4f") -> str:
    """Format a possibly-missing/NaN numeric value -- never raises on
    None/NaN/non-numeric (duplicated small helper, same judgment call as
    tdlib.loop/tdlib.oos/tdlib.diag's own private ``_fmt``)."""
    if value is None:
        return "nan"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _cell(auc, n) -> str:
    """One headline-table cell: "{auc} (n={n})", or "n/a" if the source was
    missing/skipped entirely (no auc at all)."""
    if _is_missing(auc):
        return "n/a"
    n_str = "n/a" if n is None else str(int(n))
    return f"{_fmt(auc)} (n={n_str})"


def _read_gbc_test_metric(path: Path) -> tuple:
    """Best-effort read of a persisted ``metrics.json``'s gbc/test
    roc_auc + n. Returns ``(None, None)`` if the file is missing/unparsable
    or doesn't have the expected shape (mirrors ``tdlib.oos``'s/
    ``tdlib.diag``'s own best-effort read convention -- a missing prior run
    is real, reportable information, not a report-writing failure)."""
    try:
        with open(path) as f:
            metrics = json.load(f)
        test = metrics["gbc"]["test"]
        return test.get("roc_auc"), test.get("n")
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None, None


def _table_cell(table: pd.DataFrame, combo: str, truth_kind: str, feature_set: str, col: str):
    row = table[
        (table["combo"] == combo) & (table["truth_kind"] == truth_kind)
        & (table["feature_set"] == feature_set) & (table["model"] == "gbc") & (table["split"] == "test")
    ]
    if row.empty:
        return None
    return row.iloc[0][col]


def write_alt_report(table: pd.DataFrame, base_dir: str) -> str:
    """Write ``{base_dir}/alt_truth/report.md`` and return its path.

    Unlike ``tdlib.diag.write_diag_report`` (which takes an explicit
    ``loop_summary_path`` so its baseline read need not assume a shared
    directory), this function has no such parameter -- ``strict_full``/
    ``strict_noshape`` are read straight off ``{base_dir}/iter_01/{combo}/
    metrics.json`` and ``{base_dir}/diag/{combo}/metrics.json``, ASSUMING
    ``base_dir`` is the same directory the original loop run and diag run
    both used. This matches normal usage (``run_alt.py``, env-driven off
    the same ``config.artifacts_dir()`` every other driver in this
    experiment uses) and the brief's own simpler signature for this
    function specifically.

    Headline table: one row per combo, one column per truth_kind x
    feature_set cell (``strict_full``, ``strict_noshape``, ``plain_full``,
    ``plain_noshape``, ``fwd_full``, ``fwd_noshape``) -- each cell is gbc
    test AUC + n (``_cell``), or "n/a" if that source is missing/skipped
    entirely. ``strict_*`` comes from the OTHER modules' own prior runs;
    ``plain_*``/``fwd_*`` come directly from ``table`` (no disk re-read
    needed for data already in hand).

    Interpretation (one line per combo, computed off the "full" feature-set
    column specifically -- a deliberate judgment call: the brief names the
    3 quantities as strict/plain/fwd without saying full-vs-noshape, and
    "full" is this experiment's actual real-world/production feature set,
    listed first of each pair in the headline table; the "noshape" cells
    remain visible there for anyone who wants that comparison too):
    label share = strict_full - plain_full (how much the clean-entry gate
    alone is worth), entry-fill share = plain_full - fwd_full (how much the
    race+fill mechanics, minus that gate, are worth), future share =
    fwd_full - 0.5 (whatever's left -- genuine forward-looking separation).
    Any missing operand -> "n/a" for that share, never a fabricated number.

    Detail sections: per combo x truth_kind (``plain``, ``fwd`` -- "strict"
    has no detail section here, its own screen/importance tables already
    live in the loop's/diag's own reports) x feature_set (``full``,
    ``noshape``, both shown, nested under each combo x truth_kind heading),
    the top-10 screen/importance tables read back from that cell's own
    persisted CSVs.
    """
    alt_dir = Path(base_dir) / _ALT_SUBDIR
    alt_dir.mkdir(parents=True, exist_ok=True)
    report_path = alt_dir / "report.md"

    base = Path(base_dir)

    lines = [
        "# Truth decomposition report",
        "",
        "Same points (move_class_sym0 +-2), same combos, same models/split -- only "
        "the TRUTH DEFINITION varies: 'strict' (profit_strict, existing mark_truth), "
        "'plain' (race + pessimistic entry fill, no clean-entry gate), 'fwd' (pure "
        "realized forward direction, zero label mechanics). Each is shown against both "
        "the full feature set and the shape-excluded ('noshape', tdlib.diag) one.",
        "",
        "## Headline: gbc test AUC (n) per combo x truth_kind x feature_set",
        "",
        "| combo | strict_full | strict_noshape | plain_full | plain_noshape | fwd_full | fwd_noshape |",
        "|---|---|---|---|---|---|---|",
    ]

    headline: dict = {}  # combo -> {"strict_full": (auc, n), ...} for the interpretation section below

    for tf, side in _EXPECTED_ALT_COMBOS:
        name = combo_name(tf, side)

        strict_full = _read_gbc_test_metric(base / "iter_01" / name / "metrics.json")
        strict_noshape = _read_gbc_test_metric(base / "diag" / name / "metrics.json")
        plain_full = (_table_cell(table, name, "plain", "full", "roc_auc"), _table_cell(table, name, "plain", "full", "n"))
        plain_noshape = (
            _table_cell(table, name, "plain", "noshape", "roc_auc"), _table_cell(table, name, "plain", "noshape", "n"),
        )
        fwd_full = (_table_cell(table, name, "fwd", "full", "roc_auc"), _table_cell(table, name, "fwd", "full", "n"))
        fwd_noshape = (
            _table_cell(table, name, "fwd", "noshape", "roc_auc"), _table_cell(table, name, "fwd", "noshape", "n"),
        )

        headline[name] = {
            "strict_full": strict_full, "strict_noshape": strict_noshape,
            "plain_full": plain_full, "plain_noshape": plain_noshape,
            "fwd_full": fwd_full, "fwd_noshape": fwd_noshape,
        }

        lines.append(
            f"| {name} | {_cell(*strict_full)} | {_cell(*strict_noshape)} | "
            f"{_cell(*plain_full)} | {_cell(*plain_noshape)} | {_cell(*fwd_full)} | {_cell(*fwd_noshape)} |"
        )
    lines.append("")

    # --- interpretation ------------------------------------------------------------
    lines.append("## Interpretation (full feature set)")
    lines.append("")
    lines.append(
        "label_share = strict_full - plain_full (clean-entry gate's own contribution); "
        "entry_fill_share = plain_full - fwd_full (race+fill mechanics, minus that gate); "
        "future_share = fwd_full - 0.5 (genuine forward-looking separation left over)."
    )
    lines.append("")
    for tf, side in _EXPECTED_ALT_COMBOS:
        name = combo_name(tf, side)
        cells = headline[name]
        strict_auc = cells["strict_full"][0]
        plain_auc = cells["plain_full"][0]
        fwd_auc = cells["fwd_full"][0]

        label_share = None if (_is_missing(strict_auc) or _is_missing(plain_auc)) else strict_auc - plain_auc
        entry_fill_share = None if (_is_missing(plain_auc) or _is_missing(fwd_auc)) else plain_auc - fwd_auc
        future_share = None if _is_missing(fwd_auc) else fwd_auc - 0.5

        lines.append(
            f"- {name}: label_share={_fmt(label_share, '+.4f') if label_share is not None else 'n/a'}, "
            f"entry_fill_share={_fmt(entry_fill_share, '+.4f') if entry_fill_share is not None else 'n/a'}, "
            f"future_share={_fmt(future_share, '+.4f') if future_share is not None else 'n/a'}"
        )
    lines.append("")

    # --- detail sections: per combo x truth_kind x feature_set ----------------------
    present = set(zip(table["combo"], table["truth_kind"], table["feature_set"])) if not table.empty else set()

    for tf, side in _EXPECTED_ALT_COMBOS:
        name = combo_name(tf, side)
        lines.append(f"## {name}")
        lines.append("")

        for truth_kind in _TRUTH_KINDS:
            lines.append(f"### {truth_kind}")
            lines.append("")

            for feature_set in _FEATURE_SETS:
                lines.append(f"#### {feature_set}")
                lines.append("")

                if (name, truth_kind, feature_set) not in present:
                    lines.append(f"skipped (n < {MIN_COMBO_POINTS})")
                    lines.append("")
                    continue

                combo_dir = alt_dir / name / f"{truth_kind}_{feature_set}"

                lines.append("Top screened features (train-side):")
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

                lines.append("Top importance (train-side):")
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
