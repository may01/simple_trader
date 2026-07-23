"""tdlib/oos.py -- Task 6: frozen OOS validation (Layer 6).

The last code layer. Selection is CLOSED at Layer 5 (``tdlib.loop``): a
best.json already names the winning iteration's transform/horizon and, per
combo, a frozen bundle + selected_features.json. This layer does ONE thing
-- score those frozen artifacts against an out-of-sample slim (the oos2m
holdout in production) with ZERO refit anywhere. Nothing here ever calls
``.fit()`` on anything; every classifier and every ``FreezeStats`` is loaded
straight off disk and only ever ``.transform()``/``.predict_proba()``'d.

Five things live here:

1. ``crosscheck_gates(oos_slim)`` -- a whole-frame sanity gate, independent
   of which combos actually got trained: for every tf in ``ANALYSIS_TFS``
   that carries a baked ``{tf}_move_class_sym0`` column (an OPTIONAL column
   -- see ``tdlib.config``'s slim whitelist; the oos2m slim carries it,
   older/synthetic slims may not), ``tdlib.points.crosscheck_baked`` must
   agree on at least ``_CROSSCHECK_MIN_FRACTION`` (0.999) of closed rows or
   this raises ``RuntimeError`` naming the tf and the actual fraction -- a
   drift between ``tdlib.config.MOVE_CUTS`` and whatever cuts the OOS
   dataset was actually baked with is a silent-corruption risk this
   experiment cannot afford to only notice downstream, in degraded metrics.
   A tf missing the baked column entirely just warns and is recorded
   "skipped" -- there is nothing to cross-check against, and that is
   expected for some datasets, not an error.

   Public (no leading underscore, mirroring ``tdlib.loop.select_best``'s own
   promotion rationale): ``run_oos`` calls it internally to enforce the
   gate, and it has a SECOND real caller -- ``run_oos.py``'s driver needs
   the same ``gates`` dict to hand to ``write_oos_report`` (whose signature,
   fixed by the brief, takes ``gates`` as an explicit argument rather than
   reading it off ``run_oos``'s own return value, which is pinned to a bare
   ``pd.DataFrame``). Calling this twice (once inside ``run_oos``, once from
   the driver) is a deliberately cheap redundancy -- one pass over a single
   tf's closed-row column -- in exchange for never inventing a side-channel
   (e.g. smuggling a dict through ``DataFrame.attrs``) to avoid it.

2. ``_reproduce_transform(X, transform, selected_features)`` -- reproduces
   Layer 5's ``apply_transform`` column-shaping EXACTLY, from
   ``selected_features.json`` alone, never by recomputing importance:
   "baseline"/"horizon_n2" are no-ops (horizon's own effect already lives in
   the ``feature_matrix(..., horizon=...)`` call above it); "prune_top40" is
   just re-selecting the named columns (the frozen list itself already IS
   the pruned set -- nothing to recompute); "interact_time_left" rebuilds
   each ``a__x__b`` product column by parsing the name and multiplying the
   two base columns, when both are present. Any ``selected_features`` name
   this cannot resolve is silently left ABSENT from the result -- this
   function never raises on an unresolvable name by itself. That is
   deliberate: the one caller (``run_oos``) already needs a "does X now
   satisfy the bundle's frozen feature_order" check for an entirely
   different reason (see point 3), so an unresolvable interaction name and a
   genuinely-missing engineered column funnel through that SAME check and
   the SAME error message, rather than two parallel raise sites that could
   drift apart.

3. ``run_oos(oos_slim, train_base)`` -- per combo in ``best.json``: rebuild
   X via ``feature_matrix`` (the FULL default feature set, never pre-
   restricted to ``selected_features`` -- see point 2), reproduce the
   frozen transform, then reindex to the loaded bundle's OWN
   ``feature_order`` (``load_bundle`` already restores this from
   feature_order.json -- see ``tdlib.models``'s own module docstring). This
   reindex is a POLICY choice, not something ``FreezeStats.transform``
   itself would force: that method tolerates a fit-time column being
   absent from the frame it is scoring (silently imputes it at the frozen
   median -- see its own docstring) and only raises on a column it never
   saw at fit time. Silently median-filling a genuinely-missing ENGINEERED
   feature would be a silent validation lie (a real reconstruction bug
   masquerading as "just some missing data"), so this module is stricter
   than ``FreezeStats`` on purpose: every ``feature_order`` column must
   actually be present after transform-reproduction, or this raises
   ``ValueError`` naming every missing one, before ``FreezeStats.transform``
   ever gets a chance to paper over it. The reindex ALSO drops any extra
   column ``X`` carries that the bundle never saw at fit time -- required
   for a different reason: ``FreezeStats.transform`` raises on those
   ("unseen at fit time"), and the oos2m slim in production carries a wider
   raw-column whitelist than the 2y training slim did (479 vs 471 columns --
   see task-6-brief.md), so extras are an expected, benign case here, not a
   bug.

   A combo whose oos-side usable point count (``n_long + n_short``, from
   ``tdlib.truth.truth_counts`` -- computed FIRST, before ``feature_matrix``/
   transform/bundle machinery is ever touched, mirroring
   ``tdlib.loop.run_iteration``'s own skip-before-anything-else order
   exactly) falls under ``_MIN_OOS_POINTS`` (30) is marked ``skipped`` --
   both rows (one per model) carry real ``n``/``n_long``/``n_short`` but NaN
   for robustness AND every model-dependent metric; neither
   ``feature_matrix``, ``load_bundle`` nor ``eval_classifier`` is ever
   called for it (matches ``tdlib.loop``'s own ``_skipped_combo`` --
   ``robustness={}`` -- convention). This ordering is load-bearing, not just
   a performance nicety: ``feature_matrix``'s own ``X.dropna(axis=1,
   how="all")`` step drops EVERY column when 0 rows are selected (``.all()``
   over an empty NaN-check array is vacuously True), so an earlier version
   of this function that built X before checking the point count would let
   a zero-point combo reach the missing-``feature_order`` check below with
   ALL of ``feature_order`` reported "missing" -- a misleading
   "could not be reproduced" ``ValueError`` that aborted scoring for every
   OTHER combo in the same ``run_oos`` call too (the per-combo loop in
   ``run_oos`` does not catch it). Realistic in production: tf=240 has only
   ~360 closed candles in a 2-month OOS window, and the +-2 strong-move
   tails can legitimately be empty on one side.

   Every non-skipped combo gets a before/after byte-compare of its bundle's
   ``FreezeStats.to_json`` output (to a throwaway temp file) wrapped around
   both models' ``eval_classifier`` calls -- a defensive, always-on
   assertion (not just a test-time monkeypatch) that this layer truly never
   refits anything, anywhere, even after a future refactor.

   Returns a tidy DataFrame, 2 rows (one per ``{"logistic", "gbc"}``) per
   combo in ``best.json``, columns exactly per task-6-brief.md's spec.

4. ``write_oos_report(table, gates, train_base, out_dir)`` -- renders
   ``{out_dir}/oos_report.md`` (crosscheck gate table, a test-vs-oos metrics
   table per combo x model, an oos robustness table, and a one-line verdict
   per combo) plus ``{out_dir}/oos_table.csv`` (``table`` itself). Test-side
   metrics are read back from ``{train_base}/iter_{best_iter:02d}/{combo}/
   metrics.json`` -- Layer 5's own per-combo persistence -- via
   ``best.json``'s own ``best_iter``, so this module never needs its own
   redundant copy of "which iteration is best" (``select_best`` already
   answered that at Layer 5; here it is simply read back off disk).

   The per-combo verdict is deliberately judged off the ``gbc`` model only
   (not a combined/averaged rule across both models): ``tdlib.loop``'s own
   ``IterResult.mean_test_auc`` -- the metric Layer 5's entire keep/reject
   decision runs on -- is ALREADY gbc-only (see its docstring), so judging
   Layer 6's verdict against logistic instead (or against some blend) would
   let this report disagree with the very selection criterion that chose
   the frozen iteration in the first place. "Holds" iff the oos roc_auc
   exceeds 0.5 AND oos ``lift_long``'s sign matches the frozen iteration's
   OWN test-split ``lift_long`` sign AND oos ``lift_short``'s sign matches
   its OWN test-split ``lift_short`` sign (both read from that same
   metrics.json) -- BOTH signs, not ``lift_long`` alone: this is a two-sided
   experiment (every combo, "up" or "dn" alike, produces a real
   long-vs-short prediction -- see ``tdlib.features.feature_matrix``'s own
   y-encoding, which is side-agnostic), and side-picking which single lift
   to judge (e.g. "dn combos are short-native, judge them on lift_short")
   would be actively wrong here: per rsi_side_stats, a side=-2 combo's OWN
   base tendency skews LONG, not short. Any NaN among the four signed
   quantities (oos/test x long/short) -> "fails", never a silent "holds".
   "skipped" if the combo's own row is skipped; "fails" otherwise
   (including when the test-side metrics file cannot be found/parsed at
   all -- a missing comparison point is not a silent "holds").

5. ``_freeze_json_bytes`` / ``_parse_combo_name`` / ``_fmt`` -- small local
   helpers. ``_parse_combo_name`` is the exact inverse of
   ``tdlib.loop._combo_name`` (``"{tf}_up"``/``"{tf}_dn"`` -> ``(tf, +-2)``);
   duplicated here rather than imported (that helper is private to
   ``tdlib.loop``) since Layer 6 needs it independently to route each
   ``best.json`` combo entry to ``feature_matrix``/``strong_points``.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import warnings
from pathlib import Path

import pandas as pd

from tdlib.config import ANALYSIS_TFS
from tdlib.features import feature_matrix
from tdlib.models import eval_classifier, load_bundle
from tdlib.points import crosscheck_baked, strong_points
from tdlib.truth import fwd_log_return, mark_truth, robustness_agreement, truth_counts

# A combo needs at least this many usable (long or short) oos-marked points
# to be worth scoring at all -- task-6-brief.md's exact floor (note: NOT the
# same number as tdlib.loop's own _MIN_COMBO_POINTS=60 train-time floor;
# this is a separate, lower, OOS-side bar).
_MIN_OOS_POINTS = 30

# tdlib.points.crosscheck_baked's required agreement fraction for the
# cut-provenance gate -- task-6-brief.md's exact number.
_CROSSCHECK_MIN_FRACTION = 0.999

_MODELS = ("logistic", "gbc")

# The model write_oos_report's per-combo verdict is judged off -- see the
# module docstring point 4 for why this must match tdlib.loop.IterResult.
# mean_test_auc's own gbc-only convention.
_VERDICT_MODEL = "gbc"

# task-6-brief.md's exact tidy-table column order.
_TABLE_COLUMNS = [
    "combo", "tf", "side", "model", "n", "n_long", "n_short", "base_rate",
    "roc_auc", "acc", "prec_top_decile", "prec_bottom_decile",
    "lift_long", "lift_short", "long_fwd1", "long_fwd4", "short_fwd1", "short_fwd4",
    "skipped",
]

_COMBO_NAME_RE = re.compile(r"^(\d+)_(up|dn)$")

# Greedy .+ on both sides -> splits on the RIGHTMOST "__x__" if a name ever
# contained more than one occurrence. Matches tdlib.loop.apply_transform's
# own construction exactly (f"{col}__x__{time_left_col}" -- a single
# "__x__" insertion appending the time_left column as a trailing suffix) --
# safe because no base feature name in this codebase's naming conventions
# ever itself contains the literal substring "__x__".
_INTERACT_RE = re.compile(r"^(.+)__x__(.+)$")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _parse_combo_name(name: str) -> tuple:
    """Exact inverse of ``tdlib.loop._combo_name``: ``"15_up"`` -> ``(15,
    2)``, ``"240_dn"`` -> ``(240, -2)``. Raises ``ValueError`` naming the
    unrecognized name if it doesn't match ``"{tf}_{up,dn}"``."""
    m = _COMBO_NAME_RE.match(name)
    if not m:
        raise ValueError(f"run_oos: unrecognized combo name {name!r} (expected '{{tf}}_up' or '{{tf}}_dn')")
    tf = int(m.group(1))
    side = 2 if m.group(2) == "up" else -2
    return tf, side


def _freeze_json_bytes(freeze) -> bytes:
    """``freeze.to_json`` written to a throwaway temp file, read back as
    bytes, temp file removed -- the exact "to_json to temp -> byte-compare"
    mechanism task-6-brief.md's no-refit purity check names."""
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        freeze.to_json(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        os.unlink(tmp_path)


def _fmt(value, spec: str = ".4f") -> str:
    """Format a possibly-missing/NaN numeric value for a markdown cell --
    never raises on None/NaN/non-numeric (mirrors tdlib.loop's own private
    ``_fmt`` helper; duplicated rather than imported since it is private to
    that module)."""
    if value is None:
        return "nan"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# crosscheck_gates
# ---------------------------------------------------------------------------


def crosscheck_gates(oos_slim: pd.DataFrame) -> dict:
    """Cut-provenance drift gate over EVERY tf in ``ANALYSIS_TFS`` (not just
    tfs with a combo in some particular best.json -- this is a whole-frame
    sanity check on the OOS dataset itself). See the module docstring point
    1 for the full rationale and why this is public.

    Returns ``{tf: {"status": "passed"|"skipped", "fraction": float|None}}``.
    Raises ``RuntimeError`` (naming the tf and the actual fraction) the
    moment any present baked column's agreement falls under
    ``_CROSSCHECK_MIN_FRACTION`` -- the whole gate stops at the first
    failure rather than collecting every tf's result first, since one
    failure already means "do not trust this OOS dataset's cut provenance,"
    full stop.
    """
    gates: dict = {}
    for tf in ANALYSIS_TFS:
        baked_col = f"{tf}_move_class_sym0"
        if baked_col not in oos_slim.columns:
            warnings.warn(
                f"run_oos: crosscheck gate: {baked_col!r} not present in oos slim (tf={tf}); "
                f"skipping cut-provenance check for this tf"
            )
            gates[tf] = {"status": "skipped", "fraction": None}
            continue

        fraction = crosscheck_baked(oos_slim, tf)
        if fraction < _CROSSCHECK_MIN_FRACTION:
            raise RuntimeError(
                f"run_oos: crosscheck gate FAILED for tf={tf}: sym0_class computed from "
                f"{tf}_rsi_ma8_diff agrees with the baked {baked_col} column on only "
                f"{fraction:.6f} of closed rows (required >= {_CROSSCHECK_MIN_FRACTION}) -- "
                f"cut-provenance drift between tdlib.config.MOVE_CUTS and this OOS slim's own "
                f"production baking"
            )
        gates[tf] = {"status": "passed", "fraction": fraction}
    return gates


# ---------------------------------------------------------------------------
# _reproduce_transform
# ---------------------------------------------------------------------------


def _reproduce_transform(X: pd.DataFrame, transform: str, selected_features: list) -> pd.DataFrame:
    """Reproduce ``tdlib.loop.apply_transform``'s column-shaping on ``X``
    (a fresh ``feature_matrix`` build over the OOS slim, with the full
    default feature set -- never itself pre-restricted to
    ``selected_features``), from ``selected_features`` alone. See the module
    docstring point 2 for the full per-transform rationale and why an
    unresolvable name is left absent here rather than raised here.
    """
    if transform in ("baseline", "horizon_n2"):
        return X

    if transform == "prune_top40":
        # The frozen selected_features list already IS the pruned set --
        # selection by list is the reproduction, no importance recompute.
        cols = [c for c in selected_features if c in X.columns]
        return X[cols]

    if transform == "interact_time_left":
        X = X.copy()
        for name in selected_features:
            if name in X.columns:
                continue
            m = _INTERACT_RE.match(name)
            if m and m.group(1) in X.columns and m.group(2) in X.columns:
                X[name] = X[m.group(1)] * X[m.group(2)]
            # else: left unresolved -- run_oos's own feature_order check
            # raises, naming it, right after this returns.
        return X

    raise ValueError(f"run_oos: unknown transform {transform!r}")


# ---------------------------------------------------------------------------
# run_oos
# ---------------------------------------------------------------------------


def _score_combo(oos_slim: pd.DataFrame, train_base: str, combo_name: str, combo_info: dict,
                  transform: str, horizon: str) -> list:
    tf, side = _parse_combo_name(combo_name)

    # Counts + skip decision FIRST, before ANY feature/transform/bundle
    # machinery is touched -- mirrors tdlib.loop.run_iteration's own order
    # exactly (strong_points -> mark_truth -> truth_counts -> skip check,
    # loop.py:330-337). See the module docstring point 3 for why this
    # ordering is load-bearing (a genuinely 0-row combo would otherwise
    # make feature_matrix's own dropna(how="all") drop every column, and
    # the missing-feature_order check below would then misreport the
    # entire frozen feature set as "could not be reproduced" -- aborting
    # every OTHER combo in this run_oos call too).
    pts = strong_points(oos_slim, tf, side)
    marked = mark_truth(pts, tf, horizon)
    counts = truth_counts(marked)
    n_long, n_short = counts["long"], counts["short"]
    n_total = n_long + n_short
    skipped = n_total < _MIN_OOS_POINTS

    base_row = {
        "combo": combo_name, "tf": tf, "side": side,
        "n": n_total, "n_long": n_long, "n_short": n_short,
        "skipped": skipped,
    }

    if skipped:
        # Nothing beyond counts is computed for a skipped combo -- no
        # robustness, no feature_matrix, no bundle load -- matching
        # tdlib.loop's own _skipped_combo convention (robustness={}).
        base_row.update({
            "long_fwd1": float("nan"), "long_fwd4": float("nan"),
            "short_fwd1": float("nan"), "short_fwd4": float("nan"),
        })
        return [
            {
                **base_row, "model": model_name,
                "base_rate": float("nan"), "roc_auc": float("nan"), "acc": float("nan"),
                "prec_top_decile": float("nan"), "prec_bottom_decile": float("nan"),
                "lift_long": float("nan"), "lift_short": float("nan"),
            }
            for model_name in _MODELS
        ]

    fwd1 = fwd_log_return(oos_slim, tf, 1).loc[pts.index]
    fwd4 = fwd_log_return(oos_slim, tf, 4).loc[pts.index]
    robustness = robustness_agreement(marked, fwd1, fwd4)
    base_row.update({
        "long_fwd1": robustness.get("long_fwd1"), "long_fwd4": robustness.get("long_fwd4"),
        "short_fwd1": robustness.get("short_fwd1"), "short_fwd4": robustness.get("short_fwd4"),
    })

    features_path = Path(train_base) / combo_info["features"]
    with open(features_path) as f:
        selected_features = json.load(f)

    X, y = feature_matrix(oos_slim, tf, side, horizon=horizon)
    X = _reproduce_transform(X, transform, selected_features)

    bundle_dir = Path(train_base) / combo_info["bundle_dir"]
    bundle = load_bundle(bundle_dir)  # FileNotFoundError naming missing file(s), propagated as-is
    feature_order = list(bundle["freeze"].stats.keys())

    missing = [c for c in feature_order if c not in X.columns]
    if missing:
        raise ValueError(
            f"run_oos: combo {combo_name!r}: feature(s) required by the frozen bundle could not "
            f"be reproduced on the OOS slim: {missing!r}"
        )
    X = X[feature_order]

    before = _freeze_json_bytes(bundle["freeze"])
    rows = []
    for model_name in _MODELS:
        m = eval_classifier(bundle, model_name, X, y)
        rows.append({
            **base_row, "model": model_name,
            "base_rate": m["base_rate"], "roc_auc": m["roc_auc"], "acc": m["acc"],
            "prec_top_decile": m["prec_top_decile"], "prec_bottom_decile": m["prec_bottom_decile"],
            "lift_long": m["lift_long"], "lift_short": m["lift_short"],
        })
    after = _freeze_json_bytes(bundle["freeze"])
    if before != after:
        raise AssertionError(
            f"run_oos: combo {combo_name!r}: FreezeStats bytes changed during scoring -- a refit "
            f"occurred where none should ever happen"
        )

    return rows


def run_oos(oos_slim: pd.DataFrame, train_base: str) -> pd.DataFrame:
    """Score the frozen best 2y iteration's artifacts (``{train_base}/
    best.json`` + per-combo bundles) against ``oos_slim`` with ZERO refit.
    See the module docstring point 3 for the full per-combo contract.
    """
    best_path = Path(train_base) / "best.json"
    if not best_path.exists():
        raise FileNotFoundError(f"run_oos: best.json not found at {best_path}")
    with open(best_path) as f:
        best = json.load(f)

    crosscheck_gates(oos_slim)  # raises RuntimeError on drift; see module docstring point 1

    transform = best["transform"]
    horizon = best["horizon"]

    rows: list = []
    for combo_name, combo_info in best["combos"].items():
        rows.extend(_score_combo(oos_slim, train_base, combo_name, combo_info, transform, horizon))

    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)


# ---------------------------------------------------------------------------
# write_oos_report
# ---------------------------------------------------------------------------


def _load_test_metrics(train_base: str, iter_dir_name: str | None, combo_name: str) -> dict:
    """Best-effort read of Layer 5's own persisted ``{combo}/metrics.json``
    -- a missing/unparsable file is not fatal to report-writing (defensive,
    mirrors tdlib.loop.write_iter_report's own "never crash a report over
    one bad cell" convention), just reported as "n/a" downstream."""
    if iter_dir_name is None:
        return {}
    metrics_path = Path(train_base) / iter_dir_name / combo_name / "metrics.json"
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_oos_report(table: pd.DataFrame, gates: dict, train_base: str, out_dir: str) -> str:
    """Write ``{out_dir}/oos_report.md`` (+ ``{out_dir}/oos_table.csv`` =
    ``table`` verbatim) and return the md path. See the module docstring
    point 4 for the verdict rule and why it is gbc-only.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "oos_report.md"

    table.to_csv(out / "oos_table.csv", index=False)

    best_path = Path(train_base) / "best.json"
    best: dict = {}
    iter_dir_name = None
    try:
        with open(best_path) as f:
            best = json.load(f)
        iter_dir_name = f"iter_{int(best['best_iter']):02d}"
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        best = {}
        iter_dir_name = None

    combo_names = list(dict.fromkeys(table["combo"]))  # first-seen order, de-duplicated

    lines = ["# OOS validation report (frozen artifacts, zero refit)", "", f"- train_base: {train_base}"]
    if best:
        lines.append(
            f"- best_iter: {best.get('best_iter')} "
            f"(transform={best.get('transform')}, horizon={best.get('horizon')})"
        )
    lines.append("")

    # --- crosscheck gate -----------------------------------------------------
    lines.append("## Crosscheck gate (cut-provenance drift check)")
    lines.append("")
    lines.append("| tf | status | fraction |")
    lines.append("|---|---|---|")
    for tf in sorted(gates):
        g = gates[tf]
        frac = g.get("fraction")
        lines.append(f"| tf={tf} | {g.get('status')} | {_fmt(frac) if frac is not None else 'n/a'} |")
    lines.append("")

    # --- per-combo metrics: test vs oos ---------------------------------------
    lines.append("## Per-combo metrics (test vs oos)")
    lines.append("")
    lines.append(
        "| combo | model | split | roc_auc | acc | base_rate | prec_top_decile | "
        "prec_bottom_decile | lift_long | lift_short | n |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for combo_name in combo_names:
        test_metrics = _load_test_metrics(train_base, iter_dir_name, combo_name)
        for model in _MODELS:
            tm = test_metrics.get(model, {}).get("test", {})
            lines.append(
                f"| {combo_name} | {model} | test | {_fmt(tm.get('roc_auc'))} | {_fmt(tm.get('acc'))} | "
                f"{_fmt(tm.get('base_rate'))} | {_fmt(tm.get('prec_top_decile'))} | "
                f"{_fmt(tm.get('prec_bottom_decile'))} | {_fmt(tm.get('lift_long'))} | "
                f"{_fmt(tm.get('lift_short'))} | {tm.get('n', 'n/a')} |"
            )
            oos_rows = table[(table["combo"] == combo_name) & (table["model"] == model)]
            if oos_rows.empty:
                continue
            om = oos_rows.iloc[0]
            lines.append(
                f"| {combo_name} | {model} | oos | {_fmt(om['roc_auc'])} | {_fmt(om['acc'])} | "
                f"{_fmt(om['base_rate'])} | {_fmt(om['prec_top_decile'])} | "
                f"{_fmt(om['prec_bottom_decile'])} | {_fmt(om['lift_long'])} | "
                f"{_fmt(om['lift_short'])} | {int(om['n'])} |"
            )
    lines.append("")

    # --- robustness (oos) -------------------------------------------------------
    lines.append("## Robustness (oos, truth vs realized forward return)")
    lines.append("")
    lines.append("| combo | long_fwd1 | long_fwd4 | short_fwd1 | short_fwd4 |")
    lines.append("|---|---|---|---|---|")
    for combo_name in combo_names:
        rows = table[table["combo"] == combo_name]
        if rows.empty:
            continue
        r = rows.iloc[0]
        lines.append(
            f"| {combo_name} | {_fmt(r['long_fwd1'])} | {_fmt(r['long_fwd4'])} | "
            f"{_fmt(r['short_fwd1'])} | {_fmt(r['short_fwd4'])} |"
        )
    lines.append("")

    # --- verdicts ------------------------------------------------------------------
    lines.append("## Verdicts")
    lines.append("")
    lines.append(
        f"Per combo, judged off the {_VERDICT_MODEL} model (this pipeline's own primary/"
        f"higher-capacity model -- see tdlib.loop.IterResult.mean_test_auc, which is "
        f"{_VERDICT_MODEL}-only): 'holds' iff oos roc_auc > 0.5 AND oos lift_long's sign matches "
        f"the frozen iteration's own test-split lift_long sign AND oos lift_short's sign matches "
        f"its own test-split lift_short sign -- BOTH signs, not lift_long alone (this is a "
        f"two-sided experiment; a side=-2 'dn' combo is not 'short-native', so judging it on "
        f"lift_short alone -- or either side alone -- would be arbitrary, not principled). Any "
        f"NaN among those four signed quantities -> 'fails'. 'skipped' if the oos slim had too "
        f"few marked points to score; 'fails' otherwise."
    )
    lines.append("")
    for combo_name in combo_names:
        rows = table[(table["combo"] == combo_name) & (table["model"] == _VERDICT_MODEL)]
        if rows.empty:
            lines.append(f"- {combo_name}: fails (no {_VERDICT_MODEL} row found)")
            continue
        row = rows.iloc[0]
        if bool(row["skipped"]):
            lines.append(f"- {combo_name}: skipped (n={int(row['n'])} < {_MIN_OOS_POINTS})")
            continue

        test_metrics = _load_test_metrics(train_base, iter_dir_name, combo_name)
        test_lift_long = test_metrics.get(_VERDICT_MODEL, {}).get("test", {}).get("lift_long")
        test_lift_short = test_metrics.get(_VERDICT_MODEL, {}).get("test", {}).get("lift_short")
        oos_auc = row["roc_auc"]
        oos_lift_long = row["lift_long"]
        oos_lift_short = row["lift_short"]

        signed_values = (test_lift_long, test_lift_short, oos_auc, oos_lift_long, oos_lift_short)
        verdict = "fails"
        if not any(pd.isna(v) for v in signed_values):
            if (
                oos_auc > 0.5
                and (oos_lift_long > 0) == (test_lift_long > 0)
                and (oos_lift_short > 0) == (test_lift_short > 0)
            ):
                verdict = "holds"

        lines.append(
            f"- {combo_name}: {verdict} (oos_auc={_fmt(oos_auc)}, "
            f"oos_lift_long={_fmt(oos_lift_long)}, "
            f"test_lift_long={_fmt(test_lift_long) if test_lift_long is not None else 'n/a'}, "
            f"oos_lift_short={_fmt(oos_lift_short)}, "
            f"test_lift_short={_fmt(test_lift_short) if test_lift_short is not None else 'n/a'})"
        )
    lines.append("")

    report_path.write_text("\n".join(lines))
    return str(report_path)
