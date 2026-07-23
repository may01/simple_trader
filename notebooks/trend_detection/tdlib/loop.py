"""tdlib/loop.py -- Task 5: the autonomous improvement loop (Layer 5).

This is the experiment orchestrator: it runs the whole hypothesis -> screen
-> classify -> report -> improve cycle over every (tf, side) combo in
``tdlib.config.ANALYSIS_TFS x SIDES``, iteration by iteration, and freezes
the winning iteration's artifacts (bundles + selected features) in a form
Layer 6 (frozen OOS validation) can load with zero refitting.

Five things live here:

1. ``IterConfig`` / ``ComboResult`` / ``IterResult`` -- the three small
   dataclasses that carry one iteration's configuration and results.
   ``IterResult`` carries three bookkeeping fields beyond the brief's own
   ``cfg``/``combos`` pair (``kept``, ``best_before``, ``eps_auc``, all
   defaulted) -- ``write_iter_report`` takes only ``(res, base_dir)`` per its
   spec'd signature, so the kept/rejected verdict and the auc delta it
   reports have nowhere else to live except on ``res`` itself.
   ``importance_top`` is a top-40 slice (not top-10): it doubles as the
   in-memory payload ``improvement_loop`` feeds forward as next iteration's
   ``importance_prev`` (see ``apply_transform``'s "prune_top40" -- top 40 is
   the largest number this spec ever needs from it), while a FULL,
   untruncated ranking always goes to ``importance.csv`` on disk. Report
   tables truncate it further to 10 rows at write time. It is always
   computed on the TRAIN split (see ``run_iteration``) -- ranking on test
   would let feature selection see the very rows ``mean_test_auc`` is later
   judged on, since ``chrono_split`` is a deterministic positional split.

2. ``chrono_split`` / ``apply_transform`` -- the two small pure-data-shaping
   helpers ``run_iteration`` composes each combo through: a positional
   (never shuffled) 70/30 train/test split, then one of four named column
   transforms.

3. ``run_iteration`` -- one iteration's full combo loop: point selection,
   truth marking, robustness scoring, feature assembly, screening, model
   fitting/eval/importance, and on-disk persistence per combo.

4. ``improvement_loop`` -- drives iterations 1..max_iters, applying
   ``IMPROVEMENTS[iter_no - 1]`` from iteration 2 on (iteration 1 is always
   "baseline"), keeping an iteration only when its mean gbc test AUC beats
   the best-so-far by more than ``eps_auc``, and stopping on two consecutive
   rejections, transform exhaustion, or ``max_iters``.

5. ``write_iter_report`` / ``write_summary`` -- markdown (+ csv/json)
   reporting. Neither ever reads the wall clock -- no wall-clock-derived
   content appears anywhere in this module (checked by a dedicated test) --
   two runs over the same input must be able to produce byte-identical
   output.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tdlib.config import ANALYSIS_TFS, artifacts_dir
from tdlib.features import feature_matrix
from tdlib.models import eval_classifier, fit_classifiers, importance_table, save_bundle
from tdlib.points import strong_points
from tdlib.screen import screen_charts, univariate_screen
from tdlib.truth import fwd_log_return, mark_truth, robustness_agreement, truth_counts

logger = logging.getLogger(__name__)

# Strong-point side codes this experiment classifies (tdlib.points'
# +2/-2 "strong move" extremes) -- every combo is one of ANALYSIS_TFS x SIDES.
SIDES = [2, -2]

# Iteration k (1-indexed) applies IMPROVEMENTS[k - 1]; iteration 1 is always
# "baseline" (IMPROVEMENTS[0]). Order matters -- this is a fixed improvement
# SCHEDULE, not a pool to pick from.
IMPROVEMENTS = ["baseline", "prune_top40", "interact_time_left", "horizon_n2"]

# A combo needs at least this many usable (long or short) marked points to be
# worth screening/fitting at all -- task-5-brief.md's exact floor. Public (no
# leading underscore): tdlib.diag (T7) reuses this SAME threshold ("skip
# combo if n<60 like loop") rather than a second hardcoded 60 that could
# silently drift from this one.
MIN_COMBO_POINTS = 60

# apply_transform's "prune_top40" / "interact_time_left" cutoffs, and the
# size of the in-memory importance slice ComboResult.importance_top carries
# forward as next iteration's importance_prev (see module docstring point 1
# -- 40 is the largest of the two, so one constant serves both transforms).
_PRUNE_TOP_N = 40
_INTERACT_TOP_N = 8

# Top-N rows kept in each combo's in-memory screen_top field (write_iter_report's
# own "top-10 screen table" -- screen_top has no OTHER consumer, unlike
# importance_top, so this is exactly the report's own display size).
_SCREEN_TOP_N = 10

# write_iter_report / write_summary's own markdown table row cap, applied
# defensively at write time regardless of how large the in-memory
# screen_top/importance_top fields happen to be.
_REPORT_TOP_N = 10

_TIME_LEFT_RE = re.compile(r"^time_left_(\d+)$")


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IterConfig:
    iter_no: int
    transform: str
    horizon: str = "n1"
    notes: str = ""


@dataclass
class ComboResult:
    tf: int
    side: int
    counts: dict
    robustness: dict
    n_features: int
    screen_top: pd.DataFrame
    metrics: dict
    importance_top: pd.DataFrame | None
    chart_paths: list
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class IterResult:
    cfg: IterConfig
    combos: dict  # {(tf, side): ComboResult}
    # Bookkeeping improvement_loop fills in AFTER scoring this iteration (see
    # module docstring point 1) -- a freshly-constructed IterResult (e.g. in
    # a test) defaults to "not yet judged", never crashes mean_test_auc/
    # write_iter_report.
    kept: bool = False
    best_before: float = float("-inf")
    eps_auc: float = 0.005

    @property
    def mean_test_auc(self) -> float:
        """Mean gbc test roc_auc over non-skipped combos, NaN-safe: a
        combo's own roc_auc can individually be NaN (single-class y_te --
        see eval_classifier) without poisoning the whole mean, and skipped
        combos never enter the average at all. Empty/all-NaN input ->
        NaN (nothing to average), never a crash or a fabricated 0.0.
        """
        values = []
        for combo in self.combos.values():
            if combo.skipped:
                continue
            auc = combo.metrics.get("gbc", {}).get("test", {}).get("roc_auc")
            if auc is None or (isinstance(auc, float) and math.isnan(auc)):
                continue
            values.append(float(auc))
        if not values:
            return float("nan")
        return float(sum(values) / len(values))


# ---------------------------------------------------------------------------
# combo naming -- shared by run_iteration / write_iter_report / write_summary
# AND by tdlib.diag (T7) -- public (no leading underscore) so the diagnostic
# layer's combo directories/table rows use the IDENTICAL "{tf}_up"/"{tf}_dn"
# naming as the loop's own, rather than a second hand-rolled copy that could
# drift from it.
# ---------------------------------------------------------------------------


def combo_name(tf: int, side: int) -> str:
    """``f"{tf}_up"``/``f"{tf}_dn"`` -- the on-disk combo directory name AND
    best.json's combo key, verbatim per task-5/6 briefs."""
    return f"{tf}_{'up' if side > 0 else 'dn'}"


# ---------------------------------------------------------------------------
# chrono_split
# ---------------------------------------------------------------------------


def chrono_split(X: pd.DataFrame, y: pd.Series, frac: float = 0.7) -> tuple:
    """Positional (NOT shuffled) split: the first ``frac`` fraction of rows
    (by ROW POSITION, not any label/value) is train, the rest is test.
    ``X``/``y`` are assumed already in chronological order (true of every
    ``feature_matrix`` output -- see its own docstring), so this is a plain
    ``iloc`` slice, no sorting.

    Returns ``(X_tr, X_te, y_tr, y_te)``.
    """
    n = len(X)
    split = int(n * frac)
    X_tr = X.iloc[:split]
    X_te = X.iloc[split:]
    y_tr = y.iloc[:split]
    y_te = y.iloc[split:]
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------


def _lowest_time_left_col(columns) -> str | None:
    """Among ``columns``, the ``time_left_{htf}`` column with the SMALLEST
    htf -- e.g. ``time_left_60`` for a tf=15 combo (``HIGHER_TF[15] =
    [60, 240, 1440]``), ``time_left_240`` for tf=60, ``time_left_1440`` for
    tf=240. Read straight off the column names rather than threaded through
    as an explicit tf/htf argument -- ``apply_transform``'s signature has no
    room for one (task-5-brief.md's interface is exact), and every combo's
    own ``time_left_*`` columns already encode which htfs are in play.
    Returns None if no ``time_left_*`` column is present at all.
    """
    htfs = []
    for col in columns:
        m = _TIME_LEFT_RE.match(col)
        if m:
            htfs.append(int(m.group(1)))
    if not htfs:
        return None
    return f"time_left_{min(htfs)}"


def apply_transform(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    transform: str,
    importance_prev: pd.DataFrame | None = None,
) -> tuple:
    """Apply one named column transform to a (train, test) pair, returning
    the (possibly) reshaped ``(X_tr, X_te)``.

    - "baseline" / "horizon_n2": columns unchanged (horizon_n2's own effect
      is entirely in ``run_iteration`` passing ``horizon="n2"`` to
      ``feature_matrix`` -- nothing left for this function to do to X).
    - "prune_top40": keep the top ``_PRUNE_TOP_N`` (40) columns of
      ``importance_prev`` by ``imp_mean`` descending (re-sorted defensively
      here -- callers may hand in an already-sorted ``importance_table``
      output, or a hand-built test DataFrame that isn't). Fewer than 40
      available (in ``importance_prev`` OR still present in ``X_tr``) ->
      keep all of them. No ``importance_prev`` (or an empty one) -> X
      unchanged (nothing to prune by).
    - "interact_time_left": for the top ``_INTERACT_TOP_N`` (8) features of
      ``importance_prev`` EXCLUDING any ``time_*`` column itself, add
      ``f"{col}__x__{time_left_col}"`` = ``X[col] * X[time_left_col]`` on
      train and test SEPARATELY (each from its own columns only -- no
      cross-split statistic, so this can never leak test into train or vice
      versa), where ``time_left_col`` is the LOWEST-htf ``time_left_*``
      column present (see ``_lowest_time_left_col``). No ``time_left_*``
      column present, or no ``importance_prev`` -> X unchanged.

    Any other ``transform`` string raises ``ValueError`` naming it.
    """
    if transform in ("baseline", "horizon_n2"):
        return X_tr, X_te

    if transform == "prune_top40":
        if importance_prev is None or len(importance_prev) == 0:
            return X_tr, X_te
        ranked = importance_prev.sort_values("imp_mean", ascending=False)["feature"].tolist()
        top_cols = [c for c in ranked if c in X_tr.columns][:_PRUNE_TOP_N]
        return X_tr[top_cols], X_te[top_cols]

    if transform == "interact_time_left":
        time_left_col = _lowest_time_left_col(X_tr.columns)
        if time_left_col is None or importance_prev is None or len(importance_prev) == 0:
            return X_tr, X_te

        ranked = importance_prev.sort_values("imp_mean", ascending=False)["feature"].tolist()
        base_cols = [c for c in ranked if c in X_tr.columns and not c.startswith("time_")][:_INTERACT_TOP_N]

        X_tr_new = X_tr.copy()
        X_te_new = X_te.copy()
        for col in base_cols:
            new_col = f"{col}__x__{time_left_col}"
            X_tr_new[new_col] = X_tr[col] * X_tr[time_left_col]
            X_te_new[new_col] = X_te[col] * X_te[time_left_col]
        return X_tr_new, X_te_new

    raise ValueError(f"apply_transform: unknown transform {transform!r}")


# ---------------------------------------------------------------------------
# run_iteration
# ---------------------------------------------------------------------------


def _skipped_combo(tf: int, side: int, counts: dict) -> ComboResult:
    return ComboResult(
        tf=tf, side=side, counts=counts, robustness={}, n_features=0,
        screen_top=pd.DataFrame(), metrics={}, importance_top=None,
        chart_paths=[], skipped=True, skip_reason=f"points<{MIN_COMBO_POINTS}",
    )


def run_iteration(cfg: IterConfig, slim: pd.DataFrame, importance_prev: dict | None = None) -> IterResult:
    """One iteration's full pass over every (tf, side) combo.

    Per combo: strong-point selection + truth marking + skip check (fewer
    than ``MIN_COMBO_POINTS`` usable long/short points -> skipped, nothing
    else runs for it) -> robustness scoring -> feature assembly -> chrono
    split -> ``cfg.transform`` applied -> train-only screening (+ top-15
    charts) -> classifier fit/eval -> gbc permutation importance, computed
    on the TRAIN split (X_tr/y_tr) -- NEVER the test split, since this
    importance feeds next iteration's feature selection and chrono_split is
    deterministic; ranking on X_te would let feature selection see exactly
    the rows mean_test_auc is later judged on -- persist everything to
    ``{artifacts_dir()}/iter_{cfg.iter_no:02d}/{tf}_{up,dn}/``.

    ``importance_prev`` is the dict ``apply_transform``'s "prune_top40"/
    "interact_time_left" transforms read from -- keyed ``(tf, side)``, each
    value the corresponding PREVIOUS KEPT iteration's ``importance_top``
    (see ``improvement_loop``). Absent/None is treated as "nothing to prune/
    interact by yet" (safe on iteration 1, where no previous iteration
    exists).
    """
    importance_prev = importance_prev or {}
    base = artifacts_dir()
    iter_dir_name = f"iter_{cfg.iter_no:02d}"
    combos: dict = {}

    for tf in ANALYSIS_TFS:
        for side in SIDES:
            combo_key = (tf, side)
            # NOTE: named "name", not "combo_name" -- this function body also
            # assigns the module-level combo_name() FUNCTION result here, and
            # a local variable of the same name as a module-level function
            # called on the same line would make Python treat that name as
            # local for the whole function (UnboundLocalError on this exact
            # line). "name" avoids the shadowing entirely.
            name = combo_name(tf, side)

            pts = strong_points(slim, tf, side)
            marked = mark_truth(pts, tf, cfg.horizon)
            counts = truth_counts(marked)

            if counts["long"] + counts["short"] < MIN_COMBO_POINTS:
                logger.info(
                    "iter=%d transform=%s tf=%d side=%d n_long=%d n_short=%d gbc_test_auc=skip (%s)",
                    cfg.iter_no, cfg.transform, tf, side, counts["long"], counts["short"],
                    f"points<{MIN_COMBO_POINTS}",
                )
                combos[combo_key] = _skipped_combo(tf, side, counts)
                continue

            fwd1 = fwd_log_return(slim, tf, 1).loc[pts.index]
            fwd4 = fwd_log_return(slim, tf, 4).loc[pts.index]
            robustness = robustness_agreement(marked, fwd1, fwd4)

            X, y = feature_matrix(slim, tf, side, horizon=cfg.horizon)
            X_tr, X_te, y_tr, y_te = chrono_split(X, y)
            X_tr, X_te = apply_transform(X_tr, X_te, cfg.transform, importance_prev.get(combo_key))

            scr = univariate_screen(X_tr, y_tr)
            combo_dir = Path(base) / iter_dir_name / name
            combo_dir.mkdir(parents=True, exist_ok=True)
            chart_paths = screen_charts(scr, X_tr, y_tr, combo_dir / "charts", top_k=15)

            bundle = fit_classifiers(X_tr, y_tr)
            metrics = {
                name: {
                    "train": eval_classifier(bundle, name, X_tr, y_tr),
                    "test": eval_classifier(bundle, name, X_te, y_te),
                }
                for name in ("logistic", "gbc")
            }

            # TRAIN-side, deliberately -- never X_te/y_te. This table feeds
            # importance_prev (apply_transform's "prune_top40"/
            # "interact_time_left" for the NEXT iteration), gets persisted
            # to importance.csv, and is sliced into ComboResult.importance_top.
            # chrono_split is a DETERMINISTIC positional split, so a later
            # iteration's X_te is (row-for-row, when horizon is unchanged)
            # the SAME test rows this iteration's X_te would be -- ranking
            # features on X_te here would let feature selection peek at
            # exactly the rows mean_test_auc is later judged on, inflating
            # apparent iteration-over-iteration gains. Train-side importance
            # has no such channel back into the test-set metric.
            imp = importance_table(bundle, X_tr, y_tr)

            scr.to_csv(combo_dir / "screen.csv")
            with open(combo_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, sort_keys=True, indent=2)
            imp.to_csv(combo_dir / "importance.csv", index=False)
            with open(combo_dir / "selected_features.json", "w") as f:
                json.dump(list(X_tr.columns), f)
            save_bundle(bundle, combo_dir / "bundle")

            gbc_test_auc = metrics["gbc"]["test"]["roc_auc"]
            logger.info(
                "iter=%d transform=%s tf=%d side=%d n_long=%d n_short=%d gbc_test_auc=%.4f",
                cfg.iter_no, cfg.transform, tf, side, counts["long"], counts["short"], gbc_test_auc,
            )

            combos[combo_key] = ComboResult(
                tf=tf, side=side, counts=counts, robustness=robustness, n_features=X_tr.shape[1],
                screen_top=scr.head(_SCREEN_TOP_N), metrics=metrics, importance_top=imp.head(_PRUNE_TOP_N),
                chart_paths=chart_paths, skipped=False, skip_reason="",
            )

    return IterResult(cfg=cfg, combos=combos)


# ---------------------------------------------------------------------------
# improvement_loop
# ---------------------------------------------------------------------------


def improvement_loop(slim: pd.DataFrame, max_iters: int = 4, eps_auc: float = 0.005) -> list:
    """Drive iterations 1..``max_iters``, one ``run_iteration`` call each.

    Iteration 1 is always ``IMPROVEMENTS[0]`` ("baseline"); iteration k
    (k >= 2) applies ``IMPROVEMENTS[k - 1]`` (horizon_n2's own cfg carries
    ``horizon="n2"``, every other transform keeps ``horizon="n1"``).

    Keep rule: an iteration is KEPT iff its ``mean_test_auc`` is finite AND
    exceeds the best KEPT mean_test_auc seen so far by more than
    ``eps_auc``. A kept iteration becomes the new best and its per-combo
    ``importance_top`` feeds ``importance_prev`` for every later iteration
    (until a LATER iteration is itself kept, replacing it) -- a rejected
    iteration changes neither.

    Stops (after fully processing and reporting the triggering iteration)
    on: two consecutive rejections, the ``IMPROVEMENTS`` schedule running
    out, or ``max_iters`` reached. EVERY iteration -- kept or rejected -- is
    fully persisted (via ``run_iteration``) and reported (via
    ``write_iter_report``); only the FINAL ``write_summary`` call depends on
    how the loop ended.

    Returns the list of ``IterResult`` (one per iteration actually run, in
    order).
    """
    base = artifacts_dir()
    all_iters: list = []
    best_auc = float("-inf")
    importance_prev: dict = {}
    consecutive_rejections = 0

    for iter_no in range(1, max_iters + 1):
        if iter_no - 1 >= len(IMPROVEMENTS):
            break  # transforms exhausted

        transform = IMPROVEMENTS[iter_no - 1]
        horizon = "n2" if transform == "horizon_n2" else "n1"
        cfg = IterConfig(iter_no=iter_no, transform=transform, horizon=horizon)

        result = run_iteration(cfg, slim, importance_prev)

        mean_auc = result.mean_test_auc
        kept = (not math.isnan(mean_auc)) and (mean_auc > best_auc + eps_auc)

        result.kept = kept
        result.best_before = best_auc
        result.eps_auc = eps_auc

        all_iters.append(result)
        write_iter_report(result, base)

        if kept:
            best_auc = mean_auc
            importance_prev = {
                key: combo.importance_top
                for key, combo in result.combos.items()
                if not combo.skipped and combo.importance_top is not None
            }
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1

        if consecutive_rejections >= 2:
            break

    write_summary(all_iters, base)
    return all_iters


# ---------------------------------------------------------------------------
# write_iter_report
# ---------------------------------------------------------------------------


def _fmt(value, spec: str = ".4f") -> str:
    """Format a possibly-missing/NaN numeric value for a markdown cell --
    never raises on None/NaN/non-numeric, so a report never fails to write
    over one bad cell."""
    if value is None:
        return "nan"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def write_iter_report(res: IterResult, base_dir: str) -> str:
    """Write ``{base_dir}/iter_{res.cfg.iter_no:02d}/report.md`` and return
    its path. Deterministic given ``res`` -- no wall-clock read anywhere in
    this function or this module (see the module docstring's point 5).

    Sections: header (iter/transform/horizon), per-combo counts table,
    robustness table, metrics table (per combo x {logistic,gbc} x
    {train,test}), top-10 screen table per combo, top-10 importance per
    combo, relative chart paths, and a final verdict line (kept/rejected,
    ``mean_test_auc``, ``best_before``, and the delta between them).
    """
    iter_dir = Path(base_dir) / f"iter_{res.cfg.iter_no:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    report_path = iter_dir / "report.md"

    lines = [
        f"# Iteration {res.cfg.iter_no:02d} -- {res.cfg.transform}",
        "",
        f"- transform: {res.cfg.transform}",
        f"- horizon: {res.cfg.horizon}",
    ]
    if res.cfg.notes:
        lines.append(f"- notes: {res.cfg.notes}")
    lines.append("")

    # --- counts ------------------------------------------------------------
    lines.append("## Counts")
    lines.append("")
    lines.append("| combo | tf | side | n_long | n_short | both | neither | nan | skipped |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for (tf, side), combo in res.combos.items():
        name = combo_name(tf, side)
        c = combo.counts
        lines.append(
            f"| {name} | {tf} | {side} | {c.get('long', 0)} | {c.get('short', 0)} | "
            f"{c.get('both', 0)} | {c.get('neither', 0)} | {c.get('nan', 0)} | {combo.skipped} |"
        )
    lines.append("")

    # --- robustness ----------------------------------------------------------
    lines.append("## Robustness (truth vs realized forward return)")
    lines.append("")
    lines.append("| combo | long_fwd1 | long_fwd4 | short_fwd1 | short_fwd4 |")
    lines.append("|---|---|---|---|---|")
    for (tf, side), combo in res.combos.items():
        if combo.skipped:
            continue
        name = combo_name(tf, side)
        r = combo.robustness
        lines.append(
            f"| {name} | {_fmt(r.get('long_fwd1'))} | {_fmt(r.get('long_fwd4'))} | "
            f"{_fmt(r.get('short_fwd1'))} | {_fmt(r.get('short_fwd4'))} |"
        )
    lines.append("")

    # --- metrics -------------------------------------------------------------
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| combo | model | split | roc_auc | acc | base_rate | prec_top_decile | "
        "prec_bottom_decile | lift_long | lift_short | n |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (tf, side), combo in res.combos.items():
        if combo.skipped:
            continue
        name = combo_name(tf, side)
        for model in ("logistic", "gbc"):
            for split in ("train", "test"):
                m = combo.metrics.get(model, {}).get(split, {})
                lines.append(
                    f"| {name} | {model} | {split} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('acc'))} | "
                    f"{_fmt(m.get('base_rate'))} | {_fmt(m.get('prec_top_decile'))} | "
                    f"{_fmt(m.get('prec_bottom_decile'))} | {_fmt(m.get('lift_long'))} | "
                    f"{_fmt(m.get('lift_short'))} | {m.get('n', 0)} |"
                )
    lines.append("")

    # --- top screened features ------------------------------------------------
    lines.append("## Top screened features")
    lines.append("")
    for (tf, side), combo in res.combos.items():
        if combo.skipped:
            continue
        name = combo_name(tf, side)
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| feature | auc | auc_low95 | ks |")
        lines.append("|---|---|---|---|")
        for feature, row in combo.screen_top.head(_REPORT_TOP_N).iterrows():
            lines.append(f"| {feature} | {_fmt(row.get('auc'))} | {_fmt(row.get('auc_low95'))} | {_fmt(row.get('ks_stat'))} |")
        lines.append("")

    # --- top importance --------------------------------------------------------
    lines.append("## Top importance")
    lines.append("")
    for (tf, side), combo in res.combos.items():
        if combo.skipped or combo.importance_top is None:
            continue
        name = combo_name(tf, side)
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| feature | imp_mean | imp_std |")
        lines.append("|---|---|---|")
        for _, row in combo.importance_top.head(_REPORT_TOP_N).iterrows():
            lines.append(f"| {row.get('feature')} | {_fmt(row.get('imp_mean'))} | {_fmt(row.get('imp_std'))} |")
        lines.append("")

    # --- charts ------------------------------------------------------------------
    lines.append("## Charts")
    lines.append("")
    for (tf, side), combo in res.combos.items():
        if combo.skipped or not combo.chart_paths:
            continue
        name = combo_name(tf, side)
        lines.append(f"### {name}")
        for p in combo.chart_paths:
            try:
                rel = Path(p).relative_to(iter_dir)
            except ValueError:
                rel = Path(p)  # not actually nested under iter_dir -- show as-is
            lines.append(f"- {rel}")
        lines.append("")

    # --- verdict -------------------------------------------------------------------
    mean_auc = res.mean_test_auc
    delta = mean_auc - res.best_before if math.isfinite(res.best_before) else float("nan")
    verdict = "KEPT" if res.kept else "REJECTED"
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"{verdict}: mean_test_auc={_fmt(mean_auc)}, best_before={_fmt(res.best_before)}, "
        f"delta={_fmt(delta, '+.4f')} (eps_auc={_fmt(res.eps_auc)})"
    )
    lines.append("")

    report_path.write_text("\n".join(lines))
    return str(report_path)


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------


def select_best(all_iters: list):
    """The best iteration by ``mean_test_auc`` among KEPT iterations -- since
    the keep rule only ever raises the bar (each kept iteration's
    mean_test_auc strictly exceeds every earlier kept one by more than
    eps_auc), this is always the LAST kept iteration, but computed via `max`
    rather than assumed, so it stays correct even if that invariant ever
    changes. Falls back to the last iteration overall if literally nothing
    was ever kept (should not happen in practice -- iteration 1 vs the
    initial best_auc=-inf keeps unless every combo skipped/NaN'd).

    Public (no leading underscore): ``write_summary`` uses this to build
    ``best.json``, and ``run_loop.py``'s console summary reuses the SAME
    function -- so the two can never disagree about which iteration is
    "best," including in the all-rejected fallback case.
    """
    kept_iters = [r for r in all_iters if r.kept]
    if kept_iters:
        return max(kept_iters, key=lambda r: r.mean_test_auc)
    return all_iters[-1] if all_iters else None


def write_summary(all_iters: list, base_dir: str) -> str:
    """Write ``{base_dir}/summary.md`` and ``{base_dir}/best.json``; return
    the summary.md path.

    ``summary.md``: iteration history table (iter, transform, horizon,
    mean_test_auc, kept/rejected) + the best iteration's per-combo test
    metrics.

    ``best.json``: ``{"best_iter", "transform", "horizon", "mean_test_auc",
    "combos": {"{tf}_{up,dn}": {"bundle_dir", "features", "n_test"}}}`` --
    ``bundle_dir``/``features`` are paths RELATIVE to ``base_dir`` (e.g.
    ``"iter_01/15_up/bundle"``), exactly what Layer 6 / a host-side copy
    need: no absolute path baked in that would break the moment these
    artifacts are copied anywhere else.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    best = select_best(all_iters)

    lines = ["# Improvement loop summary", "", "## Iteration history", ""]
    lines.append("| iter | transform | horizon | mean_test_auc | status |")
    lines.append("|---|---|---|---|---|")
    for r in all_iters:
        status = "kept" if r.kept else "rejected"
        lines.append(f"| {r.cfg.iter_no:02d} | {r.cfg.transform} | {r.cfg.horizon} | {_fmt(r.mean_test_auc)} | {status} |")
    lines.append("")

    combos_json: dict = {}
    if best is not None:
        lines.append(f"## Best iteration: {best.cfg.iter_no:02d} ({best.cfg.transform}, horizon={best.cfg.horizon})")
        lines.append("")
        lines.append("| combo | model | roc_auc | acc | lift_long | lift_short | n |")
        lines.append("|---|---|---|---|---|---|---|")
        iter_dir_name = f"iter_{best.cfg.iter_no:02d}"
        for (tf, side), combo in best.combos.items():
            if combo.skipped:
                continue
            name = combo_name(tf, side)
            for model in ("logistic", "gbc"):
                m = combo.metrics.get(model, {}).get("test", {})
                lines.append(
                    f"| {name} | {model} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('acc'))} | "
                    f"{_fmt(m.get('lift_long'))} | {_fmt(m.get('lift_short'))} | {m.get('n', 0)} |"
                )
            combos_json[name] = {
                "bundle_dir": f"{iter_dir_name}/{name}/bundle",
                "features": f"{iter_dir_name}/{name}/selected_features.json",
                "n_test": int(combo.metrics.get("gbc", {}).get("test", {}).get("n", 0)),
            }
        lines.append("")

    summary_path = base / "summary.md"
    summary_path.write_text("\n".join(lines))

    best_json = {
        "best_iter": best.cfg.iter_no if best is not None else None,
        "transform": best.cfg.transform if best is not None else None,
        "horizon": best.cfg.horizon if best is not None else None,
        "mean_test_auc": best.mean_test_auc if best is not None else float("nan"),
        "combos": combos_json,
    }
    with open(base / "best.json", "w") as f:
        json.dump(best_json, f, sort_keys=True, indent=2)

    return str(summary_path)
