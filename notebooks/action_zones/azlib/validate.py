"""azlib/validate.py — Task 8: train/OOS validation harness (Layer 8).

Implements design spec §8 (external/docs/superpowers/specs/2026-07-18-zone-
selection-design.md) and task-8-brief.md: the full-pipeline orchestration
layer that glues Tasks 1-7 together end to end —

    run_train(train_env, tf, direction) -> results_path
        add labels -> fit+FREEZE indicator stats -> fit a regression per
        indicator attribute group -> fuse (inverse-variance) -> sweep+select
        Y -> select R/R levels (``first_touch_rr_grid``, Fix 2 -- see that
        function's own docstring and ``.superpowers/sdd/task-fix2-
        report.md``; REPLACES the earlier ``azlib.zones.build_rr_levels``/
        ``select_levels_safe`` marginal-reach-probability selection, which
        remains intact and importable but is no longer on this pipeline's
        critical path) -> write a ResultsFile + its sidecar artifacts.
        Returns the ResultsFile's path.

    run_oos(results_path, oos_env) -> zoned OOS df
        loads the FROZEN ResultsFile + sidecars, applies them to the OOS
        wide df (attribute z-scoring, per-group predict, fusion, Y, zone
        marking) with NO refit of anything — never calls fit_stats or
        fit_regression.

    metrics(zoned_df, tf, direction, label_params) -> dict
        {"strict_coverage", "non_strict_coverage", "realized_rr",
        "reach_drift_max"} — design spec §8's own success metrics
        (maximize strict_coverage AND require profitable realized_rr;
        SELECTION already happened in Layers 5-6, this only REPORTS).

Train-only freeze / no leakage (critical, task-8-brief.md's own top
constraint)
----------------------------------------------------------------------------
Every statistic `run_train` fits (`azlib.indicators.fit_stats`'s per-
(tf,indicator,attr) mean/std, each group's `azlib.models.fit_regression`,
the selected Y and (tgt_x, sl_x)) is fit on TRAIN and serialized to disk
(the `ResultsFile` itself, plus sidecar artifacts alongside it — see
"Sidecar artifacts" below). `run_oos` loads those frozen artifacts back
(`FreezeStats.from_json`, `azlib.models.load_result`, the `ResultsFile`'s
own `y`/`tgt_x`/`sl_x` fields, a `levels.json`/`reach_train.json` sidecar
pair) and NEVER calls `fit_stats`/`fit_regression`/`sweep_y`/`select_y`/
`first_touch_rr_grid` itself — every one of those "fit on this data"
functions appears ONLY inside `run_train`. `tests/test_layer8_validate.py` proves this
directly: `run_train` builds a real results path first (unpatched), THEN
`fit_stats`/`fit_regression` are monkeypatched to raise, THEN `run_oos` is
called on that same results path and must still succeed.

`azlib.space.price_levels` is deliberately called UNFROZEN on both train and
OOS (no stats to freeze there — it is a pure, live function of a frame's own
OHLC columns, not a fitted statistic; see that module's own docstring). This
is intentional, not a leakage gap: the "frozen" artifacts are specifically
the ones Task 3/4/5/6 themselves call "fit"/"freeze" (indicator z-score
stats, per-group regressions, the selected Y and R/R levels).

Sidecar artifacts (task-8-brief.md: "save those as sidecar artifacts next to
the ResultsFile ... reference them by path in the ResultsFile fields
(freeze_stats_path, reg_models). Do NOT change ResultsFile's field names.")
----------------------------------------------------------------------------
`ResultsFile`'s own fields carry everything Task 7 defined; this module adds
exactly THREE more files, all written into that SAME directory
(`os.path.dirname(freeze_stats_path)` — a fixed, documented filename
convention, so nothing beyond the existing `freeze_stats_path` field is
needed to locate them):

  - `stats.json`      -- `freeze_stats_path` itself (Task 3's FreezeStats).
  - `<indicator>_<attr>_linear.json` (+ sibling `.joblib`, Task 4's own
    `save_result` convention) -- one per fitted single-attribute regression
    group; `reg_models` is the literal list of these JSON paths (Task 7's
    own docstring only requires "enough for a later re-inference run to
    know which saved RegResult files to reload" — a full path is strictly
    more useful for that than a bare `"rsi:linear"`-style identifier, and
    is still just a `str`, so `reg_models: list[str]` is unchanged). The
    filename's OWN 3 underscore-separated parts (`indicator`, `attr`,
    `kind`) are how `run_oos` recovers which attribute column + kind each
    saved model needs — see `_reg_model_key`.
  - `levels.json`      -- the FULL `{"tgt_x","sl_x","rr","exp_ret"}` dict
    `first_touch_rr_grid` returned (Fix 2 -- same 4-key shape
    `build_rr_levels`/`select_levels_safe` always returned; not just the 2
    numbers `ResultsFile.tgt_x`/`sl_x` keep) — `run_oos` needs the whole
    dict unchanged to hand straight to `azlib.zones.build_zoned_dataset`
    (which also reads `levels["rr"]` to detect the no-profitable-levels
    sentinel, a REUSED Task 7 behavior — see below).
  - `reach_train.json`  -- the two frozen TRAIN extreme-diff arrays
    (`{tf}_high_diff_prc`, negated `{tf}_low_diff_prc` — the exact
    `azlib.rr` sign convention `azlib.zones.build_rr_levels` itself uses,
    still used here for `run_oos`'s own `reach_drift_max` sanity check —
    see below; unrelated to Fix 2's own first-touch selection) needed to
    rebuild the SAME `reach_prob_estimator` callables `run_oos` computes
    `reach_drift_max` against, ONCE, right there (`azlib.rr
    .reach_freq_drift` takes a CALLABLE, not a precomputed array —
    reconstructing it from its own frozen input is the only way to get
    that callable back without re-fitting it on OOS). The resulting scalar
    is written onto the returned zoned df as an ordinary broadcast column,
    `az_reach_drift_max` — NOT recomputed later by `metrics()` from
    whatever rows/columns happen to still be present on its own `zoned_df`
    argument (see `run_oos`'s own docstring for why: a caller-filtered
    `zoned_df`, e.g. a Task 9 notebook slicing to a date range, must not
    silently get a different — or a bare, easily-confused-with-"no zoned
    entries" — drift number).

Reuse (task-8-brief.md: "REUSE select_levels_safe + build_rr_levels ... DO
NOT duplicate that logic in validate.py"; superseded by Fix 2 for the R/R
SELECTION itself — see below)
----------------------------------------------------------------------------
`run_train` calls THIS module's own `first_touch_rr_grid` exactly once
(Fix 2, ".superpowers/sdd/task-fix2-report.md") to get the R/R `levels`
dict — `azlib.zones.build_rr_levels`/`select_levels_safe`'s marginal-
reach-probability selection is no longer on this path (it degenerates to a
near-target/far-stop mirage, see `first_touch_rr_grid`'s own docstring),
though both functions remain intact/importable for other callers/tests.
`azlib.zones.build_zoned_dataset` (which recognizes `first_touch_rr_grid`'s
own NaN sentinel — the SAME 4-key shape `select_levels_safe`'s sentinel
uses — and produces an all-False zone / NaN tgt-sl-rr automatically) is
still REUSED unchanged for the actual per-row zoned columns, on BOTH the
train-side sweep's `rr_fn` (see `run_train`'s own docstring) and OOS. No
zone-marking or down-side-negation logic is reimplemented here — only the
R/R SELECTION formula itself (`first_touch_rr_grid`) is new.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from azlib.config import default_label_params
from azlib.indicators import INDICATORS, FreezeStats, attribute_frame, fit_stats
from azlib.infer import fuse_inverse_variance, inferred_coeff, select_y, sweep_y
from azlib.loader import LabelParams, add_labels, label_col, load_wide_df
from azlib.models import fit_regression, groups, load_result, predict_reg, save_result
from azlib.rr import reach_freq_drift, reach_prob_estimator
from azlib.space import label_coeff, price_levels
from azlib.zones import (
    ResultsFile, build_rr_levels, build_zoned_dataset, zone_limit_price,
)
from helpers import dataset_folder

# --- module-level pipeline knobs -------------------------------------------
#
# Not pinned by any earlier task's brief -- Task 8's OWN driver defaults for
# "run the full pipeline end to end" (design spec's own [0, 2.0]/[-2.0, 2.0]
# sweep ranges, coarsened for a fast, deterministic, warning-free Docker test
# run; a real notebook driver, Task 9, is free to sweep finer grids).
_REG_KIND = "linear"  # single fixed kind for every attribute group (YAGNI —
# azlib.models offers "linear"/"poly2"/"gbr" per group, but selecting the
# best kind per group is a model-selection concern out of this task's scope;
# "linear" is deterministic, fast, and raises no sklearn convergence
# warnings under this project's `-W error` policy).
_X_GRID = np.round(np.arange(0.25, 2.01, 0.5), 2)  # target/stop-X sweep grid.
# Starts at 0.25, NOT 0.0 (design spec's own range is "[0, 2.0]"):
# `azlib.rr.rr_grid`'s expected-return-after-fees formula scales the RISK
# term by `sl_x` itself (`risk = sl_x * candle_size`), so `sl_x = 0.0` always
# zeroes that term out REGARDLESS of how likely the (zero-distance) stop is
# to trigger -- a mathematically-consistent ("a stop AT your own entry price
# costs nothing to be stopped out of") but economically degenerate edge case
# that this coarse a grid can end up selecting outright (empirically
# observed against `two_synthetic_datasets` during this task's own TDD
# loop -- see task-8-report.md). Excluding exactly 0.0 keeps every selected
# `(tgt_x, sl_x)` a genuine, non-zero-distance trade for this driver's own
# default grid; a real notebook driver (Task 9) sweeping a finer grid is
# free to include 0.0 again if that edge case is ever actually wanted.
_Y_GRID = np.round(np.arange(-2.0, 2.01, 0.5), 2)  # Y sweep grid.
_MIN_FIT_POINTS = 10  # a group with fewer non-NaN (X, y) rows than this is
# skipped entirely (not enough signal to fit anything meaningful) rather
# than handed to sklearn, which would raise on a near-empty/degenerate fit.


# --- env-file application (task-8-brief.md: "set env via the passed
# env-file path so a single Docker invocation covers both datasets") -------


def _apply_env_file(path: str) -> None:
    """Parse a plain ``KEY=VALUE`` env file (same format as ``configs/*.env``
    -- e.g. ``ROOT_FOLDER=short``, blank lines and ``#``-comments ignored)
    and apply every line to ``os.environ`` (call-time mutation, matching
    ``helpers.py``'s own "all os.environ reads happen at CALL TIME" design).

    Deliberately hand-rolled (no ``python-dotenv`` dependency) — the format
    this project's own ``configs/*.env`` files use is a strict subset (no
    quoting, no variable expansion, no multi-line values) that a 6-line
    parser covers exactly, and this module must not add a new dependency to
    the experiment image for it.
    """
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ[key.strip()] = value.strip()


def _fee() -> float:
    """Fractional trading-fee rate, from the just-applied env's
    ``EXCHANGE_FEE`` (matches ``configs/*.env``'s own key) -- defaults to
    ``0.001`` (0.1%, this project's own configs' typical value) if absent.
    """
    return float(os.environ.get("EXCHANGE_FEE", "0.001"))


def _results_dir(tf: int, direction: str) -> str:
    """Env-supplied artifact root for one ``(tf, direction)`` run (global
    constraint: "artifacts write to a caller/env-supplied path ... never
    into the worktree" — the "env" here is ``helpers.dataset_folder()``,
    itself driven by the ``ROOT_FOLDER``/``DATA_ROOT``/``DATA_SET_NAME``/
    ``PAIR`` env vars ``_apply_env_file`` just set).

    ``{dataset_folder()}action_zones/{direction}/{tf}/results/`` — matches
    Task 7's own ``ResultsFile.freeze_stats_path`` docstring EXAMPLE path
    verbatim (``"/vol/action_zones/short/60/results/stats.json"``). Created
    (``exist_ok=True``) if missing.

    ``dataset_folder`` is imported at module level specifically so tests can
    ``monkeypatch.setattr("azlib.validate.dataset_folder", ...)`` — same
    already-bound-reference convention ``azlib.loader.wide_df_path`` uses
    (see ``tests/test_layer1_loader.py``) — and point this at ``tmp_path``
    instead of the real (and, for ``ROOT_FOLDER=short``, not even mounted in
    this Docker service) ``/trader_data*`` volume root.
    """
    base = os.path.join(dataset_folder(), "action_zones", direction, str(tf), "results")
    os.makedirs(base, exist_ok=True)
    return base


# --- candle_size (azlib.rr's own "recommended, caller-computed" knob) ------


def _candle_size(wide_df: pd.DataFrame, tf: int) -> float:
    """Median COMPLETED ``{tf}`` candle range (``high - low``), in price.

    ``azlib.rr``'s module docstring recommends exactly this ("the MEDIAN
    ``|{tf}_high - {tf}_low|`` in PRICE, over TRAIN, at the SAME ``tf`` ...
    NOT computed by this module [azlib.rr] -- the caller/driver passes the
    number in") -- this is that caller. Reduced to COMPLETED candles via
    ``{tf}_is_closed`` (not the raw per-minute forming ``{tf}_high``/
    ``{tf}_low`` cummax/cummin) so early-in-bucket minutes' artificially
    narrow forming range never biases the median low — same completed-candle
    philosophy ``azlib.space.price_levels`` uses (see that module's
    docstring), applied here to a different statistic.
    """
    closed = wide_df[f"{tf}_is_closed"].astype(bool)
    rng = (wide_df.loc[closed, f"{tf}_high"] - wide_df.loc[closed, f"{tf}_low"]).dropna()
    return float(rng.median())


# --- per-attribute-group regression fitting/application --------------------


def _reg_model_filename(indicator: str, attrs: str | list[str] | tuple[str, ...], kind: str) -> str:
    """Filename convention ``run_oos`` parses back via ``_reg_model_key``
    (``indicator``/``kind`` are underscore-free identifiers from
    ``azlib.indicators.INDICATORS``/``azlib.models.REG_KINDS``; each attr in
    ``attrs`` -- ``azlib.indicators.ATTRS`` values -- is ALSO underscore-free
    -- a plain 3-way ``"_"``-join round-trips exactly via ``str.split("_")``).

    ``attrs`` is either a single attr name (the legacy single-attribute-group
    call shape, e.g. ``"position"`` -> ``"rsi_position_linear.json"``,
    unchanged) or a list/tuple of attr names (Task Ss10's multi-attribute
    HUMAN-SELECTED-model path, e.g. ``["position", "slope", "distance"]`` ->
    ``"rsi_position-slope-distance_gbr.json"`` -- joined with ``"-"``, NOT
    ``"_"``, so the filename's own 3 ``"_"``-separated parts still round-trip
    via ``_reg_model_key``'s plain ``str.split("_")``).
    """
    attrs_str = "-".join(attrs) if isinstance(attrs, (list, tuple)) else attrs
    return f"{indicator}_{attrs_str}_{kind}.json"


def _reg_model_key(path: str) -> tuple[str, str, str]:
    """Inverse of ``_reg_model_filename`` -- ``(indicator, attrs_str, kind)``
    from a saved ``RegResult`` JSON path's stem. ``attrs_str`` is either a
    single attr name (legacy 1D group) or ``"-"``-joined attr names (Task
    §10 multi-attribute selected-model group) -- callers split it on
    ``"-"`` themselves (see ``_predict_all_groups``) to recover the
    individual attribute list; this function's own ``"_"``-split is
    unaffected either way since attr names never contain ``"_"`` or ``"-"``.
    """
    indicator, attrs_str, kind = Path(path).stem.split("_")
    return indicator, attrs_str, kind


def _predict_safe(res, X_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``azlib.models.predict_reg``, but safe to call on an ``X_full`` that
    contains warm-up NaN rows.

    ``predict_reg``'s own docstring implies plain NaN-in/NaN-out
    propagation, but the underlying fitted sklearn estimator's ``.predict``
    does NOT actually behave that way for every ``kind`` — ``sklearn``'s own
    input validation REJECTS a NaN-containing ``X`` outright (``ValueError:
    Input X contains NaN``) rather than silently propagating NaN through the
    arithmetic, for ``LinearRegression`` (this module's own fixed
    ``_REG_KIND``) among others. This wrapper is what actually delivers the
    NaN-safe, 1:1-aligned-with-every-row-of-wide_df contract
    ``_fit_all_groups``/``_predict_all_groups`` need: predicts ONLY on the
    non-NaN rows of ``X_full`` and fills every other row with NaN directly,
    never handing sklearn a NaN row at all.
    """
    n = X_full.shape[0]
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    mask = ~np.isnan(X_full).any(axis=1)
    if np.any(mask):
        mean[mask], std[mask] = predict_reg(res, X_full[mask])
    return mean, std


def _fit_all_groups(
    wide_df: pd.DataFrame,
    tf: int,
    stats: FreezeStats,
    target: np.ndarray,
    results_dir: str,
    selected_models: list[dict] | None = None,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Fit + save every regression group this run needs, then predict each
    over the FULL row range -- either a caller-SELECTED set of (possibly
    multi-attribute) models (Task §10: "make run_train fit/fuse a HUMAN-
    SELECTED set of regression models ... instead of the hardcoded
    1D-linear-per-indicator, so the buy/sell zones reflect the good
    models"), or, when ``selected_models`` is absent/empty, the ORIGINAL
    legacy fan-out: one ``_REG_KIND`` regression per single-attribute group
    (``azlib.models.groups(indicator, 1)``, every indicator in
    ``azlib.indicators.INDICATORS`` -- "run the selected regression models
    over the dataset -> per-GROUP (mean, std)", design spec §11.1).

    ``selected_models``: a non-empty list of ``{"indicator": str, "attrs":
    list[str], "kind": str}`` dicts -- e.g. ``[{"indicator": "rsi", "attrs":
    ["position", "slope", "distance"], "kind": "gbr"}]`` fits EXACTLY that
    one 3D RSI gbr group instead of the legacy 8-model 1D-linear fan-out.
    Each spec's ``attrs`` selects MULTIPLE columns out of
    ``attribute_frame(wide_df, tf, indicator, stats)`` at once (``af[attrs]``
    -- same-indicator, same-space, per Layer 4's own "never mixes attributes
    across indicators" rule, just no longer restricted to ``dim=1``), fit
    with THAT spec's own ``kind`` (not the module-level ``_REG_KIND``) via
    ``azlib.models.fit_regression`` (which already accepts multi-column
    ``X`` unchanged). Backward compatible: ``selected_models=None`` (the
    default) or ``[]`` runs the ORIGINAL legacy branch below, byte-for-byte
    unchanged -- every existing caller/test that never passes this argument
    sees no behavior change at all.

    Both branches fit each group on its own non-NaN (frozen-attribute(s),
    ``target``) rows only (sklearn cannot fit through NaN) via a boolean
    ``mask``, then predict over the FULL (including warm-up-NaN) row range
    so every returned ``mean``/``std`` array is aligned 1:1 with ``wide_df``
    — NaN rows predict NaN (ordinary NaN propagation), matching this
    pipeline's existing NaN-safe convention elsewhere. A group with fewer
    than ``_MIN_FIT_POINTS`` usable rows is skipped entirely (see that
    constant's docstring).

    Returns ``(reg_model_paths, means, stds)`` -- parallel lists, one entry
    per FITTED group (skipped groups contribute nothing to any of the
    three).
    """
    reg_model_paths: list[str] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []

    if selected_models:
        for spec in selected_models:
            indicator = spec["indicator"]
            attrs = list(spec["attrs"])
            kind = spec["kind"]

            af = attribute_frame(wide_df, tf, indicator, stats)
            X_full = af[attrs].to_numpy(dtype=float)
            mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(target)
            if int(mask.sum()) < _MIN_FIT_POINTS:
                continue

            res = fit_regression(X_full[mask], target[mask], kind=kind)
            path = os.path.join(results_dir, _reg_model_filename(indicator, attrs, kind))
            save_result(res, path)

            mean, std = _predict_safe(res, X_full)
            reg_model_paths.append(path)
            means.append(mean)
            stds.append(std)

        return reg_model_paths, means, stds

    # --- legacy path: every indicator x every single-attribute group -------
    for indicator in INDICATORS:
        af = attribute_frame(wide_df, tf, indicator, stats)
        for (attr,) in groups(indicator, 1):
            X_full = af[[attr]].to_numpy(dtype=float)
            mask = ~np.isnan(X_full).any(axis=1) & ~np.isnan(target)
            if int(mask.sum()) < _MIN_FIT_POINTS:
                continue

            res = fit_regression(X_full[mask], target[mask], kind=_REG_KIND)
            path = os.path.join(results_dir, _reg_model_filename(indicator, attr, _REG_KIND))
            save_result(res, path)

            mean, std = _predict_safe(res, X_full)
            reg_model_paths.append(path)
            means.append(mean)
            stds.append(std)

    return reg_model_paths, means, stds


def _predict_all_groups(
    wide_df: pd.DataFrame, tf: int, stats: FreezeStats, reg_model_paths: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """OOS-side counterpart of ``_fit_all_groups``'s predict half: reload
    each FROZEN ``RegResult`` (``azlib.models.load_result`` — no refit) and
    predict over ``wide_df`` (the OOS frame) using that SAME frozen
    ``stats`` for z-scoring (``attribute_frame(..., stats=stats)``).

    ``attrs_str`` (``_reg_model_key``'s second element) is split on ``"-"``
    to recover the ORIGINAL attr list a Task §10 multi-attribute selected
    model was fit on (e.g. ``"position-slope-distance"`` ->
    ``["position", "slope", "distance"]``), so ``X_full`` is built with the
    SAME columns, in the SAME order, ``run_train``'s ``_fit_all_groups``
    used. Legacy single-attribute paths (``attrs_str`` has no ``"-"``) split
    into a 1-element list -- identical behavior to the old hardcoded
    ``af[[attr]]`` lookup.

    Returns the fused ``(mean, std)`` -- ``azlib.infer.fuse_inverse_variance``
    over every loaded group's per-point prediction, exactly mirroring
    ``run_train``'s own train-side fusion.
    """
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for path in reg_model_paths:
        indicator, attrs_str, _kind = _reg_model_key(path)
        attrs = attrs_str.split("-")
        res = load_result(path)
        af = attribute_frame(wide_df, tf, indicator, stats)
        X_full = af[attrs].to_numpy(dtype=float)
        mean, std = _predict_safe(res, X_full)
        means.append(mean)
        stds.append(std)
    return fuse_inverse_variance(np.vstack(means), np.vstack(stds))


# --- run_train ---------------------------------------------------------------


def run_train(train_env: str, tf: int, direction: str, selected_models: list[dict] | None = None) -> str:
    """Run the full pipeline (Layers 1-7) on the TRAIN set; freeze every fit
    statistic + write a ``ResultsFile`` (+ sidecars, see module docstring);
    return the ``ResultsFile``'s path.

    Uses ``azlib.config.default_label_params(tf)`` unconditionally for the
    labels this run's target/coverage are built against — matches the
    brief's own exact call signature (``run_train(train_env, tf,
    direction)``, no further knobs).

    ``selected_models`` (Task §10, OPTIONAL, backward compatible): a
    non-empty list of ``{"indicator": str, "attrs": list[str], "kind": str}``
    dicts fits EXACTLY those (possibly multi-attribute) regression groups
    instead of the legacy hardcoded 1D-linear-per-indicator fan-out -- e.g.
    ``[{"indicator": "rsi", "attrs": ["position", "slope", "distance"],
    "kind": "gbr"}]`` fits ONE 3D RSI gbr model, so the buy/sell zones this
    run selects reflect a HUMAN-picked good model rather than every
    indicator's default 1D fit. ``None``/``[]`` (the default) is the
    ORIGINAL legacy behavior, unchanged -- see ``_fit_all_groups``'s own
    docstring. Threaded straight into ``_fit_all_groups`` and persisted
    verbatim onto the returned ``ResultsFile.selected_models`` (documents
    what was selected; ``run_oos`` does not need it to re-predict, since it
    always reloads ``reg_models`` -- see ``azlib.zones.ResultsFile``'s own
    docstring).

    Pipeline, in order (design spec §8):
      1. ``_apply_env_file(train_env)`` + ``load_wide_df()`` -- env-driven
         data (module docstring's own train-only-freeze section explains
         why this is the ONLY module-level ``load_wide_df`` reference, so
         tests can monkeypatch it).
      2. ``add_labels`` (Task 1) -- strict/non-strict label columns.
      3. ``fit_stats`` (Task 3) -- FROZEN per-(tf,indicator,attr) mean/std,
         saved to ``stats.json``.
      4. ``label_coeff`` (Task 2) -- the regression target.
      5. ``_fit_all_groups`` (Task 4 fan-out, see its own docstring) -- one
         regression per single-attribute group, saved + predicted.
      6. ``fuse_inverse_variance`` (Task 5) -- per-point fused (mean, std).
      7. ``first_touch_rr_grid`` (Fix 2, REPLACES the earlier ``build_rr_levels``
         call -- see that function's own docstring) -- the ``(tgt_x, sl_x)``
         R/R levels selected by REALIZED first-touch expected R (or the
         no-profitable-levels NaN sentinel).
      8. ``sweep_y``/``select_y`` (Task 5, Y-DEPENDENT as of Fix 1 -- see
         ".superpowers/sdd/task-fix1-report.md"). Before the sweep, the
         FIXED per-candle target/stop PRICE columns (``az_tgt``/``az_sl``)
         for the selected ``(tgt_x, sl_x)`` are computed once, exactly as
         ``azlib.zones.build_zoned_dataset`` computes them (reusing
         ``price_levels``) -- these do NOT depend on ``Y`` (tgt/sl are
         fixed price-space distances from each row's own entry). ``rr_fn``
         is then a closure that, for a given ``Y``'s zone marking, calls
         ``_realized_rr_for_marking(wide_df, tf, direction, az_tgt_fixed,
         az_sl_fixed, zone_marking, p.n)`` -- the REALIZED mean R-multiple
         of entries actually inside that ``Y``'s zone (real forward-touch
         walk, same definition ``metrics()``'s own ``realized_rr`` uses).
         This is genuinely Y-DEPENDENT: a wider marking (larger ``|Y|`` in
         the widening direction) admits worse, further-from-center entries,
         so the mean R tends to FALL as the zone widens -- fixing the old
         degeneracy where a Y-invariant ``rr_fn`` (returning the one
         ``levels["exp_ret"]`` for every ``Y``) made a wider zone always
         "free" and the sweep always walked out to the grid boundary.
         ``select_y`` then picks the ``Y`` maximizing TOTAL realized profit
         (``n_zoned * realized_rr``) among rows whose ``realized_rr`` is
         profitable (``> 0.0``) -- self-limiting, since widening raises
         ``n_zoned`` but lowers ``realized_rr``, so the product peaks at a
         genuinely selective interior ``Y`` rather than the widest one.
      9. ``zone_limit_price`` (Task 7) at the selected Y -- the final
         entry threshold.
     10. Sidecars (``levels.json``, ``reach_train.json`` -- module
         docstring) + ``ResultsFile.save`` -- the returned path.

    **No-profitable-R/R case** (task-8-brief.md: "must record a valid
    'no-zone' ResultsFile and not crash"): when ``first_touch_rr_grid``
    returns its own NaN sentinel (Fix 2 -- same 4-key shape
    ``select_levels_safe``'s sentinel always used), ``rr_fn`` returns NaN for every
    ``Y`` -- ``NaN > 0.0`` is ``False`` (no RuntimeWarning, ordinary NaN
    comparison semantics, matching every earlier layer's own convention),
    so ``select_y`` finds no profitable ``Y`` and returns its OWN documented
    default (``0.0``) rather than raising. ``ResultsFile.tgt_x``/``sl_x``
    come out NaN (from the sentinel), exactly like a real
    ``select_levels_safe`` caller elsewhere in this pipeline -- no special
    casing is needed HERE, it falls out of Tasks 5-7's own existing
    NaN-safe/sentinel-safe designs.
    """
    _apply_env_file(train_env)
    wide_df = load_wide_df()

    p = default_label_params(tf)
    add_labels(wide_df, p)

    stats = fit_stats(wide_df, tf)
    results_dir = _results_dir(tf, direction)
    stats_path = os.path.join(results_dir, "stats.json")
    stats.to_json(stats_path)

    target = label_coeff(wide_df, tf, direction).to_numpy(dtype=float)
    reg_model_paths, means, stds = _fit_all_groups(
        wide_df, tf, stats, target, results_dir, selected_models=selected_models
    )
    if not reg_model_paths:
        raise RuntimeError(
            f"run_train: no single-attribute regression group had >= "
            f"{_MIN_FIT_POINTS} usable (non-NaN) rows for tf={tf} -- "
            "not enough data to fit anything"
        )
    fused_mean, fused_std = fuse_inverse_variance(np.vstack(means), np.vstack(stds))

    candle_size = _candle_size(wide_df, tf)
    fee = _fee()

    # Fix 2b (.superpowers/sdd/task-fix2-report.md, "Fix 2b (regularized)"):
    # strict_label is computed HERE (moved up from below its original
    # post-levels position -- add_labels already ran in step 2, so the
    # column exists) so first_touch_rr_grid can evaluate candidates on the
    # STRICT-labeled entries (the actual zoned subset the levels are used
    # on) instead of a broad random sample -- see that function's own
    # docstring for the exact fallback rule.
    strict_label = (wide_df[label_col(p, direction, strict=True)] == 1.0).to_numpy(dtype=bool)

    # Fix 2 (.superpowers/sdd/task-fix2-report.md): first_touch_rr_grid
    # REPLACES azlib.zones.build_rr_levels (marginal, independent
    # p_target/p_stop reach-probability selection -- degenerates to a
    # near-target/far-stop mirage) with a REALIZED first-touch expected-R
    # selection over the SAME _X_GRID -- see that function's own docstring.
    # Fix 2b additionally regularizes it: (1) scores on the strict-label
    # entries (the actual zoned subset), not an unconstrained random
    # sample -- reduces train->OOS overfit; (2) restricts candidates to a
    # sane reward:risk band, excluding both the tiny-target/far-stop mirage
    # AND its opposite (extreme far-target) corner.
    # Settled on Fix 1 (reach-based level selection). The first-touch grid
    # (`first_touch_rr_grid`, Fix 2/2b) was explored but OVERFIT the 2-month
    # OOS (realized R worse on 4/6 combos), so run_train uses build_rr_levels.
    # `first_touch_rr_grid` is kept below as an unused, documented artifact.
    levels = build_rr_levels(wide_df, tf, direction, _X_GRID, fee, candle_size)

    # Fix 1 (Y-dependent rr_fn -- see this function's own docstring, step 8,
    # and .superpowers/sdd/task-fix1-report.md): the FIXED per-candle
    # target/stop PRICE columns for the selected (tgt_x, sl_x), exactly the
    # same mapping azlib.zones.build_zoned_dataset uses (reusing
    # price_levels, not reimplemented). These do NOT vary with Y -- only
    # which ROWS a given Y's zone marks varies.
    tgt_high, tgt_low = price_levels(wide_df, tf, x=levels["tgt_x"])
    sl_high, sl_low = price_levels(wide_df, tf, x=levels["sl_x"])
    if direction == "long":
        az_tgt_fixed = tgt_high.to_numpy(dtype=float)
        az_sl_fixed = sl_low.to_numpy(dtype=float)
    else:
        az_tgt_fixed = tgt_low.to_numpy(dtype=float)
        az_sl_fixed = sl_high.to_numpy(dtype=float)

    def rr_fn(zone_marking: pd.Series) -> float:
        return _realized_rr_for_marking(
            wide_df, tf, direction, az_tgt_fixed, az_sl_fixed, zone_marking, int(p.n)
        )

    sweep = sweep_y(
        fused_mean, fused_std, wide_df, tf, direction, price_levels, strict_label, _Y_GRID, rr_fn,
    )
    y = select_y(sweep)

    coeff_y = inferred_coeff(fused_mean, fused_std, y)
    high_level, low_level = price_levels(wide_df, tf)
    zone_limit = zone_limit_price(coeff_y, low_level.to_numpy(), high_level.to_numpy())
    _ = build_zoned_dataset(wide_df, tf, direction, zone_limit, levels)  # sanity: pipeline runs end to end

    levels_path = os.path.join(results_dir, "levels.json")
    with open(levels_path, "w") as f:
        json.dump(levels, f)

    high_diff_train = wide_df[f"{tf}_high_diff_prc"].dropna().to_numpy(dtype=float)
    low_diff_train_neg = (-wide_df[f"{tf}_low_diff_prc"]).dropna().to_numpy(dtype=float)
    reach_path = os.path.join(results_dir, "reach_train.json")
    with open(reach_path, "w") as f:
        json.dump(
            {"high_diff_prc": high_diff_train.tolist(), "low_diff_prc_neg": low_diff_train_neg.tolist()}, f
        )

    rf = ResultsFile(
        tf=tf,
        direction=direction,
        label_params=asdict(p),
        freeze_stats_path=stats_path,
        reg_models=reg_model_paths,
        y=y,
        tgt_x=levels["tgt_x"],
        sl_x=levels["sl_x"],
        fee=fee,
        selected_models=list(selected_models) if selected_models else [],
    )
    results_path = os.path.join(results_dir, "results.json")
    rf.save(results_path)
    return results_path


# --- run_oos -----------------------------------------------------------------


def run_oos(results_path: str, oos_env: str) -> pd.DataFrame:
    """Load the FROZEN ``ResultsFile`` (+ sidecars) and apply it to the OOS
    wide df -- NO refit of anything (module docstring's "Train-only freeze /
    no leakage" section; proved directly by
    ``test_run_oos_never_refits_frozen_artifacts``).

    Returns the zoned OOS ``pd.DataFrame`` (``azlib.zones
    .build_zoned_dataset``'s own columns), PLUS the extra columns
    ``metrics()`` needs and ``build_zoned_dataset`` itself does not produce
    (it only ever sees a plain ``wide_df``/``zone_limit``/``levels``, not
    this module's own label/reach bookkeeping):

      - ``az_label_strict``/``az_label_nonstrict`` (bool): the SAME strict/
        non-strict label columns ``run_train`` used, recomputed here from
        the ResultsFile-frozen ``label_params`` (a pure, non-fitted
        function of OOS's own OHLC/ATR data — see module docstring's
        "price_levels is deliberately unfrozen" note, same reasoning
        applies to labels: computing GROUND-TRUTH labels on OOS for later
        evaluation is not "refitting" anything, it never touches a
        train-derived statistic).
      - ``1_high``/``1_low`` (float): OOS's own 1-minute extremes --
        ``metrics()``'s ``realized_rr`` needs these for its own
        forward-touch walk.
      - ``az_reach_drift_max`` (float, BROADCAST — same value every row):
        the max ``|train P - OOS realized reach-freq|`` drift (module
        docstring's "reach_train.json" section; ``azlib.rr
        .reach_freq_drift``, REUSED), computed HERE, ONCE, from the FULL
        frozen train arrays (``reach_train.json``) and the FULL OOS
        ``{tf}_high_diff_prc``/``{tf}_low_diff_prc`` series straight off
        ``wide_df`` — deliberately NOT left for ``metrics()`` to recompute
        from whatever rows/columns happen to still be present on its own
        ``zoned_df`` argument. An earlier version of this module stashed
        the frozen train arrays on ``zoned_df.attrs`` and had ``metrics()``
        recompute the drift from there + the (possibly already row-
        filtered) ``zoned_df`` — broken two ways: (1) `pandas.DataFrame
        .attrs` is dropped by common operations (concat, some merges,
        older-pandas `.copy()`), and (2) even where `.attrs` DOES survive,
        recomputing the "OOS realized frequency" side from a row-FILTERED
        `zoned_df` silently changes the answer (fewer/different rows ->
        a different empirical frequency) — a Task 9 notebook that slices
        `zoned_df` to a date range before calling `metrics()` would get a
        subtly WRONG drift number either way, not just a missing one. A
        plain broadcast COLUMN fixes both: it is computed once against the
        complete OOS data right here, and an ordinary column (unlike
        `.attrs`) survives row-filtering/`.copy()`/concat intact — see
        `tests/test_layer8_validate.py`'s
        `test_reach_drift_max_survives_row_filtering_and_copy` (this exact
        scenario, RED under the old `.attrs` design, GREEN now).
    """
    rf = ResultsFile.load(results_path)
    _apply_env_file(oos_env)
    wide_df = load_wide_df()

    results_dir = os.path.dirname(rf.freeze_stats_path)
    stats = FreezeStats.from_json(rf.freeze_stats_path)

    fused_mean, fused_std = _predict_all_groups(wide_df, rf.tf, stats, rf.reg_models)
    coeff_y = inferred_coeff(fused_mean, fused_std, rf.y)

    high_level, low_level = price_levels(wide_df, rf.tf)
    zone_limit = zone_limit_price(coeff_y, low_level.to_numpy(), high_level.to_numpy())

    with open(os.path.join(results_dir, "levels.json")) as f:
        levels = json.load(f)

    zdf = build_zoned_dataset(wide_df, rf.tf, rf.direction, zone_limit, levels)

    p = LabelParams(**rf.label_params)
    add_labels(wide_df, p)
    zdf["az_label_strict"] = (wide_df[label_col(p, rf.direction, strict=True)] == 1.0).to_numpy()
    zdf["az_label_nonstrict"] = (wide_df[label_col(p, rf.direction, strict=False)] == 1.0).to_numpy()

    zdf["1_high"] = wide_df["1_high"].to_numpy()
    zdf["1_low"] = wide_df["1_low"].to_numpy()

    with open(os.path.join(results_dir, "reach_train.json")) as f:
        reach = json.load(f)
    train_up = np.asarray(reach["high_diff_prc"], dtype=float)
    train_down = np.asarray(reach["low_diff_prc_neg"], dtype=float)
    oos_up = wide_df[f"{rf.tf}_high_diff_prc"].to_numpy(dtype=float)
    oos_down = (-wide_df[f"{rf.tf}_low_diff_prc"]).to_numpy(dtype=float)
    zdf["az_reach_drift_max"] = _compute_reach_drift_max(train_up, train_down, oos_up, oos_down)

    return zdf


# --- metrics -------------------------------------------------------------


def _coverage(label: np.ndarray, zoned: np.ndarray) -> float:
    """Fraction of ``label``-True rows that are also ``zoned``-True.

    NaN-safe default (matches ``azlib.infer.sweep_y``'s own
    ``strict_coverage`` convention exactly): 0 label-True rows -> ``NaN``
    (coverage of an empty set is undefined), guarded via a plain Python
    ``if``/scalar division so no ``RuntimeWarning`` either.
    """
    n_label = int(label.sum())
    if n_label == 0:
        return float("nan")
    return float(np.sum(label & zoned)) / n_label


def _realized_rr_for_marking(
    wide_df: pd.DataFrame,
    tf: int,
    direction: str,
    az_tgt,
    az_sl,
    marking,
    n: int,
) -> float:
    """Core forward-touch mean-R computation (Fix 1: the reusable,
    Y-DEPENDENT half of ``_realized_rr`` -- callable on ANY boolean
    marking, not just a frozen ``zoned_df``'s own zone column). Both
    ``_realized_rr`` (below, unchanged callers) and ``run_train``'s per-Y
    ``rr_fn`` closure (this module's own docstring, "Fix 1") delegate here.

    **Exact definition** (mirrors ``indicators.labels._forward_labels``'
    first-touch walk conceptually, generalized to per-row ``az_tgt``/
    ``az_sl`` PRICES instead of a fixed ``m*atr``/``x*atr`` distance, and to
    a realized MAGNITUDE rather than a binary label):

    For every row where ``marking`` is True (an actual entry) with a
    non-NaN ``az_tgt``/``az_sl`` (excludes the no-profitable-levels
    sentinel, whose price_levels(..., x=nan) columns are all-NaN) and at
    least ``win = max(1, n) * tf`` future 1-minute rows available:

    ::

        reward_dist = |az_tgt - entry_price|   # entry_price: 1_low (long) / 1_high (short)
        risk_dist   = |entry_price - az_sl|
        # first-touch walk over the next `win` 1-minute rows (index p+1..p+win),
        # SAME pessimistic same-minute tie rule as _forward_labels (stop wins):
        target_first = (target touched at some minute) AND
                        (no earlier or same minute stop touch)
        stop_first   = (stop touched, and not target_first)
        R = +reward_dist / risk_dist   if target_first
            -1.0                       if stop_first
            (excluded from the mean)   if neither touched within `win`,
                                           or risk_dist <= 0

    ``realized_rr`` = mean ``R`` over every row resolved this way. This is
    the standard "average R-multiple" trading metric: +1R is "won exactly
    the configured risk amount", the actual mean reflects the CONFIGURED
    reward:risk ratio (``reward_dist / risk_dist``) weighted by how often
    the target was actually reached first, in real forward OHLC data --
    unlike ``azlib.rr``'s ``expected_return_after_fees`` (a train-fit
    PROBABILITY-weighted estimate), this is a REALIZED, walked-forward
    number, and (unlike that formula) is NOT fee-adjusted: ``metrics()``'s
    own signature (task-8-brief.md's fixed interface) takes no ``fee``
    argument, so there is no fee to subtract here — see module docstring.

    ``az_tgt``/``az_sl``/``marking`` are array-like (bare numpy array or
    ``pd.Series``), one value per row of ``wide_df``, consumed PURELY
    POSITIONALLY (same convention ``azlib.infer.sweep_y`` uses for
    ``fused_mean``/``fused_std``/``strict_label``) -- this is what lets
    ``run_train``'s ``rr_fn`` hand in a FIXED, held-constant ``(az_tgt,
    az_sl)`` price pair (computed once outside the Y sweep) alongside a
    freshly-computed per-Y ``zone_marking`` Series, without either one
    needing to already be a column of the same frame.

    NaN-safe default: zero resolved rows (empty ``marking`` -- e.g. a
    tight/one-sided ``Y`` whose zone contains no rows at all; or every
    marked row too close to the frame's end for a full forward window; or
    an all-NaN ``az_tgt``/``az_sl`` pair, e.g. the no-profitable-levels
    sentinel) -> ``NaN`` -- a documented sentinel EXCLUDED from
    ``select_y``'s profitability filter (``NaN > 0.0`` is ``False``, no
    RuntimeWarning). Missing ``1_high``/``1_low`` columns on ``wide_df`` ->
    ``NaN`` too, rather than ``KeyError``.
    """
    required = {"1_high", "1_low"}
    if not required <= set(wide_df.columns):
        return float("nan")

    zoned = np.asarray(marking, dtype=bool)
    tgt = np.asarray(az_tgt, dtype=float)
    sl = np.asarray(az_sl, dtype=float)
    one_high = wide_df["1_high"].to_numpy(dtype=float)
    one_low = wide_df["1_low"].to_numpy(dtype=float)
    entry_col = "1_low" if direction == "long" else "1_high"
    entry = wide_df[entry_col].to_numpy(dtype=float)

    n_rows = len(wide_df)
    win = max(1, int(n)) * tf
    if win >= n_rows:
        return float("nan")

    positions = np.arange(n_rows)
    candidate = zoned & ~np.isnan(tgt) & ~np.isnan(sl) & (positions + win <= n_rows - 1)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return float("nan")

    sw_high = np.lib.stride_tricks.sliding_window_view(one_high, win)
    sw_low = np.lib.stride_tricks.sliding_window_view(one_low, win)
    wh = sw_high[idx + 1]
    wl = sw_low[idx + 1]

    if direction == "long":
        target_hit = wh > tgt[idx, None]
        stop_hit = wl < sl[idx, None]
        reward_dist = tgt[idx] - entry[idx]
        risk_dist = entry[idx] - sl[idx]
    else:
        target_hit = wl < tgt[idx, None]
        stop_hit = wh > sl[idx, None]
        reward_dist = entry[idx] - tgt[idx]
        risk_dist = sl[idx] - entry[idx]

    t_first = np.where(target_hit.any(axis=1), target_hit.argmax(axis=1), win)
    s_first = np.where(stop_hit.any(axis=1), stop_hit.argmax(axis=1), win)

    target_won = t_first < s_first
    stop_lost = (~target_won) & (s_first < win)
    resolved = target_won & (risk_dist > 0.0) | stop_lost & (risk_dist > 0.0)
    if not np.any(resolved):
        return float("nan")

    risk_safe = np.where(risk_dist > 0.0, risk_dist, 1.0)  # dummy denom where unused (resolved excludes it)
    r_multiple = np.where(target_won, reward_dist / risk_safe, -1.0)
    return float(np.mean(r_multiple[resolved]))


# --- first_touch_rr_grid (Fix 2 -- REPLACES the marginal-reach-probability
# (tgt_x, sl_x) selection azlib.zones.build_rr_levels/azlib.rr.select_levels
# used to perform for run_train) --------------------------------------------

# Sentinel dict this module returns when NO (tgt_x, sl_x) candidate scores a
# strictly positive first-touch expected R -- same 4 keys
# azlib.zones.select_levels_safe's own NaN sentinel uses (_NO_PROFITABLE_LEVELS
# there), duplicated here (not imported) so this module does not need to
# reach into azlib.zones' own private (``_``-prefixed) constant -- both are
# just "the 4 ResultsFile/build_zoned_dataset fields, all NaN", the shape
# every downstream consumer (``run_train``'s own ``build_zoned_dataset``
# sanity call, ``ResultsFile.tgt_x``/``sl_x``) already expects.
_NO_PROFITABLE_FIRST_TOUCH_LEVELS = {
    "tgt_x": float("nan"),
    "sl_x": float("nan"),
    "rr": float("nan"),
    "exp_ret": float("nan"),
}

# Fix 2b (.superpowers/sdd/task-fix2-report.md, "Fix 2b (regularized)"):
# regularization knobs added after Fix 2's plain first-touch selection was
# found to OVERFIT train (train->OOS realized R got WORSE on 4/6 combos,
# e.g. long/240 +1.48R -> -0.45R). Two independent regularizers:
#
#   1. Evaluate on the STRICT-labeled entries (the actual zoned subset the
#      selected levels are USED on), not an unconstrained random sample of
#      every candle -- see first_touch_rr_grid's own "Evaluation set"
#      docstring section. _MIN_STRICT_LABEL_COUNT is the fallback floor:
#      below this many strict-labeled (and otherwise valid) rows there is
#      not enough signal to trust a strict-label-only score, so the
#      function falls back to the ORIGINAL (Fix 2) random-sample-of-all-
#      valid-rows pool instead -- same fallback philosophy as
#      ``azlib.indicators``' ``_MIN_FIT_POINTS``: too little data silently
#      degrades to a broader, still-deterministic default rather than
#      erroring or picking on a handful of noisy points.
#   2. Restrict candidates to a sane reward:risk band (_RR_MIN.._RR_MAX) --
#      excludes BOTH the tiny-target/far-stop mirage corner (e.g. 0.25:1.75,
#      ratio ~0.14) AND its opposite extreme (a far target with a razor-
#      thin stop, ratio > 3) -- both ends of the grid are corners a small,
#      noisy train sample can spuriously favor; only genuinely moderate R/R
#      shapes are even eligible to be scored.
_MIN_STRICT_LABEL_COUNT = 200
_RR_MIN = 1.0
_RR_MAX = 3.0


def first_touch_rr_grid(
    wide_df: pd.DataFrame,
    tf: int,
    direction: str,
    x_grid: np.ndarray,
    n: int,
    fee: float,
    candle_size: float,
    strict_label: np.ndarray | pd.Series | None = None,
    sample_size: int = 40_000,
    seed: int = 0,
    min_strict_count: int = _MIN_STRICT_LABEL_COUNT,
    rr_min: float = _RR_MIN,
    rr_max: float = _RR_MAX,
) -> dict:
    """Fix 2: replace ``azlib.zones.build_rr_levels``'s degenerate marginal
    (INDEPENDENT ``p_target``/``p_stop`` reach-probability) ``(tgt_x, sl_x)``
    selection with one that scores every grid candidate by its REALIZED
    FIRST-TOUCH expected R on train, exactly the quantity a real trade
    actually earns/loses.

    **Why Fix 2** (see ``.superpowers/sdd/task-fix2-report.md`` for the full
    writeup): ``azlib.rr.rr_grid``/``select_levels`` maximize
    ``reward*p_target - risk*p_stop - 2*fee`` where ``p_target``/``p_stop``
    are INDEPENDENT marginal "ever reached within the horizon" probabilities
    -- this ignores which level is hit FIRST, so it degenerates to "tiny
    target you almost always eventually touch (``tgt_x`` near the grid's
    lower bound), far stop you almost never touch (``sl_x`` near the grid's
    upper bound)" -- a mirage: a genuinely terrible reward:risk ratio (e.g.
    ``0.25:1.75`` -> a win pays only ``+0.14R`` while a loss costs the full
    ``-1.0R``) that the marginal/independent formula scores as attractive
    purely because the near level's ISOLATED touch probability is high and
    the far level's ISOLATED touch probability is low -- but on a REAL
    price path the two are not independent draws, they are a RACE: which
    one gets touched first. This function runs that race directly.

    **Score formula** (this function's own documented design choice, no
    earlier task pins one): for every ``(tgt_x, sl_x)`` in the
    ``x_grid x x_grid`` cross product (matching ``azlib.rr.rr_grid``'s own
    full cross-product grid, so this is a drop-in replacement for the same
    ``_X_GRID`` ``run_train`` already sweeps)::

        az_tgt, az_sl = the SAME per-direction price mapping
                         azlib.zones.build_zoned_dataset uses:
                           long:  az_tgt = price_levels(wide_df, tf, x=tgt_x)[0]  # high side
                                  az_sl  = price_levels(wide_df, tf, x=sl_x)[1]   # low side
                           short: az_tgt = price_levels(wide_df, tf, x=tgt_x)[1]  # low side
                                  az_sl  = price_levels(wide_df, tf, x=sl_x)[0]   # high side
        mean_r = _realized_rr_for_marking(wide_df, tf, direction, az_tgt, az_sl,
                                           eval_marking, n)
        fee_r  = 2 * fee / candle_size   # entry+exit fees, expressed as a
                                          # constant R-unit penalty -- see
                                          # "Fee term" below
        score  = mean_r - fee_r

    ``mean_r`` is EXACTLY ``_realized_rr_for_marking``'s realized-first-touch
    mean R-multiple (Fix 1's own reusable core, REUSED not reimplemented --
    same forward-touch walk, same same-minute-tie-favors-stop rule) over
    ``eval_marking`` (see "Evaluation set" below), for THIS candidate's own
    ``(az_tgt, az_sl)`` price pair. Candidates where ``mean_r`` is ``NaN``
    (zero resolved rows in the evaluation set -- e.g. a target so far it is
    never reached) are skipped entirely (never selected, never compared).

    **Fee term.** ``fee`` is the same fractional per-side trading-fee rate
    ``azlib.rr.rr_grid``'s own ``fee`` parameter is (e.g. ``0.001`` = 0.1%);
    ``2 * fee`` is "one fee charge on the way in, one on the way out" (same
    "entry+exit fees" wording ``azlib.rr``'s own module docstring uses).
    Expressing that as an R-unit penalty needs SOME price-space reference to
    divide by -- ``candle_size`` (the same "one typical candle's price
    range" caller-computed number ``azlib.rr``'s own module docstring
    documents and ``run_train`` already passes to it) is reused here as
    that reference, giving the constant, candidate-INDEPENDENT penalty
    ``fee_r = 2*fee/candle_size`` subtracted from every candidate's
    ``mean_r`` alike. This is a deliberately SIMPLE choice (documented, per
    this task's own brief, as one acceptable option among several) rather
    than a per-candidate risk-normalized fee (which would need
    ``2*fee*entry_price/risk_dist``, varying per row AND per candidate) --
    it still does exactly what a fee term needs to here: a real per-round-
    trip cost that must be cleared before a candidate counts as
    "profitable", and (see ``test_run_train_no_profitable_rr_records_nan_zone_results_file``)
    an outrageous ``fee`` still drives EVERY candidate's score negative,
    matching ``azlib.rr.rr_grid``'s own fee-guards-profitability behavior.
    ``candle_size <= 0`` (degenerate/empty train data) makes ``fee_r`` fall
    back to ``0.0`` rather than raising or dividing by zero.

    **Evaluation set** (Fix 2b regularization #1 --
    ``.superpowers/sdd/task-fix2-report.md``'s "Fix 2b (regularized)"
    section): PREFERS the STRICT-labeled rows -- ``strict_label`` (a bool
    array/Series aligned 1:1 with ``wide_df``, typically
    ``(wide_df[label_col(p, direction, strict=True)] == 1.0)``, computed by
    the caller since ``add_labels`` already ran before this function is
    called) intersected with this function's own "valid" mask (see below)
    -- over an unconstrained random sample of every candle, WHEN there are
    at least ``min_strict_count`` (default ``_MIN_STRICT_LABEL_COUNT`` =
    ``200``) such rows: the selected levels are ultimately USED on exactly
    this strict-labeled subset (the zoned entries), so scoring them on that
    SAME subset (rather than on a broad, mostly-irrelevant random sample of
    every candle in the frame) is what actually reduces train->OOS overfit
    -- Fix 2's plain first-touch selection, scored on an unconstrained
    random sample, was found to overfit train (train->OOS realized R got
    WORSE on 4/6 combos on real data).

    Either way (``strict_label`` provided and large enough, or the
    fallback), the candidate POOL is then a SEEDED (``seed``, default
    ``0`` -- fixed integer, NEVER wall-clock/unseeded, per this codebase's
    own determinism rule) uniform-random sample of up to ``sample_size``
    (default ``40_000``) rows from that pool -- "valid" (the base mask
    both the strict-label pool and the fallback pool are intersected
    with/equal to) meaning: a non-NaN entry price (``1_low`` for long /
    ``1_high`` for short), a non-NaN reference ``price_levels`` band at
    that row (checked at the grid's own first ``x_grid`` value --
    ``price_levels``'s warm-up-NaN PATTERN, which rows are NaN, does not
    depend on ``x`` itself, only on the rolling-window/shift machinery
    producing the ma/std it is offset from -- so any single representative
    ``x`` correctly identifies every warm-up row for every candidate at
    once, without recomputing ``price_levels`` once per grid value just to
    build this mask), and at least ``win = max(1, n) * tf`` future 1-minute
    rows still available before the frame ends (same ``win``
    ``_realized_rr_for_marking`` itself requires). When the pool's row
    count is ``<= sample_size``, EVERY row in it is used (no sampling
    needed) -- fully deterministic either way. ``strict_label=None`` (the
    default) always uses the fallback pool -- every existing caller that
    predates Fix 2b sees no behavior change from this parameter alone. The
    SAME sampled row set (one boolean ``marking`` array) is reused for
    EVERY grid candidate, so every candidate's score is comparable
    apples-to-apples against the identical evaluation rows.

    **Reward:risk band** (Fix 2b regularization #2): a candidate
    ``(tgt_x, sl_x)`` is only even SCORED when ``sl_x > 0`` (guards the
    ``tgt_x / sl_x`` division) AND its reward:risk ratio ``tgt_x / sl_x``
    falls inside ``[rr_min, rr_max]`` (default ``[_RR_MIN, _RR_MAX]`` =
    ``[1.0, 3.0]``) -- excludes BOTH the tiny-target/far-stop mirage corner
    (e.g. ``0.25:1.75``, ratio ``~0.14``) AND the opposite extreme (a far
    target paired with a razor-thin stop, ratio ``> 3``) up front, before
    any first-touch scoring happens -- narrowing the search to genuinely
    moderate R/R shapes a small, noisy train sample is far less likely to
    spuriously favor. Candidates outside the band are skipped entirely
    (never scored, never selected) exactly like a ``NaN`` ``mean_r``.

    **Selection.** Returns the ``{"tgt_x", "sl_x", "rr", "exp_ret"}`` dict
    (same 4 keys ``azlib.rr.select_levels``/``azlib.zones.select_levels_safe``
    return, so this is a drop-in replacement for ``build_rr_levels``'s own
    return value everywhere it is consumed -- ``ResultsFile.tgt_x``/``sl_x``,
    ``azlib.zones.build_zoned_dataset``, ``run_train``'s own Fix-1 ``rr_fn``
    closure) for the in-band candidate with the MAXIMUM ``score`` among
    candidates with ``score > 0.0`` -- ``rr`` is ``tgt_x / sl_x`` (the
    winning candidate's own reward:risk ratio -- always inside
    ``[rr_min, rr_max]`` by construction now) and ``exp_ret`` is that
    winning ``score`` itself. If NO in-band candidate scores strictly
    positive (or every in-band candidate's ``mean_r`` was ``NaN``, or the
    band excludes every grid combo entirely), returns
    ``_NO_PROFITABLE_FIRST_TOUCH_LEVELS`` -- the same 4-key, all-``NaN``
    sentinel shape ``select_levels_safe`` returns, so every existing
    no-profitable-levels code path downstream (``build_zoned_dataset``'s
    ``_is_no_profitable_sentinel`` check, ``ResultsFile.tgt_x``/``sl_x``
    coming out ``NaN``) keeps working unchanged.

    Raises ``ValueError`` if ``direction`` is not ``"long"``/``"short"``
    (same guard every other direction-aware function in this pipeline uses).
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    x_grid = np.asarray(x_grid, dtype=float)
    entry_col = "1_low" if direction == "long" else "1_high"
    entry = wide_df[entry_col].to_numpy(dtype=float)
    n_rows = len(wide_df)
    win = max(1, int(n)) * tf

    # Warm-up mask: checked at ONE representative x (price_levels' NaN
    # PATTERN is x-independent, see docstring above) rather than
    # recomputed per grid value.
    ref_x = float(x_grid[0]) if x_grid.size else 1.0
    ref_high, ref_low = price_levels(wide_df, tf, x=ref_x)
    ref_valid = ~np.isnan(ref_high.to_numpy(dtype=float)) & ~np.isnan(ref_low.to_numpy(dtype=float))

    positions = np.arange(n_rows)
    valid = ref_valid & ~np.isnan(entry) & (positions + win <= n_rows - 1)

    # Fix 2b regularization #1: prefer the STRICT-labeled (and otherwise
    # valid) rows -- the actual zoned subset the selected levels are USED
    # on -- over the unconstrained "every valid candle" pool, but only when
    # there are enough of them (min_strict_count) to trust; otherwise fall
    # back to the original Fix 2 pool. See this function's own "Evaluation
    # set" docstring section.
    if strict_label is not None:
        strict_valid = valid & np.asarray(strict_label, dtype=bool)
    else:
        strict_valid = None

    if strict_valid is not None and int(strict_valid.sum()) >= min_strict_count:
        pool = strict_valid
    else:
        pool = valid

    idx_pool = np.flatnonzero(pool)

    rng = np.random.default_rng(seed)
    if idx_pool.size > sample_size:
        idx_sample = rng.choice(idx_pool, size=sample_size, replace=False)
    else:
        idx_sample = idx_pool

    marking = np.zeros(n_rows, dtype=bool)
    marking[idx_sample] = True

    fee_r = 2.0 * fee / candle_size if candle_size > 0.0 else 0.0

    # Precompute price_levels ONCE per unique x_grid value (len(x_grid)
    # calls), not once per (tgt_x, sl_x) PAIR (len(x_grid)**2 calls) -- the
    # cross product below only ever indexes into this cache.
    levels_by_x = {float(x): price_levels(wide_df, tf, x=float(x)) for x in x_grid}

    best_tgt_x = None
    best_sl_x = None
    best_score = None
    for tgt_x in x_grid:
        tgt_high, tgt_low = levels_by_x[float(tgt_x)]
        for sl_x in x_grid:
            # Fix 2b regularization #2: only candidates inside the sane
            # reward:risk band are even scored -- excludes both the mirage
            # corner and its far-target/thin-stop opposite up front.
            if sl_x <= 0.0:
                continue
            ratio = float(tgt_x) / float(sl_x)
            if not (rr_min <= ratio <= rr_max):
                continue

            sl_high, sl_low = levels_by_x[float(sl_x)]

            if direction == "long":
                az_tgt = tgt_high.to_numpy(dtype=float)
                az_sl = sl_low.to_numpy(dtype=float)
            else:
                az_tgt = tgt_low.to_numpy(dtype=float)
                az_sl = sl_high.to_numpy(dtype=float)

            mean_r = _realized_rr_for_marking(wide_df, tf, direction, az_tgt, az_sl, marking, int(n))
            if math.isnan(mean_r):
                continue

            score = mean_r - fee_r
            if best_score is None or score > best_score:
                best_tgt_x, best_sl_x, best_score = float(tgt_x), float(sl_x), score

    if best_score is None or best_score <= 0.0:
        return dict(_NO_PROFITABLE_FIRST_TOUCH_LEVELS)

    rr = best_tgt_x / best_sl_x if best_sl_x != 0.0 else float("nan")
    return {"tgt_x": best_tgt_x, "sl_x": best_sl_x, "rr": rr, "exp_ret": best_score}


def _realized_rr(zoned_df: pd.DataFrame, tf: int, direction: str, n: int) -> float:
    """Realized reward/risk (R-multiple) of ZONED entries, from ACTUAL
    forward touches of ``az_tgt``/``az_sl`` (task-8-brief.md: "define
    clearly") -- thin wrapper around ``_realized_rr_for_marking`` (Fix 1)
    that extracts the frozen zone/target/stop columns off an already-built
    ``zoned_df`` (e.g. ``run_oos``'s return value, or ``metrics()``'s own
    argument). See that function's own docstring for the exact forward-
    touch definition and NaN-safe semantics -- unchanged here; every
    existing caller of ``_realized_rr`` (``metrics()``, this module's own
    tests) keeps its original call signature and behavior byte-for-byte.

    NaN-safe default: missing ``az_zone_{direction}_{tf}``/``az_tgt``/
    ``az_sl``/``1_high``/``1_low`` columns on ``zoned_df`` (a
    caller-constructed frame that only wants ``strict_coverage``, say) ->
    ``NaN``, rather than ``KeyError`` -- checked HERE against ``zoned_df``'s
    own columns (the delegate's own required-column check only covers
    ``1_high``/``1_low``, since it has no zone/tgt/sl column names of its
    own to check -- those are passed in as plain arrays instead).
    """
    zone_col = f"az_zone_{direction}_{tf}"
    required = {zone_col, "az_tgt", "az_sl", "1_high", "1_low"}
    if not required <= set(zoned_df.columns):
        return float("nan")

    marking = zoned_df[zone_col].to_numpy(dtype=bool)
    az_tgt = zoned_df["az_tgt"].to_numpy(dtype=float)
    az_sl = zoned_df["az_sl"].to_numpy(dtype=float)
    return _realized_rr_for_marking(zoned_df, tf, direction, az_tgt, az_sl, marking, n)


def _compute_reach_drift_max(
    train_up: np.ndarray, train_down: np.ndarray, oos_up: np.ndarray, oos_down: np.ndarray
) -> float:
    """Max ``|train P - OOS realized reach-freq|`` across ``_X_GRID``, both
    up (high) and down (negated low) sides, via ``azlib.rr
    .reach_freq_drift`` (REUSED, not reimplemented -- task-8-brief.md).

    Pure-array helper — takes the frozen TRAIN extreme-diff arrays and the
    FULL OOS extreme-diff arrays directly (both already the exact `up`/
    `down`-side-convention arrays `azlib.rr` expects — see
    `azlib.zones.build_rr_levels`'s own "down-side reach wiring" docstring:
    `up` = `{tf}_high_diff_prc`, `down` = NEGATED `{tf}_low_diff_prc`, on
    BOTH sides here). Called exactly ONCE, by `run_oos`, against the
    complete OOS series — never re-derived later from a possibly row-
    filtered `zoned_df` (see `run_oos`'s own docstring for why: this is
    what makes the resulting `az_reach_drift_max` COLUMN a stable, filter-
    proof broadcast value rather than something `metrics()` recomputes).

    NaN-safe: a frozen train array too small for `reach_prob_estimator`
    (< 2 points) -> NaN (documented "not computable" case — distinct from a
    caller-built frame simply missing the `az_reach_drift_max` column
    entirely, which `metrics()` treats as a hard error — see that
    function's docstring).
    """
    train_up = np.asarray(train_up, dtype=float)
    train_down = np.asarray(train_down, dtype=float)
    if train_up.size < 2 or train_down.size < 2:
        return float("nan")

    reach_up_est = reach_prob_estimator(train_up)
    reach_down_est = reach_prob_estimator(train_down)

    drift_up = reach_freq_drift(reach_up_est, np.asarray(oos_up, dtype=float), _X_GRID)
    drift_down = reach_freq_drift(reach_down_est, np.asarray(oos_down, dtype=float), _X_GRID)

    return float(max(drift_up["drift"].abs().max(), drift_down["drift"].abs().max()))


def _reach_drift_max_from_column(zoned_df: pd.DataFrame) -> float:
    """Extract ``reach_drift_max`` from ``zoned_df``'s ``az_reach_drift_max``
    column (a broadcast scalar, ``run_oos`` computes it once against the
    FULL OOS data and writes the SAME value to every row — see that
    function's own docstring for why a real column, not ``.attrs``).

    Raises ``KeyError`` if the column is ABSENT — deliberately NOT a silent
    NaN default: a bare NaN here would be indistinguishable from "no zoned
    entries" (this pipeline's existing NaN-safe convention for
    ``strict_coverage``/``realized_rr``), the wrong failure mode for a
    regime-drift SAFETY check whose whole point is to be noticed. A
    ``zoned_df`` produced by ``run_oos`` always has this column already; a
    hand-constructed test frame that does not care about this ONE metric
    must add the column explicitly (even as an explicit NaN value) rather
    than get a silent pass-through default — see
    ``tests/test_layer8_validate.py``'s hand-check tests for the pattern.
    """
    if "az_reach_drift_max" not in zoned_df.columns:
        raise KeyError(
            "metrics(): zoned_df is missing the 'az_reach_drift_max' "
            "column -- run_oos() always adds it (a broadcast scalar "
            "computed once against the FULL OOS data); a caller-"
            "constructed zoned_df must add it explicitly (even as an "
            "explicit NaN) rather than silently getting a NaN "
            "reach_drift_max indistinguishable from 'no zoned entries'"
        )
    return float(zoned_df["az_reach_drift_max"].iloc[0])


def metrics(zoned_df: pd.DataFrame, tf: int, direction: str, label_params: dict) -> dict:
    """``{"strict_coverage", "non_strict_coverage", "realized_rr",
    "reach_drift_max"}`` -- design spec §8's own success metrics (module
    docstring).

    ``strict_coverage``/``non_strict_coverage``: fraction of
    ``az_label_strict``/``az_label_nonstrict`` rows (``run_train``/
    ``run_oos`` already resolve these to FIXED column names, from whatever
    the ResultsFile-frozen ``label_params`` actually named them -- see
    ``run_oos``'s docstring) that fall inside ``az_zone_{direction}_{tf}``.

    ``realized_rr``: see ``_realized_rr``'s own docstring for the exact
    definition. ``reach_drift_max``: read straight off ``zoned_df``'s
    ``az_reach_drift_max`` column — see ``_reach_drift_max_from_column``'s
    docstring (raises if that column is absent, rather than a silent NaN).

    ``label_params`` is used ONLY to size ``realized_rr``'s forward-touch
    window (``label_params.get("n", 1)``, defaulting to ``1`` — the same
    default ``azlib.config.default_label_params`` itself uses) — NOT to
    locate the label columns (those are already fixed-named on
    ``zoned_df``, independent of the suffix any particular
    ``LabelParams`` would produce via ``azlib.loader.label_col``). This is
    why an EMPTY ``label_params={}`` is a valid call (task-8-brief.md's own
    integration test does exactly this) rather than needing every
    ``LabelParams`` field.

    Each of the 4 metrics is computed independently from whatever columns
    it needs and returns its OWN NaN default when they are absent (rather
    than the whole call raising ``KeyError``) — this lets a hand-crafted
    unit-test frame supply only the columns the ONE metric under test
    actually needs (e.g. a ``strict_coverage`` hand-check needs no
    ``az_tgt``/``az_sl``/``1_high``/``1_low``, and a ``realized_rr``
    hand-check needs no label columns at all).
    """
    zone_col = f"az_zone_{direction}_{tf}"
    zoned = zoned_df[zone_col].to_numpy(dtype=bool) if zone_col in zoned_df.columns else None
    strict = zoned_df["az_label_strict"].to_numpy(dtype=bool) if "az_label_strict" in zoned_df.columns else None
    non_strict = (
        zoned_df["az_label_nonstrict"].to_numpy(dtype=bool) if "az_label_nonstrict" in zoned_df.columns else None
    )

    n = int(label_params.get("n", 1)) if label_params else 1

    strict_coverage = _coverage(strict, zoned) if strict is not None and zoned is not None else float("nan")
    non_strict_coverage = (
        _coverage(non_strict, zoned) if non_strict is not None and zoned is not None else float("nan")
    )

    return {
        "strict_coverage": strict_coverage,
        "non_strict_coverage": non_strict_coverage,
        "realized_rr": _realized_rr(zoned_df, tf, direction, n),
        "reach_drift_max": _reach_drift_max_from_column(zoned_df),
    }
