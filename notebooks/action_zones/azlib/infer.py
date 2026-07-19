"""azlib/infer.py — Task 5: inference, inverse-variance fusion & Y-sweep.

Implements design spec §11
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):

  5.1 (Task 4, not here) run selected regression models over the dataset ->
      per-GROUP (mean, std) of ``label_coeff`` for every point.
  5.2 ``fuse_inverse_variance`` -- collapse those per-group (mean, std)
      pairs into a single per-point (mean, std) via inverse-variance
      weighting.
  5.3/5.4 ``inferred_coeff`` = ``fused_mean + Y * fused_std``; ``sweep_y``
      sweeps ``Y`` over a grid, converts each ``Y``'s ``inferred_coeff`` to
      a price-space zone (via the SAME ``low_level + coeff*(high_level -
      low_level)`` inverse map Task 7's ``zone_limit_price`` will formalize
      -- see ``sweep_y``'s docstring), marks 1-minute rows inside that
      zone, and reports ``strict_coverage``/``realized_rr``/``n_zoned`` per
      ``Y``. ``select_y`` picks the ``Y`` that maximizes ``strict_coverage``
      among rows whose ``realized_rr`` is profitable.

Dependency inversion (task-5-brief.md): Task 6 (``azlib/rr.py``, the real
realized-R/R calculation from reach-probabilities) does not exist yet.
``sweep_y`` therefore takes an INJECTED ``rr_fn`` callback
(``zone_marking -> float``) instead of importing anything from a not-yet-
existent module -- this keeps Layer 5 fully testable today (tests pass a
stub ``rr_fn``) and Layer 6 pluggable later without any signature change
here. ``sweep_y`` similarly takes an injected ``price_levels_fn`` (matching
``azlib.space.price_levels``'s ``(wide_df, tf) -> (price_high_level,
price_low_level)`` signature) rather than importing ``azlib.space``
directly -- Task 5's OWN tests use a fully controllable stub instead of
Task 2's real per-candle, warm-up-NaN levels (see ``sweep_y``'s docstring
and ``tests/conftest.py``'s ``synthetic_pipeline`` fixture), while a real
notebook driver (Task 9) passes the real ``azlib.space.price_levels``
unchanged.

All functions here are pure -- no argument is mutated in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- fuse_inverse_variance ---------------------------------------------------

# Floor applied to every group's std BEFORE squaring/inverting into a
# fusion weight (design spec §11.2's "weight = 1/std^2 per point").
# Two jobs, both from task-5-brief.md's "Semantics" section:
#   1. Guard div-by-zero -- a group whose std is EXACTLY 0 at some point
#      (e.g. a degenerate/constant-prediction region) would otherwise
#      compute `1.0 / 0.0 ** 2`, which raises a numpy RuntimeWarning
#      ("divide by zero") under this project's `-W error` test policy, and
#      would poison the fused result with +inf/nan.
#   2. Avoid a model with an UNDERSTATED std getting near-infinite fusion
#      weight -- e.g. `gbr`'s in-sample residual std (azlib.models) can be
#      spuriously tiny in a training region the model has essentially
#      memorized, even though real out-of-sample uncertainty there is not
#      actually near-zero. Without a floor, one such point would swamp
#      every other (better-calibrated) group's contribution at that point.
# Value: 1e-3. `label_coeff` (the quantity these stds describe) lives in
# [0, 1], and task-4-brief.md's own fixtures produce residual stds on the
# order of 0.05-0.3 for realistic fits -- 1e-3 is roughly two orders of
# magnitude below that "typical" range (so it only bites pathologically
# small/zero stds, never an ordinary well-fit model's own std) while still
# being comfortably above float64 zero (weight caps at 1e6, not inf).
_STD_FLOOR = 1e-3


def fuse_inverse_variance(means: np.ndarray, stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance fusion of per-group (mean, std) into one (mean, std).

    ``means``/``stds``: shape ``[n_groups, n_points]`` -- row ``g`` is one
    regression group's per-point ``predict_reg`` output (design spec
    §11.1/§11.2). Returns ``(fused_mean, fused_std)``, each shape
    ``[n_points]``:

    ::

        weight[g, p] = 1 / max(stds[g, p], _STD_FLOOR) ** 2
        fused_mean[p] = sum_g(weight[g, p] * means[g, p]) / sum_g(weight[g, p])
        fused_std[p]  = sqrt(1 / sum_g(weight[g, p]))

    **Std floor** (``_STD_FLOOR``, see module-level constant docstring
    above): applied to every group's std BEFORE squaring/inverting, so a
    zero (or near-zero) std never produces an infinite weight / div-by-zero
    warning, and no single group's spuriously tiny reported std can
    dominate to the point of ignoring every other group entirely.

    **NaN handling** (per point, per group): a group that is NaN (mean OR
    std) at point ``p`` is EXCLUDED from that point's fusion -- its weight
    is forced to 0 rather than propagating NaN through the weighted sum.
    If ALL groups are NaN at point ``p``, both ``fused_mean[p]`` and
    ``fused_std[p]`` are NaN (there is nothing to fuse). Implemented via
    the same "swap in a dummy non-zero denominator, then overwrite the
    result" pattern as ``azlib.space.coeff`` -- the actual 0/0 division is
    never performed, so no RuntimeWarning is raised at an all-NaN point
    either.
    """
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    if means.shape != stds.shape:
        raise ValueError(f"means/stds shape mismatch: {means.shape} vs {stds.shape}")

    valid = ~np.isnan(means) & ~np.isnan(stds)

    floored_std = np.maximum(stds, _STD_FLOOR)
    weight = 1.0 / floored_std**2
    weight = np.where(valid, weight, 0.0)
    means_safe = np.where(valid, means, 0.0)  # value irrelevant where weight is 0

    sum_w = weight.sum(axis=0)
    sum_wm = (weight * means_safe).sum(axis=0)

    all_nan = sum_w == 0.0  # every group NaN (or n_groups == 0) at this point
    safe_sum_w = np.where(all_nan, 1.0, sum_w)  # dummy denominator, never actually 0 -> no warning

    fused_mean = np.where(all_nan, np.nan, sum_wm / safe_sum_w)
    fused_std = np.where(all_nan, np.nan, np.sqrt(1.0 / safe_sum_w))

    return fused_mean, fused_std


# --- inferred_coeff -----------------------------------------------------------


def inferred_coeff(mean: np.ndarray, std: np.ndarray, y: float) -> np.ndarray:
    """``inferred_coeff = mean + y * std`` (design spec §11.3/§11.4).

    ``y=0`` -> ``mean`` unchanged; ``y=1`` -> ``mean + std`` (one std above
    the fused prediction). ``y`` ranges over ``[-2.0, 2.0]`` in ``sweep_y``
    below, but this function itself accepts any float -- no clamping here
    (the resulting ``inferred_coeff`` is intentionally allowed to fall
    outside ``[0, 1]``; it is a linear extrapolation of the coeff scale,
    not itself a coeff -- see ``sweep_y``'s docstring for how it is turned
    into a price-space zone).
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return mean + y * std


# --- sweep_y ------------------------------------------------------------------

# "Profitable" threshold for select_y (task-5-brief.md: "rr > 1 or
# exp-return>0 -- define clearly"). realized_rr here is whatever `rr_fn`
# returns for a given Y's zone marking; until Task 6 (azlib.rr) exists,
# that is a reward:risk RATIO (the classic R/R convention design spec §5
# and §11 both use, "R/R = P(target) / P(stop)") -- profitable means that
# ratio exceeds 1 (expected reward bigger than expected risk), not an
# absolute expected-return-after-fees test (that quantity only exists once
# Task 6's real p_target/p_stop/fee machinery is wired in).
_PROFITABLE_RR_THRESHOLD = 1.0

# select_y's NaN-safe default when NO row in the sweep is profitable
# (task-5-brief.md: "returns a documented default"). 0.0 -- the midpoint
# of the [-2.0, 2.0] sweep range, i.e. "trust the fused mean with no
# std-based adjustment" -- is always a valid, in-range Y regardless of
# what the sweep grid actually contained, and reduces inferred_coeff back
# to the plain fused mean (inferred_coeff(mean, std, 0.0) == mean), the
# least-opinionated fallback available.
_SELECT_Y_DEFAULT = 0.0


def _entry_price_column(direction: str) -> str:
    """Column of ``wide_df`` holding each row's pessimistic-fill entry
    price -- ``1_low`` for long, ``1_high`` for short. Same choice as
    ``azlib.space.label_coeff``'s own entry-extreme convention (see that
    module's docstring); kept here as a tiny private helper so ``sweep_y``
    validates ``direction`` in exactly one place.
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    return "1_low" if direction == "long" else "1_high"


def sweep_y(
    fused_mean,
    fused_std,
    wide_df: pd.DataFrame,
    tf: int,
    direction: str,
    price_levels_fn,
    strict_label,
    y_grid,
    rr_fn,
) -> pd.DataFrame:
    """Sweep ``Y`` over ``y_grid``; per ``Y``, form a zone and score it.

    **Final signature** (documented here for Task 7/8, which call this):
    ``sweep_y(fused_mean, fused_std, wide_df, tf, direction,
    price_levels_fn, strict_label, y_grid, rr_fn) -> pd.DataFrame``.

    Parameters
    ----------
    fused_mean, fused_std : array-like, one value per row of ``wide_df``
        ``fuse_inverse_variance``'s output (or any per-row (mean, std)
        pair) -- IN THE SAME ROW ORDER as ``wide_df`` (a bare numpy array
        is taken positionally; a ``pd.Series`` has its OWN index discarded
        and is re-wrapped onto ``wide_df.index`` positionally too -- this
        function never re-aligns by label, only by row position, matching
        how ``fuse_inverse_variance`` and ``predict_reg`` produce plain
        arrays in the caller's original row order).
    wide_df : the wide df those predictions were made over.
    tf : timeframe the zone applies to (passed through to
        ``price_levels_fn``).
    direction : ``"long"`` or ``"short"`` -- picks the entry-extreme column
        (``1_low``/``1_high``, matching ``azlib.space.label_coeff``) and
        the zone-membership inequality (see below).
    price_levels_fn : ``(wide_df, tf) -> (price_high_level, price_low_level)``
        -- matches ``azlib.space.price_levels``'s call signature (called
        with 2 positional args only; that function's ``window``/``x`` keep
        their own defaults). Injected rather than imported directly so
        Task 5's own tests can hand in a fully controllable, NaN-free
        stub instead of depending on Task 2's real per-candle warm-up
        behavior (see ``tests/conftest.py``'s ``synthetic_pipeline``
        fixture) -- a real notebook driver passes
        ``azlib.space.price_levels`` unchanged.
    strict_label : boolean array-like, one value per row of ``wide_df``
        (same row-order convention as ``fused_mean``/``fused_std`` above)
        -- True where that row is a strict-positive label (design spec §8;
        ``azlib.loader.add_labels`` + ``label_col(..., strict=True)``).
    y_grid : iterable of floats, the ``Y`` values to sweep (design spec
        §11.4: ``Y in [-2.0, 2.0]``, step 0.1 typical -- this function does
        not itself clamp/validate the grid, the caller decides its range).
    rr_fn : ``(zone_marking: pd.Series[bool]) -> float`` -- Task 6's
        realized-R/R calculation, injected (dependency inversion:
        ``azlib/rr.py`` does not exist yet, see module docstring). Called
        once per ``Y`` with that ``Y``'s zone-membership boolean Series
        (indexed like ``wide_df``).

    Zone construction (per ``Y``, design spec §11.5/§7):

    ::

        coeff_y     = inferred_coeff(fused_mean, fused_std, Y)      # unclamped
        zone_limit  = price_low_level + coeff_y * (price_high_level - price_low_level)
        # ^ inverse of azlib.space.coeff's price->[0,1] map -- Task 7's
        #   zone_limit_price will formalize this exact formula; duplicated
        #   inline here (not imported) because azlib/zones.py does not
        #   exist yet -- see module docstring's dependency-inversion note.
        marked      = (1_low < zone_limit)   if direction == "long"
                    = (1_high > zone_limit)  if direction == "short"

    A row whose ``fused_mean``/``fused_std``/price-level inputs are NaN
    (e.g. Task 2's real ``price_levels`` warm-up rows) gets ``zone_limit ==
    NaN`` -- the ``<``/``>`` comparison against NaN is `False` by plain
    IEEE-754 comparison semantics (no RuntimeWarning; NaN comparisons,
    unlike NaN arithmetic, never warn), so that row is simply never marked,
    with no special-casing needed.

    **Monotonicity in Y (documented direction, task-5-brief.md)**: assuming
    ``fused_std >= 0`` (it is a std) and ``price_high_level >
    price_low_level`` (the expected/typical case for
    ``azlib.space.price_levels``'s output -- an upper level above a lower
    level), ``coeff_y`` and therefore ``zone_limit`` are both
    non-decreasing in ``Y``. For **long** (``1_low < zone_limit``), a
    non-decreasing threshold can only ADD rows to the marked set as ``Y``
    increases -- the zone WIDENS (``n_zoned`` non-decreasing in ``Y``, and
    so is ``strict_coverage``, since the marked set only grows). For
    **short** (``1_high > zone_limit``), a non-decreasing threshold can
    only REMOVE rows -- the zone NARROWS (``n_zoned`` non-increasing in
    ``Y``).

    Returns a tidy ``pd.DataFrame`` with exactly the columns ``{"y",
    "strict_coverage", "realized_rr", "n_zoned"}``, one row per
    ``y_grid`` entry, in ``y_grid``'s own order.

    ``strict_coverage`` = ``|strict_label & marked| / |strict_label|`` --
    the fraction of strict-labeled rows that land inside that ``Y``'s zone
    (always in ``[0, 1]`` when there is at least one strict-labeled row).
    **NaN-safe default**: if ``strict_label`` has zero True rows at all
    (coverage of an empty set is undefined), every row's
    ``strict_coverage`` is ``NaN`` -- guarded with a plain Python
    ``if``/scalar division (not an array-level `0/0`), so no RuntimeWarning
    either.
    """
    entry_col = _entry_price_column(direction)
    index = wide_df.index

    mean_arr = fused_mean.to_numpy(dtype=float) if isinstance(fused_mean, pd.Series) else np.asarray(fused_mean, dtype=float)
    std_arr = fused_std.to_numpy(dtype=float) if isinstance(fused_std, pd.Series) else np.asarray(fused_std, dtype=float)
    strict_arr = strict_label.to_numpy(dtype=bool) if isinstance(strict_label, pd.Series) else np.asarray(strict_label, dtype=bool)

    price_high_level, price_low_level = price_levels_fn(wide_df, tf)
    high_arr = np.asarray(price_high_level, dtype=float)
    low_arr = np.asarray(price_low_level, dtype=float)

    entry_arr = wide_df[entry_col].to_numpy(dtype=float)

    n_strict = int(strict_arr.sum())

    rows = []
    for y in y_grid:
        y = float(y)
        coeff_y = inferred_coeff(mean_arr, std_arr, y)
        zone_limit = low_arr + coeff_y * (high_arr - low_arr)

        if direction == "long":
            marked_arr = entry_arr < zone_limit
        else:
            marked_arr = entry_arr > zone_limit
        marked = pd.Series(marked_arr, index=index)

        n_zoned = int(marked_arr.sum())

        if n_strict == 0:
            strict_coverage = float("nan")
        else:
            strict_coverage = float(np.sum(strict_arr & marked_arr)) / n_strict

        realized_rr = rr_fn(marked)

        rows.append(
            {
                "y": y,
                "strict_coverage": strict_coverage,
                "realized_rr": realized_rr,
                "n_zoned": n_zoned,
            }
        )

    return pd.DataFrame(rows, columns=["y", "strict_coverage", "realized_rr", "n_zoned"])


# --- select_y -------------------------------------------------------------


def select_y(sweep: pd.DataFrame) -> float:
    """``Y`` maximizing ``strict_coverage`` among profitable rows.

    "Profitable" = ``realized_rr > _PROFITABLE_RR_THRESHOLD`` (``1.0`` --
    see that constant's docstring above: a reward:risk RATIO greater than
    1, the design spec's own R/R convention, not an absolute
    expected-return test). Among rows passing that filter, returns the
    ``y`` of whichever has the largest ``strict_coverage`` (NaN
    ``strict_coverage`` values, e.g. from a sweep with zero strict-labeled
    rows, are skipped by ``pandas``' default ``idxmax`` ``skipna=True``
    behavior, so they never win a comparison against a real number but
    also never raise merely by being present).

    **NaN-safe default** (documented, task-5-brief.md): if NO row is
    profitable (including the empty-sweep and all-rows-non-profitable
    cases), or every profitable row's ``strict_coverage`` is itself NaN,
    returns ``_SELECT_Y_DEFAULT`` (``0.0`` -- the midpoint of the
    ``[-2.0, 2.0]`` sweep range; ``inferred_coeff(mean, std, 0.0) ==
    mean``, i.e. "trust the fused mean with no std-based adjustment", the
    least-opinionated fallback).
    """
    if sweep.empty:
        return _SELECT_Y_DEFAULT

    profitable = sweep["realized_rr"] > _PROFITABLE_RR_THRESHOLD
    candidates = sweep.loc[profitable]

    if candidates.empty or candidates["strict_coverage"].isna().all():
        return _SELECT_Y_DEFAULT

    best_idx = candidates["strict_coverage"].idxmax()
    return float(candidates.loc[best_idx, "y"])
