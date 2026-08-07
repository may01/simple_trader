"""azlib/zones.py — Task 7: zones + inference-capable results file.

Implements design spec §7
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):

  §7.1 ``zone_limit_price`` -- the exact algebraic inverse of
      ``azlib.space.coeff``'s price -> ``[0, 1]`` map: ``low_level +
      coeff*(high_level - low_level)``. Given an inferred coeff (Task 5's
      ``inferred_coeff`` = ``fused_mean + Y*fused_std``) and the SAME
      per-candle, held-constant ``price_high_level``/``price_low_level``
      Task 2's ``price_levels`` produces, converts back to a price-space
      entry threshold ("``zone_limit = price(inferred_label_coeff)``",
      design spec §7 step 1, verbatim).
  §7.2 ``mark_zones`` -- marks every 1-minute row whose entry extreme
      (pessimistic fill, same convention as ``azlib.space.label_coeff``:
      ``1_low`` for long, ``1_high`` for short) is on the profitable side of
      ``zone_limit`` (design spec §7 step 2).
  §7.3 ``build_zoned_dataset`` -- assembles the full per-row zoned dataset:
      the zone-membership boolean, the entry threshold, and target/stop
      PRICES derived from Task 6's selected ``(tgt_x, sl_x)`` via
      ``azlib.space.price_levels`` (DRY -- reuses Task 2's own
      percent->price conversion, just evaluated at a different ``x``;
      design spec §7 step 3).
  §7.4 ``ResultsFile`` -- the "inference-capable results file" the design
      spec's Restrictions section requires ("produces an inference-capable
      results file so OOS and future datasets can be re-marked") -- every
      knob needed to reproduce a zone-marking run on a new dataset, JSON
      round-trippable.

Carry-forwards from Task 6 (task-7-brief.md's "Constraints / notes",
REQUIRED):

  1. **Down-side reach wiring.** ``build_rr_levels`` (below) is the one
     place this module constructs ``azlib.rr``'s two reach-probability
     callables for a given ``(wide_df, tf, direction)``: ``reach_up =
     reach_prob_estimator({tf}_high_diff_prc_train)`` ALWAYS; ``reach_down =
     reach_prob_estimator(-{tf}_low_diff_prc_train)`` ALWAYS (negated) --
     ``azlib.rr``'s own sign convention (see that module's docstring),
     independent of ``direction`` (``rr_grid``'s ``direction`` argument
     alone decides which of the two is the target vs. the stop side).
     Getting a SHORT's negation wrong is the highest-risk mistake here (a
     short's PROFIT side is precisely the down/negated one) -- see
     ``test_layer7_zones.py``'s
     ``test_build_rr_levels_short_uses_negated_low_diff_prc_wiring``.
  2. **``select_levels`` raises when no profitable combo.**
     ``select_levels_safe`` wraps ``azlib.rr.select_levels``, catching the
     ``ValueError`` and returning a documented NaN sentinel dict (same 4
     keys ``select_levels`` itself returns, all ``float("nan")``) instead of
     propagating. ``build_zoned_dataset`` recognizes that sentinel (a NaN
     ``rr``) and returns an all-False zone with NaN ``tgt``/``sl``/``rr`` for
     that ``(tf, direction)`` rather than crashing the whole run.

All functions here are pure -- no argument is mutated in place. Artifacts
(zoned datasets, results files) are written to a caller-supplied path only
(``ResultsFile.save``'s ``path`` argument) -- this module never decides
where on disk anything lives; that is a driver's job (Task 8/9), per the
design spec's Restrictions ("artifacts written only to the mounted volume,
never into the worktree").
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from azlib.rr import reach_prob_estimator, rr_grid, select_levels
from azlib.space import price_levels

# --- zone_limit_price -----------------------------------------------------


def zone_limit_price(
    inferred_coeff: np.ndarray, low_level: np.ndarray, high_level: np.ndarray
) -> np.ndarray:
    """Inverse of ``azlib.space.coeff``'s price -> ``[0, 1]`` map.

    ``low_level + inferred_coeff * (high_level - low_level)`` -- the exact
    algebraic inverse of ``coeff``'s ``(price - low_level) / (high_level -
    low_level)`` (design spec §7 step 1, "zone_limit =
    price(inferred_label_coeff)").

    Unlike ``coeff``, this is **not** clamped: ``inferred_coeff`` (Task 5's
    ``mean + Y*std``) is allowed to fall outside ``[0, 1]`` by design (see
    ``azlib.infer.inferred_coeff``'s own docstring) -- extrapolating the
    zone threshold beyond the observed high/low band is a valid, intentional
    outcome of a large ``|Y|``, not an error to clamp away.

    DRY note (task-7-brief.md): ``azlib.infer.sweep_y`` computes this exact
    same formula inline (kept there rather than imported from here, to avoid
    an ``infer``<->``zones`` import cycle -- see that module's docstring).
    Both implementations must be kept in agreement --
    ``test_layer7_zones.py``'s
    ``test_zone_limit_price_agrees_with_sweep_y_inline_copy`` cross-checks
    them against each other.
    """
    inferred_coeff = np.asarray(inferred_coeff, dtype=float)
    low_level = np.asarray(low_level, dtype=float)
    high_level = np.asarray(high_level, dtype=float)
    return low_level + inferred_coeff * (high_level - low_level)


# --- mark_zones -------------------------------------------------------------


def _entry_price_column(direction: str) -> str:
    """Column of ``wide_df`` holding each row's pessimistic-fill entry
    price -- ``1_low`` for long, ``1_high`` for short. Same choice as
    ``azlib.space.label_coeff``/``azlib.infer.sweep_y``'s own entry-extreme
    convention (see those modules' docstrings).
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    return "1_low" if direction == "long" else "1_high"


def mark_zones(wide_df: pd.DataFrame, tf: int, direction: str, zone_limit: np.ndarray) -> pd.Series:
    """Bool per row: is this row's entry extreme inside the zone (design spec §7 step 2)?

    ``long``: ``1_low < zone_limit``; ``short``: ``1_high > zone_limit`` --
    same pessimistic-fill entry-extreme convention as
    ``azlib.space.label_coeff``/``azlib.infer.sweep_y``.

    ``tf`` is not itself used in the comparison (``zone_limit`` is already a
    fully-formed price-space array, one value per row of ``wide_df``) -- kept
    to name the returned Series ``az_zone_{direction}_{tf}``, the exact
    column-name convention ``build_zoned_dataset`` needs (task-7-brief.md's
    "Column prefix ``az_`` ... so a later ``join_action_zones`` [Layer 10]
    can surface them without name clashes").

    A NaN ``zone_limit`` at some row (e.g. Task 2's real ``price_levels``
    warm-up rows) makes that row's comparison ``False`` by plain IEEE-754 NaN
    comparison semantics -- never marked, no ``RuntimeWarning`` (NaN
    comparisons, unlike NaN arithmetic, never warn -- same reasoning as
    ``azlib.infer.sweep_y``'s own docstring).

    Raises ``ValueError`` if ``direction`` is not ``"long"``/``"short"``.
    """
    entry_col = _entry_price_column(direction)
    entry_price = wide_df[entry_col].to_numpy(dtype=float)
    zone_limit_arr = np.asarray(zone_limit, dtype=float)

    if direction == "long":
        marked = entry_price < zone_limit_arr
    else:
        marked = entry_price > zone_limit_arr

    return pd.Series(marked, index=wide_df.index, name=f"az_zone_{direction}_{tf}")


# --- select_levels_safe (Task 6 carry-forward #2) ---------------------------

# Sentinel dict returned by select_levels_safe when azlib.rr.select_levels
# raises ValueError (no (tgt_x, sl_x) combo has a positive
# expected_return_after_fees) -- same 4 keys select_levels itself returns,
# all NaN. build_zoned_dataset recognizes this sentinel (any of the 4 keys'
# NaN-ness suffices -- see _is_no_profitable_sentinel) and produces an
# all-False zone / NaN tgt-sl-rr instead of crashing.
_NO_PROFITABLE_LEVELS = {
    "tgt_x": float("nan"),
    "sl_x": float("nan"),
    "rr": float("nan"),
    "exp_ret": float("nan"),
}


def select_levels_safe(grid: pd.DataFrame) -> dict:
    """``azlib.rr.select_levels``, with the "no profitable combo"
    ``ValueError`` caught and converted into a documented NaN sentinel
    (task-7-brief.md's required Task 6 carry-forward #2) instead of
    propagating.

    Returns ``select_levels(grid)`` unchanged when at least one combo is
    profitable; otherwise returns a dict with the exact same 4 keys
    (``{"tgt_x", "sl_x", "rr", "exp_ret"}``), all ``float("nan")`` -- so
    callers never need a ``try``/``except`` of their own and can treat the
    return value uniformly.
    """
    try:
        return select_levels(grid)
    except ValueError:
        return dict(_NO_PROFITABLE_LEVELS)


def _is_no_profitable_sentinel(levels: dict) -> bool:
    """True if ``levels`` is (or matches the shape of) ``select_levels_safe``'s
    NaN sentinel -- checked via ``rr``'s NaN-ness alone. ``azlib.rr``'s
    ``select_levels`` never returns a NaN ``rr`` for a genuinely profitable
    combo: its own selection filter (``expected_return_after_fees > 0.0``)
    already excludes any NaN row via plain NaN-comparison-is-False
    semantics, so a real winning row's ``rr`` is always finite.
    """
    return math.isnan(levels["rr"])


# --- build_rr_levels (Task 6 carry-forward #1: down-side reach wiring) ------


def build_rr_levels(
    wide_df: pd.DataFrame,
    tf: int,
    direction: str,
    x_grid: np.ndarray,
    fee: float,
    candle_size: float,
    min_bin: int = 50,
) -> dict:
    """One-call per-``(tf, direction)`` R/R level selection.

    Builds both reach-probability estimators with ``azlib.rr``'s documented
    sign convention, runs the ``(tgt_x, sl_x)`` grid search, and returns the
    winning combo (or the no-profitable-levels sentinel -- see
    ``select_levels_safe``).

    ``reach_up``/``reach_down`` are built the SAME way regardless of
    ``direction`` (task-7-brief.md's required carry-forward #1 --
    ``azlib.rr``'s own module docstring: "no separate down code path")::

        reach_up   = reach_prob_estimator(wide_df[f"{tf}_high_diff_prc"].dropna())
        reach_down = reach_prob_estimator(-wide_df[f"{tf}_low_diff_prc"].dropna())  # NEGATED

    ``rr_grid``'s own ``direction`` argument is what then decides which of
    the two is the target vs. the stop side (for ``"short"``, ``reach_down``
    -- the negated one -- becomes the TARGET side, since a short's profit
    comes from a down move; forgetting the negation here would silently make
    every short's target probability come from the WRONG tail of the
    distribution). ``x_grid``/``fee``/``candle_size``/``min_bin`` are passed
    straight through to ``reach_prob_estimator``/``rr_grid`` unchanged (see
    ``azlib.rr``'s own docstrings for their meaning/units).

    Returns ``select_levels_safe(grid)`` -- either the winning ``{"tgt_x",
    "sl_x", "rr", "exp_ret"}`` dict, or the NaN no-profitable-levels
    sentinel.
    """
    high_diff_prc_train = wide_df[f"{tf}_high_diff_prc"].dropna().to_numpy()
    low_diff_prc_train = wide_df[f"{tf}_low_diff_prc"].dropna().to_numpy()

    reach_up = reach_prob_estimator(high_diff_prc_train, min_bin=min_bin)
    reach_down = reach_prob_estimator(-low_diff_prc_train, min_bin=min_bin)  # negated -- see docstring

    grid = rr_grid(reach_up, reach_down, x_grid, fee=fee, candle_size=candle_size, direction=direction)
    return select_levels_safe(grid)


# --- build_zoned_dataset ------------------------------------------------------


def build_zoned_dataset(
    wide_df: pd.DataFrame, tf: int, direction: str, zone_limit: np.ndarray, levels: dict
) -> pd.DataFrame:
    """Assemble the full per-row zoned dataset (design spec §7 steps 2-3).

    ``levels``: a ``{"tgt_x", "sl_x", "rr", ...}`` dict -- typically
    ``azlib.rr.select_levels``'s (or this module's ``select_levels_safe``'s)
    return value. Columns of the returned ``pd.DataFrame`` (indexed exactly
    like ``wide_df``):

    - ``az_zone_{direction}_{tf}`` (bool): ``mark_zones(wide_df, tf,
      direction, zone_limit)``.
    - ``az_entry`` (float): ``zone_limit`` itself, broadcast to a Series.
    - ``az_tgt``/``az_sl`` (float, price space): the selected ``(tgt_x,
      sl_x)`` converted to PRICES via ``azlib.space.price_levels`` -- reused
      (DRY), not reimplemented, evaluated at each of the two ``x`` values
      task-7-brief.md's "tgt/sl price mapping" pins EXACTLY::

          long:  az_tgt = price_levels(wide_df, tf, x=levels["tgt_x"])[0]  # up/high side
                 az_sl  = price_levels(wide_df, tf, x=levels["sl_x"])[1]   # low side
          short: az_tgt = price_levels(wide_df, tf, x=levels["tgt_x"])[1]  # down/low side
                 az_sl  = price_levels(wide_df, tf, x=levels["sl_x"])[0]   # high side

      (a long's target is an UP move, priced off the high-side level at
      ``tgt_x``; its stop is a DOWN move, priced off the low-side level at
      ``sl_x``; a short is the mirror image.)
    - ``az_rr`` (float): ``levels["rr"]``, broadcast to every row.

    **No-profitable-levels handling** (task-7-brief.md's required Task 6
    carry-forward #2): when ``levels`` is ``select_levels_safe``'s NaN
    sentinel (recognized via a NaN ``levels["rr"]`` -- see
    ``_is_no_profitable_sentinel``), the zone column is forced all-``False``
    (regardless of what ``zone_limit`` itself would otherwise mark -- there
    is no profitable exit for this ``(tf, direction)``, so no entry should
    be taken here at all) rather than calling ``mark_zones``.
    ``az_tgt``/``az_sl`` come out all-NaN automatically in this case too:
    ``price_levels(..., x=float("nan"))`` propagates NaN through its
    ``ma +/- x*std`` formula at every row, the same ordinary NaN-arithmetic
    propagation ``azlib.space`` already relies on elsewhere (no special
    casing needed, no warning under this project's ``-W error`` policy).
    ``az_rr`` is ``levels["rr"]`` unchanged (already NaN in this case).

    Raises ``ValueError`` if ``direction`` is not ``"long"``/``"short"``
    (same guard ``mark_zones``/``_entry_price_column`` use, checked up front
    here too so a bad ``direction`` fails before any ``price_levels`` work).
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    zone_col = f"az_zone_{direction}_{tf}"
    index = wide_df.index

    if _is_no_profitable_sentinel(levels):
        zone_series = pd.Series(False, index=index, name=zone_col)
    else:
        zone_series = mark_zones(wide_df, tf, direction, zone_limit)

    tgt_high, tgt_low = price_levels(wide_df, tf, x=levels["tgt_x"])
    sl_high, sl_low = price_levels(wide_df, tf, x=levels["sl_x"])

    if direction == "long":
        az_tgt, az_sl = tgt_high, sl_low
    else:
        az_tgt, az_sl = tgt_low, sl_high

    out = pd.DataFrame(index=index)
    out[zone_col] = zone_series.to_numpy()
    out["az_entry"] = np.asarray(zone_limit, dtype=float)
    out["az_tgt"] = az_tgt.to_numpy()
    out["az_sl"] = az_sl.to_numpy()
    out["az_rr"] = levels["rr"]
    return out


# --- ResultsFile ---------------------------------------------------------------


@dataclass
class ResultsFile:
    """Inference-capable results file (design spec's Restrictions: "produces
    an inference-capable results file so OOS and future datasets can be
    re-marked"). Bundles every knob needed to reproduce a zone-marking run
    on a NEW dataset without re-fitting anything:

    - ``tf``/``direction``: which ``(timeframe, side)`` this result is for.
    - ``label_params``: the ``azlib.loader.LabelParams`` knobs used to build
      the strict/non-strict labels this result was tuned against (kept as a
      plain ``dict`` here, not the dataclass itself, so this module does not
      need to import ``azlib.loader`` just to round-trip a results file --
      a caller reconstructs a real ``LabelParams`` from it if needed, e.g.
      via ``LabelParams(**label_params)``).
    - ``freeze_stats_path``: where the ``azlib.indicators.FreezeStats``
      (train-frozen indicator mean/std) that the regression models this
      result depends on were fit against was saved -- a path reference, not
      the stats themselves, so a results file stays small and the stats
      stay a single source of truth on disk.
    - ``reg_models``: the regression-model identifiers (e.g.
      ``"rsi:linear"``) fused (Task 5) to produce this result's inferred
      coeff -- enough for a later re-inference run to know which saved
      ``RegResult`` files (``azlib.models.save_result``) to reload.
    - ``y``: the selected Y (``azlib.infer.select_y``'s output) this
      result's zone was built at.
    - ``tgt_x``/``sl_x``: the selected R/R levels (``azlib.rr.select_levels``
      /``build_rr_levels``'s ``tgt_x``/``sl_x``), or ``float("nan")`` for
      the Task 6 carry-forward's no-profitable-levels sentinel (see
      ``select_levels_safe``).
    - ``fee``: the fractional trading-fee rate the R/R grid search used
      (``azlib.rr.rr_grid``'s own ``fee`` parameter).
    - ``selected_models`` (Task §10, OPTIONAL -- defaults to ``[]``): the
      HUMAN-SELECTED ``{"indicator", "attrs", "kind"}`` spec list
      ``azlib.validate.run_train``'s own ``selected_models`` argument was
      called with, if any (e.g. ``[{"indicator": "rsi", "attrs":
      ["position", "slope", "distance"], "kind": "gbr"}]``) -- persisted
      verbatim purely to DOCUMENT which model(s) this result's zone was
      built from; ``[]`` for the legacy "every indicator x 1D linear"
      fan-out. Not consulted by ``run_oos`` for re-inference -- ``reg_models``
      (the saved ``RegResult`` file paths) is the only thing OOS prediction
      actually reloads; this field is a record, not a control input.

    ``save``/``load`` round-trip every field through plain JSON -- every
    field here is itself JSON-native (``int``/``str``/``dict``/``list``/
    ``float``, Python's ``json`` module round-trips ``NaN`` too, by
    default), so no special-casing is needed here (unlike, e.g.,
    ``azlib.models.save_result``, which additionally persists a
    non-JSON-native fitted sklearn estimator via a sibling ``.joblib`` file
    -- there is no such non-JSON-native payload in a ``ResultsFile``).
    ``selected_models`` defaults to ``field(default_factory=list)`` so an
    OLDER saved ``results.json`` (written before this field existed, with no
    ``"selected_models"`` key at all) still loads cleanly via
    ``cls(**payload)`` -- backward-compatible round-trip, not just a
    forward one.
    """

    tf: int
    direction: str
    label_params: dict
    freeze_stats_path: str
    reg_models: list[str]
    y: float
    tgt_x: float
    sl_x: float
    fee: float
    selected_models: list = field(default_factory=list)

    def save(self, path: str) -> None:
        """Write every field to ``path`` as plain JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str) -> "ResultsFile":
        """Reconstruct a ``ResultsFile`` saved by ``save`` -- exact round trip."""
        with open(path) as f:
            payload = json.load(f)
        return cls(**payload)
