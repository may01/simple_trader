"""tdlib/points.py -- Task 2: strong-point selection on the SLIM frame.

Three functions:

1. ``sym0_class(diff, cuts)`` -- bucket an ``{tf}_rsi_ma8_diff`` series into
   the same -2..+2 "symmetric zero" classes the real pipeline bakes into
   ``{tf}_move_class_sym0`` (see
   worktrees/rsi-quantile-sym0/indicators/library/classification.py::
   MoveClassSym0Field / _apply_cuts -- this is a standalone re-implementation
   of that exact formula against a plain ``list[float]`` cuts argument
   instead of a stats-file lookup, so tdlib has no runtime dependency on the
   indicators package).

2. ``strong_points(slim, tf, side)`` -- the SLIM rows where tf's candle is
   CLOSED and its RSI momentum is at a "strong" extreme (class +-2). This is
   the point-selection step: strong RSI-momentum extremes are the rows this
   experiment builds trend-detection ground truth FROM.

3. ``crosscheck_baked(slim, tf)`` -- sanity-check ``sym0_class`` against the
   real pipeline's own baked ``{tf}_move_class_sym0`` column, when present,
   so a drift between this module's formula and production's stats-driven
   one is caught by a fraction-agreement metric rather than silently
   diverging.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tdlib.config import MOVE_CUTS

_VALID_SIDES = (2, -2)


def sym0_class(diff: pd.Series, cuts: list) -> pd.Series:
    """Bucket ``diff`` into 5 classes (-2..+2) by the 4 ascending ``cuts``.

    ``np.digitize(diff.values, cuts, right=False) - 2``: with ``right=False``
    a value sitting exactly ON a cut lands in the UPPER tier (numpy's own
    boundary rule for increasing bins -- see the module docstring's
    production cross-reference for the identical formula/wording). So:

      - diff >= cuts[3]              -> class +2
      - cuts[2] <= diff < cuts[3]    -> class +1
      - cuts[1] <= diff < cuts[2]    -> class  0
      - cuts[0] <= diff < cuts[1]    -> class -1
      - diff <  cuts[0]              -> class -2

    NaN inputs: ``np.digitize`` itself puts NaN in the TOP bin (NaN compares
    False against everything, so it sorts as "past the last cut"), which
    would otherwise silently read as class +2 -- a real, non-missing signal.
    That's wrong for a missing value, so NaN positions are explicitly
    overwritten to class 0 (the middle/neutral class) AFTER digitizing,
    matching production's ``_apply_cuts`` (NaN -> the middle of its 5-tier
    scale).

    Returns an ``int8`` Series on ``diff``'s own index (order preserved).
    """
    values = diff.values
    idx = np.digitize(values, cuts, right=False) - 2
    idx = np.where(np.isnan(values), 0, idx)
    return pd.Series(idx, index=diff.index, dtype=np.int8)


def strong_points(slim: pd.DataFrame, tf: int, side: int) -> pd.DataFrame:
    """Rows of ``slim`` where tf's candle is closed AND its RSI-momentum
    class equals ``side`` (+2 or -2 -- the two "strong move" extremes; the
    +-1/0 classes are not "strong" and are never selected here).

    ``side`` must be +2 or -2 -- anything else is a caller bug, raised as
    ``ValueError`` (not silently coerced/ignored). ``tf`` must be a key of
    ``MOVE_CUTS`` (15/60/240) -- an unknown tf raises the natural
    ``KeyError`` straight out of the ``MOVE_CUTS[tf]`` lookup; that lookup
    happens before any ``slim`` column access and is not caught here, so the
    error names the actual bad key.

    Returns the matching row subset of ``slim`` (a view/shallow copy is
    fine -- callers only ever read from it downstream, never mutate it back
    into ``slim``). If nothing matches, returns an EMPTY frame that still
    carries ``slim``'s full column set, and emits a ``UserWarning`` naming
    the (tf, side) that produced no points -- an empty selection is valid
    (e.g. a quiet tf/side combo in a short sample) but surprising enough to
    be worth a heads-up rather than a silent empty result.
    """
    if side not in _VALID_SIDES:
        raise ValueError(f"strong_points: side must be one of {_VALID_SIDES}, got {side!r}")

    cuts = MOVE_CUTS[tf]  # KeyError propagates for an unknown tf -- not caught here

    closed = slim[f"{tf}_is_closed"] == True  # noqa: E712 - explicit bool-column compare
    cls = sym0_class(slim[f"{tf}_rsi_ma8_diff"], cuts)
    mask = closed & (cls == side)

    result = slim[mask]
    if result.empty:
        warnings.warn(f"strong_points: no rows selected for tf={tf}, side={side}")
    return result


def crosscheck_baked(slim: pd.DataFrame, tf: int) -> float:
    """Fraction of tf's CLOSED rows where ``sym0_class`` (this module's
    formula) agrees with the real pipeline's baked ``{tf}_move_class_sym0``
    column, when that column is present in ``slim`` -- ``slim`` is a
    dataset-specific reduction and won't always carry it (see task-1
    brief's SLIM whitelist: move_class_sym0 is an OPTIONAL per-tf suffix,
    silently dropped if absent from the source wide df).

    Returns ``float("nan")`` if the baked column is absent (nothing to
    cross-check against -- not a 0/1 answer), else a fraction in [0, 1]
    computed only over tf's closed rows (non-closed rows have no
    trustworthy baked value to compare against either way).
    """
    baked_col = f"{tf}_move_class_sym0"
    if baked_col not in slim.columns:
        return float("nan")

    closed = slim[f"{tf}_is_closed"] == True  # noqa: E712
    subset = slim.loc[closed]

    computed = sym0_class(subset[f"{tf}_rsi_ma8_diff"], MOVE_CUTS[tf])
    baked = subset[baked_col]

    agree = computed.to_numpy() == baked.to_numpy()
    return float(agree.mean())
