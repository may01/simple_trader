"""azlib/space.py — Task 2: action space (price levels + coeff) + label_coeff.

Implements design spec §1-2
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):
percentage-change levels (``diff_prc_ma +/- X*std``), converted to price
levels referenced against the *previous completed same-TF candle*, then a
price-space ``coeff`` mapping any price onto ``[0, 1]`` between those two
levels. ``label_coeff`` plugs the entry extreme (pessimistic fill: 1-min low
for long, 1-min high for short) through that mapping — the regression target
for later layers.

Completed-candle reduction (the core correctness requirement — see
.superpowers/sdd/task-2-brief.md and the design spec's §1). ``{tf}_high`` /
``{tf}_low`` on the wide df are per-minute *forming* cummax/cummin within the
current tf-period bucket — NOT a candle's final high/low until the row where
``{tf}_is_closed`` is True. ``price_levels`` therefore:

  1. Reduces to one row per COMPLETED candle via ``{tf}_is_closed`` — that
     closed row's ``{tf}_high``/``{tf}_low`` is the candle's final value.
  2. Computes ``diff_prc``/``diff_prc_ma``/``diff_prc_std`` on that
     per-candle sequence (candle-to-candle), never on the 1-minute forming
     series.
  3. For a forming candle *c*, uses the PREVIOUS completed candle's
     (index *c-1*) ma/std/reference-high/reference-low — never *c*'s own
     (still-forming, not-yet-final) data. This is what makes the result
     look-ahead free: a forming candle's own high/low can be mutated freely
     without changing its own levels (see
     ``test_price_levels_no_look_ahead_from_own_forming_candle`` in
     ``tests/test_layer2_space.py``).
  4. Broadcasts each candle's two level values to every 1-minute row of
     that candle (held constant for the whole bucket, including that
     candle's own closing minute — the closing minute is still part of
     *that* candle's forming period, not the next one's).

This is a rework of the version implemented in commit 40b5fe7, which
computed levels via a plain ``.shift(1)`` on the 1-minute *forming*
``{tf}_high``/``{tf}_low`` columns — a minute-to-minute forming increment,
not the candle-to-candle change the design spec calls for, and not
held-constant/look-ahead-free. See task-2-report.md for the full
before/after writeup.

Reuse vs compute (see the design spec's Reuse map and task-2-brief.md):
  - The wide df's ``{tf}_{side}_diff_prc`` / ``{tf}_{side}_diff_prc_rm_{w}``
    1-minute columns are NEVER reused here, even when present — they are
    forming-candle (minute-to-minute) values, not the closed-candle
    (candle-to-candle) sequence this module needs. ``price_levels`` always
    computes its own diff/ma/std internally from the completed-candle
    reduction described above.
  - Plain (whole-window) std is NEVER precomputed anywhere in the existing
    codebase — only one-sided ``_std_above``/``_std_below`` subsets exist,
    and per the design spec's reuse map those are explicitly NOT used here
    (a one-sided std of a skewed window is a different, smaller statistic
    than a plain std of the whole window — using it would silently narrow
    every level). ``diff_prc_std`` is therefore always computed fresh.

All functions here are pure and read-only on their ``wide_df`` argument —
no column is added or mutated in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def diff_prc(series: pd.Series) -> pd.Series:
    """1-step percentage change of a series, in PERCENT (already x100).

    ``(s - s.shift(1)) / s.shift(1) * 100``. First row is NaN (no previous
    value to diff against). Generic Series->Series helper — the caller
    decides what series to feed it; ``price_levels`` below feeds it the
    *completed-candle* high/low sequence, never the raw 1-minute forming
    series. Reimplements — does not import — the identical formula in
    ``indicators.library.price_derivatives._DiffPrcBase.compute``: azlib is
    read-only on existing simple_trader code and that formula lives inside
    an ``IndicatorField`` class tied to the production indicator framework,
    not importable as a standalone function.
    """
    prev = series.shift(1)
    return (series - prev) / prev * 100.0


def diff_prc_ma(diff: pd.Series, window: int = 6) -> pd.Series:
    """Rolling mean of a diff_prc series over ``window`` rows.

    Generic Series->Series helper — whatever index ``diff`` carries (raw
    1-minute or, as ``price_levels`` uses it, the completed-candle
    sequence), the rolling window is over that series' own rows.
    """
    return diff.rolling(window).mean()


def diff_prc_std(diff: pd.Series, window: int = 6) -> pd.Series:
    """PLAIN rolling std of a diff_prc series over ``window`` rows.

    "Plain" = the whole window's standard deviation, as opposed to the
    existing wide-df's one-sided ``_std_above``/``_std_below`` columns
    (std of just the above-window-mean or below-window-mean subset). Those
    sided columns are NOT used anywhere in this module (see module
    docstring) — this is always a fresh computation.

    Uses pandas' rolling ``.std()`` default: ``ddof=1`` (sample standard
    deviation, Bessel-corrected). Documented here since the interface only
    says "plain rolling std" without pinning ddof explicitly.
    """
    return diff.rolling(window).std()


def _completed_candle_high_low(wide_df: pd.DataFrame, tf: int) -> tuple[pd.Series, pd.Series]:
    """Reduce ``wide_df`` to one row per COMPLETED ``tf`` candle.

    Selects rows where ``{tf}_is_closed`` is True — at that row,
    ``{tf}_high``/``{tf}_low`` (the forming cummax/cummin) equal that
    candle's final high/low, since it's the candle's last minute. Returned
    Series are indexed by each candle's own closing-row timestamp, in
    chronological order — one entry per completed candle, in the order
    those candles closed.
    """
    closed = wide_df[f"{tf}_is_closed"].astype(bool)
    high = wide_df.loc[closed, f"{tf}_high"]
    low = wide_df.loc[closed, f"{tf}_low"]
    return high, low


def price_levels(
    wide_df: pd.DataFrame, tf: int, window: int = 6, x: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Percentage-change levels converted to price, per design spec §1.

    ::

        # per COMPLETED candle k:
        high_diff_prc[k] = (high[k] - high[k-1]) / high[k-1] * 100   # candle-to-candle
        high_diff_prc_ma[k] = rolling mean over the last `window` completed candles
        high_std[k]         = plain rolling std over the last `window` completed candles

        # for forming candle c, from the PREVIOUS completed candle c-1:
        high_level = high_diff_prc_ma[c-1] + x * high_std[c-1]
        low_level  = low_diff_prc_ma[c-1]  - x * low_std[c-1]
        price_high_level = high[c-1] * (1 + high_level / 100)
        price_low_level  = low[c-1]  * (1 + low_level  / 100)

    Every 1-minute row of candle *c* (its whole forming period, including
    its own closing minute) gets the SAME ``(price_high_level,
    price_low_level)`` pair — computed purely from candle *c-1* and earlier,
    never from candle *c*'s own (possibly still-forming) high/low. This is
    what makes the result look-ahead free.

    Implementation: the per-candle level is computed once, at each closed
    candle's own closing-row timestamp (using that candle's OWN final
    high/low + its own trailing window's ma/std) — call this
    ``level_at_close[k]``. That is exactly the level candle *k+1* needs
    (its "c-1" reference). Reindexing ``level_at_close`` onto every 1-minute
    row (non-NaN only at each candle's closing minute), shifting by exactly
    one row, and forward-filling delays each candle's own computed level so
    it first becomes visible at the FIRST minute of the FOLLOWING candle and
    then stays constant (via ffill) through every minute of that following
    candle, including that candle's own closing minute — which is superseded
    only starting at the first minute of the candle after that. This relies
    on the wide df being contiguous 1-minute data (every row immediately
    following a closing row is the first row of the next candle), which
    holds throughout this experiment.

    Returns ``(price_high_level, price_low_level)``, each a ``pd.Series``
    aligned to ``wide_df``'s full index. NaN wherever there is no previous
    completed candle at all (the very first candle) or fewer than ``window``
    completed candles are available yet (rolling-window warm-up).
    """
    closed_high, closed_low = _completed_candle_high_low(wide_df, tf)

    high_diff = diff_prc(closed_high)
    low_diff = diff_prc(closed_low)
    high_ma = diff_prc_ma(high_diff, window)
    low_ma = diff_prc_ma(low_diff, window)
    high_std = diff_prc_std(high_diff, window)
    low_std = diff_prc_std(low_diff, window)

    high_level_pct = high_ma + x * high_std
    low_level_pct = low_ma - x * low_std

    # Value computed AT each completed candle's own closing minute, using
    # that candle's own final high/low as the reference price — this is
    # exactly the "previous completed candle" payload the FOLLOWING
    # (forming) candle must broadcast to all of its own 1-minute rows.
    level_at_close_high = closed_high * (1.0 + high_level_pct / 100.0)
    level_at_close_low = closed_low * (1.0 + low_level_pct / 100.0)

    # Delay by one row (own closing minute -> next candle's first minute),
    # then hold constant (ffill) through the whole of the next candle.
    price_high_level = level_at_close_high.reindex(wide_df.index).shift(1).ffill()
    price_low_level = level_at_close_low.reindex(wide_df.index).shift(1).ffill()

    price_high_level = price_high_level.rename("price_high_level")
    price_low_level = price_low_level.rename("price_low_level")

    return price_high_level, price_low_level


def coeff(price: np.ndarray, low_level: np.ndarray, high_level: np.ndarray) -> np.ndarray:
    """Map ``price`` onto ``[0, 1]`` between ``low_level`` (-> 0) and
    ``high_level`` (-> 1), clamped outside that range.

    ``clamp((price - low_level) / (high_level - low_level), 0, 1)``.

    Degenerate case (``high_level == low_level`` at some position, e.g. a
    collapsed action space): that position's result is 0.0, with NO
    divide-by-zero / invalid-value warning — the actual division is never
    performed there (``np.where`` swaps in a dummy non-zero denominator
    first, then the result at those positions is overwritten with 0.0). Any
    NaN in the inputs (unwarmed rolling windows) propagates to NaN in the
    output, as plain NaN arithmetic/comparison — no warning either.
    """
    price = np.asarray(price, dtype=float)
    low_level = np.asarray(low_level, dtype=float)
    high_level = np.asarray(high_level, dtype=float)

    span = high_level - low_level
    degenerate = span == 0.0  # NaN spans compare False here -> handled by normal NaN propagation below
    safe_span = np.where(degenerate, 1.0, span)  # dummy denominator, never actually 0 -> no warning

    raw = (price - low_level) / safe_span
    raw = np.where(degenerate, 0.0, raw)

    return np.clip(raw, 0.0, 1.0)


def label_coeff(
    wide_df: pd.DataFrame, tf: int, direction: str, window: int = 6, x: float = 2.0
) -> pd.Series:
    """``coeff`` of the entry extreme, per design spec §2 (pessimistic fill).

    Entry price = ``1_low`` for ``direction="long"``, ``1_high`` for
    ``direction="short"`` — NOT close (close is only the generic marker used
    to illustrate the general ``coeff`` price->position map; the concrete
    price plugged in here is decided by use, and label_coeff's use is the
    entry extreme, matching the profit-label definitions in
    ``indicators.labels``). Each row's entry price is mapped through that
    SAME row's held-constant, previous-completed-candle
    ``price_low_level``/``price_high_level`` (see ``price_levels``).

    Defined for every row of ``wide_df`` (independent of any label column) —
    NaN only where ``price_levels``' completed-candle warm-up hasn't
    happened yet. Returns a ``pd.Series`` aligned to ``wide_df``'s index.
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    price_high_level, price_low_level = price_levels(wide_df, tf, window, x)

    entry_col = "1_low" if direction == "long" else "1_high"
    entry_price = wide_df[entry_col]

    values = coeff(
        entry_price.to_numpy(dtype=float),
        price_low_level.to_numpy(dtype=float),
        price_high_level.to_numpy(dtype=float),
    )
    return pd.Series(values, index=wide_df.index, name="label_coeff")
