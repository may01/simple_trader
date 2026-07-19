"""azlib/space.py — Task 2: action space (price levels + coeff) + label_coeff.

Implements design spec §1-2
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):
percentage-change levels (``diff_prc_ma +/- X*std``), converted to price
levels referenced against the *previous same-TF candle*, then a price-space
``coeff`` mapping any price onto ``[0, 1]`` between those two levels.
``label_coeff`` plugs the entry extreme (pessimistic fill: 1-min low for
long, 1-min high for short) through that mapping — the regression target
for later layers.

Reuse vs compute (see the design spec's Reuse map and task-2-brief.md):
  - ``{tf}_{side}_diff_prc`` and ``{tf}_{side}_diff_prc_rm_{window}`` are
    already columns on the real wide df (``indicators/library/
    price_derivatives.py``) — ``price_levels`` uses them when present
    instead of recomputing, falling back to raw ``{tf}_high``/``{tf}_low``
    (via ``diff_prc``/``diff_prc_ma`` below) only when absent.
  - Plain (whole-window) std is NEVER precomputed anywhere in the existing
    codebase — only one-sided ``_std_above``/``_std_below`` subsets exist,
    and per the design spec's reuse map those are explicitly NOT used here
    (a one-sided std of a skewed window is a different, smaller statistic
    than a plain std of the whole window — using it would silently narrow
    every level). ``diff_prc_std`` is therefore always computed fresh from
    the (possibly reused) diff_prc column.

All functions here are pure and read-only on their ``wide_df`` argument —
no column is added or mutated in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def diff_prc(series: pd.Series) -> pd.Series:
    """1-step percentage change of a price series, in PERCENT (already x100).

    ``(s - s.shift(1)) / s.shift(1) * 100``. First row is NaN (no previous
    value to diff against). Reimplements — does not import — the identical
    formula in ``indicators.library.price_derivatives._DiffPrcBase.compute``:
    azlib is read-only on existing simple_trader code and that formula lives
    inside an ``IndicatorField`` class tied to the production indicator
    framework, not importable as a standalone function.
    """
    prev = series.shift(1)
    return (series - prev) / prev * 100.0


def diff_prc_ma(diff: pd.Series, window: int = 6) -> pd.Series:
    """Rolling mean of a diff_prc series over ``window`` rows.

    Equivalent to the existing ``{src}_diff_prc_rm_{window}`` wide-df column
    when ``window`` matches what that column was built with (``price_levels``
    below reuses that column directly rather than calling this when
    possible; this function is the fallback / the plain building block).
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


def _diff_and_ma(wide_df: pd.DataFrame, tf: int, side: str, window: int) -> tuple[pd.Series, pd.Series]:
    """Return (diff, ma) for one side ("high" or "low"), preferring existing
    wide-df columns over recomputing from raw ``{tf}_{side}``.

    Three cases, checked independently (a real wide df may have diff_prc
    without the matching *_rm_{window} column, e.g. if window != 6):
      1. Both ``{tf}_{side}_diff_prc`` and ``{tf}_{side}_diff_prc_rm_{window}``
         present -> both reused as-is.
      2. Only ``{tf}_{side}_diff_prc`` present -> ma computed from it.
      3. Neither present -> both recomputed from raw ``{tf}_{side}``.
    """
    diff_col = f"{tf}_{side}_diff_prc"
    ma_col = f"{tf}_{side}_diff_prc_rm_{window}"

    if diff_col in wide_df.columns:
        diff = wide_df[diff_col]
    else:
        diff = diff_prc(wide_df[f"{tf}_{side}"])

    if ma_col in wide_df.columns:
        ma = wide_df[ma_col]
    else:
        ma = diff_prc_ma(diff, window)

    return diff, ma


def price_levels(
    wide_df: pd.DataFrame, tf: int, window: int = 6, x: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Percentage-change levels converted to price, per design spec §1.

    ::

        high_level(x) = high_diff_prc_ma + x * high_std
        low_level(x)  = low_diff_prc_ma  - x * low_std
        price_high_level = prev_high * (1 + high_level(x) / 100)
        price_low_level  = prev_low  * (1 + low_level(x)  / 100)

    ``prev_high``/``prev_low`` are ``{tf}_high``/``{tf}_low`` shifted by one
    same-TF row — the previous same-TF candle, matching diff_prc's own
    reference (see module docstring / diff_prc). ``high_std``/``low_std``
    are the PLAIN rolling std of the high/low diff_prc series (never the
    sided ``_std_above``/``_std_below`` columns — see ``diff_prc_std``).

    Returns ``(price_high_level, price_low_level)``, each a ``pd.Series``
    aligned to ``wide_df``'s index (NaN wherever the rolling window or the
    one-row shift hasn't warmed up yet).
    """
    high_diff, high_ma = _diff_and_ma(wide_df, tf, "high", window)
    low_diff, low_ma = _diff_and_ma(wide_df, tf, "low", window)

    high_std = diff_prc_std(high_diff, window)
    low_std = diff_prc_std(low_diff, window)

    high_level = high_ma + x * high_std
    low_level = low_ma - x * low_std

    prev_high = wide_df[f"{tf}_high"].shift(1)
    prev_low = wide_df[f"{tf}_low"].shift(1)

    price_high_level = prev_high * (1.0 + high_level / 100.0)
    price_low_level = prev_low * (1.0 + low_level / 100.0)

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
    ``indicators.labels``).

    Defined for every row of ``wide_df`` (independent of any label column) —
    NaN only where ``price_levels``' rolling windows/shift haven't warmed up.
    Returns a ``pd.Series`` aligned to ``wide_df``'s index.
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
