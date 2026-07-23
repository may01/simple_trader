"""tdlib/truth.py -- Task 2: ground-truth labeling for strong points.

Four functions, layered:

1. ``mark_truth(pts, tf, horizon)`` -- turns the pair of profit_strict
   long/short 0/1/NaN label columns (see ``tdlib.config.LABEL_COLS``) into
   ONE categorical ground-truth column per row: "long" / "short" / "both" /
   "neither" / "nan".

2. ``truth_counts(marked)`` -- category tally of a ``mark_truth`` result,
   always all five keys present (a category with zero rows is real
   information, not an absence to omit).

3. ``fwd_log_return(slim, tf, bars)`` -- REALIZED forward log return over
   ``bars`` closed tf-candles. Pure lookahead (reads the FUTURE close) --
   this exists ONLY to score truth/robustness against what actually
   happened next; it must NEVER be fed into a model as a feature (it is not
   knowable at prediction time).

4. ``robustness_agreement(marked, fwd1, fwd4)`` -- how often the
   profit_strict truth categories agree in DIRECTION with the raw realized
   forward return, at two horizons -- a sanity metric on the labels
   themselves, independent of any model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tdlib.config import LABEL_COLS

_CATEGORIES = ("long", "short", "both", "neither", "nan")


def mark_truth(pts: pd.DataFrame, tf: int, horizon: str = "n1") -> pd.Series:
    """Categorize each row of ``pts`` from tf's profit_strict long/short
    label pair at the given ``horizon``.

    Reads ``LABEL_COLS[tf][f"pslong_{horizon}"]`` /
    ``LABEL_COLS[tf][f"psshort_{horizon}"]`` -- an unknown ``horizon``
    (anything other than what LABEL_COLS actually carries, i.e. "n1"/"n2")
    raises the natural ``KeyError`` straight out of that dict lookup. Note
    LABEL_COLS only carries n1/n2 for the profit_strict ("ps") pair -- the
    PLAIN long/short pair (``LABEL_COLS[tf]["plong_n1"]`` etc.) is n1-only
    and is never read by this function; ``horizon="n2"`` still resolves the
    ps pair, same shape as "n1".

    Categories (row-wise, from the (pslong, psshort) pair):
      - pslong == 1, psshort == 1 -> "both"
      - pslong == 1, psshort == 0 -> "long"
      - pslong == 0, psshort == 1 -> "short"
      - pslong == 0, psshort == 0 -> "neither"
      - either value NaN          -> "nan"     (overrides all of the above)

    Returns a Series of python ``str``, indexed exactly like ``pts``.
    """
    long_col = LABEL_COLS[tf][f"pslong_{horizon}"]
    short_col = LABEL_COLS[tf][f"psshort_{horizon}"]

    long_val = pts[long_col]
    short_val = pts[short_col]

    is_long = long_val == 1
    is_short = short_val == 1
    is_nan = long_val.isna() | short_val.isna()

    marked = pd.Series("neither", index=pts.index, dtype=object)
    marked[is_long & is_short] = "both"
    marked[is_long & ~is_short] = "long"
    marked[~is_long & is_short] = "short"
    # NaN overrides any of the above: a NaN value always compares False in
    # is_long/is_short, so a row with a NaN in only ONE of the two columns
    # could otherwise land in "long"/"short" from the other column's real
    # value above. Applied last, unconditionally, so "either label NaN"
    # always wins regardless of the other label's value.
    marked[is_nan] = "nan"

    return marked


def truth_counts(marked: pd.Series) -> dict:
    """Category tally of a ``mark_truth`` result. Always all five keys
    (long/short/both/neither/nan), zero-filled for categories that don't
    occur -- callers should never need a ``.get(cat, 0)`` of their own."""
    value_counts = marked.value_counts()
    return {cat: int(value_counts.get(cat, 0)) for cat in _CATEGORIES}


def fwd_log_return(slim: pd.DataFrame, tf: int, bars: int) -> pd.Series:
    """Realized forward log return, ``bars`` CLOSED tf-candles ahead.

    PURE LOOKAHEAD -- reads ``{tf}_close`` ``bars`` candles into the future.
    For truth/robustness scoring ONLY. Never use this as a model feature: at
    prediction time the future close does not exist yet.

    Computed on the subseries of rows where ``{tf}_is_closed`` is True (so
    "bars ahead" means bars closed tf-candles, not bars raw slim rows --
    the two differ whenever tf > 15, since the SLIM frame is on a 15-minute
    grid), then reindexed back onto ``slim``'s full index: non-closed rows
    are NaN (they were never in the closed subseries to begin with), and the
    last ``bars`` closed rows are NaN (nothing ``bars`` candles ahead of them
    yet).
    """
    closed = slim[f"{tf}_is_closed"] == True  # noqa: E712
    close = slim.loc[closed, f"{tf}_close"]

    fwd = np.log(close.shift(-bars) / close)
    return fwd.reindex(slim.index)


def robustness_agreement(marked: pd.Series, fwd1: pd.Series, fwd4: pd.Series) -> dict:
    """How often the truth categories agree in direction with the raw
    realized forward return, at two horizons.

    Over rows where ``marked == "long"``: ``P(fwd1 > 0)``, ``P(fwd4 > 0)``.
    Over rows where ``marked == "short"``: ``P(fwd1 < 0)``, ``P(fwd4 < 0)``.
    NaN rows in the relevant fwd series (see ``fwd_log_return``'s own NaN
    cases) are excluded from BOTH the numerator and denominator of each
    fraction -- an unrealized/unknown return is neither agreement nor
    disagreement, so it must not silently count as either.

    ``n_long`` / ``n_short`` are the total count of rows marked "long" /
    "short" in ``marked`` itself -- independent of fwd1/fwd4's own NaN
    pattern (which can differ between the two). They describe the CLASS
    size, not the denominator of any one fraction above.

    An empty class (no "long" or no "short" rows, or every one of its fwd
    values NaN) yields ``float("nan")`` for that class's fractions, not a
    crash or a silently-wrong 0.0.
    """

    def _fraction(index: pd.Index, fwd: pd.Series, positive: bool) -> float:
        values = fwd.loc[index].dropna()
        if values.empty:
            return float("nan")
        agree = (values > 0) if positive else (values < 0)
        return float(agree.mean())

    long_index = marked.index[marked == "long"]
    short_index = marked.index[marked == "short"]

    return {
        "long_fwd1": _fraction(long_index, fwd1, positive=True),
        "long_fwd4": _fraction(long_index, fwd4, positive=True),
        "short_fwd1": _fraction(short_index, fwd1, positive=False),
        "short_fwd4": _fraction(short_index, fwd4, positive=False),
        "n_long": int(len(long_index)),
        "n_short": int(len(short_index)),
    }
