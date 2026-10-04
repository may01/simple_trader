"""azlib/indicators.py — Task 3: indicator attributes + freeze stats (Layer 3).

Implements design spec §7
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):
three indicator families (RSI, MACD, MA), each exposing a small set of
"attributes" (position / slope / distance — MA has no distance) read
straight off the wide df, then z-scored against TRAIN-frozen (mean, std)
stats and clamped to [-3, +3] before being handed to later layers
(azlib.models, Task 4) as model features.

GRANULARITY (the key correctness requirement here — see
.superpowers/sdd/task-3-brief.md's "GRANULARITY DECISION"): indicator
attributes are POINT-IN-TIME per 1-minute row. ``raw_attribute`` reads the
wide-df indicator columns AS-IS at each row — they are already
closed-candle-based + partial-current (computed upstream by
``build_indicator_input`` on real data; the synthetic fixture's EWM-smoothed
proxy stands in for that in tests). ``slope`` is a plain ``.diff()`` over
1-minute rows; ``distance`` is the documented column difference, per row.

This is intentionally DIFFERENT from ``azlib.space``'s (Task 2)
completed-candle held-constant levels: NO ``{tf}_is_closed`` reduction and
NO held-constant broadcast happens anywhere in this module. Indicator
attributes are live entry-time signals available at minute *t*, not a
per-candle zone — every 1-minute row gets its own, independently computed
value, varying minute to minute exactly as the underlying wide-df column
does.

Freezing (design spec §7): ``fit_stats`` computes (mean, std) per
``(tf, indicator, attr)`` over a TRAIN frame's non-NaN rows only, returned
as a ``FreezeStats`` (JSON round-trippable via ``to_json``/``from_json``).
``attribute_frame`` is raw (no z-score) when ``stats`` is omitted, and
z-scored + clamped via those FROZEN stats when given one — it never
recomputes mean/std from whatever frame it is scoring, which is what makes
OOS/live application look-ahead-free with respect to the train/test split
(the OOS frame's own distribution never leaks into the normalization).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from azlib.config import MACD_COL, MACD_HIST_COL, MA_COL, RSI_COL, RSI_MA_COL

# --- interface constants (later tasks import these EXACT names) ------------

INDICATORS = ("rsi", "macd", "ma")

ATTRS = {
    "rsi": ("position", "slope", "distance"),
    "macd": ("position", "slope", "distance"),
    "ma": ("position", "slope"),  # MA has NO distance attribute.
}


def raw_attribute(wide_df: pd.DataFrame, tf: int, indicator: str, attr: str) -> pd.Series:
    """One indicator attribute, POINT-IN-TIME per 1-minute row of ``wide_df``.

    Exact formulas (design spec §7, column names from ``azlib.config``):

    - RSI:  ``position`` = ``{tf}_rsi_ma8``;
            ``slope``    = ``{tf}_rsi_ma8.diff()``;
            ``distance`` = ``{tf}_rsi_14 - {tf}_rsi_ma8``.
    - MACD: ``position`` = ``{tf}_macd_12_26_9``;
            ``slope``    = ``{tf}_macd_12_26_9.diff()``;
            ``distance`` = ``{tf}_macd_hist_12_26_9``.
    - MA:   ``position`` = ``({tf}_close - {tf}_ema_25) / {tf}_close * 100``;
            ``slope``    = ``{tf}_ema_25.diff()``.
            (``distance`` is not defined for MA — raises ``ValueError``.)

    No ``{tf}_is_closed`` reduction, no held-constant broadcast — every row
    of the returned Series is that row's own value (see module docstring's
    GRANULARITY note). Raises ``ValueError`` for an unknown ``indicator`` or
    an ``attr`` not in ``ATTRS[indicator]`` (checked BEFORE any column
    lookup, so an invalid ``(indicator, attr)`` pair fails the same way
    regardless of what ``wide_df`` contains).
    """
    if indicator not in INDICATORS:
        raise ValueError(f"unknown indicator {indicator!r}, must be one of {INDICATORS}")
    if attr not in ATTRS[indicator]:
        raise ValueError(
            f"unknown attr {attr!r} for indicator {indicator!r}, must be one of {ATTRS[indicator]}"
        )

    if indicator == "rsi":
        rsi_ma = wide_df[f"{tf}_{RSI_MA_COL}"]
        if attr == "position":
            return rsi_ma
        if attr == "slope":
            return rsi_ma.diff()
        # attr == "distance"
        rsi = wide_df[f"{tf}_{RSI_COL}"]
        return rsi - rsi_ma

    if indicator == "macd":
        macd = wide_df[f"{tf}_{MACD_COL}"]
        if attr == "position":
            return macd
        if attr == "slope":
            return macd.diff()
        # attr == "distance"
        return wide_df[f"{tf}_{MACD_HIST_COL}"]

    # indicator == "ma"
    ema = wide_df[f"{tf}_{MA_COL}"]
    if attr == "position":
        close = wide_df[f"{tf}_close"]
        return (close - ema) / close * 100.0
    # attr == "slope"
    return ema.diff()


# --- freeze stats -------------------------------------------------------


@dataclass
class FreezeStats:
    """Frozen (mean, std) per ``(tf, indicator, attr)``, JSON round-trippable.

    ``stats`` maps the 3-tuple key straight to a ``(mean, std)`` pair, per
    the interface's data model. JSON has no tuple-key support, so
    ``to_json``/``from_json`` serialize as a flat list of
    ``{"tf", "indicator", "attr", "mean", "std"}`` records instead — the
    round trip reconstructs the exact same dict keyed by 3-tuples (``tf`` an
    int, matching what ``fit_stats`` produces).
    """

    stats: dict[tuple[int, str, str], tuple[float, float]] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        records = [
            {"tf": tf, "indicator": indicator, "attr": attr, "mean": mean, "std": std}
            for (tf, indicator, attr), (mean, std) in self.stats.items()
        ]
        with open(path, "w") as f:
            json.dump(records, f)

    @classmethod
    def from_json(cls, path: str) -> "FreezeStats":
        with open(path) as f:
            records = json.load(f)
        stats = {
            (int(r["tf"]), r["indicator"], r["attr"]): (float(r["mean"]), float(r["std"]))
            for r in records
        }
        return cls(stats)


def fit_stats(train_df: pd.DataFrame, tf: int) -> FreezeStats:
    """Fit (mean, std) per ``(tf, indicator, attr)`` on ``train_df``'s
    non-NaN rows, for every indicator/attr combo in ``INDICATORS``/``ATTRS``.

    Train-only: this is the one function in this module allowed to look at
    a frame's own distribution — ``attribute_frame`` never calls this
    itself, callers freeze the result once (typically via ``to_json``) and
    reuse it unchanged for every later OOS/live application (see module
    docstring). ``std`` uses pandas' ``.std()`` default (``ddof=1``, sample
    standard deviation), matching ``azlib.space.diff_prc_std``'s convention.
    """
    stats: dict[tuple[int, str, str], tuple[float, float]] = {}
    for indicator in INDICATORS:
        for attr in ATTRS[indicator]:
            raw = raw_attribute(train_df, tf, indicator, attr).dropna()
            stats[(tf, indicator, attr)] = (float(raw.mean()), float(raw.std()))
    return FreezeStats(stats)


def zscore_clamp(series: pd.Series, mean: float, std: float, lo: float = -3.0, hi: float = 3.0) -> pd.Series:
    """``clip((series - mean) / std, lo, hi)``.

    ``std == 0`` (degenerate/constant column) returns an all-``0.0`` Series
    of the same index, WITHOUT performing the division — this is a plain
    ``if`` guard, not a ``np.where`` trick, so there is no
    divide-by-zero/invalid-value warning to suppress in the first place
    (unlike ``azlib.space.coeff``'s degenerate-span case, which does need
    the dummy-denominator trick because it operates elementwise on an
    array with only SOME positions degenerate; here ``std`` is a single
    scalar shared by the whole series, so the whole series is either
    degenerate or it isn't).
    """
    if std == 0:
        return pd.Series(0.0, index=series.index, name=series.name)
    z = (series - mean) / std
    return z.clip(lower=lo, upper=hi)


# --- attribute_frame ---------------------------------------------------------


def attribute_frame(
    wide_df: pd.DataFrame, tf: int, indicator: str, stats: "FreezeStats | None" = None
) -> pd.DataFrame:
    """One indicator's attributes as columns, raw or frozen-normalized.

    Columns = ``ATTRS[indicator]`` (``("position","slope","distance")`` for
    rsi/macd, ``("position","slope")`` for ma), each built via
    ``raw_attribute``. When ``stats`` is ``None`` the columns are the raw
    values unchanged; when a ``FreezeStats`` is given, each column is
    z-scored + clamped via THAT frozen ``(mean, std)`` for
    ``(tf, indicator, attr)`` (``zscore_clamp``) — ``wide_df``'s own
    distribution is never consulted for normalization in that case, so
    passing a train-fit ``stats`` while scoring an OOS/live ``wide_df`` is
    the intended, look-ahead-free use (see module docstring).
    """
    if indicator not in INDICATORS:
        raise ValueError(f"unknown indicator {indicator!r}, must be one of {INDICATORS}")

    columns: dict[str, pd.Series] = {}
    for attr in ATTRS[indicator]:
        raw = raw_attribute(wide_df, tf, indicator, attr)
        if stats is None:
            columns[attr] = raw
        else:
            mean, std = stats.stats[(tf, indicator, attr)]
            columns[attr] = zscore_clamp(raw, mean, std)

    return pd.DataFrame(columns, index=wide_df.index)
