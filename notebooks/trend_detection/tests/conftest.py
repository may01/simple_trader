"""Shared pytest configuration + fixtures for the trend_detection tdlib tests.

Two jobs:

1. sys.path wiring — tdlib lives at notebooks/trend_detection/tdlib/ and is
   never pip-installed; it is imported straight off disk as a plain
   top-level package (``import tdlib`` / ``from tdlib import <submodule>``).
   Its parent directory (notebooks/trend_detection/, i.e. this file's
   grandparent) must be on sys.path *before* any test module runs, so we do
   it here at collection time. Mirrors action_zones's azlib wiring exactly
   (notebooks/action_zones/tests/conftest.py in the zone-selection-
   experiment worktree) — same two-line insert, adapted to this package's
   own directory depth.

2. ``make_slim()`` / ``synthetic_slim_df`` — the shared SLIM-frame fixture
   every later trend_detection task's tests build on. "SLIM" is this
   experiment's own name for the reduced frame the real pipeline works on:
   the wide 1-minute df (see data.py::_build_wide_df) filtered down to only
   the rows where ``15_is_closed`` is True, i.e. one row per completed
   15-minute candle, still carrying every ``{tf}_*`` column for every
   configured timeframe. See make_slim()'s docstring for the exact column
   list and generation semantics (Task 0 brief).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- sys.path wiring --------------------------------------------------------
# notebooks/trend_detection/ is this file's grandparent (tests/conftest.py ->
# tests/ -> trend_detection/). Insert once, so `import tdlib` resolves both
# when pytest is invoked from /code (the Docker workdir) and from any other
# cwd.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


# --- make_slim() / synthetic_slim_df fixture --------------------------------

# Fixed default seed/days. NEVER replace with time.time()/os.urandom/an
# unseeded np.random call — every test run must reproduce byte-identical
# fixture data so a failure is reproducible from the seed alone.
_DEFAULT_SEED = 0
_DEFAULT_DAYS = 12

# Timeframes carried by the SLIM frame. config_loader.py's CANDLES
# (configs/candles_config.yaml: [1, 5, 15, 60, 240, 1440]) minus tf=1 — the
# SLIM reduction is keyed on 15_is_closed, and tdlib has no use for the raw
# 1-minute-only columns, so this experiment's SLIM frame never carries
# {1}_* columns (Task 0 brief).
_TFS = (5, 15, 60, 240, 1440)

# Timeframes that carry profit_strict / plain label columns (Task 0 brief).
_LABEL_TFS = (15, 60, 240)

# LINK/USDT-ish starting price (see project memory: NN datasets are built on
# link_usdt) — purely cosmetic, no tdlib formula in this experiment is
# scale-dependent; only here so printed frames look like real market data.
_START_PRICE = 20.0

# {tf}_ps{long,short}_n{1,2}_m1_{suffix} / {tf}_p{long,short}_n1_m1_{suffix}
# label-column suffixes, verbatim from the Task 0 brief. The profit_strict
# ("ps") pair carries both an n1 and n2 horizon variant; the plain pair is
# n1-only (per the brief's exact enumerated column list — there is no plain
# n2 variant in this experiment's label set).
_PS_SUFFIX = {15: "x0p3_l15_y0p2", 60: "x0p2_l15_y0p1", 240: "x0p1_l15_y0p1"}
_PLAIN_SUFFIX = {15: "x0p3", 60: "x0p2", 240: "x0p1"}


def _ps_label_col(tf: int, side: str, n: int) -> str:
    """profit_strict label column name, e.g. ``15_pslong_n1_m1_x0p3_l15_y0p2``."""
    return f"{tf}_ps{side}_n{n}_m1_{_PS_SUFFIX[tf]}"


def _plain_label_col(tf: int, side: str) -> str:
    """Plain label column name, e.g. ``15_plong_n1_m1_x0p3``."""
    return f"{tf}_p{side}_n1_m1_{_PLAIN_SUFFIX[tf]}"


def _tf_derived_columns(idx: pd.DatetimeIndex, tf: int) -> dict[str, object]:
    """Return ``{tf}_open_index`` / ``{tf}_is_closed`` — deterministic
    functions of the index, replicating data.py::_build_wide_df EXACTLY
    (worktree data.py, read for this task):

      - ``{tf}_open_index`` (data.py lines 196-199): ``df.index.floor(f"{tf}min")``
        — the candle-open timestamp (datetime64, tz-matching the index) that
        each row's still-forming (or just-closed) {tf}-candle belongs to.
        Used downstream for time-in-candle math, so it must be the real
        floor value, not a random/derived stand-in.
      - ``{tf}_is_closed`` (data.py lines 219-239): True at the last 1-minute
        row of each tf-period candle. Per-tf branch, copied verbatim:
          tf=1:    always True
          tf=5:    minute % 5 == 4
          tf=15:   minute % 15 == 14
          tf=60:   minute == 59
          tf=240:  minute == 59 and hour % 4 == 3
          tf=1440: minute == 59 and hour == 23
          other:   generic "next minute rolls to a new {tf}min bucket" fallback

    Not randomized: downstream completed-candle logic must see the real
    boundary semantics, not noise.
    """
    if tf == 1:
        is_closed = pd.Series(True, index=idx)
    elif tf == 5:
        is_closed = idx.minute % 5 == 4
    elif tf == 15:
        is_closed = idx.minute % 15 == 14
    elif tf == 60:
        is_closed = idx.minute == 59
    elif tf == 240:
        is_closed = (idx.minute == 59) & (idx.hour % 4 == 3)
    elif tf == 1440:
        is_closed = (idx.minute == 59) & (idx.hour == 23)
    else:
        is_closed = (idx + pd.Timedelta(minutes=1)).floor(f"{tf}min") != idx.floor(f"{tf}min")

    return {
        f"{tf}_open_index": idx.floor(f"{tf}min"),
        f"{tf}_is_closed": np.asarray(is_closed, dtype=bool),
    }


def _tf_value_columns(tf: int, rng: np.random.Generator, n: int) -> dict[str, object]:
    """Return every remaining {tf}_* column (Task 0 brief's column list)
    with independently-seeded synthetic values.

    Unlike the real wide df (and unlike action_zones's synthetic_wide_df),
    each tf's OHLC/indicator columns here are their OWN standalone random
    walk / distribution — NOT derived from a shared underlying 1-minute
    series. The SLIM fixture only needs plausible, constraint-satisfying
    numbers to exercise tdlib code paths (nothing in this task's scope reads
    across tfs), so paying for a full 1-minute simulation underneath just to
    re-derive per-tf aggregates would add real complexity/runtime for no
    test value.
    """
    cols: dict[str, object] = {}

    # --- OHLCV: positive random walk; high >= max(open, close),
    # low <= min(open, close) (both by construction, per brief) ------------
    log_returns = rng.normal(0.0, 0.0025, size=n)
    close = _START_PRICE * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate(([close[0]], close[:-1]))
    wick_up = np.abs(rng.normal(0.0015, 0.001, size=n)) * close + 1e-6
    wick_dn = np.abs(rng.normal(0.0015, 0.001, size=n)) * close + 1e-6
    high = np.maximum(open_, close) + wick_up
    low = np.clip(np.minimum(open_, close) - wick_dn, 1e-6, None)
    volume = rng.uniform(50.0, 500.0, size=n)

    cols[f"{tf}_open"] = open_
    cols[f"{tf}_high"] = high
    cols[f"{tf}_low"] = low
    cols[f"{tf}_close"] = close
    cols[f"{tf}_volume"] = volume

    # --- indicators ---------------------------------------------------------
    cols[f"{tf}_rsi_ma8"] = np.clip(rng.normal(50.0, 10.0, size=n), 0.0, 100.0)

    # std ~= 1.4: at n=1152 this reliably puts real mass beyond +-1.36 on
    # BOTH tails (P(|Z| > 1.36/1.4) ~= 33%, i.e. ~190 rows/tail) --
    # test_l0_conftest.py asserts both tails are present for tf=15.
    cols[f"{tf}_rsi_ma8_diff"] = rng.normal(0.0, 1.4, size=n)

    atr_14 = np.abs(rng.normal(0.5, 0.2, size=n)) + 0.01  # strictly positive
    cols[f"{tf}_atr_14"] = atr_14
    cols[f"{tf}_natr_14"] = atr_14 / close * 100.0

    bb_middle = close + rng.normal(0.0, 0.05, size=n)
    half_2 = np.abs(rng.normal(0.3, 0.1, size=n)) + 1e-3  # > 0, strict band separation
    cols[f"{tf}_bb_upper_20_2"] = bb_middle + half_2
    cols[f"{tf}_bb_middle_20_2"] = bb_middle
    cols[f"{tf}_bb_lower_20_2"] = bb_middle - half_2

    # "_3" band: strictly WIDER than "_2" (half_3 = half_2 + a positive
    # extra), same center — guarantees bb_upper_20_3 > bb_lower_20_3 (the
    # brief's only stated inequality for this pair) plus upper_3 > upper_2
    # and lower_3 < lower_2 (the "(wider)" note). No bb_middle_20_3 column —
    # the brief lists only upper/lower for the "_3" band.
    half_3 = half_2 + np.abs(rng.normal(0.3, 0.1, size=n)) + 1e-3
    cols[f"{tf}_bb_upper_20_3"] = bb_middle + half_3
    cols[f"{tf}_bb_lower_20_3"] = bb_middle - half_3

    cols[f"{tf}_adx_14"] = np.clip(rng.normal(25.0, 15.0, size=n), 0.0, 100.0)
    cols[f"{tf}_cci_14"] = rng.normal(0.0, 100.0, size=n)
    cols[f"{tf}_macd_hist_12_26_9"] = rng.normal(0.0, 0.5, size=n)
    cols[f"{tf}_ema_25_minus_close"] = rng.normal(0.0, 1.0, size=n)
    cols[f"{tf}_ema_50_slope"] = rng.normal(0.0, 0.05, size=n)
    cols[f"{tf}_logret"] = rng.normal(0.0, 0.01, size=n)
    cols[f"{tf}_body_ratio"] = rng.uniform(0.0, 1.0, size=n)  # body / range fraction, non-negative
    cols[f"{tf}_vol_regime"] = rng.integers(0, 5, size=n)  # int 0..4

    return cols


def _random_binary_label(rng: np.random.Generator, n: int, nan_frac: float = 0.05) -> np.ndarray:
    """0.0/1.0 draws with ~``nan_frac`` share replaced by NaN (Task 0 brief:
    "values: random 0.0/1.0 with ~5% NaN, seeded")."""
    vals = rng.integers(0, 2, size=n).astype(float)
    vals[rng.random(n) < nan_frac] = np.nan
    return vals


def make_slim(seed: int = _DEFAULT_SEED, days: int = _DEFAULT_DAYS) -> pd.DataFrame:
    """Build one synthetic SLIM frame.

    ``days*96`` rows (default 1152): 1-minute UTC timestamps starting
    2023-01-01, reduced to the 15-minute-close minutes (:14/:29/:44/:59 of
    every hour) — the same reduction the real pipeline applies via
    ``wide_df[wide_df["15_is_closed"]]``. On this reduced grid:

      - ``15_is_closed`` is True at every row (by construction: the grid
        itself IS the 15_is_closed==True rows).
      - ``60_is_closed`` is True at every 4th row (only the :59 phase of the
        4-row :14/:29/:44/:59 cycle also satisfies tf=60's own boundary).
      - ``240_is_closed`` is True at every 16th row (:59 AND hour%4==3).
      - ``1440_is_closed`` is True at every 96th row (:59 AND hour==23 — the
        single last-15m-candle-of-the-day row).

      ``days*96`` is always exactly divisible by 4/16/96 for any integer
      ``days`` (96 = 4*24 = 16*6 = 96*1), so these are exact row counts, not
      approximations.

    For tf in {5, 15, 60, 240, 1440}: ``{tf}_open_index``/``{tf}_is_closed``
    (see _tf_derived_columns — exact data.py::_build_wide_df semantics)
    plus ``{tf}_open/high/low/close/volume``, ``{tf}_rsi_ma8``,
    ``{tf}_rsi_ma8_diff``, ``{tf}_atr_14``, ``{tf}_natr_14``,
    ``{tf}_bb_{upper,middle,lower}_20_2``, ``{tf}_bb_{upper,lower}_20_3``,
    ``{tf}_adx_14``, ``{tf}_cci_14``, ``{tf}_macd_hist_12_26_9``,
    ``{tf}_ema_25_minus_close``, ``{tf}_ema_50_slope``, ``{tf}_logret``,
    ``{tf}_body_ratio``, ``{tf}_vol_regime`` (see _tf_value_columns).

    For tf in {15, 60, 240}: 6 label columns each — the profit_strict
    long/short pair at n1 and n2, plus the plain long/short pair at n1 only
    (see _ps_label_col/_plain_label_col for exact names) — random 0.0/1.0
    with ~5% NaN.

    Built from a seeded ``numpy.random.Generator`` (``seed``, default 0) so
    the same rows/values come back on every call — ``make_slim(0)`` called
    twice returns byte-identical frames (asserted by test_l0_conftest.py).
    Every later trend_detection task's tests build on this same fixture, so
    it is documented in full here rather than left as a one-off helper (same
    convention as action_zones's synthetic_wide_df).

    Columns are collected into one dict and handed to the ``pd.DataFrame``
    constructor ONCE at the end, mirroring data.py::_build_wide_df's own
    ``new_cols`` + single-``pd.concat`` pattern (data.py lines 191-193: "8
    inserts x len(CANDLES) sequential setitems fragment the frame ... and
    pandas warns on every later insert") instead of ~140 sequential
    ``df[col] = ...`` setitems on a growing frame.
    """
    rng = np.random.default_rng(seed)

    full_index = pd.date_range("2023-01-01", periods=days * 24 * 60, freq="1min", tz="UTC")
    slim_index = full_index[full_index.minute % 15 == 14]
    n = len(slim_index)

    new_cols: dict[str, object] = {}

    for tf in _TFS:
        new_cols.update(_tf_derived_columns(slim_index, tf))
        new_cols.update(_tf_value_columns(tf, rng, n))

    for tf in _LABEL_TFS:
        for side in ("long", "short"):
            new_cols[_ps_label_col(tf, side, 1)] = _random_binary_label(rng, n)
            new_cols[_ps_label_col(tf, side, 2)] = _random_binary_label(rng, n)
        for side in ("long", "short"):
            new_cols[_plain_label_col(tf, side)] = _random_binary_label(rng, n)

    df = pd.DataFrame(new_cols, index=slim_index)
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_slim_df() -> pd.DataFrame:
    """``make_slim()`` at its defaults (seed=0, days=12) — the shared SLIM
    fixture every trend_detection tdlib test builds on. See make_slim()'s
    docstring for the full column/semantics spec.

    Function-scoped (a fresh frame per test, pytest's default): later tasks'
    label-mutating helpers (set_labels()) act in place, so tests must not
    share one frame instance across the suite (same reasoning as
    action_zones's synthetic_wide_df fixture).
    """
    return make_slim()


def set_labels(
    df: pd.DataFrame,
    tf: int,
    row_positions,
    long_val: float,
    short_val: float,
) -> None:
    """Overwrite the n1 profit_strict long/short label pair for ``tf`` at
    ``row_positions`` (anything valid for ``.iloc``'s row indexer — a list/
    array of integer positions, a slice, or a single int), in place.

    Lets tests engineer known-truth label sequences on top of the otherwise-
    random ``make_slim()``/``synthetic_slim_df`` frame, e.g.::

        set_labels(df, 15, slice(10, 15), long_val=1.0, short_val=0.0)

    Only the n1 profit_strict pair is touched — the n2 and plain label
    columns are untouched. ``tf`` must be one of the label timeframes (15,
    60, 240); any other value raises the natural ``KeyError`` from the
    ``_PS_SUFFIX`` lookup inside ``_ps_label_col``.
    """
    long_col = _ps_label_col(tf, "long", 1)
    short_col = _ps_label_col(tf, "short", 1)
    df.iloc[row_positions, df.columns.get_loc(long_col)] = long_val
    df.iloc[row_positions, df.columns.get_loc(short_col)] = short_val
