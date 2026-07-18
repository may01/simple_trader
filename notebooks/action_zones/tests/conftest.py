"""Shared pytest configuration + fixtures for the action_zones azlib tests.

Two jobs:

1. sys.path wiring — azlib lives at notebooks/action_zones/azlib/ and is
   never pip-installed; it is imported straight off disk as a plain
   top-level package (``import azlib``). Its parent directory
   (notebooks/action_zones/, i.e. this file's grandparent) must be on
   sys.path *before* any test module runs, so we do it here at collection
   time.

2. ``synthetic_wide_df`` — the shared multi-timeframe fixture described in
   "Appendix A" of the action_zones plan
   (external/docs/superpowers/plans/2026-07-19-zone-selection-experiment.md).
   Every later task's tests (labels, action-space, indicator attributes,
   regression/classification, ...) build on this same fixture, so it is
   documented in detail below rather than left as a one-off helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import talib

# --- sys.path wiring --------------------------------------------------------
# notebooks/action_zones/ is this file's grandparent (tests/conftest.py ->
# tests/ -> action_zones/). Insert once, so `import azlib` resolves both when
# pytest is invoked from /code (the Docker workdir) and from any other cwd.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat "no tests collected" as success (exit 0), not pytest's default 5.

    Task 0 ships this suite empty by design (later tasks add the first real
    tests) and the Docker gate command is fixed as
    ``pytest notebooks/action_zones/tests -v`` with no extra flags, so this
    can't be handled via a CLI flag — it has to live here. Scoped to this
    conftest only: running the repo's top-level `tests/` suite (a sibling,
    unrelated directory) never loads this hook, so its exit-code behavior is
    untouched.
    """
    if session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK


# --- synthetic_wide_df fixture (Appendix A) ---------------------------------

# Fixed seed constant. NEVER replace with time.time()/os.urandom/an unseeded
# np.random call — every test run must reproduce byte-identical fixture data
# so a failure is reproducible from the seed alone.
_SEED = 42

# "A few hundred rows" per Appendix A. 400 one-minute rows gives every
# rolling/EWM warmup used below (atr_14_ma_5 needs ~18 rows, rsi/macd/ema
# unstable periods need ~35) a comfortable amount of settled, non-NaN data
# while staying small enough to build+use in every test with no I/O.
_N_ROWS = 400

# Appendix A: "for tf in {15, 60, 240}".
_TFS = (15, 60, 240)

# LINK/USDT-ish starting price (see project memory: NN datasets are built on
# link_usdt). The absolute scale doesn't matter for any azlib formula (all of
# diff_prc/coeff/zscore are scale-invariant or price-ratio based) — it's only
# here to make printed frames look like real market data during debugging.
_START_PRICE = 20.0


def _build_ohlcv(rng: np.random.Generator) -> pd.DataFrame:
    """Base 1-minute OHLCV via a seeded log-return random walk.

    high/low are the close plus/minus an independent, strictly non-negative
    "wick" so every row satisfies ``high >= close >= low`` by construction
    (required: 1_high >= 1_close >= 1_low, per the task brief).
    """
    index = pd.date_range("2024-01-01", periods=_N_ROWS, freq="1min", name="timestamp")

    log_returns = rng.normal(loc=0.0, scale=0.0015, size=_N_ROWS)
    close = _START_PRICE * np.exp(np.cumsum(log_returns))

    # abs() guarantees the wick is >= 0, so high >= close >= low always holds
    # (equality is fine — the brief only requires >=, not strict >). A small
    # additive floor keeps high strictly > low so true-range (used by ATR)
    # never collapses to exactly zero.
    wick_up = np.abs(rng.normal(loc=0.0008, scale=0.0006, size=_N_ROWS)) * close + 1e-6
    wick_dn = np.abs(rng.normal(loc=0.0008, scale=0.0006, size=_N_ROWS)) * close + 1e-6
    high = close + wick_up
    low = close - wick_dn
    open_ = np.concatenate(([close[0]], close[:-1]))
    volume = rng.uniform(50.0, 500.0, size=_N_ROWS)
    taker_base_vol = volume * rng.uniform(0.3, 0.7, size=_N_ROWS)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "taker_base_vol": taker_base_vol,
        },
        index=index,
    )


def _add_tf_columns(df: pd.DataFrame, tf: int) -> None:
    """Add every {tf}_* column from Appendix A for one timeframe, in place.

    Mirrors data.py::_build_wide_df's real semantics for the *_high/_low/_close
    triple: {tf}_high/{tf}_low are the cummax/cummin of the 1-minute high/low
    within the current tf-minute bucket (a forming-candle-aware running
    extreme), and {tf}_close is the raw 1-minute close (a candle's current
    close is always just the latest 1-minute close, whether the candle is
    still forming or not). tf=1 columns collapse to the raw high/low/close
    (cummax/cummin over a 1-row bucket is the value itself), which is why
    `1_high/1_low/1_close` are handled by the same code path with tf=1.

    The indicator columns (rsi/macd/ema) are a fixture-only simplification:
    real production indicators are computed via indicators/library's
    IndicatorField classes on data.get_df(tf)'s closed-candle slices, which
    is out of scope to replicate exactly here (that's Task 3's job). To keep
    this fixture useful for testing tf-parameterized code (instead of every
    tf accidentally producing identical numbers, which would hide a bug like
    "code ignores the tf argument"), indicators are computed on an
    EWM-smoothed proxy for "this tf's close" whose span scales with tf --
    larger tf -> smoother, slower-moving proxy series, deterministic given
    the same seeded `close` array.
    """
    bucket = df.index.floor(f"{tf}min")
    tf_high = df.groupby(bucket)["high"].cummax()
    tf_low = df.groupby(bucket)["low"].cummin()
    tf_close = df["close"]  # raw 1-min close, see docstring above

    df[f"{tf}_high"] = tf_high
    df[f"{tf}_low"] = tf_low
    df[f"{tf}_close"] = tf_close

    # --- diff_prc / diff_prc_rm_6 (Task 2 formulas, precomputed here so the
    # fixture also exercises azlib.space's "prefer existing columns" path) ---
    high_diff = tf_high.pct_change() * 100.0
    low_diff = tf_low.pct_change() * 100.0
    df[f"{tf}_high_diff_prc"] = high_diff
    df[f"{tf}_high_diff_prc_rm_6"] = high_diff.rolling(6).mean()
    df[f"{tf}_low_diff_prc"] = low_diff
    df[f"{tf}_low_diff_prc_rm_6"] = low_diff.rolling(6).mean()

    # --- {tf}_atr_14_ma_5: strictly positive once warmed up -------------
    # true_range = tf_high - tf_low >= high - low > 0 at every row (cummax
    # only ever grows the high side, cummin only ever shrinks the low side),
    # so a rolling mean of positive values is itself positive.
    true_range = tf_high - tf_low
    atr_14 = true_range.rolling(14).mean()
    df[f"{tf}_atr_14_ma_5"] = atr_14.rolling(5).mean()

    # --- indicator columns (see docstring: EWM-smoothed close proxy) -----
    span = max(tf // 5, 2)
    close_proxy = tf_close.ewm(span=span, adjust=False).mean().to_numpy(dtype=float)

    rsi_14 = talib.RSI(close_proxy, timeperiod=14)
    rsi_ma8 = talib.EMA(rsi_14, timeperiod=8)
    macd, _macd_signal, macd_hist = talib.MACD(
        close_proxy, fastperiod=12, slowperiod=26, signalperiod=9
    )
    ema_25 = talib.EMA(close_proxy, timeperiod=25)

    df[f"{tf}_rsi_14"] = rsi_14
    df[f"{tf}_rsi_ma8"] = rsi_ma8
    df[f"{tf}_macd_12_26_9"] = macd
    df[f"{tf}_macd_hist_12_26_9"] = macd_hist
    df[f"{tf}_ema_25"] = ema_25


def _make_synthetic_wide_df() -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)
    df = _build_ohlcv(rng)

    # 1_high/1_low/1_close: tf=1 case of the same aggregation (see
    # _add_tf_columns docstring) — equal to the raw high/low/close.
    _add_tf_columns(df, 1)

    for tf in _TFS:
        _add_tf_columns(df, tf)

    return df


@pytest.fixture
def synthetic_wide_df() -> pd.DataFrame:
    """Small, deterministic, multi-timeframe wide df for azlib tests.

    ~400 rows of 1-minute-indexed synthetic OHLCV plus, for tf in {1, 15, 60,
    240}: {tf}_high/{tf}_low/{tf}_close, {tf}_high_diff_prc(_rm_6),
    {tf}_low_diff_prc(_rm_6), {tf}_atr_14_ma_5, {tf}_rsi_14, {tf}_rsi_ma8,
    {tf}_macd_12_26_9, {tf}_macd_hist_12_26_9, {tf}_ema_25 (Appendix A of
    external/docs/superpowers/plans/2026-07-19-zone-selection-experiment.md).

    Built from a seeded numpy Generator (fixed integer seed, see _SEED) so
    the same rows/values come back on every call/run — required for
    reproducible test failures. Guarantees, by construction:
      - 1_high >= 1_close >= 1_low (and same for every {tf}_high/_low/_close)
      - {tf}_atr_14_ma_5 > 0 for every non-warmup row

    Function-scoped (a fresh frame per test): later tasks' `add_labels()`
    mutates the wide df in place, so tests must not share one frame instance
    across the suite.
    """
    return _make_synthetic_wide_df()
