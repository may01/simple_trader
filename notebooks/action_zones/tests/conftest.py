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


# --- synthetic_wide_df fixture (Appendix A) ---------------------------------

# Fixed seed constant. NEVER replace with time.time()/os.urandom/an unseeded
# np.random call — every test run must reproduce byte-identical fixture data
# so a failure is reproducible from the seed alone.
_SEED = 42

# "A few hundred rows" per Appendix A, bumped from the original 400 for
# Task 2's rework (see task-2-report.md): azlib.space.price_levels now
# reduces {tf}_high/{tf}_low to COMPLETED {tf}_is_closed candles before
# computing anything (task-2-brief.md's completed-candle reduction) rather
# than the prior — wrong — per-minute .shift(1). That needs real elapsed
# time, not just elapsed rows: the first non-NaN level for a given tf/window
# requires (window + 1) completed candles (to fill the rolling window on the
# closed-candle sequence) plus a few more minutes into the following forming
# candle to observe the held-constant broadcast, i.e. roughly
# (window + 2) * tf minutes. With the function's *default* window=6 that is
# (6+2)*240 = 1920 minutes for tf=240 alone -- too large for a fixture this
# fixture's other consumers (talib warmup, label coverage) need to stay
# small. 750 rows (12.5h) is chosen instead so that test_layer2_space.py's
# synthetic-fixture sanity test can use an explicit smaller window=2
# override for the label_coeff parametrization -- (2+1)*240 = 720 minutes
# fully closed (3 completed 240-candles) plus 30 forming-candle minutes
# fits inside 750 rows, giving every tf in {15, 60, 240} at least one
# non-NaN row with window=2. The full algorithmic proof (default window,
# exact hand-computed values, held-constant, no-look-ahead, warm-up) lives
# in that same file's dedicated tiny hand-crafted tf=15 frame, not here.
_N_ROWS = 750

# Appendix A: "for tf in {15, 60, 240}".
_TFS = (15, 60, 240)

# LINK/USDT-ish starting price (see project memory: NN datasets are built on
# link_usdt). The absolute scale doesn't matter for any azlib formula (all of
# diff_prc/coeff/zscore are scale-invariant or price-ratio based) — it's only
# here to make printed frames look like real market data during debugging.
_START_PRICE = 20.0


def _build_ohlcv(
    rng: np.random.Generator, n_rows: int = _N_ROWS, vol_scale: float = 1.0
) -> pd.DataFrame:
    """Base 1-minute OHLCV via a seeded log-return random walk.

    high/low are the close plus/minus an independent, strictly non-negative
    "wick" so every row satisfies ``high >= close >= low`` by construction
    (required: 1_high >= 1_close >= 1_low, per the task brief).

    ``n_rows``/``vol_scale`` (Task 8 addition, default to the original
    fixture's own 750/1.0 so every EXISTING caller of ``_build_ohlcv`` is
    unaffected): ``vol_scale`` multiplies both the log-return std and the
    wick means/stds uniformly, so a caller needing MORE label/zone
    "reachability" within a short forward window (Task 8's
    ``two_synthetic_datasets`` fixture — see below) can turn up realized
    volatility without hand-tuning three separate scale constants.
    """
    index = pd.date_range("2024-01-01", periods=n_rows, freq="1min", name="timestamp")

    log_returns = rng.normal(loc=0.0, scale=0.0015 * vol_scale, size=n_rows)
    close = _START_PRICE * np.exp(np.cumsum(log_returns))

    # abs() guarantees the wick is >= 0, so high >= close >= low always holds
    # (equality is fine — the brief only requires >=, not strict >). A small
    # additive floor keeps high strictly > low so true-range (used by ATR)
    # never collapses to exactly zero.
    wick_up = np.abs(rng.normal(loc=0.0008 * vol_scale, scale=0.0006 * vol_scale, size=n_rows)) * close + 1e-6
    wick_dn = np.abs(rng.normal(loc=0.0008 * vol_scale, scale=0.0006 * vol_scale, size=n_rows)) * close + 1e-6
    high = close + wick_up
    low = close - wick_dn
    open_ = np.concatenate(([close[0]], close[:-1]))
    volume = rng.uniform(50.0, 500.0, size=n_rows)
    taker_base_vol = volume * rng.uniform(0.3, 0.7, size=n_rows)

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

    # --- {tf}_is_closed: True at the last 1-min row of each tf-period
    # candle -- mirrors data.py::_build_wide_df's real per-tf branches
    # exactly (see data.py lines ~219-239). Required by azlib.space's Task
    # 2 rework: price_levels reduces to completed candles via this column
    # before computing anything (task-2-brief.md's "completed-candle
    # reduction") -- {tf}_high/{tf}_low above are the *forming*
    # cummax/cummin within the current bucket, not that candle's final
    # value, until the row where {tf}_is_closed is True.
    idx = df.index
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
    df[f"{tf}_is_closed"] = is_closed

    # --- {tf}_atr_14_ma_5: strictly positive once warmed up -------------
    # true_range = tf_high - tf_low >= high - low > 0 at every row (cummax
    # only ever grows the high side, cummin only ever shrinks the low side),
    # so a rolling mean of positive values is itself positive.
    true_range = tf_high - tf_low
    atr_14 = true_range.rolling(14).mean()
    df[f"{tf}_atr_14_ma_5"] = atr_14.rolling(5).mean()

    # --- {tf}_high_diff_prc / {tf}_low_diff_prc --------------------------
    # Task 6 (azlib/rr.py) reach-probability estimator input. Mirrors
    # indicators/library/price_derivatives.py's real `HighDiffPrcField` /
    # `LowDiffPrcField` production columns EXACTLY (same 1-step
    # `(series - series.shift(1)) / series.shift(1) * 100` formula, applied
    # to the FORMING `{tf}_high`/`{tf}_low` cummax/cummin columns above) --
    # a per-1-minute-row, minute-to-minute series, DELIBERATELY NOT the
    # completed-candle-to-candle sequence `azlib.space.price_levels`
    # computes internally (see that module's docstring: this wide-df column
    # is "NEVER reused" by azlib.space, for that exact reason). For
    # reach-probability's own touch/first-passage semantics (design spec
    # §6.1) this forming series is actually the RIGHT input, not a
    # substitute: `{tf}_high` is a running cummax within the still-forming
    # candle, so its row-to-row diff_prc directly tracks whether/how far a
    # new high-water mark was touched at each passing minute -- exactly
    # "has the candle touched this level yet", not just its final close.
    df[f"{tf}_high_diff_prc"] = (tf_high - tf_high.shift(1)) / tf_high.shift(1) * 100.0
    df[f"{tf}_low_diff_prc"] = (tf_low - tf_low.shift(1)) / tf_low.shift(1) * 100.0

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


def _make_synthetic_wide_df(
    seed: int = _SEED, n_rows: int = _N_ROWS, vol_scale: float = 1.0
) -> pd.DataFrame:
    """Build one synthetic wide df. ``seed``/``n_rows``/``vol_scale`` default
    to the original fixture's own values (Task 8 addition — every EXISTING
    caller, i.e. the plain ``synthetic_wide_df`` fixture below, is
    unaffected). See ``two_synthetic_datasets`` (Task 8) for why a second,
    independently-seeded/higher-volatility variant is needed.
    """
    rng = np.random.default_rng(seed)
    df = _build_ohlcv(rng, n_rows=n_rows, vol_scale=vol_scale)

    # 1_high/1_low/1_close: tf=1 case of the same aggregation (see
    # _add_tf_columns docstring) — equal to the raw high/low/close.
    _add_tf_columns(df, 1)

    for tf in _TFS:
        _add_tf_columns(df, tf)

    return df


@pytest.fixture
def synthetic_wide_df() -> pd.DataFrame:
    """Small, deterministic, multi-timeframe wide df for azlib tests.

    ~750 rows of 1-minute-indexed synthetic OHLCV plus, for tf in {1, 15, 60,
    240}: {tf}_high/{tf}_low/{tf}_close, {tf}_is_closed, {tf}_atr_14_ma_5,
    {tf}_high_diff_prc/{tf}_low_diff_prc, {tf}_rsi_14, {tf}_rsi_ma8,
    {tf}_macd_12_26_9, {tf}_macd_hist_12_26_9, {tf}_ema_25 (Appendix A of
    external/docs/superpowers/plans/2026-07-19-zone-selection-experiment.md).

    Built from a seeded numpy Generator (fixed integer seed, see _SEED) so
    the same rows/values come back on every call/run — required for
    reproducible test failures. Guarantees, by construction:
      - 1_high >= 1_close >= 1_low (and same for every {tf}_high/_low/_close)
      - {tf}_atr_14_ma_5 > 0 for every non-warmup row
      - {tf}_is_closed True at exactly the last 1-min row of each {tf}-period
        candle (see data.py::_build_wide_df, mirrored in _add_tf_columns)

    Function-scoped (a fresh frame per test): later tasks' `add_labels()`
    mutates the wide df in place, so tests must not share one frame instance
    across the suite.
    """
    return _make_synthetic_wide_df()


# --- rng_matrix fixture (Task 4/5 cross-layer) ------------------------------

# Task 4's brief specifies its Layer-5 forward integration test verbatim,
# `test_regression_feeds_fusion(rng_matrix)` -- a plain (X, y) numpy matrix,
# independent of synthetic_wide_df's wide-df/indicator-column shape (Task 4's
# azlib.models and Task 5's azlib.infer operate on bare numpy arrays, not
# wide dfs). Not specified further anywhere in the plan/design doc beyond
# that one usage (`X, y = rng_matrix`; `X[:, :1]`, `X[:, 1:2]` both used, so
# X needs >= 2 columns), so this fixture's exact shape/distribution is this
# task's own design choice, documented here for later tasks (Task 5) that
# reuse it.
_RNG_MATRIX_SEED = 123
_RNG_MATRIX_N = 200


@pytest.fixture
def rng_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic (X, y) numpy matrix for azlib.models/azlib.infer.

    X: (200, 2) independent standard-normal columns. y: a mix of a linear
    term in column 0 and a quadratic term in column 1 (plus small noise), so
    that `fit_regression(X[:, :1], y, "linear")` and
    `fit_regression(X[:, 1:2], y, "poly2")` each have real, non-degenerate
    signal to fit -- not just noise -- without needing azlib.indicators/
    azlib.space at all. Seeded (see _RNG_MATRIX_SEED) for reproducibility,
    independent of synthetic_wide_df's own _SEED.
    """
    rng = np.random.default_rng(_RNG_MATRIX_SEED)
    X = rng.normal(size=(_RNG_MATRIX_N, 2))
    noise = rng.normal(scale=0.05, size=_RNG_MATRIX_N)
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] ** 2 + noise
    return X, y


# --- synthetic_pipeline fixture (Task 5's own integration test) ------------
#
# task-5-brief.md's integration test, `test_y_sweep_produces_selectable_y`:
#   sweep = sweep_y(**synthetic_pipeline, y_grid=np.round(np.arange(-2,2.01,0.1),1))
# so this fixture is a plain dict of every `azlib.infer.sweep_y` keyword
# EXCEPT `y_grid` (the test supplies that explicitly).


def _bracketing_price_levels(wide_df: pd.DataFrame, tf: int) -> tuple[pd.Series, pd.Series]:
    """Stub `price_levels_fn` for `synthetic_pipeline` -- CONSTANT
    high/low price levels that bracket every row's actual `1_low`/`1_high`
    (half the observed min below, 1.5x the observed max above).

    Deliberately NOT `azlib.space.price_levels` (Task 2's real per-candle,
    warm-up-NaN function) -- `sweep_y`'s `price_levels_fn` parameter exists
    precisely so Task 5's own tests can inject a fully controllable,
    NaN-free zone range instead of depending on Task 2's real per-candle
    band width (a function of the synthetic RNG's own volatility, unrelated
    to what THIS layer needs to exercise) -- see task-5-brief.md's
    dependency-inversion note (`rr_fn`); the same idea applied one layer up
    to `price_levels_fn`. A real notebook driver (Task 9) passes
    `azlib.space.price_levels` itself here instead.

    Because every row's entry price is guaranteed strictly inside
    `[low, high]` by construction, sweeping `y` (-> a coeff between 0 and 1
    for `y` in the typical `[-2, 2]` sweep range with `fused_std=0.2`, see
    `synthetic_pipeline` below) sweeps `zone_limit` MONOTONICALLY across
    the entire observed entry-price range -- giving both the integration
    test and any local monotonic-in-Y unit test real, deterministic,
    warm-up-free signal.
    """
    lo = float(min(wide_df["1_low"].min(), wide_df["1_high"].min())) * 0.5
    hi = float(max(wide_df["1_low"].max(), wide_df["1_high"].max())) * 1.5
    idx = wide_df.index
    return pd.Series(hi, index=idx), pd.Series(lo, index=idx)


@pytest.fixture
def synthetic_pipeline(synthetic_wide_df) -> dict:
    """kwargs for `azlib.infer.sweep_y` (everything but `y_grid`).

    - `fused_mean`/`fused_std`: a flat mean=0.5/std=0.2 pair, standing in
      for Layer 4's fused regression output. Task 5's own unit tests
      exercise `fuse_inverse_variance` directly on real per-group inputs
      elsewhere -- this fixture only needs SOME plausible per-point
      (mean, std) to drive `sweep_y` end-to-end, not a real fitted model.
    - `wide_df`/`tf`/`direction`: `synthetic_wide_df` (see that fixture),
      tf=15, direction="long".
    - `price_levels_fn`: `_bracketing_price_levels` (see its docstring) --
      not the real `azlib.space.price_levels`.
    - `strict_label`: a sparse, deterministic boolean scattering (every
      37th row True) -- enough strict-labeled rows to make
      `strict_coverage` well-defined (non-NaN) without labeling most of
      the frame.
    - `rr_fn`: a stub standing in for Task 6's real R/R (`azlib.rr`'s
      `reach_prob_estimator`/`rr_grid`/`select_levels` -- see
      `azlib/infer.py`'s module docstring) -- `2 * zone_marking.mean() -
      0.1`, which crosses `select_y`'s `> 0.0` fee-aware
      expected-return-after-fees "profitable" threshold once a given `Y`'s
      zone covers more than 5% of `wide_df`'s rows (mirrors Task 6's own
      contract: `rr_fn` returns an expected-return-after-fees number, not a
      raw R/R ratio -- the `- 0.1` stands in for a small fixed fee/cost so
      an EMPTY zone, `2*0 - 0.1 = -0.1`, is correctly non-profitable rather
      than sitting exactly on the boundary). Combined with
      `_bracketing_price_levels` guaranteeing zone coverage sweeps from ~0%
      to ~100% across the `y_grid`, this ensures the integration test
      exercises `select_y`'s PROFITABLE-row branch (not only its NaN-safe
      default) at least once.
    """
    wide_df = synthetic_wide_df
    mean = pd.Series(0.5, index=wide_df.index)
    std = pd.Series(0.2, index=wide_df.index)

    strict_label = pd.Series(False, index=wide_df.index)
    strict_label.iloc[::37] = True

    def rr_fn(zone_marking: pd.Series) -> float:
        return 2.0 * float(zone_marking.mean()) - 0.1

    return {
        "fused_mean": mean,
        "fused_std": std,
        "wide_df": wide_df,
        "tf": 15,
        "direction": "long",
        "price_levels_fn": _bracketing_price_levels,
        "strict_label": strict_label,
        "rr_fn": rr_fn,
    }


# --- synthetic_pipeline_full fixture (Task 7's own integration test) -------
#
# task-7-brief.md's integration test, `test_zoned_dataset_and_results_roundtrip`:
#   zdf = build_zoned_dataset(**synthetic_pipeline_full)
# so this fixture is a plain dict of every `azlib.zones.build_zoned_dataset`
# keyword arg (`wide_df`, `tf`, `direction`, `zone_limit`, `levels`).


@pytest.fixture
def synthetic_pipeline_full(synthetic_wide_df) -> dict:
    """kwargs for `azlib.zones.build_zoned_dataset` (Task 7's own integration test).

    - `wide_df`/`tf`/`direction`: `synthetic_wide_df` (see that fixture),
      tf=15, direction="long" -- matching the integration test's own
      hard-coded `az_zone_long_15` column-name assertion.
    - `zone_limit`: built from the SAME `_bracketing_price_levels` stub
      `synthetic_pipeline` uses (a constant, NaN-free band bracketing every
      row's `1_low`/`1_high`) at a fixed mid-band coeff (0.5) -- deliberately
      NOT Task 2's real `azlib.space.price_levels` (which has warm-up NaN
      rows) for the ENTRY threshold, so the integration test's zone column
      has a real, deterministic mix of True/False rows regardless of Task
      2's own warm-up window. `build_zoned_dataset`'s `az_tgt`/`az_sl`
      columns still go through the REAL `azlib.space.price_levels`
      internally (not injected -- see that function's own docstring), so
      those two columns legitimately carry Task 2's warm-up NaNs; only the
      zone-membership/entry threshold here is stubbed.
    - `levels`: a plausible, already-selected (design spec §6's
      `azlib.rr.select_levels` output shape) `{"tgt_x", "sl_x", "rr",
      "exp_ret"}` dict -- not itself produced via a real R/R grid search
      here (that is `azlib.zones.build_rr_levels`'s job, exercised directly
      by its own unit tests elsewhere in this file), just a fixed,
      profitable-looking combo standing in for one.
    """
    wide_df = synthetic_wide_df
    tf = 15
    direction = "long"

    price_high_level, price_low_level = _bracketing_price_levels(wide_df, tf)
    coeff_y = 0.5
    zone_limit = (
        price_low_level.to_numpy()
        + coeff_y * (price_high_level.to_numpy() - price_low_level.to_numpy())
    )

    levels = {"tgt_x": 1.5, "sl_x": 1.0, "rr": 1.3, "exp_ret": 0.02}

    return {
        "wide_df": wide_df,
        "tf": tf,
        "direction": direction,
        "zone_limit": zone_limit,
        "levels": levels,
    }


# --- two_synthetic_datasets fixture (Task 8's own integration test) --------
#
# task-8-brief.md's required RED integration test:
#   def test_train_then_oos_metrics(monkeypatch, two_synthetic_datasets):
#       rp = run_train("train.env", tf=15, direction="long")
#       zoos = run_oos(rp, "oos.env")
#       m = metrics(zoos, 15, "long", label_params={})
#       assert 0 <= m["strict_coverage"] <= 1 and "realized_rr" in m
#
# `run_train`/`run_oos` are "env-driven": they set os.environ from a passed
# env-FILE path (azlib.validate._apply_env_file, a real KEY=VALUE parser --
# see that module) and then call `load_wide_df()` (env-driven itself, via
# helpers.wide_df_path()). This fixture therefore does two, and only two,
# things a real Docker-mounted-volume run does NOT need:
#   1. Writes two REAL, tiny env files (KEY=VALUE, same format as
#      configs/*.env) to `tmp_path` -- `_apply_env_file` itself is exercised
#      for real, un-mocked, against these files.
#   2. Monkeypatches `azlib.validate.load_wide_df` (per the global
#      constraint: never load the real 6 GB volume df in tests) to a fake
#      that picks train_df/oos_df by reading `os.environ["DATA_SET_NAME"]`
#      -- the exact env var the two files set to different values -- so the
#      fake genuinely proves `run_train`/`run_oos` applied the RIGHT env
#      file before loading data, not a hardcoded stub.
#   Also monkeypatches `azlib.validate.dataset_folder` (env-driven artifact
#   root -- see validate.py's `_results_dir`) to `tmp_path`, so
#   ResultsFile + every sidecar this task writes lands under `tmp_path`,
#   never the real (and, for ROOT_FOLDER="short", not even mounted in this
#   Docker service) `/trader_data*` volume root -- satisfies the global
#   constraint "artifacts write to a caller/env-supplied path; in tests use
#   tmp_path. Never write into the worktree."
#
# `train_df`/`oos_df`: two INDEPENDENTLY-seeded synthetic wide dfs (seeds 42
# and 99 -- 42 matches `synthetic_wide_df`'s own `_SEED`, reused here only
# for a familiar/reproducible train seed, not because it needs to be THAT
# specific value; 99 is arbitrary but fixed), each bumped to 2000 rows (up
# from the base fixture's 750) and 3x the base fixture's volatility
# (`vol_scale=3.0`). Both bumps exist for the SAME reason: Task 8's own
# `run_train` default-label-params pipeline (`azlib.config
# .default_label_params`, m=2.0/x=2.0 ATR-multiples, n=1 -> only a
# `1*15=15`-minute forward window at tf=15) needs a real, non-vacuous
# strict-positive count AND real zone/label overlap to keep
# `metrics()["strict_coverage"]` non-NaN (`0 <= nan <= 1` is `False` in
# Python -- the brief's own integration-test assertion would fail on a
# vacuous 0-positive fixture) -- test_layer1_loader.py hit the exact same
# vacuous-positives risk with the BASE (750-row, 1x-vol) fixture and worked
# around it with much looser label params (m=0.3/x=0.3); Task 8 cannot loosen
# `run_train`'s own INTERNAL default-label-params choice for the brief's
# verbatim-signature integration test (`run_train(train_env, tf, direction)`
# takes no label_params override in that call), so the fixture's DATA is
# tuned upward instead. Chosen empirically (see task-8-report.md) against
# the actual `run_train` pipeline -- not derived from a closed-form
# probability calculation.
_VALIDATE_N_ROWS = 2000
_VALIDATE_VOL_SCALE = 3.0


def _write_env_file(path, data_set_name: str) -> None:
    path.write_text(
        "ROOT_FOLDER=short\n"
        "DATA_ROOT=az_test\n"
        f"DATA_SET_NAME={data_set_name}\n"
        "PAIR=test_pair\n"
        "EXCHANGE_FEE=0.001\n"
    )


@pytest.fixture
def two_synthetic_datasets(tmp_path, monkeypatch) -> dict:
    """Everything `tests/test_layer8_validate.py` needs to drive
    `run_train`/`run_oos` end to end without the real 6 GB volume df or the
    real `/trader_data*` mount -- see the module-level comment block above
    for the full rationale.

    Returns ``{"train_env", "oos_env", "train_df", "oos_df"}`` -- the two
    env-file PATHS (str) `run_train`/`run_oos` are meant to be called with,
    plus the two underlying frames themselves (for tests that want to
    hand-inspect them, e.g. to independently recompute an expected metric).
    """
    train_df = _make_synthetic_wide_df(seed=42, n_rows=_VALIDATE_N_ROWS, vol_scale=_VALIDATE_VOL_SCALE)
    oos_df = _make_synthetic_wide_df(seed=99, n_rows=_VALIDATE_N_ROWS, vol_scale=_VALIDATE_VOL_SCALE)

    train_env_path = tmp_path / "train.env"
    oos_env_path = tmp_path / "oos.env"
    _write_env_file(train_env_path, "train")
    _write_env_file(oos_env_path, "oos")

    frames = {"train": train_df, "oos": oos_df}

    def fake_load_wide_df():
        import os

        return frames[os.environ["DATA_SET_NAME"]].copy()

    artifacts_root = tmp_path / "artifacts" / ""  # trailing sep, matches dataset_folder()'s own convention
    monkeypatch.setattr("azlib.validate.load_wide_df", fake_load_wide_df)
    monkeypatch.setattr("azlib.validate.dataset_folder", lambda: str(artifacts_root) + "/")

    return {
        "train_env": str(train_env_path),
        "oos_env": str(oos_env_path),
        "train_df": train_df,
        "oos_df": oos_df,
    }
