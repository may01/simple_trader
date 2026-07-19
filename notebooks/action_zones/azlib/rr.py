"""azlib/rr.py — Task 6: hybrid reach-probability estimator + R/R grid.

Implements design spec §6/§6.1
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):

  §6.1 ``reach_prob_estimator`` -- a **touch / first-passage** probability
      estimator (NOT a terminal Gaussian): given a train array of extreme
      diffs (``{tf}_high_diff_prc`` for the up/long-target side,
      ``-{tf}_low_diff_prc`` for the down/short-target side -- see "Sign
      convention" below), frozen on TRAIN only, hybrid body+tail:
        - body: a smoothed (linearly-interpolated) empirical survival
          function ``P(X >= level)``;
        - tail: where a level's supporting sample count drops below
          ``min_bin``, a fitted generalized-Pareto (POT) tail, so far-X
          levels get a smooth, non-zero, non-noisy estimate instead of a
          raw-empirical exact zero.
  §6  ``rr_grid`` -- sweep target-X/stop-X over a grid; ``rr =
      p_target/p_stop``; keep combos where the fee-aware expected return is
      profitable. ``select_levels`` picks the best such combo.
      ``reach_freq_drift`` compares train-frozen P_reach against realized
      OOS reach-frequency per level (regime-drift sanity check).

Sign convention (single canonical direction, no separate "down" code path)
----------------------------------------------------------------------------
``reach_prob_estimator`` always implements ONE convention:
``P(X >= level) = fraction of train values >= level`` -- monotone
NON-INCREASING in ``level``, ``P(level == min(train)) == 1.0``. This is
exactly what the UP/long-target side needs
(``P(high_diff_prc >= level)``, design spec §6.1's own formula, verbatim).

The DOWN/stop-loss side's semantics (``P(low_diff_prc <= level)`` for a
level that is typically negative) is obtained by feeding this SAME
estimator the NEGATED low-diff series and querying with a NON-NEGATIVE
magnitude:

::

    reach_up   = reach_prob_estimator(high_diff_prc_train)        # P(high_diff_prc >= level)
    reach_down = reach_prob_estimator(-low_diff_prc_train)         # P(-low_diff_prc >= level)
                                                                    # == P(low_diff_prc <= -level)

so that a caller only ever passes non-negative ``x_grid`` magnitudes to
EITHER callable (matching design spec §6's own "[0, 2.0]" sweep range) --
no sign-flipping needed inside ``rr_grid`` or by callers of it. The
integration test below (and this module's own unit tests) build ``reach_up``
and ``reach_down`` this way; ``rr_grid`` itself is agnostic to how its two
injected callables were constructed -- it only ever calls them with plain
``x_grid`` values.

Expected-return-after-fees formula (this task's own documented design
choice -- task-6-brief.md: "scale reward/risk by candle_size -- document
your exact expected-return formula")
----------------------------------------------------------------------------
::

    reward = tgt_x * candle_size          # price units
    risk   = sl_x  * candle_size          # price units
    expected_return_after_fees = reward * p_target - risk * p_stop
                                  - 2 * fee * candle_size

``fee`` is a FRACTIONAL trading-fee rate (e.g. ``0.001`` = 0.1% per side);
it is scaled by ``candle_size`` (the same price-unit reference reward/risk
use) so all three terms of the formula share one consistent unit -- with
``candle_size=1.0`` this reduces EXACTLY to ``reward*p_target -
risk*p_stop - 2*fee``, the brief's own literal "subtracts 2*fee" wording.
"Entry+exit fees" = ``2 * fee`` (one fee charge on the way in, one on the
way out).

``candle_size`` definition (task-6-brief.md: "define explicitly ... It's a
parameter passed into rr_grid")
----------------------------------------------------------------------------
Recommended (not computed by this module -- the caller/driver passes the
number in): the MEDIAN ``|{tf}_high - {tf}_low|`` in PRICE, over TRAIN, at
the SAME ``tf`` the zone/target/stop apply to -- i.e. one typical candle's
price range. Deliberately NOT ATR (design spec's own Scope decision: "No
ATR sizing for the action levels" -- this experiment's price space is
percentage-diff-based, not ATR-based; ``candle_size`` here is a plain,
simple normalizing reference, not a second action-level sizing mechanism).

Baseline normal-CDF column (design spec §6.1: "a normal-CDF estimate is
computed and logged ... only as a sanity-check reference, never used to
select levels")
----------------------------------------------------------------------------
``reach_prob_estimator``'s returned callable exposes its train array's
plain ``mean_``/``std_`` as attributes (not part of the ``Callable``
signature itself -- an implementation detail ``rr_grid`` optionally reads
via ``getattr``). When BOTH injected callables expose them, ``rr_grid``
adds ``p_target_normal``/``p_stop_normal`` baseline columns (Gaussian
survival-function probabilities at the same levels) purely for reporting;
``select_levels`` never reads them.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm

# Floor every returned reach-probability at this value -- guarantees the
# hybrid estimator NEVER returns an exact 0.0 (design spec §6.1: "far-X
# levels ... get a smooth, non-zero ... estimate"), and doubles as a guard
# against a literal 1/0 in rr_grid's `rr = p_target / p_stop` division
# (which would raise a numpy RuntimeWarning under this project's `-W error`
# test policy, and produce a useless +inf).
_PROB_FLOOR = 1e-6

# A tail excess sample smaller than this, or with (near-)zero variance
# (e.g. every peaks-over-threshold value tied at the same number), makes
# the closed-form method-of-moments GPD fit below degenerate (division by
# ~0 variance) -- guarded by falling back to a plain exponential tail
# (GPD shape=0) instead of computing the MOM formula in that case.
_MIN_EXCESS_FOR_MOM = 2
_MIN_EXCESS_VARIANCE = 1e-12


def _fit_gpd_mom(excess: np.ndarray) -> tuple[float, float]:
    """Closed-form method-of-moments (shape, scale) for a generalized-Pareto
    tail fit to ``excess`` (non-negative peaks-over-threshold values).

    Deliberately NOT ``scipy.stats.genpareto.fit`` -- that runs a numeric
    MLE optimizer, which can raise a scipy ``RuntimeWarning``/
    ``OptimizeWarning`` on small or near-degenerate excess samples
    (forbidden under this project's `-W error` test policy: see
    task-6-brief.md's "scipy fit warnings -- guard/seed to avoid"). MOM is
    closed-form -- no optimizer, no warnings, fully deterministic given the
    same excess sample:

    ::

        m, v = excess.mean(), excess.var(ddof=1)
        shape = 0.5 * (m**2 / v - 1)
        scale = 0.5 * m * (m**2 / v + 1)

    Degenerate guard (documented above at ``_MIN_EXCESS_FOR_MOM``/
    ``_MIN_EXCESS_VARIANCE``): too few points, non-positive variance, or a
    non-positive MOM scale all fall back to ``(shape=0.0, scale=max(mean,
    tiny floor))`` -- a plain exponential tail (GPD's ``shape=0`` special
    case), never a literal division by (near-)zero.

    **Shape clamped to >= 0.** A NEGATIVE GPD shape implies a FINITE upper
    support bound (``-scale/shape``) -- ``genpareto.sf`` is an EXACT
    ``0.0`` beyond it, which would make far-enough levels genuinely
    impossible-to-reach rather than "smooth, non-zero" (design spec §6.1's
    own requirement) -- a real market has no such hard ceiling, and a small
    or oddly-shaped excess sample can easily produce a spuriously negative
    MOM shape that has nothing to do with the true underlying tail. Clamped
    to the nearest ALWAYS-unbounded-support family member instead
    (``max(shape, 0.0)``, keeping the MOM ``scale`` unchanged) -- this is
    the module-level ``_PROB_FLOOR`` clip's structural counterpart: that
    floor guards the numeric edge case (never return a literal ``0.0``),
    this guards the SHAPE OF THE CURVE itself (never claim a level is
    truly unreachable).
    """
    excess = np.asarray(excess, dtype=float)
    m = float(excess.mean())
    if excess.size < _MIN_EXCESS_FOR_MOM:
        return 0.0, max(m, 1e-6)

    v = float(excess.var(ddof=1))
    if v <= _MIN_EXCESS_VARIANCE:
        return 0.0, max(m, 1e-6)

    shape = 0.5 * (m**2 / v - 1.0)
    scale = 0.5 * m * (m**2 / v + 1.0)
    if scale <= 0.0:
        return 0.0, max(m, 1e-6)
    return max(shape, 0.0), scale


class _HybridReachEstimator:
    """Frozen ``P(X >= level)`` callable -- smoothed empirical body +
    generalized-Pareto (POT) tail. See ``reach_prob_estimator``'s docstring
    for the full contract; this is a class (not a closure) only so the
    returned callable can also carry ``.mean_``/``.std_`` (the train
    array's own plain mean/std, for ``rr_grid``'s optional normal-CDF
    baseline logging column -- see module docstring's "Baseline" section).
    """

    def __init__(self, train_extreme_diff: np.ndarray, min_bin: int):
        arr = np.asarray(train_extreme_diff, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = arr.size
        if n < 2:
            raise ValueError(
                f"reach_prob_estimator needs >= 2 non-NaN train values, got {n}"
            )

        self.n = n
        self.mean_ = float(arr.mean())
        self.std_ = float(arr.std(ddof=1)) if n > 1 else 0.0

        sorted_arr = np.sort(arr)
        self._sorted = sorted_arr
        # Survival value AT each sorted (ascending) point:
        # P(X >= sorted_arr[i]) = (n - i) / n -- e.g. i=0 (the minimum)
        # -> n/n == 1.0 exactly; i=n-1 (the maximum) -> 1/n.
        self._survival_at_points = (n - np.arange(n)) / n

        # Crossover threshold `u`: the level with exactly `k = min(min_bin,
        # n)` training points still >= it -- task-6-brief.md's "tail kicks
        # in only where bin count < min_bin". Levels <= u use the body
        # (>= min_bin supporting points); levels > u use the parametric
        # tail (< min_bin supporting points, by construction).
        k = min(min_bin, n)
        self._u = float(sorted_arr[n - k])
        self._tail_p0 = k / n  # empirical survival EXACTLY at the threshold

        excess = sorted_arr[n - k :] - self._u  # >= 0, size k -- POT excesses
        self._shape, self._scale = _fit_gpd_mom(excess)

    def __call__(self, levels) -> np.ndarray:
        levels = np.atleast_1d(np.asarray(levels, dtype=float))

        # Body: linear interpolation between order statistics' own survival
        # values ("smoothed empirical ECDF", design spec §6.1) -- np.interp
        # constant-extrapolates below the observed minimum to 1.0 (its
        # first tabulated y-value), matching "P(level == min) == 1.0" for
        # levels at or below the observed minimum too.
        out = np.interp(levels, self._sorted, self._survival_at_points)

        # Tail: levels past the crossover threshold get the POT/GPD
        # extrapolation instead (continuous with the body AT `u` by
        # construction: genpareto.sf(0, ...) == 1, so the tail formula
        # evaluates to exactly `tail_p0` right at the threshold too).
        tail_mask = levels > self._u
        if np.any(tail_mask):
            excess_levels = levels[tail_mask] - self._u
            out[tail_mask] = self._tail_p0 * genpareto.sf(
                excess_levels, self._shape, loc=0.0, scale=self._scale
            )

        return np.clip(out, _PROB_FLOOR, 1.0)


def reach_prob_estimator(
    train_extreme_diff: np.ndarray, min_bin: int = 50
) -> Callable[[np.ndarray], np.ndarray]:
    """Hybrid empirical-body + generalized-Pareto-tail reach-probability
    estimator, frozen on ``train_extreme_diff`` (design spec §6.1).

    ``train_extreme_diff``: 1-D array of a train-only EXTREME diff_prc
    series (e.g. ``{tf}_high_diff_prc`` for the up/long-target side,
    ``-{tf}_low_diff_prc`` for the down/stop side -- see module docstring's
    "Sign convention"). NaNs are dropped before fitting. Requires >= 2
    non-NaN values (``ValueError`` otherwise -- a single point has no
    meaningful tail to fit).

    Returns a callable ``f(levels: array-like) -> np.ndarray`` where
    ``f(level) == P(X >= level)`` (touch/first-passage probability, NOT a
    terminal-distribution CDF):

    - **Body** (level <= the ``min_bin``-crossover threshold ``u`` --
      i.e. >= ``min_bin`` training points remain at/above that level):
      smoothed (linearly-interpolated between order statistics) empirical
      survival function.
    - **Tail** (level > ``u`` -- < ``min_bin`` supporting points): a
      generalized-Pareto peaks-over-threshold fit (method-of-moments, see
      ``_fit_gpd_mom``), continuous with the body at ``u``. This is what
      guarantees far-tail levels get a smooth, NON-ZERO estimate (a plain
      empirical fraction would be an exact ``0.0`` past the observed
      maximum) -- design spec §6.1 explicitly forbids a Gaussian here (fat
      tails would be underestimated); this uses generalized-Pareto instead.

    Every returned probability is floored at ``_PROB_FLOOR`` (``1e-6``) --
    never an exact ``0.0`` regardless of how far past the observed data a
    queried level is.

    The returned callable also exposes ``.mean_``/``.std_`` (the frozen
    train array's own plain mean/std, ``ddof=1``) as attributes -- an
    implementation detail ``rr_grid`` optionally reads for its normal-CDF
    baseline logging column (module docstring's "Baseline" section); not
    part of the ``Callable[[np.ndarray], np.ndarray]`` interface itself.
    """
    return _HybridReachEstimator(train_extreme_diff, min_bin)


# --- rr_grid ------------------------------------------------------------------


def rr_grid(
    reach_up: Callable[[np.ndarray], np.ndarray],
    reach_down: Callable[[np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    fee: float,
    candle_size: float,
    direction: str,
) -> pd.DataFrame:
    """Full (tgt_x, sl_x) cross-product grid-search over ``x_grid`` (design
    spec §6: "Sweep target-X and SL-X over [0, 2.0]").

    ``reach_up``/``reach_down``: the two reach-probability callables (see
    module docstring's "Sign convention" -- both take plain, non-negative
    ``x_grid``-style magnitudes and return ``P(reach)`` in ``[0, 1]``).
    ``direction`` picks which one is the TARGET side and which is the STOP
    side: for ``"long"``, profit requires an UP move (target -> ``reach_up``)
    and loss is triggered by a DOWN move (stop -> ``reach_down``); for
    ``"short"`` it is the reverse.

    Per ``(tgt_x, sl_x)`` pair (every combination of ``x_grid`` with
    itself -- ``len(x_grid) ** 2`` rows):

    ::

        p_target = target_fn(tgt_x)
        p_stop   = stop_fn(sl_x)
        rr       = p_target / p_stop
        reward   = tgt_x * candle_size
        risk     = sl_x  * candle_size
        expected_return_after_fees = reward*p_target - risk*p_stop - 2*fee*candle_size

    (see module docstring's "Expected-return-after-fees formula" for the
    full rationale/units discussion, and "candle_size definition" for the
    recommended ``candle_size`` computation, which is NOT done inside this
    function -- it is a plain ``float`` parameter, computed by the caller).

    ``rr``'s own denominator is floored at ``_PROB_FLOOR`` before dividing
    (belt-and-suspenders alongside ``reach_prob_estimator``'s own floor --
    protects against an externally-supplied ``reach_down``/``reach_up``
    callable that does NOT itself guarantee a non-zero result), so ``rr``
    is always finite, never a ``RuntimeWarning``-raising ``x/0``.

    Optional baseline columns ``p_target_normal``/``p_stop_normal`` (design
    spec §6.1's normal-CDF sanity reference, "logged ... never used to
    select levels") are added ONLY when the corresponding callable exposes
    ``.mean_``/``.std_`` attributes (as ``reach_prob_estimator``'s own
    return value does) -- absent otherwise, no error.

    Returns a tidy ``pd.DataFrame``, columns (at least) ``{"tgt_x", "sl_x",
    "p_target", "p_stop", "rr", "expected_return_after_fees"}``, one row per
    ``(tgt_x, sl_x)`` combination.
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    x_grid = np.asarray(x_grid, dtype=float)
    tgt_mesh, sl_mesh = np.meshgrid(x_grid, x_grid, indexing="ij")
    tgt_flat = tgt_mesh.ravel()
    sl_flat = sl_mesh.ravel()

    if direction == "long":
        target_fn, stop_fn = reach_up, reach_down
    else:
        target_fn, stop_fn = reach_down, reach_up

    p_target = np.asarray(target_fn(tgt_flat), dtype=float)
    p_stop = np.asarray(stop_fn(sl_flat), dtype=float)

    p_stop_safe = np.maximum(p_stop, _PROB_FLOOR)
    rr = p_target / p_stop_safe

    reward = tgt_flat * candle_size
    risk = sl_flat * candle_size
    expected_return_after_fees = reward * p_target - risk * p_stop - 2.0 * fee * candle_size

    data = {
        "tgt_x": tgt_flat,
        "sl_x": sl_flat,
        "p_target": p_target,
        "p_stop": p_stop,
        "rr": rr,
        "expected_return_after_fees": expected_return_after_fees,
    }

    target_mean, target_std = getattr(target_fn, "mean_", None), getattr(target_fn, "std_", None)
    stop_mean, stop_std = getattr(stop_fn, "mean_", None), getattr(stop_fn, "std_", None)
    if target_mean is not None and target_std is not None:
        data["p_target_normal"] = norm.sf(tgt_flat, loc=target_mean, scale=max(target_std, 1e-12))
    if stop_mean is not None and stop_std is not None:
        data["p_stop_normal"] = norm.sf(sl_flat, loc=stop_mean, scale=max(stop_std, 1e-12))

    return pd.DataFrame(data)


# --- select_levels --------------------------------------------------------


def select_levels(grid: pd.DataFrame) -> dict:
    """Best ``(tgt_x, sl_x)`` combo from an ``rr_grid`` output: max
    ``expected_return_after_fees`` among rows where that value is
    STRICTLY POSITIVE (design spec §6/task-6-brief.md: "excluding rows
    with exp_ret <= 0" -- a row worth exactly ``0.0`` breaks even after
    fees, not genuinely profitable, so it is excluded too).

    Returns ``{"tgt_x", "sl_x", "rr", "exp_ret"}`` (exact keys) for the
    winning row.

    Raises ``ValueError`` if NO row is profitable (``grid`` empty, or every
    row's ``expected_return_after_fees <= 0.0``) -- unlike
    ``azlib.infer.select_y``'s NaN-safe default (a documented NEUTRAL ``Y``
    always exists, ``0.0``), there is no equivalent "neutral" ``(tgt_x,
    sl_x)`` pair here: every combo either has learned a genuine edge or it
    doesn't, and silently returning some placeholder pair for later layers
    to build real target/stop prices from would be a much more dangerous
    silent failure than the loud one here.
    """
    profitable = grid.loc[grid["expected_return_after_fees"] > 0.0]
    if profitable.empty:
        raise ValueError(
            "select_levels: no (tgt_x, sl_x) combo has a positive "
            "expected_return_after_fees -- nothing to select"
        )

    best_idx = profitable["expected_return_after_fees"].idxmax()
    best = profitable.loc[best_idx]
    return {
        "tgt_x": float(best["tgt_x"]),
        "sl_x": float(best["sl_x"]),
        "rr": float(best["rr"]),
        "exp_ret": float(best["expected_return_after_fees"]),
    }


# --- reach_freq_drift ----------------------------------------------------


def reach_freq_drift(
    train_est: Callable[[np.ndarray], np.ndarray],
    oos_extreme_diff: np.ndarray,
    x_grid: np.ndarray,
) -> pd.DataFrame:
    """Train-frozen ``P_reach`` vs. realized OOS reach-frequency, per level
    (design spec §6.1's "Drift check" -- regime-drift sanity report).

    Per ``level`` in ``x_grid``:

    ::

        train_p  = train_est(level)                              # frozen train-side estimate
        oos_freq = fraction of oos_extreme_diff >= level          # realized OOS frequency, SAME "P(X >= level)" convention as reach_prob_estimator
        drift    = oos_freq - train_p                             # signed: >0 means OOS reaches that level MORE often than train predicted

    NaNs in ``oos_extreme_diff`` are dropped before computing ``oos_freq``
    (same convention as ``reach_prob_estimator``'s own train-side NaN
    handling); an all-NaN/empty ``oos_extreme_diff`` yields ``oos_freq``
    (and therefore ``drift``) of ``NaN`` at every level rather than a
    ``0/0`` warning.

    Returns a ``pd.DataFrame`` with columns ``{"level", "train_p",
    "oos_freq", "drift"}``, one row per ``x_grid`` entry, in ``x_grid``'s
    own order.
    """
    levels = np.asarray(x_grid, dtype=float)
    train_p = np.asarray(train_est(levels), dtype=float)

    oos = np.asarray(oos_extreme_diff, dtype=float)
    oos = oos[~np.isnan(oos)]

    if oos.size == 0:
        oos_freq = np.full(levels.shape, np.nan)
    else:
        oos_freq = np.array([float(np.mean(oos >= level)) for level in levels])

    drift = oos_freq - train_p

    return pd.DataFrame(
        {
            "level": levels,
            "train_p": train_p,
            "oos_freq": oos_freq,
            "drift": drift,
        }
    )
