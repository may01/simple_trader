"""Layer 6 tests — azlib.rr (hybrid reach-probability estimator + R/R grid).

Covers: reach_prob_estimator, rr_grid, select_levels, reach_freq_drift
(azlib/rr.py). See .superpowers/sdd/task-6-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §6/§6.1.

Filename carries "layer6" so `pytest -k layer6` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.rr existing, so — same
convention as test_layer2_space.py/test_layer5_infer.py — the whole module
imports it at top level.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from azlib.rr import reach_freq_drift, reach_prob_estimator, rr_grid, select_levels


# --- reach_prob_estimator -----------------------------------------------------


def test_monotone_non_increasing_and_min_level_prob_is_one():
    rng = np.random.default_rng(1)
    arr = rng.normal(0.0, 1.0, 300)
    est = reach_prob_estimator(arr, min_bin=50)

    levels = np.linspace(arr.min(), arr.max(), 60)
    p = est(levels)

    assert np.all(np.diff(p) <= 1e-9)  # non-increasing (float-tolerant)
    # P(level == min(train)) -- every point is >= its own minimum.
    assert p[0] == pytest.approx(1.0, abs=1e-9)


def test_body_region_matches_raw_empirical_at_a_training_point():
    # A level chosen deep in the body (far more than min_bin points remain
    # at/above it) must match the plain "count(arr >= level) / n" empirical
    # fraction EXACTLY -- it lands exactly on a training order statistic, so
    # the body's smoothed (linearly-interpolated) ECDF has no interpolation
    # artifact there (interpolating between a point and itself).
    rng = np.random.default_rng(4)
    arr = rng.normal(0.0, 1.0, 400)
    est = reach_prob_estimator(arr, min_bin=50)

    sorted_arr = np.sort(arr)
    probe = sorted_arr[50]  # 400 - 50 = 350 points >= probe, way above min_bin
    expected = float(np.sum(arr >= probe)) / arr.size

    got = est(np.array([probe]))[0]
    assert got == pytest.approx(expected, rel=1e-9)


def test_far_tail_returns_smooth_non_zero_not_raw_zero():
    # Beyond max(train), the RAW empirical fraction is exactly 0.0 (no
    # training point is >= a level past the observed max) -- design spec
    # §6.1 explicitly forbids that ("far-X levels get a smooth, non-zero,
    # non-noisy estimate", not "raw-zero"). A sparse, unbounded
    # (exponential-shaped, shape~0) far tail keeps the parametric fit's
    # support unbounded, so this holds at any finite query level.
    rng = np.random.default_rng(2)
    bulk = rng.normal(0.0, 0.5, 300)
    tail = 5.0 + rng.exponential(scale=1.0, size=8)  # sparse, far right tail
    arr = np.concatenate([bulk, tail])
    est = reach_prob_estimator(arr, min_bin=50)

    beyond_max = float(arr.max()) + 0.5
    further = float(arr.max()) + 1.5

    p_beyond, p_further = est(np.array([beyond_max, further]))

    assert int(np.sum(arr >= beyond_max)) == 0  # raw ECDF would say 0.0 here
    assert p_beyond > 0.0 and np.isfinite(p_beyond)
    assert p_further > 0.0 and np.isfinite(p_further)
    assert p_further < p_beyond  # still decaying, not flat/noisy


def test_tail_crossover_matches_documented_mom_gpd_formula():
    # Directly pins the documented formula (method-of-moments GPD fit on
    # peaks-over-threshold excesses, threshold = the level with exactly
    # `min_bin` supporting points) -- "tail kicks in only where bin count <
    # min_bin" (task-6-brief.md), verified by construction: the probed
    # level has strictly fewer than min_bin supporting training points.
    rng = np.random.default_rng(3)
    bulk = rng.normal(0.0, 0.5, 300)
    tail = 5.0 + rng.exponential(scale=1.0, size=8)
    arr = np.concatenate([bulk, tail])
    min_bin = 50
    est = reach_prob_estimator(arr, min_bin=min_bin)

    n = arr.size
    sorted_arr = np.sort(arr)
    k = min(min_bin, n)
    u = sorted_arr[n - k]
    excess = sorted_arr[n - k :] - u
    m = float(excess.mean())
    v = float(excess.var(ddof=1))
    xi = max(0.5 * (m**2 / v - 1.0), 0.0)  # rr.py clamps shape >= 0, see _fit_gpd_mom's docstring
    sigma = 0.5 * m * (m**2 / v + 1.0)

    query_level = u + 1.0
    assert int(np.sum(arr >= query_level)) < min_bin  # sanity: genuinely sparse there

    expected_p = (k / n) * genpareto.sf(query_level - u, xi, loc=0.0, scale=sigma)
    got_p = est(np.array([query_level]))[0]
    assert got_p == pytest.approx(expected_p, rel=1e-6)


def test_tail_and_body_agree_exactly_at_the_crossover_threshold():
    rng = np.random.default_rng(5)
    arr = rng.normal(0.0, 1.0, 300)
    min_bin = 50
    est = reach_prob_estimator(arr, min_bin=min_bin)

    n = arr.size
    sorted_arr = np.sort(arr)
    u = sorted_arr[n - min_bin]

    got = est(np.array([u]))[0]
    assert got == pytest.approx(min_bin / n, rel=1e-9)


def test_degenerate_sparse_tail_fit_raises_no_warning_under_dash_w_error():
    # Tiny array, ties at the minimum, n < min_bin (every point is
    # nominally "tail") -- task-6-brief.md's "handle empty/degenerate tail
    # fits without RuntimeWarning" guard.
    arr = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = reach_prob_estimator(arr, min_bin=50)
        p = est(np.array([0.0, 1.0, 2.0, 10.0]))

    assert np.isfinite(p).all()
    assert (p > 0.0).all()


def test_drops_nan_from_train_array_without_warning():
    arr = np.array([np.nan, 1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est = reach_prob_estimator(arr, min_bin=3)
        p = est(np.array([1.0, 6.0]))
    assert np.isfinite(p).all()


def test_reach_prob_estimator_rejects_too_few_points():
    with pytest.raises(ValueError):
        reach_prob_estimator(np.array([1.0]))


# --- rr_grid -------------------------------------------------------------


def _linear_reach(intercept: float, slope: float):
    """Deterministic stub reach-prob callable: clip(intercept - slope*level, eps, 1)."""

    def f(levels):
        levels = np.asarray(levels, dtype=float)
        return np.clip(intercept - slope * levels, 1e-6, 1.0)

    return f


def test_rr_grid_returns_required_columns_and_full_cross_product():
    const = _linear_reach(0.5, 0.0)
    grid = rr_grid(const, const, np.array([0.0, 1.0]), fee=0.001, candle_size=1.0, direction="long")

    assert {"tgt_x", "sl_x", "p_target", "p_stop", "rr", "expected_return_after_fees"} <= set(grid.columns)
    assert len(grid) == 4  # full 2x2 cross product of the x_grid


def test_rr_grid_rr_is_p_target_over_p_stop_and_direction_routes_correctly():
    reach_up = _linear_reach(1.0, 0.3)
    reach_down = _linear_reach(1.0, 0.1)
    x_grid = np.array([0.0, 1.0, 2.0])

    grid_long = rr_grid(reach_up, reach_down, x_grid, fee=0.0, candle_size=2.0, direction="long")
    row = grid_long[(grid_long["tgt_x"] == 1.0) & (grid_long["sl_x"] == 2.0)].iloc[0]

    expected_p_target = 1.0 - 0.3 * 1.0  # long target -> reach_up
    expected_p_stop = 1.0 - 0.1 * 2.0  # long stop -> reach_down
    assert row["p_target"] == pytest.approx(expected_p_target)
    assert row["p_stop"] == pytest.approx(expected_p_stop)
    assert row["rr"] == pytest.approx(expected_p_target / expected_p_stop)

    grid_short = rr_grid(reach_up, reach_down, x_grid, fee=0.0, candle_size=2.0, direction="short")
    row_short = grid_short[(grid_short["tgt_x"] == 1.0) & (grid_short["sl_x"] == 2.0)].iloc[0]

    expected_p_target_short = 1.0 - 0.1 * 1.0  # short target -> reach_down
    expected_p_stop_short = 1.0 - 0.3 * 2.0  # short stop -> reach_up
    assert row_short["p_target"] == pytest.approx(expected_p_target_short)
    assert row_short["p_stop"] == pytest.approx(expected_p_stop_short)


def test_rr_grid_expected_return_subtracts_2fee_and_scales_reward_risk_by_candle_size():
    reach_up = _linear_reach(1.0, 0.3)
    reach_down = _linear_reach(1.0, 0.1)

    # tgt_x = sl_x = 0 -> reward = risk = 0, so exp_ret is exactly -2*fee*candle_size
    # (== -2*fee when candle_size=1.0, matching the brief's literal wording).
    grid_zero = rr_grid(reach_up, reach_down, np.array([0.0]), fee=0.001, candle_size=1.0, direction="long")
    row_zero = grid_zero.iloc[0]
    assert row_zero["expected_return_after_fees"] == pytest.approx(-2 * 0.001)

    x_grid = np.array([0.0, 1.0, 2.0])
    candle_size = 2.0
    fee = 0.001
    grid = rr_grid(reach_up, reach_down, x_grid, fee=fee, candle_size=candle_size, direction="long")
    row = grid[(grid["tgt_x"] == 1.0) & (grid["sl_x"] == 2.0)].iloc[0]

    reward = 1.0 * candle_size
    risk = 2.0 * candle_size
    expected_exp_ret = reward * row["p_target"] - risk * row["p_stop"] - 2 * fee * candle_size
    assert row["expected_return_after_fees"] == pytest.approx(expected_exp_ret)


def test_rr_grid_larger_tgt_x_gives_lower_p_target():
    rng = np.random.default_rng(6)
    arr = rng.normal(0.0, 1.0, 300)
    est = reach_prob_estimator(arr, min_bin=50)

    x_grid = np.array([0.0, 0.5, 1.0, 1.5])
    grid = rr_grid(est, est, x_grid, fee=0.0, candle_size=1.0, direction="long")

    fixed_sl = grid[grid["sl_x"] == 0.5].sort_values("tgt_x")
    p_target = fixed_sl["p_target"].to_numpy()
    assert np.all(np.diff(p_target) <= 1e-9)
    assert p_target[0] > p_target[-1]


def test_rr_grid_rejects_bad_direction():
    const = _linear_reach(0.5, 0.0)
    with pytest.raises(ValueError):
        rr_grid(const, const, np.array([0.0]), fee=0.0, candle_size=1.0, direction="sideways")


# --- select_levels ---------------------------------------------------------


def test_select_levels_picks_max_exp_ret_excluding_nonpositive():
    grid = pd.DataFrame(
        {
            "tgt_x": [0.5, 1.0, 1.5, 2.0],
            "sl_x": [0.5, 0.5, 0.5, 0.5],
            "p_target": [0.6, 0.4, 0.2, 0.1],
            "p_stop": [0.5, 0.5, 0.5, 0.5],
            "rr": [1.2, 0.8, 0.4, 0.2],
            "expected_return_after_fees": [-0.01, 0.05, 0.09, -0.02],
        }
    )
    lv = select_levels(grid)
    assert set(lv.keys()) == {"tgt_x", "sl_x", "rr", "exp_ret"}
    assert lv["tgt_x"] == pytest.approx(1.5)
    assert lv["sl_x"] == pytest.approx(0.5)
    assert lv["rr"] == pytest.approx(0.4)
    assert lv["exp_ret"] == pytest.approx(0.09)


def test_select_levels_raises_when_none_profitable():
    grid = pd.DataFrame(
        {
            "tgt_x": [0.5, 1.0],
            "sl_x": [0.5, 0.5],
            "p_target": [0.6, 0.4],
            "p_stop": [0.5, 0.5],
            "rr": [1.2, 0.8],
            "expected_return_after_fees": [0.0, -0.02],  # exactly-0 excluded too
        }
    )
    with pytest.raises(ValueError):
        select_levels(grid)


# --- reach_freq_drift --------------------------------------------------------


def test_reach_freq_drift_identical_train_oos_is_near_zero():
    rng = np.random.default_rng(7)
    arr = rng.normal(0.0, 1.0, 500)
    est = reach_prob_estimator(arr, min_bin=50)

    x_grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # solidly in the body
    drift_df = reach_freq_drift(est, arr, x_grid)

    assert {"level", "train_p", "oos_freq", "drift"} <= set(drift_df.columns)
    assert len(drift_df) == len(x_grid)
    assert drift_df["drift"].abs().max() < 0.05


def test_reach_freq_drift_detects_a_shifted_oos_distribution():
    rng = np.random.default_rng(8)
    train = rng.normal(0.0, 1.0, 500)
    oos = rng.normal(2.0, 1.0, 500)  # clearly shifted -- reaches higher levels more often
    est = reach_prob_estimator(train, min_bin=50)

    x_grid = np.array([0.5, 1.0, 1.5])
    drift_df = reach_freq_drift(est, oos, x_grid)

    # oos (shifted up) touches these levels MORE often than train predicted.
    assert (drift_df["drift"] > 0.0).all()


# --- integration -> Layer 7 (task-6-brief.md's own required test) ----------


def test_rr_levels_feed_zone_calc(synthetic_wide_df):
    hi = synthetic_wide_df["15_high_diff_prc"].dropna().to_numpy()
    est = reach_prob_estimator(hi)
    grid = rr_grid(
        est, est, np.round(np.arange(0, 2.01, 0.25), 2), fee=0.001, candle_size=1.0, direction="long"
    )
    lv = select_levels(grid)
    assert 0 <= lv["tgt_x"] <= 2.0 and lv["rr"] > 0
