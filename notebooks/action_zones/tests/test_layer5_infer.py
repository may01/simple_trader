"""Layer 5 tests — azlib.infer (inverse-variance fusion + Y-sweep).

Covers: fuse_inverse_variance, inferred_coeff, sweep_y, select_y
(azlib/infer.py). See .superpowers/sdd/task-5-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §11.

Filename carries "layer5" so `pytest -k layer5` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.infer existing, so — same
convention as test_layer2_space.py/test_layer3_indicators.py — the whole
module imports it at top level (Task 4's own forward-dep test, in
test_layer4_models.py, is the one that needed an importorskip guard; this
module IS the Task 5 implementation, so no guard is needed here).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from azlib.infer import fuse_inverse_variance, inferred_coeff, select_y, sweep_y


# --- fuse_inverse_variance ----------------------------------------------------


def test_fuse_two_equal_groups_fused_std_is_std_over_sqrt2():
    means = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    stds = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])

    fused_mean, fused_std = fuse_inverse_variance(means, stds)

    np.testing.assert_allclose(fused_mean, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(fused_std, 0.2 / np.sqrt(2))


def test_fuse_downweights_huge_std_group_fused_mean_tracks_confident_group():
    # group 0 is confident (small std) around mean=0.2; group 1 is very
    # unsure (huge std) around a very different mean=0.9 -- fused_mean
    # should land close to the confident group's own mean, not a plain
    # (unweighted) average of the two (which would be 0.55).
    means = np.array([[0.2], [0.9]])
    stds = np.array([[0.01], [50.0]])

    fused_mean, fused_std = fuse_inverse_variance(means, stds)

    assert fused_mean[0] == pytest.approx(0.2, abs=1e-3)
    assert fused_std[0] < 0.01 + 1e-6  # fusing in a 2nd (near-useless) group can only help/be neutral


def test_fuse_std_floor_guards_div_by_zero_and_bounds_weight():
    # a single group reporting std == 0 at every point: without the floor,
    # 1 / std**2 is a literal 1/0 -> RuntimeWarning under -W error, and the
    # fused result would be exactly 0 (nonsensical -- zero uncertainty).
    means = np.array([[0.7, 0.7]])
    stds = np.array([[0.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fused_mean, fused_std = fuse_inverse_variance(means, stds)

    assert np.isfinite(fused_mean).all()
    assert np.isfinite(fused_std).all()
    assert (fused_std > 0.0).all()
    # single group -> weight = 1/floor**2 -> fused_std == floor exactly.
    np.testing.assert_allclose(fused_std, 1e-3)
    np.testing.assert_allclose(fused_mean, [0.7, 0.7])


def test_fuse_nan_group_excluded_per_point_result_matches_remaining_group():
    # point 0: group 0 is NaN, group 1 is valid -> fused == group 1's own
    # values exactly (group 0 contributes nothing at that point).
    means = np.array([[np.nan, 0.3], [0.6, 0.3]])
    stds = np.array([[np.nan, 0.1], [0.05, 0.1]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fused_mean, fused_std = fuse_inverse_variance(means, stds)

    assert fused_mean[0] == pytest.approx(0.6)
    assert fused_std[0] == pytest.approx(0.05)
    # point 1: both groups agree exactly -> fused mean == that shared value,
    # fused std == the two-equal-groups case (std / sqrt(2)).
    assert fused_mean[1] == pytest.approx(0.3)
    assert fused_std[1] == pytest.approx(0.1 / np.sqrt(2))


def test_fuse_all_nan_point_returns_nan():
    means = np.array([[np.nan, 0.4], [np.nan, 0.5]])
    stds = np.array([[np.nan, 0.1], [np.nan, 0.1]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fused_mean, fused_std = fuse_inverse_variance(means, stds)

    assert np.isnan(fused_mean[0])
    assert np.isnan(fused_std[0])
    assert not np.isnan(fused_mean[1])
    assert not np.isnan(fused_std[1])


def test_fuse_rejects_shape_mismatch():
    means = np.zeros((2, 3))
    stds = np.zeros((2, 4))
    with pytest.raises(ValueError):
        fuse_inverse_variance(means, stds)


# --- inferred_coeff -------------------------------------------------------


def test_inferred_coeff_y0_returns_mean():
    mean = np.array([0.1, 0.5, 0.9])
    std = np.array([0.05, 0.1, 0.2])
    out = inferred_coeff(mean, std, 0.0)
    np.testing.assert_allclose(out, mean)


def test_inferred_coeff_y1_returns_mean_plus_std():
    mean = np.array([0.1, 0.5, 0.9])
    std = np.array([0.05, 0.1, 0.2])
    out = inferred_coeff(mean, std, 1.0)
    np.testing.assert_allclose(out, mean + std)


def test_inferred_coeff_negative_y():
    mean = np.array([0.5])
    std = np.array([0.2])
    out = inferred_coeff(mean, std, -2.0)
    np.testing.assert_allclose(out, [0.1])


# --- sweep_y ---------------------------------------------------------------


def _bracketing_levels(wide_df: pd.DataFrame, tf: int) -> tuple[pd.Series, pd.Series]:
    """Same idea as conftest.py's `_bracketing_price_levels` (module-local
    copy so this test module doesn't reach into conftest internals): a
    constant (high, low) pair bracketing every row's `1_low`/`1_high`, so
    sweeping `y` sweeps `zone_limit` across the WHOLE observed entry-price
    range with no warm-up NaNs.
    """
    lo = float(min(wide_df["1_low"].min(), wide_df["1_high"].min())) * 0.5
    hi = float(max(wide_df["1_low"].max(), wide_df["1_high"].max())) * 1.5
    idx = wide_df.index
    return pd.Series(hi, index=idx), pd.Series(lo, index=idx)


def _count_rr_fn(zone_marking: pd.Series) -> float:
    """Stub rr_fn recording nothing but a deterministic function of the
    zone -- how many rows are marked. Used to check sweep_y actually hands
    the zone marking through to rr_fn (not some other value).
    """
    return float(zone_marking.sum())


@pytest.fixture
def sweep_inputs(synthetic_wide_df) -> dict:
    wide_df = synthetic_wide_df
    mean = pd.Series(0.5, index=wide_df.index)
    std = pd.Series(0.2, index=wide_df.index)
    strict_label = pd.Series(False, index=wide_df.index)
    strict_label.iloc[::20] = True
    return {
        "fused_mean": mean,
        "fused_std": std,
        "wide_df": wide_df,
        "tf": 15,
        "price_levels_fn": _bracketing_levels,
        "strict_label": strict_label,
    }


def test_sweep_y_returns_required_columns(sweep_inputs):
    y_grid = np.round(np.arange(-2.0, 2.01, 0.5), 1)
    sweep = sweep_y(
        **sweep_inputs, direction="long", y_grid=y_grid, rr_fn=_count_rr_fn
    )

    assert {"y", "strict_coverage", "realized_rr", "n_zoned"} <= set(sweep.columns)
    assert len(sweep) == len(y_grid)
    np.testing.assert_allclose(sorted(sweep["y"]), sorted(y_grid))


def test_sweep_y_strict_coverage_in_unit_interval(sweep_inputs):
    y_grid = np.round(np.arange(-2.0, 2.01, 0.5), 1)
    sweep = sweep_y(
        **sweep_inputs, direction="long", y_grid=y_grid, rr_fn=_count_rr_fn
    )

    coverage = sweep["strict_coverage"].dropna()
    assert not coverage.empty
    assert (coverage >= 0.0).all() and (coverage <= 1.0).all()


def test_sweep_y_long_zone_widens_with_larger_y(sweep_inputs):
    y_grid = np.round(np.arange(-2.0, 2.01, 0.5), 1)
    sweep = sweep_y(
        **sweep_inputs, direction="long", y_grid=y_grid, rr_fn=_count_rr_fn
    ).sort_values("y")

    n_zoned = sweep["n_zoned"].to_numpy()
    assert np.all(np.diff(n_zoned) >= 0)  # non-decreasing as y increases
    assert n_zoned[0] < n_zoned[-1]  # and genuinely different end to end


def test_sweep_y_short_zone_narrows_with_larger_y(sweep_inputs):
    y_grid = np.round(np.arange(-2.0, 2.01, 0.5), 1)
    inputs = dict(sweep_inputs)
    sweep = sweep_y(
        **inputs, direction="short", y_grid=y_grid, rr_fn=_count_rr_fn
    ).sort_values("y")

    n_zoned = sweep["n_zoned"].to_numpy()
    assert np.all(np.diff(n_zoned) <= 0)  # non-increasing as y increases
    assert n_zoned[0] > n_zoned[-1]


def test_sweep_y_passes_zone_marking_through_to_rr_fn(sweep_inputs):
    seen = {}

    def spy_rr_fn(zone_marking: pd.Series) -> float:
        seen["marking"] = zone_marking
        return 1.0

    y_grid = [0.5]
    sweep = sweep_y(**sweep_inputs, direction="long", y_grid=y_grid, rr_fn=spy_rr_fn)

    assert int(seen["marking"].sum()) == int(sweep.loc[0, "n_zoned"])
    assert isinstance(seen["marking"], pd.Series)
    assert seen["marking"].dtype == bool


def test_sweep_y_handles_nan_warmup_rows_without_warning(synthetic_wide_df):
    wide_df = synthetic_wide_df
    n = len(wide_df)
    warmup = 10

    mean = pd.Series(0.5, index=wide_df.index)
    mean.iloc[:warmup] = np.nan
    std = pd.Series(0.2, index=wide_df.index)
    std.iloc[:warmup] = np.nan

    def levels_with_nan_warmup(df: pd.DataFrame, tf: int) -> tuple[pd.Series, pd.Series]:
        hi, lo = _bracketing_levels(df, tf)
        hi = hi.copy()
        lo = lo.copy()
        hi.iloc[:warmup] = np.nan
        lo.iloc[:warmup] = np.nan
        return hi, lo

    strict_label = pd.Series(False, index=wide_df.index)
    strict_label.iloc[::15] = True

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sweep = sweep_y(
            fused_mean=mean,
            fused_std=std,
            wide_df=wide_df,
            tf=15,
            direction="long",
            price_levels_fn=levels_with_nan_warmup,
            strict_label=strict_label,
            y_grid=[0.0, 1.0],
            rr_fn=_count_rr_fn,
        )

    assert (sweep["n_zoned"] <= n - warmup).all()


def test_sweep_y_rejects_bad_direction(sweep_inputs):
    with pytest.raises(ValueError):
        sweep_y(**sweep_inputs, direction="sideways", y_grid=[0.0], rr_fn=_count_rr_fn)


# --- select_y ----------------------------------------------------------------


def test_select_y_picks_profitable_max_coverage_row():
    sweep = pd.DataFrame(
        {
            "y": [-1.0, 0.0, 1.0, 2.0],
            "strict_coverage": [0.9, 0.2, 0.5, 0.8],
            "realized_rr": [0.5, 1.5, 1.2, 1.1],  # y=-1.0 not profitable
            "n_zoned": [10, 20, 30, 40],
        }
    )
    # profitable rows: y in {0.0, 1.0, 2.0} -- max coverage among those is
    # y=2.0 (coverage 0.8), even though y=-1.0 has higher coverage overall
    # (0.9) it is excluded for not being profitable.
    assert select_y(sweep) == pytest.approx(2.0)


def test_select_y_returns_documented_default_when_none_profitable():
    sweep = pd.DataFrame(
        {
            "y": [-1.0, 0.0, 1.0],
            "strict_coverage": [0.9, 0.5, 0.1],
            "realized_rr": [0.5, 0.8, 0.99],  # none > 1.0
            "n_zoned": [5, 10, 15],
        }
    )
    y = select_y(sweep)
    assert y == pytest.approx(0.0)
    assert -2.0 <= y <= 2.0


def test_select_y_empty_sweep_returns_default():
    sweep = pd.DataFrame(columns=["y", "strict_coverage", "realized_rr", "n_zoned"])
    assert select_y(sweep) == pytest.approx(0.0)


def test_select_y_all_profitable_coverage_nan_returns_default():
    sweep = pd.DataFrame(
        {
            "y": [0.0, 1.0],
            "strict_coverage": [np.nan, np.nan],
            "realized_rr": [1.5, 2.0],
            "n_zoned": [10, 20],
        }
    )
    assert select_y(sweep) == pytest.approx(0.0)


# --- integration -> Layers 6/7 (task-5-brief.md's own required test) --------


def test_y_sweep_produces_selectable_y(synthetic_pipeline):
    y_grid = np.round(np.arange(-2, 2.01, 0.1), 1)
    sweep = sweep_y(**synthetic_pipeline, y_grid=y_grid)

    assert {"y", "strict_coverage", "n_zoned"} <= set(sweep.columns)

    y = select_y(sweep)
    assert -2.0 <= y <= 2.0
