"""Layer 7 tests — azlib.zones (zone marking + zoned dataset + results file).

Covers: zone_limit_price, mark_zones, select_levels_safe, build_rr_levels,
build_zoned_dataset, ResultsFile (azlib/zones.py). See
.superpowers/sdd/task-7-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §7.

Also covers task-7-brief.md's two REQUIRED Task 6 carry-forwards:
  1. Down-side reach wiring for a SHORT must negate low_diff_prc (see
     test_build_rr_levels_short_uses_negated_low_diff_prc_wiring).
  2. select_levels's ValueError (no profitable combo) must be caught per
     (tf, direction), not crash the run (see the select_levels_safe and
     build_zoned_dataset no-profitable-levels tests below).

Filename carries "layer7" so `pytest -k layer7` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.zones existing, so — same
convention as test_layer2_space.py/test_layer5_infer.py/test_layer6_rr.py —
the whole module imports it at top level.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from azlib.infer import inferred_coeff, sweep_y
from azlib.rr import reach_prob_estimator, rr_grid, select_levels
from azlib.space import coeff, price_levels
from azlib.zones import (
    ResultsFile,
    build_rr_levels,
    build_zoned_dataset,
    mark_zones,
    select_levels_safe,
    zone_limit_price,
)


# --- zone_limit_price ---------------------------------------------------------


def test_zone_limit_price_is_exact_inverse_of_coeff_round_trip():
    # coeff clamps outside [0, 1] -- pick `frac` strictly inside [0, 1] so the
    # forward map is never clamped and the round trip is exact, not just
    # approximately-recovered-modulo-clamping.
    rng = np.random.default_rng(11)
    low = rng.uniform(10.0, 20.0, 50)
    high = low + rng.uniform(1.0, 10.0, 50)
    frac = rng.uniform(0.0, 1.0, 50)
    price = low + frac * (high - low)

    c = coeff(price, low, high)
    back = zone_limit_price(c, low, high)

    assert back == pytest.approx(price, rel=1e-9)


def test_zone_limit_price_is_not_clamped_extrapolates_beyond_band():
    # Unlike `coeff` (which clamps its OUTPUT to [0, 1]), `zone_limit_price`
    # is a pure linear inverse -- an out-of-[0, 1] inferred_coeff (design
    # spec §11.3's `mean + Y*std`, intentionally unclamped -- see
    # azlib.infer.inferred_coeff's docstring) must extrapolate the price
    # BEYOND [low_level, high_level], not clamp back into it.
    low = np.array([10.0])
    high = np.array([20.0])

    above = zone_limit_price(np.array([1.5]), low, high)
    below = zone_limit_price(np.array([-0.5]), low, high)

    assert above[0] == pytest.approx(25.0)  # 10 + 1.5*10
    assert below[0] == pytest.approx(5.0)  # 10 + (-0.5)*10


def test_zone_limit_price_agrees_with_sweep_y_inline_copy(synthetic_pipeline):
    # DRY note (task-7-brief.md): azlib.infer.sweep_y keeps an INLINE copy of
    # this exact formula (documented there as a deliberate non-import, to
    # avoid an infer<->zones import cycle) -- both implementations must stay
    # in agreement. Cross-checks sweep_y's own `n_zoned` output for one Y
    # against independently recomputing the same zone via
    # zone_limit_price + mark_zones (this module's own functions).
    kwargs = dict(synthetic_pipeline)
    y = 0.3
    sweep = sweep_y(**kwargs, y_grid=[y])
    n_zoned_from_sweep = int(sweep.loc[sweep["y"] == y, "n_zoned"].iloc[0])

    wide_df = kwargs["wide_df"]
    tf = kwargs["tf"]
    direction = kwargs["direction"]
    mean = kwargs["fused_mean"].to_numpy()
    std = kwargs["fused_std"].to_numpy()
    price_high_level, price_low_level = kwargs["price_levels_fn"](wide_df, tf)

    coeff_y = inferred_coeff(mean, std, y)
    zl = zone_limit_price(coeff_y, price_low_level.to_numpy(), price_high_level.to_numpy())
    marked = mark_zones(wide_df, tf, direction, zl)

    assert int(marked.sum()) == n_zoned_from_sweep
    assert n_zoned_from_sweep > 0  # sanity: this Y actually zones SOME rows


# --- mark_zones ----------------------------------------------------------------


def _tiny_entry_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"1_low": [1.0, 2.0, 3.0, 4.0], "1_high": [1.5, 2.5, 3.5, 4.5]},
        index=pd.RangeIndex(4),
    )


def test_mark_zones_long_rule_is_1_low_lt_zone_limit():
    wide_df = _tiny_entry_df()
    zone_limit = np.full(4, 2.5)

    marked = mark_zones(wide_df, 15, "long", zone_limit)

    assert list(marked) == [True, True, False, False]  # 1_low: 1,2 < 2.5; 3,4 not
    assert marked.name == "az_zone_long_15"
    assert marked.index.equals(wide_df.index)


def test_mark_zones_short_rule_is_1_high_gt_zone_limit():
    wide_df = _tiny_entry_df()
    zone_limit = np.full(4, 2.5)

    marked = mark_zones(wide_df, 60, "short", zone_limit)

    assert list(marked) == [False, False, True, True]  # 1_high: 1.5,2.5 not > 2.5; 3.5,4.5 are
    assert marked.name == "az_zone_short_60"


def test_mark_zones_nan_zone_limit_row_is_never_marked_without_warning():
    import warnings

    wide_df = _tiny_entry_df()
    zone_limit = np.array([2.5, np.nan, 2.5, np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        marked = mark_zones(wide_df, 15, "long", zone_limit)

    assert list(marked) == [True, False, False, False]


def test_mark_zones_rejects_bad_direction():
    wide_df = _tiny_entry_df()
    with pytest.raises(ValueError):
        mark_zones(wide_df, 15, "sideways", np.full(4, 2.5))


# --- select_levels_safe (Task 6 carry-forward #2) -----------------------------


def _grid(exp_ret) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tgt_x": [1.0],
            "sl_x": [0.5],
            "p_target": [0.5],
            "p_stop": [0.3],
            "rr": [1.67],
            "expected_return_after_fees": [exp_ret],
        }
    )


def test_select_levels_safe_passes_through_the_winning_row_when_profitable():
    lv = select_levels_safe(_grid(0.05))
    expected = select_levels(_grid(0.05))
    assert lv == expected
    assert not math.isnan(lv["rr"])


def test_select_levels_safe_returns_nan_sentinel_when_none_profitable():
    lv = select_levels_safe(_grid(-0.02))
    assert set(lv.keys()) == {"tgt_x", "sl_x", "rr", "exp_ret"}
    assert all(math.isnan(v) for v in lv.values())


def test_select_levels_safe_never_raises_on_empty_grid():
    empty = pd.DataFrame(
        columns=["tgt_x", "sl_x", "p_target", "p_stop", "rr", "expected_return_after_fees"]
    )
    lv = select_levels_safe(empty)
    assert all(math.isnan(v) for v in lv.values())


# --- build_rr_levels (Task 6 carry-forward #1: down-side reach wiring) -------


def test_build_rr_levels_short_uses_negated_low_diff_prc_wiring(synthetic_wide_df):
    # Coordinator-review-style regression guard (mirrors
    # test_layer6_rr.py's own down-side sign test, one layer up): confirms
    # THIS module's own R/R wiring (build_rr_levels), not just azlib.rr
    # itself, negates {tf}_low_diff_prc before building the down-reach
    # estimator for a SHORT. A caller who forgot to negate would build
    # reach_down straight off the raw (un-negated) low_diff_prc series --
    # provably different levels, checked directly against both the correct
    # and the (deliberately reconstructed) wrong wiring.
    tf = 15
    wide_df = synthetic_wide_df
    x_grid = np.round(np.arange(0.0, 2.01, 0.25), 2)
    fee = 0.0
    candle_size = 1.0

    got = build_rr_levels(wide_df, tf, "short", x_grid, fee, candle_size)

    high_diff = wide_df[f"{tf}_high_diff_prc"].dropna().to_numpy()
    low_diff = wide_df[f"{tf}_low_diff_prc"].dropna().to_numpy()

    reach_up = reach_prob_estimator(high_diff)
    reach_down_correct = reach_prob_estimator(-low_diff)  # negated, per azlib.rr's sign convention
    grid_correct = rr_grid(reach_up, reach_down_correct, x_grid, fee=fee, candle_size=candle_size, direction="short")
    expected = select_levels_safe(grid_correct)

    reach_down_wrong = reach_prob_estimator(low_diff)  # the bug: forgot to negate
    grid_wrong = rr_grid(reach_up, reach_down_wrong, x_grid, fee=fee, candle_size=candle_size, direction="short")
    wrong = select_levels_safe(grid_wrong)

    for key in ("tgt_x", "sl_x", "rr", "exp_ret"):
        assert got[key] == pytest.approx(expected[key])

    # The wrong (un-negated) wiring must produce a genuinely DIFFERENT
    # result -- otherwise this test would not actually be able to catch a
    # forgotten negation.
    assert not math.isclose(got["exp_ret"], wrong["exp_ret"], rel_tol=1e-9) or not math.isclose(
        got["rr"], wrong["rr"], rel_tol=1e-9
    )


def test_build_rr_levels_long_uses_high_diff_prc_directly(synthetic_wide_df):
    tf = 15
    wide_df = synthetic_wide_df
    x_grid = np.round(np.arange(0.0, 2.01, 0.25), 2)

    got = build_rr_levels(wide_df, tf, "long", x_grid, fee=0.0, candle_size=1.0)

    high_diff = wide_df[f"{tf}_high_diff_prc"].dropna().to_numpy()
    low_diff = wide_df[f"{tf}_low_diff_prc"].dropna().to_numpy()
    reach_up = reach_prob_estimator(high_diff)
    reach_down = reach_prob_estimator(-low_diff)
    grid = rr_grid(reach_up, reach_down, x_grid, fee=0.0, candle_size=1.0, direction="long")
    expected = select_levels_safe(grid)

    for key in ("tgt_x", "sl_x", "rr", "exp_ret"):
        assert got[key] == pytest.approx(expected[key])
    assert not math.isnan(got["rr"])  # sanity: a real, profitable combo was found


def test_build_rr_levels_returns_nan_sentinel_when_none_profitable_no_crash(synthetic_wide_df):
    # Task 6 carry-forward #2: an outrageous fee guarantees no (tgt_x, sl_x)
    # combo is profitable -- build_rr_levels must NOT raise, it must return
    # the documented NaN sentinel.
    got = build_rr_levels(synthetic_wide_df, 15, "long", np.array([0.1, 0.2]), fee=1000.0, candle_size=1.0)
    assert all(math.isnan(v) for v in got.values())


# --- build_zoned_dataset ------------------------------------------------------


def test_build_zoned_dataset_has_exactly_the_required_columns(synthetic_wide_df):
    wide_df = synthetic_wide_df
    zone_limit = np.full(len(wide_df), float(wide_df["1_low"].median()))
    levels = {"tgt_x": 1.0, "sl_x": 0.5, "rr": 1.8, "exp_ret": 0.02}

    zdf = build_zoned_dataset(wide_df, 15, "long", zone_limit, levels)

    assert set(zdf.columns) == {"az_zone_long_15", "az_entry", "az_tgt", "az_sl", "az_rr"}
    assert len(zdf) == len(wide_df)
    assert zdf.index.equals(wide_df.index)


def test_build_zoned_dataset_long_tgt_sl_price_mapping_matches_price_levels(synthetic_wide_df):
    wide_df = synthetic_wide_df
    tf = 15
    zone_limit = np.full(len(wide_df), float(wide_df["1_low"].median()))
    levels = {"tgt_x": 1.2, "sl_x": 0.7, "rr": 1.5, "exp_ret": 0.03}

    zdf = build_zoned_dataset(wide_df, tf, "long", zone_limit, levels)

    expected_tgt = price_levels(wide_df, tf, x=1.2)[0]  # up/high side
    expected_sl = price_levels(wide_df, tf, x=0.7)[1]  # low side

    pd.testing.assert_series_equal(zdf["az_tgt"], expected_tgt, check_names=False)
    pd.testing.assert_series_equal(zdf["az_sl"], expected_sl, check_names=False)
    assert (zdf["az_rr"] == 1.5).all()
    assert (zdf["az_entry"].to_numpy() == zone_limit).all()


def test_build_zoned_dataset_short_tgt_sl_price_mapping_matches_price_levels(synthetic_wide_df):
    wide_df = synthetic_wide_df
    tf = 60
    zone_limit = np.full(len(wide_df), float(wide_df["1_high"].median()))
    levels = {"tgt_x": 0.9, "sl_x": 1.1, "rr": 1.1, "exp_ret": 0.01}

    zdf = build_zoned_dataset(wide_df, tf, "short", zone_limit, levels)

    expected_tgt = price_levels(wide_df, tf, x=0.9)[1]  # down/low side
    expected_sl = price_levels(wide_df, tf, x=1.1)[0]  # high side

    pd.testing.assert_series_equal(zdf["az_tgt"], expected_tgt, check_names=False)
    pd.testing.assert_series_equal(zdf["az_sl"], expected_sl, check_names=False)
    assert (zdf["az_rr"] == 1.1).all()


def test_build_zoned_dataset_zone_column_matches_mark_zones(synthetic_wide_df):
    wide_df = synthetic_wide_df
    tf = 15
    zone_limit = np.full(len(wide_df), float(wide_df["1_low"].median()))
    levels = {"tgt_x": 1.0, "sl_x": 1.0, "rr": 1.2, "exp_ret": 0.01}

    zdf = build_zoned_dataset(wide_df, tf, "long", zone_limit, levels)
    expected = mark_zones(wide_df, tf, "long", zone_limit)

    assert (zdf["az_zone_long_15"].to_numpy() == expected.to_numpy()).all()
    assert zdf["az_zone_long_15"].to_numpy().any()  # genuinely marks some rows
    assert not zdf["az_zone_long_15"].to_numpy().all()  # not literally every row either


def test_build_zoned_dataset_rejects_bad_direction(synthetic_wide_df):
    wide_df = synthetic_wide_df
    levels = {"tgt_x": 1.0, "sl_x": 1.0, "rr": 1.0, "exp_ret": 0.1}
    with pytest.raises(ValueError):
        build_zoned_dataset(wide_df, 15, "sideways", np.zeros(len(wide_df)), levels)


def test_build_zoned_dataset_no_profitable_levels_gives_allfalse_zone_and_nan_tgt_sl_rr(synthetic_wide_df):
    # Task 6 carry-forward #2: build_zoned_dataset must recognize
    # select_levels_safe's NaN sentinel and produce an all-False zone with
    # NaN tgt/sl/rr, rather than (incorrectly) still marking rows against a
    # zone_limit whose exit levels were never actually profitable.
    wide_df = synthetic_wide_df
    tf = 15
    # A zone_limit that WOULD mark real rows if build_zoned_dataset ignored
    # the sentinel and called mark_zones anyway -- proves the all-False
    # result is the sentinel-handling branch, not a coincidentally-empty
    # zone_limit.
    zone_limit = np.full(len(wide_df), float(wide_df["1_low"].median()))
    levels = select_levels_safe(_grid(-0.05))  # deliberately non-profitable

    zdf = build_zoned_dataset(wide_df, tf, "long", zone_limit, levels)

    assert not zdf["az_zone_long_15"].any()
    assert zdf["az_tgt"].isna().all()
    assert zdf["az_sl"].isna().all()
    assert zdf["az_rr"].isna().all()

    # Sanity: the SAME zone_limit, with genuinely profitable levels, marks
    # real rows -- confirms the all-False result above is due to the
    # sentinel, not this zone_limit being degenerate.
    profitable_levels = {"tgt_x": 1.0, "sl_x": 1.0, "rr": 1.2, "exp_ret": 0.01}
    zdf_profitable = build_zoned_dataset(wide_df, tf, "long", zone_limit, profitable_levels)
    assert zdf_profitable["az_zone_long_15"].any()


# --- ResultsFile ---------------------------------------------------------------


def test_results_file_round_trips_all_fields(tmp_path):
    rf = ResultsFile(
        tf=60,
        direction="short",
        label_params={"n": 1, "m": 2.0, "x": 2.0, "l": 60, "y": 1.0},
        freeze_stats_path="/vol/action_zones/short/60/results/stats.json",
        reg_models=["rsi:linear", "macd:poly2"],
        y=-0.4,
        tgt_x=1.3,
        sl_x=0.9,
        fee=0.0015,
    )
    path = str(tmp_path / "rf.json")

    rf.save(path)
    loaded = ResultsFile.load(path)

    assert loaded == rf


def test_results_file_round_trips_nan_tgt_sl_sentinel(tmp_path):
    # The no-profitable-levels sentinel (Task 6 carry-forward #2) must also
    # round-trip cleanly through save/load -- a results file for a
    # (tf, direction) with no profitable exit still needs to be persisted
    # (and later recognized) rather than crashing the save step.
    rf = ResultsFile(
        tf=15,
        direction="long",
        label_params={},
        freeze_stats_path="s.json",
        reg_models=[],
        y=0.0,
        tgt_x=float("nan"),
        sl_x=float("nan"),
        fee=0.001,
    )
    path = str(tmp_path / "rf_nan.json")

    rf.save(path)
    loaded = ResultsFile.load(path)

    assert math.isnan(loaded.tgt_x)
    assert math.isnan(loaded.sl_x)
    assert loaded.y == pytest.approx(0.0)


def test_results_file_save_writes_only_to_caller_supplied_path(tmp_path):
    # Global constraint: artifacts write to a caller-supplied path, never
    # into the worktree -- confirmed here by using tmp_path exclusively and
    # checking no stray file lands next to it (e.g. save() does not also
    # write a sibling model file the way azlib.models.save_result does).
    rf = ResultsFile(
        tf=15, direction="long", label_params={}, freeze_stats_path="s.json",
        reg_models=["rsi:linear"], y=0.1, tgt_x=1.0, sl_x=1.0, fee=0.001,
    )
    path = tmp_path / "rf.json"
    rf.save(str(path))

    assert path.exists()
    assert list(tmp_path.iterdir()) == [path]


# --- integration -> Layer 8 (task-7-brief.md's own required test) ------------


def test_zoned_dataset_and_results_roundtrip(synthetic_pipeline_full, tmp_path):
    zdf = build_zoned_dataset(**synthetic_pipeline_full)
    assert {"az_zone_long_15", "az_entry", "az_tgt", "az_sl", "az_rr"} <= set(zdf.columns)

    rf = ResultsFile(
        tf=15, direction="long", label_params={}, freeze_stats_path="s.json",
        reg_models=["rsi:linear"], y=0.3, tgt_x=1.5, sl_x=1.0, fee=0.001,
    )
    rf.save(str(tmp_path / "rf.json"))
    assert ResultsFile.load(str(tmp_path / "rf.json")).y == 0.3
