"""Layer 8 tests — azlib.validate (train/OOS validation harness + metrics).

Covers: run_train, run_oos, metrics (azlib/validate.py). See
.superpowers/sdd/task-8-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §8.

Filename carries "layer8" so `pytest -k layer8` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.validate existing, so — same
convention as every earlier layer's own test module — the whole module
imports it at top level.
"""
from __future__ import annotations

import math
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from azlib.validate import first_touch_rr_grid, metrics, run_oos, run_train
from azlib.zones import ResultsFile


def _raise(*_args, **_kwargs):
    raise AssertionError("must not be called — see docstring of the test that patches this in")


# --- _apply_env_file ---------------------------------------------------------


def test_apply_env_file_sets_os_environ(tmp_path, monkeypatch):
    from azlib.validate import _apply_env_file

    path = tmp_path / "some.env"
    path.write_text(
        "# a comment, ignored\n"
        "\n"
        "ROOT_FOLDER=short\n"
        "DATA_SET_NAME=foo\n"
        "EXCHANGE_FEE=0.0025\n"
    )
    monkeypatch.delenv("ROOT_FOLDER", raising=False)
    monkeypatch.delenv("DATA_SET_NAME", raising=False)
    monkeypatch.delenv("EXCHANGE_FEE", raising=False)

    _apply_env_file(str(path))

    assert os.environ["ROOT_FOLDER"] == "short"
    assert os.environ["DATA_SET_NAME"] == "foo"
    assert os.environ["EXCHANGE_FEE"] == "0.0025"


# --- run_train -> artifacts ---------------------------------------------------


def test_run_train_produces_results_path_and_artifacts(two_synthetic_datasets):
    ds = two_synthetic_datasets
    rp = run_train(ds["train_env"], tf=15, direction="long")

    assert os.path.isfile(rp)
    rf = ResultsFile.load(rp)
    assert rf.tf == 15
    assert rf.direction == "long"
    assert os.path.isfile(rf.freeze_stats_path)
    assert len(rf.reg_models) > 0
    for path in rf.reg_models:
        assert os.path.isfile(path)
        assert os.path.isfile(path[: -len(".json")] + ".joblib")

    results_dir = os.path.dirname(rf.freeze_stats_path)
    assert os.path.isfile(os.path.join(results_dir, "levels.json"))
    assert os.path.isfile(os.path.join(results_dir, "reach_train.json"))

    # Global constraint: artifacts land under tmp_path (this fixture's
    # monkeypatched dataset_folder), never the worktree.
    assert "zone-selection-experiment/notebooks" not in rp


# --- run_train: selected_models (Task §10 human-selected-model wiring) -----
#
# run_train(..., selected_models=[...]) fits EXACTLY the caller-selected
# (indicator, attrs, kind) combos instead of the legacy "every indicator x
# 1D linear" fan-out -- so the buy/sell zones can reflect a human-picked
# regression model (e.g. RSI 3D position/slope/distance gbr) instead of the
# hardcoded 1D-linear-per-indicator default. Backward compatible: omitting
# selected_models (or passing None/[]) is byte-for-byte the existing legacy
# behavior -- see test_run_train_produces_results_path_and_artifacts above
# and test_run_train_legacy_path_selected_models_empty_and_many_1d_models
# below, both still green.


def test_reg_model_filename_multi_attr_join():
    from azlib.validate import _reg_model_filename, _reg_model_key

    fn = _reg_model_filename("rsi", ["position", "slope", "distance"], "gbr")
    assert fn == "rsi_position-slope-distance_gbr.json"
    assert _reg_model_key("/some/dir/" + fn) == ("rsi", "position-slope-distance", "gbr")

    # single attr (legacy call shape) -- filename/key round-trip unchanged.
    fn1 = _reg_model_filename("rsi", "position", "linear")
    assert fn1 == "rsi_position_linear.json"
    assert _reg_model_key("/some/dir/" + fn1) == ("rsi", "position", "linear")


def test_run_train_selected_models_fits_exactly_one_multi_attr_model(two_synthetic_datasets):
    ds = two_synthetic_datasets
    selected = [{"indicator": "rsi", "attrs": ["position", "slope", "distance"], "kind": "gbr"}]
    rp = run_train(ds["train_env"], tf=15, direction="long", selected_models=selected)

    rf = ResultsFile.load(rp)
    assert rf.selected_models == selected

    results_dir = os.path.dirname(rf.freeze_stats_path)
    expected_path = os.path.join(results_dir, "rsi_position-slope-distance_gbr.json")
    assert rf.reg_models == [expected_path]  # exactly ONE model -- the selected combo, nothing else
    assert os.path.isfile(expected_path)
    assert os.path.isfile(expected_path[: -len(".json")] + ".joblib")

    # No stray legacy per-attribute 1D models were also fit alongside it --
    # exclude the non-reg-model sidecar/results JSON files this same run
    # also writes (stats.json/levels.json/reach_train.json/results.json).
    _non_reg_model_files = {"stats.json", "levels.json", "reach_train.json", "results.json"}
    reg_json_files = [
        f for f in os.listdir(results_dir) if f.endswith(".json") and f not in _non_reg_model_files
    ]
    assert reg_json_files == ["rsi_position-slope-distance_gbr.json"]


def test_run_oos_selected_models_multi_attr_no_refit(two_synthetic_datasets, monkeypatch):
    # Same no-leakage proof as test_run_oos_never_refits_frozen_artifacts,
    # for the selected_models path: run_train (unpatched) first, THEN patch
    # fit_stats/fit_regression to raise, THEN run_oos must still succeed --
    # proving it reloads the frozen multi-attribute gbr model rather than
    # refitting it.
    ds = two_synthetic_datasets
    selected = [{"indicator": "rsi", "attrs": ["position", "slope", "distance"], "kind": "gbr"}]
    rp = run_train(ds["train_env"], tf=15, direction="long", selected_models=selected)

    monkeypatch.setattr("azlib.validate.fit_stats", _raise)
    monkeypatch.setattr("azlib.validate.fit_regression", _raise)

    zoos = run_oos(rp, ds["oos_env"])

    assert isinstance(zoos, pd.DataFrame)
    assert "az_zone_long_15" in zoos.columns
    assert len(zoos) == len(ds["oos_df"])


def test_run_train_legacy_path_selected_models_empty_and_many_1d_models(two_synthetic_datasets):
    # No selected_models argument at all -- the EXISTING "every indicator x
    # 1D linear" fan-out, byte-for-byte unchanged (this is the backward-
    # compat guarantee: selected_models=None must not alter legacy output).
    ds = two_synthetic_datasets
    rp = run_train(ds["train_env"], tf=15, direction="long")

    rf = ResultsFile.load(rp)
    assert rf.selected_models == []
    assert len(rf.reg_models) > 1  # multiple single-attribute models, legacy fan-out
    for path in rf.reg_models:
        assert path.endswith("_linear.json")  # legacy _REG_KIND, unchanged


# --- run_train: Fix 1 -- rr_fn is Y-DEPENDENT, not degenerate-widest -------
#
# .superpowers/sdd/task-fix1-report.md: before Fix 1, run_train's rr_fn
# closure returned the ONE levels["exp_ret"] constant for every Y (Y-
# invariant) and select_y maximized strict_coverage among profitable rows
# -- together these always walked the sweep out to the grid boundary (the
# widest zone was always "free"). Fix 1 makes rr_fn genuinely Y-dependent
# (the REAL forward-touch mean R of entries inside each Y's own zone) and
# select_y maximize TOTAL profit (n_zoned * realized_rr) instead -- this
# test proves the degeneracy is actually gone, at the run_train level (not
# just in azlib.infer's own unit tests).


def test_run_train_rr_fn_is_y_dependent_not_constant(two_synthetic_datasets, monkeypatch):
    # Spy on azlib.validate's own module-level sweep_y reference (same
    # monkeypatch-on-the-imported-name convention this module's no-refit
    # tests already use) to capture the internal sweep DataFrame run_train
    # builds, without changing run_train's own public return value (still
    # just a results path).
    from azlib import validate as validate_mod

    real_sweep_y = validate_mod.sweep_y
    captured = {}

    def spy_sweep_y(*args, **kwargs):
        result = real_sweep_y(*args, **kwargs)
        captured["sweep"] = result
        return result

    monkeypatch.setattr(validate_mod, "sweep_y", spy_sweep_y)

    ds = two_synthetic_datasets
    run_train(ds["train_env"], tf=15, direction="long")

    sweep = captured["sweep"]
    finite = sweep["realized_rr"].dropna()
    # Under the OLD Y-invariant rr_fn, every finite realized_rr in the
    # sweep was the exact same float (levels["exp_ret"]) -- nunique() == 1
    # (or 0, if every Y happened to resolve to NaN). Fix 1's rr_fn recomputes
    # a genuine forward-touch mean R per Y's own zone marking, so at least
    # two DIFFERENT values must appear across a 9-point Y grid on real
    # (synthetic but non-trivial) data.
    assert len(finite) >= 2
    assert finite.nunique() > 1


# --- run_oos: no refit on OOS (the critical no-leakage proof) ---------------


def test_run_oos_never_refits_frozen_artifacts(two_synthetic_datasets, monkeypatch):
    # 1. Build a REAL results path first, with fit_stats/fit_regression
    #    UNPATCHED — run_train genuinely needs them.
    ds = two_synthetic_datasets
    rp = run_train(ds["train_env"], tf=15, direction="long")

    # 2. NOW monkeypatch both "fit" functions (validate.py's own imported
    #    references — the names run_train actually calls) to raise.
    monkeypatch.setattr("azlib.validate.fit_stats", _raise)
    monkeypatch.setattr("azlib.validate.fit_regression", _raise)

    # 3. run_oos must still succeed — it only ever loads frozen artifacts
    #    (FreezeStats.from_json, azlib.models.load_result), never calls
    #    either patched-to-raise function.
    zoos = run_oos(rp, ds["oos_env"])

    assert isinstance(zoos, pd.DataFrame)
    assert "az_zone_long_15" in zoos.columns
    assert len(zoos) == len(ds["oos_df"])


# --- run_train: no-profitable-R/R -> valid no-zone ResultsFile, no crash ---


def test_run_train_no_profitable_rr_records_nan_zone_results_file(tmp_path, two_synthetic_datasets):
    ds = two_synthetic_datasets
    # An outrageous EXCHANGE_FEE guarantees no (tgt_x, sl_x) combo is
    # profitable (mirrors test_layer7_zones.py's own
    # test_build_rr_levels_returns_nan_sentinel_when_none_profitable_no_crash) —
    # append it to the existing train.env (later lines win, _apply_env_file
    # applies them in file order).
    train_env = tmp_path / "train_outrageous_fee.env"
    original = pathlib.Path(ds["train_env"]).read_text()
    train_env.write_text(original + "\nEXCHANGE_FEE=1000\n")

    rp = run_train(str(train_env), tf=15, direction="long")  # must not raise

    rf = ResultsFile.load(rp)
    assert math.isnan(rf.tgt_x)
    assert math.isnan(rf.sl_x)

    zoos = run_oos(rp, ds["oos_env"])
    assert not zoos["az_zone_long_15"].any()  # sentinel -> all-False zone (Task 7's own handling)
    assert zoos["az_tgt"].isna().all()
    assert zoos["az_sl"].isna().all()


# --- metrics: strict_coverage / non_strict_coverage (hand-checkable) -------


def _tiny_zoned_df(zoned, strict, non_strict) -> pd.DataFrame:
    n = len(zoned)
    return pd.DataFrame(
        {
            "az_zone_long_15": np.array(zoned, dtype=bool),
            "az_label_strict": np.array(strict, dtype=bool),
            "az_label_nonstrict": np.array(non_strict, dtype=bool),
            # metrics() requires az_reach_drift_max to be PRESENT (raises if
            # absent — see _reach_drift_max_from_column's docstring); these
            # coverage-only hand-checks don't care about its value, so an
            # explicit NaN column (present, just not a real number) is the
            # correct way to opt out of that one metric.
            "az_reach_drift_max": np.full(n, np.nan),
        },
        index=pd.RangeIndex(n),
    )


def test_metrics_strict_coverage_hand_check():
    # 4 rows: strict-labeled at 0,1,2 (3 rows); zoned at 0,1,3 (3 rows).
    # Strict rows INSIDE the zone: 0,1 -> 2/3.
    zdf = _tiny_zoned_df(
        zoned=[True, True, False, True],
        strict=[True, True, True, False],
        non_strict=[True, True, True, True],
    )
    m = metrics(zdf, 15, "long", label_params={})

    assert m["strict_coverage"] == pytest.approx(2 / 3)
    assert 0 <= m["strict_coverage"] <= 1


def test_metrics_non_strict_coverage_hand_check():
    # non-strict-labeled at all 4 rows; zoned at 0,1,3 -> 3/4.
    zdf = _tiny_zoned_df(
        zoned=[True, True, False, True],
        strict=[False, False, False, False],
        non_strict=[True, True, True, True],
    )
    m = metrics(zdf, 15, "long", label_params={})

    assert m["non_strict_coverage"] == pytest.approx(3 / 4)
    assert math.isnan(m["strict_coverage"])  # zero strict-labeled rows -> NaN, documented default


# --- metrics: realized_rr (actual forward tgt/sl touches) ------------------


def test_metrics_realized_rr_hand_check_target_and_stop_outcomes():
    # See azlib.validate._realized_rr's own docstring for the exact
    # definition this hand-computes against.
    n = 40
    one_high = np.full(n, 100.0)
    one_low = np.full(n, 100.0)
    one_high[6] = 111.0  # row 0's forward window (rows 1..15): touches target (110) at offset 5
    one_low[26] = 94.0  # row 20's forward window (rows 21..35): touches stop (95) at offset 5

    zone = np.zeros(n, dtype=bool)
    zone[[0, 20]] = True
    tgt = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    tgt[[0, 20]] = 110.0
    sl[[0, 20]] = 95.0

    zdf = pd.DataFrame(
        {
            "az_zone_long_15": zone,
            "az_tgt": tgt,
            "az_sl": sl,
            "1_high": one_high,
            "1_low": one_low,
            "az_reach_drift_max": np.full(n, np.nan),  # present but unused by this test -- see _tiny_zoned_df
        },
        index=pd.RangeIndex(n),
    )

    m = metrics(zdf, 15, "long", label_params={"n": 1})

    # Row 0: reward_dist=10, risk_dist=5, target touched first -> R = +2.0.
    # Row 20: stop touched first -> R = -1.0.
    # mean = (2.0 + -1.0) / 2 = 0.5.
    assert m["realized_rr"] == pytest.approx(0.5)


def test_realized_rr_for_marking_matches_realized_rr_delegation():
    # Fix 1 (.superpowers/sdd/task-fix1-report.md): _realized_rr_for_marking
    # is the reusable core _realized_rr now delegates to. Same hand-crafted
    # scenario as test_metrics_realized_rr_hand_check_target_and_stop_outcomes
    # above, called two ways: (1) directly against a plain wide_df + bare
    # az_tgt/az_sl/marking arrays (the shape run_train's own rr_fn closure
    # uses), and (2) via _realized_rr's existing zoned_df-column call shape
    # -- both must agree exactly, proving the refactor changed nothing for
    # _realized_rr's existing callers (metrics()).
    from azlib.validate import _realized_rr, _realized_rr_for_marking

    n = 40
    one_high = np.full(n, 100.0)
    one_low = np.full(n, 100.0)
    one_high[6] = 111.0  # row 0's forward window (rows 1..15): touches target (110) at offset 5
    one_low[26] = 94.0  # row 20's forward window (rows 21..35): touches stop (95) at offset 5

    marking = np.zeros(n, dtype=bool)
    marking[[0, 20]] = True
    tgt = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    tgt[[0, 20]] = 110.0
    sl[[0, 20]] = 95.0

    wide_df = pd.DataFrame({"1_high": one_high, "1_low": one_low}, index=pd.RangeIndex(n))

    # Row 0: reward_dist=10, risk_dist=5, target touched first -> R = +2.0.
    # Row 20: stop touched first -> R = -1.0. mean = (2.0 + -1.0) / 2 = 0.5.
    direct = _realized_rr_for_marking(wide_df, 15, "long", tgt, sl, marking, n=1)
    assert direct == pytest.approx(0.5)

    zoned_df = pd.DataFrame(
        {
            "az_zone_long_15": marking,
            "az_tgt": tgt,
            "az_sl": sl,
            "1_high": one_high,
            "1_low": one_low,
        },
        index=pd.RangeIndex(n),
    )
    delegated = _realized_rr(zoned_df, 15, "long", n=1)
    assert delegated == pytest.approx(direct)


def test_realized_rr_for_marking_empty_marking_returns_nan():
    # "Handle empty marking -> NaN" (Fix 1 brief): a Y whose zone contains
    # zero rows must not crash or divide by zero -- it is simply excluded
    # from select_y's profitability filter via the ordinary NaN-safe
    # default, same as every other "nothing resolved" case.
    from azlib.validate import _realized_rr_for_marking

    n = 40
    wide_df = pd.DataFrame(
        {"1_high": np.full(n, 100.0), "1_low": np.full(n, 100.0)}, index=pd.RangeIndex(n)
    )
    marking = np.zeros(n, dtype=bool)
    tgt = np.full(n, 110.0)
    sl = np.full(n, 95.0)

    out = _realized_rr_for_marking(wide_df, 15, "long", tgt, sl, marking, n=1)
    assert math.isnan(out)


def test_metrics_realized_rr_nan_when_no_zoned_rows_resolve():
    n = 10  # too small for a full tf=15 forward window at any row
    zdf = pd.DataFrame(
        {
            "az_zone_long_15": np.array([True] + [False] * (n - 1)),
            "az_tgt": np.array([110.0] + [np.nan] * (n - 1)),
            "az_sl": np.array([95.0] + [np.nan] * (n - 1)),
            "1_high": np.full(n, 100.0),
            "1_low": np.full(n, 100.0),
            "az_reach_drift_max": np.full(n, np.nan),
        },
        index=pd.RangeIndex(n),
    )
    m = metrics(zdf, 15, "long", label_params={"n": 1})
    assert math.isnan(m["realized_rr"])


# --- reach_drift_max: _compute_reach_drift_max (pure array math) -----------


def test_compute_reach_drift_max_near_zero_for_identical_train_oos():
    from azlib.validate import _compute_reach_drift_max

    rng = np.random.default_rng(7)
    diff = rng.normal(0.0, 1.0, 500)

    # identical train/oos on BOTH sides -> realized OOS frequency should
    # track the train-frozen estimator closely at every x_grid level.
    drift_max = _compute_reach_drift_max(
        train_up=diff, train_down=diff, oos_up=diff, oos_down=diff
    )

    assert drift_max == pytest.approx(0.0, abs=0.05)


def test_compute_reach_drift_max_nan_when_train_array_too_small():
    from azlib.validate import _compute_reach_drift_max

    drift_max = _compute_reach_drift_max(
        train_up=np.array([1.0]),  # < 2 points -> reach_prob_estimator can't fit
        train_down=np.array([1.0, 2.0]),
        oos_up=np.zeros(5),
        oos_down=np.zeros(5),
    )
    assert math.isnan(drift_max)


# --- metrics: reach_drift_max is read from the az_reach_drift_max column ---


def test_metrics_reads_reach_drift_max_from_column():
    zdf = _tiny_zoned_df(zoned=[True, False], strict=[True, False], non_strict=[True, False])
    zdf["az_reach_drift_max"] = 0.42  # broadcast, as run_oos would write it

    m = metrics(zdf, 15, "long", label_params={})

    assert m["reach_drift_max"] == pytest.approx(0.42)


def test_metrics_raises_when_reach_drift_max_column_absent():
    # Deliberately NOT using _tiny_zoned_df (which always includes the
    # column) -- a caller-constructed frame missing az_reach_drift_max
    # entirely must get a loud, clear error, NOT a silent NaN that looks
    # like "no zoned entries" (task-8 coordinator review).
    zdf = pd.DataFrame(
        {
            "az_zone_long_15": [True, False],
            "az_label_strict": [True, False],
            "az_label_nonstrict": [True, False],
        },
        index=pd.RangeIndex(2),
    )
    with pytest.raises(KeyError, match="az_reach_drift_max"):
        metrics(zdf, 15, "long", label_params={})


# --- reach_drift_max survives row-filtering / .copy() (regression guard) ---


def test_reach_drift_max_survives_row_filtering_and_copy(two_synthetic_datasets):
    # Regression guard for the .attrs -> column fix (task-8 coordinator
    # review): pandas .attrs can be silently dropped by ordinary
    # row-filtering/.copy()/concat, AND (even where .attrs survives)
    # recomputing "OOS realized frequency" from a row-filtered zoned_df
    # silently changes the answer -- a Task 9 notebook that filters
    # zoned_df to a date range before calling metrics() must NOT silently
    # get a different (or missing) drift number. az_reach_drift_max is an
    # ordinary broadcast COLUMN, computed ONCE by run_oos against the FULL
    # OOS data, precisely so it survives exactly this kind of transform.
    #
    # This test is RED under the old .attrs-recompute-from-zoned_df design
    # (verified during this fix: filtering to zoos.iloc[10:] changed the
    # recomputed value from ~0.0205 to ~0.0209 -- a real, silent drift in
    # the metric, not just an .attrs-dropped-to-NaN failure) and GREEN now.
    ds = two_synthetic_datasets
    rp = run_train(ds["train_env"], tf=15, direction="long")
    zoos = run_oos(rp, ds["oos_env"])

    baseline = metrics(zoos, 15, "long", label_params={})["reach_drift_max"]
    assert not math.isnan(baseline)  # sanity: a real number to preserve

    filtered = zoos.iloc[10:].copy()  # row-filter + .copy() -- attrs-dropping transform
    m_filtered = metrics(filtered, 15, "long", label_params={})

    assert m_filtered["reach_drift_max"] == pytest.approx(baseline)


# --- integration: task-8-brief.md's own required RED test ------------------


def test_train_then_oos_metrics(monkeypatch, two_synthetic_datasets):
    ds = two_synthetic_datasets
    rp = run_train(ds["train_env"], tf=15, direction="long")
    zoos = run_oos(rp, ds["oos_env"])
    m = metrics(zoos, 15, "long", label_params={})

    assert 0 <= m["strict_coverage"] <= 1
    assert "realized_rr" in m


# --- Fix 2 / Fix 2b: first_touch_rr_grid (anti-mirage regression) ----------
#
# See .superpowers/sdd/task-fix2-report.md. azlib.zones.build_rr_levels'
# marginal (INDEPENDENT p_target/p_stop) reach-probability selection
# degenerates to a "tiny target you almost always eventually touch, far stop
# you almost never touch" mirage -- ignoring which level is hit FIRST.
# first_touch_rr_grid replaces that selection with one scored by the
# REALIZED first-touch mean R on a hand-built synthetic frame below, where a
# near-target/far-stop combo (tgt_x=0.25, sl_x=1.75 -- the exact corner the
# design brief calls out) is engineered to LOSE (its stop gets hit first far
# more often than the marginal/independent method would predict), while the
# grid's OTHER combos are engineered to be genuinely profitable.
#
# Fix 2b (regularization, ".superpowers/sdd/task-fix2-report.md"'s own "Fix
# 2b (regularized)" section): plain Fix 2, scored on an unconstrained random
# sample of every valid candle, was found to OVERFIT train (train->OOS
# realized R got WORSE on 4/6 combos on real data). Two regularizers were
# added: (1) prefer scoring on the STRICT-labeled entries (the actual zoned
# subset the levels are used on) when there are enough of them
# (`min_strict_count`), falling back to the broad pool otherwise; (2)
# restrict candidates to a sane reward:risk band ([1.0, 3.0] by default) --
# this EXCLUDES the (1.0, 0.25) combo (ratio 4.0) this test suite's own
# PREVIOUS revision asserted was selected: that pick, while genuinely not
# the near-target/far-stop mirage, had its own far-target/thin-stop-style
# skew (`rr=4.0`) outside the new sane band, so it is no longer selectable
# either -- see below for the updated, in-band expectation.
#
# Frame construction (tf=2, deterministic, no randomness anywhere -- UNCHANGED
# from Fix 2's own version):
#   - A "background" 2-minute-candle series (`2_high`/`2_low`) that is
#     FLAT (std=0 -> every price_levels(x=...) threshold collapses to the
#     reference price itself -> risk_dist==0 -> automatically EXCLUDED from
#     _realized_rr_for_marking's `resolved` set, see that function's own
#     docstring) EXCEPT for a short (`_BURST_CANDLES`) alternating +-`_M`
#     percent burst immediately before each of `_N_EVENTS` designed
#     "entries" -- this is what gives each entry's own reference candle a
#     real, well-defined, non-zero std (`_M * sqrt(6/5)`, the exact std of
#     an alternating +-M series over ANY 6-window, phase-independent).
#   - The 1-MINUTE price (`1_high`==`1_low`, since only price magnitude, not
#     the high/low spread, matters for this test) is DECOUPLED from the
#     `2_high`/`2_low` series (tf=2, not tf=1) and, by default, "self-
#     tracks" the CURRENT price_levels reference (the preceding completed
#     candle's own close) at every row -- this is what price_levels' own
#     threshold is centered on, so an ORDINARY (non-designed) row always has
#     entry == reference (risk_dist/reward_dist offset by a clean x*std_pct
#     amount, never a spurious leftover from unrelated background noise).
#     Only TWO rows per event deviate from this default: the entry row
#     itself (already matches the self-tracking default, kept explicit for
#     clarity) and the ONE row immediately after it, which jumps to
#     `entry_ref * (1 +/- mag * std_pct / 100)` -- the single forward-touch
#     row `win = max(1, n) * tf = 1 * 2 = 2` needs to see for the entry to
#     resolve win/loss.
#   - Up-events (target side, long direction): magnitudes `[1.2]*5 +
#     [0.4]*3 + [0.15]*2` (in x-grid units) -- most reach x=1.0, none reach
#     x=1.75, all reach x=0.25.
#   - Down-events (stop side): magnitudes `[2.4]*4 + [0.6]*15` -- a much
#     LARGER and MORE FREQUENT set than the up-side (deliberately fat/
#     frequent downside -- the real-world asymmetry the marginal method
#     misses): ALL 19 reach x=0.25, only the 4 largest reach x=1.0/x=1.75.
#   - `entry_rows` (returned by `_build_anti_mirage_frame`) doubles as the
#     STRICT-labeled set these tests pass as `strict_label`: a plain bool
#     mask True at exactly those 29 rows, mirroring the real
#     `(wide_df[label_col(p, direction, strict=True)] == 1.0)` shape
#     `run_train` builds -- there is no real label pipeline on this
#     hand-built frame, so the mask is constructed directly.
#
# Verified empirically (this test's own construction):
#   - Scored on the (small, 29-row) STRICT-labeled pool (`min_strict_count`
#     lowered so the fallback does NOT kick in -- 29 is below the
#     production default of 200), the in-band argmax is (1.0, 1.0) --
#     `ratio=1.0`, a genuinely BALANCED combo, `exp_ret ~= 0.111`.
#   - Scored on the broad fallback pool (no `strict_label`, or
#     `min_strict_count` left at its production default so 29 < 200 falls
#     back), EVERY in-band candidate on this frame scores <= 0 -> the
#     no-profitable sentinel (contamination from off-event background rows
#     in the unconstrained pool -- exactly the overfit-prone behavior Fix 2b
#     regularizes away by preferring the strict-labeled pool instead).
#   - Restricting the grid to just the mirage's two levels ([0.25, 1.75])
#     against the strict-labeled pool also returns the sentinel: the
#     (0.25, 1.75) corner itself has `ratio=0.14` (outside [1, 3], never
#     even scored) and neither (0.25, 0.25) nor (1.75, 1.75) (the only two
#     IN-band combos in that restricted grid) clears `score > 0.0` there
#     either.

_M = 0.05  # background alternating step, percent
_BURST_CANDLES = 10  # alternating candles feeding each entry's own std
_GAP_CANDLES = 15  # flat (std==0, auto-excluded) candles between events
_FLAT_PRICE = 100.0
_TF = 2  # candle tf for price_levels; WIN = max(1, n) * _TF, n=1 -> WIN=2
_UP_LIST = [1.2] * 5 + [0.4] * 3 + [0.15] * 2
_DOWN_LIST = [2.4] * 4 + [0.6] * 15
_ANTI_MIRAGE_X_GRID = np.array([0.25, 1.0, 1.75])
# Below the production default (200) on purpose -- this frame's own
# strict-labeled set has only 29 rows; a test-only override is how the
# strict-label pathway itself gets exercised without needing a 200+-event
# frame (see min_strict_count's own docstring: it's a caller-tunable knob,
# not a hardcoded constant this test would otherwise be unable to reach).
_ANTI_MIRAGE_MIN_STRICT_COUNT = 10


def _build_anti_mirage_frame() -> tuple[pd.DataFrame, list[int]]:
    """Hand-built, fully deterministic frame (see module comment block
    above) where the near-target/far-stop mirage corner (0.25, 1.75) loses
    on first-touch and a structurally different combo is genuinely
    profitable. Returns ``(wide_df, entry_rows)`` -- ``entry_rows`` (the
    designed entries' own row positions) is returned for tests that want to
    hand-inspect just those rows; ``first_touch_rr_grid`` itself uses its
    own generic (non-warmup, non-NaN) row mask, not this list.
    """
    seq = [(True, mag) for mag in _UP_LIST] + [(False, mag) for mag in _DOWN_LIST]
    std_pct = _M * np.sqrt(6.0 / 5.0)

    candle_vals = [_FLAT_PRICE] * (_BURST_CANDLES + 2)
    event_ref_candle_idx = []
    for _is_up, _mag in seq:
        candle_vals += [_FLAT_PRICE] * _GAP_CANDLES
        sign = 1
        c = _FLAT_PRICE
        for _ in range(_BURST_CANDLES):
            c = c * (1 + sign * _M / 100.0)
            candle_vals.append(c)
            sign *= -1
        event_ref_candle_idx.append(len(candle_vals) - 1)
        candle_vals.append(_FLAT_PRICE)  # entry candle's own close: back to the flat anchor
    candle_vals += [_FLAT_PRICE] * _GAP_CANDLES

    n_candles = len(candle_vals)
    n_rows = n_candles * _TF
    tf_col = np.repeat(np.array(candle_vals, dtype=float), _TF)
    is_closed = np.zeros(n_rows, dtype=bool)
    is_closed[_TF - 1 :: _TF] = True

    # 1-minute price self-tracks the CURRENT price_levels reference (the
    # preceding completed candle's own close) at every row by default.
    candle_idx_per_row = np.arange(n_rows) // _TF
    ref_candle_per_row = np.clip(candle_idx_per_row - 1, 0, None)
    one = np.array([candle_vals[c] for c in ref_candle_per_row], dtype=float)

    entry_rows = []
    for k, (is_up, mag) in enumerate(seq):
        ref_candle = event_ref_candle_idx[k]
        entry_candle = ref_candle + 1
        entry_row = entry_candle * _TF
        ref_price = candle_vals[ref_candle]
        one[entry_row] = ref_price
        entry_rows.append(entry_row)
        direction_mult = 1.0 if is_up else -1.0
        jump_price = ref_price * (1 + direction_mult * mag * std_pct / 100.0)
        one[entry_row + 1] = jump_price  # the ONE forward row win=2 needs to see

    idx = pd.RangeIndex(n_rows)
    df = pd.DataFrame(index=idx)
    df[f"{_TF}_high"] = tf_col
    df[f"{_TF}_low"] = tf_col
    df[f"{_TF}_close"] = tf_col
    df[f"{_TF}_is_closed"] = is_closed
    df["1_high"] = one
    df["1_low"] = one
    return df, entry_rows


def _strict_label_mask(df: pd.DataFrame, entry_rows: list[int]) -> np.ndarray:
    """The anti-mirage frame's own `entry_rows` as a bool mask, mirroring
    the real `(wide_df[label_col(p, direction, strict=True)] == 1.0)` shape
    `run_train` passes as `first_touch_rr_grid`'s `strict_label` argument.
    """
    mask = np.zeros(len(df), dtype=bool)
    mask[entry_rows] = True
    return mask


def test_first_touch_rr_grid_avoids_near_target_far_stop_mirage():
    # THE anti-mirage regression test (Fix 2's own TDD brief, Fix 2b's
    # regularized selection): on a frame where the (0.25, 1.75) near-
    # target/far-stop corner genuinely LOSES on first-touch (its far stop
    # gets hit often enough, by the fat/frequent down-side, to swamp its
    # tiny 0.25:1.75 reward:risk payoff), assert first_touch_rr_grid --
    # scored on the STRICT-labeled entries, the actual zoned subset the
    # levels are used on -- picks something else: NOT that corner, INSIDE
    # the sane [1, 3] reward:risk band, and genuinely profitable.
    df, entry_rows = _build_anti_mirage_frame()
    strict_label = _strict_label_mask(df, entry_rows)

    out = first_touch_rr_grid(
        df,
        _TF,
        "long",
        _ANTI_MIRAGE_X_GRID,
        n=1,
        fee=0.0,
        candle_size=1.0,
        seed=0,
        strict_label=strict_label,
        min_strict_count=_ANTI_MIRAGE_MIN_STRICT_COUNT,
    )

    # A genuinely profitable pick was found at all (not the no-profitable sentinel).
    assert not math.isnan(out["tgt_x"])
    assert out["exp_ret"] > 0.0

    # Not the mirage corner.
    assert not (out["tgt_x"] == pytest.approx(0.25) and out["sl_x"] == pytest.approx(1.75))
    # In the sane reward:risk band -- Change 2's own constraint, checked
    # directly against the SELECTED ratio (not just re-deriving it from
    # tgt_x/sl_x, though those agree too -- see the exact-value assertions
    # below).
    assert 1.0 <= out["rr"] <= 3.0

    # Exact deterministic regression values (fully hand-built, seeded frame
    # -- see .superpowers/sdd/task-fix2-report.md's "Fix 2b (regularized)"
    # section): a genuinely BALANCED combo (tgt_x == sl_x, ratio exactly
    # 1.0) -- not just "not the corner", the SAME moderate shape the
    # anti-mirage brief describes as the desired outcome.
    assert out["tgt_x"] == pytest.approx(1.0)
    assert out["sl_x"] == pytest.approx(1.0)
    assert out["rr"] == pytest.approx(1.0)
    assert out["exp_ret"] == pytest.approx(0.11111111111096698)


def test_first_touch_rr_grid_mirage_corner_alone_has_no_profitable_combo():
    # Directly isolates the mirage's own first-touch outcome: restricting
    # the grid to JUST the two mirage levels (so the only 4 candidates are
    # (0.25,0.25), (0.25,1.75) -- the corner itself --, (1.75,0.25),
    # (1.75,1.75)) finds NOTHING profitable on the strict-labeled pool --
    # the (0.25, 1.75) corner's ratio (0.14) is outside the [1, 3] band and
    # never even scored, and neither in-band survivor here, (0.25, 0.25) or
    # (1.75, 1.75), clears score > 0.0 either.
    df, entry_rows = _build_anti_mirage_frame()
    strict_label = _strict_label_mask(df, entry_rows)

    out = first_touch_rr_grid(
        df,
        _TF,
        "long",
        np.array([0.25, 1.75]),
        n=1,
        fee=0.0,
        candle_size=1.0,
        seed=0,
        strict_label=strict_label,
        min_strict_count=_ANTI_MIRAGE_MIN_STRICT_COUNT,
    )

    assert math.isnan(out["tgt_x"])
    assert math.isnan(out["sl_x"])
    assert math.isnan(out["rr"])
    assert math.isnan(out["exp_ret"])


def test_first_touch_rr_grid_reward_risk_band_excludes_out_of_band_ratio_winner():
    # Change 2's own regression: Fix 2 (pre-2b) selected (tgt_x=1.0,
    # sl_x=0.25) on this exact frame+pool -- a real, positive-expectancy
    # combo, but with ratio=4.0, OUTSIDE the new [1, 3] band. Prove the band
    # actually excludes it (not just that a DIFFERENT combo happens to win)
    # by widening the grid to include 0.25 again and confirming the winner
    # is no longer that out-of-band point.
    df, entry_rows = _build_anti_mirage_frame()
    strict_label = _strict_label_mask(df, entry_rows)

    out = first_touch_rr_grid(
        df,
        _TF,
        "long",
        _ANTI_MIRAGE_X_GRID,
        n=1,
        fee=0.0,
        candle_size=1.0,
        seed=0,
        strict_label=strict_label,
        min_strict_count=_ANTI_MIRAGE_MIN_STRICT_COUNT,
    )

    assert not (out["tgt_x"] == pytest.approx(1.0) and out["sl_x"] == pytest.approx(0.25))
    assert 1.0 <= out["rr"] <= 3.0


def test_first_touch_rr_grid_strict_label_fallback_when_too_few():
    # Change 1's own fallback rule: a strict_label whose (valid-intersected)
    # count is BELOW min_strict_count must fall back to the broad pool --
    # this frame's own strict-labeled set has only 29 rows, well under the
    # PRODUCTION default (min_strict_count left unset -> 200), so passing
    # strict_label here at the default threshold must behave IDENTICALLY to
    # passing no strict_label at all (both hit the same fallback pool).
    df, entry_rows = _build_anti_mirage_frame()
    strict_label = _strict_label_mask(df, entry_rows)
    assert strict_label.sum() < 200  # sanity: genuinely below the default floor

    out_with_strict = first_touch_rr_grid(
        df, _TF, "long", _ANTI_MIRAGE_X_GRID, n=1, fee=0.0, candle_size=1.0, seed=0,
        strict_label=strict_label,  # min_strict_count left at its production default (200)
    )
    out_without_strict = first_touch_rr_grid(
        df, _TF, "long", _ANTI_MIRAGE_X_GRID, n=1, fee=0.0, candle_size=1.0, seed=0,
    )

    assert out_with_strict == out_without_strict
    # And on the broad (contaminated, off-event-heavy) fallback pool with
    # the ratio band applied, nothing on this frame actually clears the
    # profitability bar -- the sentinel, not a spurious pick -- which is
    # exactly the overfit-prone behavior Fix 2b's strict-label preference
    # (once enough labeled rows exist) is meant to avoid relying on.
    assert math.isnan(out_without_strict["tgt_x"])


def test_first_touch_rr_grid_no_profitable_combo_returns_sentinel():
    # An outrageous fee (same "guarantee nothing clears the bar" technique
    # test_run_train_no_profitable_rr_records_nan_zone_results_file already
    # uses against azlib.zones.build_rr_levels) must drive EVERY candidate's
    # score negative here too -- first_touch_rr_grid's own fee_r = 2*fee/
    # candle_size term dominates any realistic mean_r -- and fall back to
    # the same 4-key NaN sentinel select_levels_safe uses, not a crash.
    df, entry_rows = _build_anti_mirage_frame()
    strict_label = _strict_label_mask(df, entry_rows)

    out = first_touch_rr_grid(
        df,
        _TF,
        "long",
        _ANTI_MIRAGE_X_GRID,
        n=1,
        fee=1_000_000.0,
        candle_size=1.0,
        seed=0,
        strict_label=strict_label,
        min_strict_count=_ANTI_MIRAGE_MIN_STRICT_COUNT,
    )

    assert set(out.keys()) == {"tgt_x", "sl_x", "rr", "exp_ret"}
    assert math.isnan(out["tgt_x"])
    assert math.isnan(out["sl_x"])
    assert math.isnan(out["rr"])
    assert math.isnan(out["exp_ret"])


def test_first_touch_rr_grid_deterministic_same_seed():
    # Same seed -> same selection, even when sample_size actually forces
    # real (seeded, NEVER unseeded/wall-clock) subsampling of the
    # strict-labeled candidate pool -- a sample_size smaller than the
    # 29-row strict-labeled pool guarantees the sampling path is genuinely
    # exercised, not a no-op.
    df, entry_rows = _build_anti_mirage_frame()
    strict_label = _strict_label_mask(df, entry_rows)

    kwargs = dict(
        n=1,
        fee=0.0,
        candle_size=1.0,
        seed=0,
        strict_label=strict_label,
        min_strict_count=_ANTI_MIRAGE_MIN_STRICT_COUNT,
        sample_size=10,
    )
    out1 = first_touch_rr_grid(df, _TF, "long", _ANTI_MIRAGE_X_GRID, **kwargs)
    out2 = first_touch_rr_grid(df, _TF, "long", _ANTI_MIRAGE_X_GRID, **kwargs)

    assert out1 == out2
