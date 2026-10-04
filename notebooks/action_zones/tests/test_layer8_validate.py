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

from azlib.validate import metrics, run_oos, run_train
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
