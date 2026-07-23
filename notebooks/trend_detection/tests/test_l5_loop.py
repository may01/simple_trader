"""Layer 5 tests -- tdlib.loop (the autonomous improvement loop: iteration
scheduling, keep/reject bookkeeping, per-combo persistence, markdown/json
reporting). See task-5-brief.md.

Filename carries "l5" so `pytest -k l5` selects every test in this module
(mirrors test_l0_conftest.py / test_l1_extract.py / test_l2_points_truth.py /
test_l3_features.py / test_l4_screen_models.py's own "l0".."l4" naming).

Two fixture-construction helpers do double duty across most of this file:

- ``_force_strong_points_both_sides`` -- a two-sided extension of
  test_l3_features.py / test_l4_screen_models.py's own single-sided
  ``_force_strong_points`` helper: neutralizes ``{tf}_rsi_ma8_diff`` to 0.0
  across the WHOLE frame, then forces class +2 at one disjoint position set
  and class -2 at another.
- ``_engineered_slim`` -- applies that at tf=15 to manufacture >=60 usable
  long/short points per side (task-5-brief.md's own "Context you cannot
  infer" recipe), while leaving tf=60/240 untouched -- at seed=0/days=12
  those are naturally far too sparse (<<60, verified against a throwaway
  probe against the real fixture) to ever pass the 60-point floor, which
  exercises the skip path "for free" in the very same fixture used by the
  full integration test.

Most unit tests below (chrono_split, apply_transform, the keep/reject rule,
write_iter_report, write_summary) build small hand-controlled X/y/IterResult
objects directly -- faster and easier to hand-verify than routing everything
through make_slim + feature_matrix + real model fits, matching
test_l4_screen_models.py's own stated convention. Only the skip-path test and
the two full-pipeline tests (determinism, and the L5 -> L6 integration test)
need the real fixture and a real (non-monkeypatched) run_iteration/
improvement_loop call.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdlib.config import LABEL_COLS, MOVE_CUTS
from tdlib.features import feature_matrix
from tdlib.loop import (
    SIDES,
    ComboResult,
    IterConfig,
    IterResult,
    apply_transform,
    chrono_split,
    improvement_loop,
    run_iteration,
    select_best,
    write_iter_report,
    write_summary,
)
from tdlib.models import load_bundle
from tdlib.points import strong_points
from tdlib.truth import mark_truth, truth_counts

from .conftest import make_slim, set_labels

# --- shared fixture-construction helpers -------------------------------------


def _force_strong_points_both_sides(df: pd.DataFrame, tf: int, up_positions: list, dn_positions: list) -> None:
    """In place: neutralize ``{tf}_rsi_ma8_diff`` to 0.0 across the WHOLE
    frame (class 0), then force class +2 at ``up_positions`` and class -2 at
    ``dn_positions`` (must be disjoint) -- so ``strong_points(df, tf, 2)``
    selects exactly ``up_positions`` and ``strong_points(df, tf, -2)``
    selects exactly ``dn_positions``, nothing more, nothing less.
    """
    diff_col = df.columns.get_loc(f"{tf}_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0
    df.iloc[up_positions, diff_col] = MOVE_CUTS[tf][3] + 1.0
    df.iloc[dn_positions, diff_col] = MOVE_CUTS[tf][0] - 1.0


def _assign_mixed_labels(df: pd.DataFrame, tf: int, positions: list) -> None:
    """Cycle ``positions`` (already chronologically sorted) through
    long/short/both/neither, in that repeating order -- exercises every
    ``mark_truth`` category except "nan", and keeps long/short perfectly
    INTERLEAVED throughout ``positions`` (every 4th position is long, every
    4th is short) so a later chronological 70/30 split always has both
    classes on both sides of the cut.
    """
    pattern = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]  # long, short, both, neither
    for i, (long_val, short_val) in enumerate(pattern):
        subset = positions[i::4]
        if subset:
            set_labels(df, tf, subset, long_val=long_val, short_val=short_val)


def _strong_point_positions(n_rows: int) -> tuple:
    """The exact disjoint up/dn position lists ``_engineered_slim`` forces
    tf=15 strong points at -- factored out (rather than inlined in
    ``_engineered_slim``) so a test that wants to layer EXTRA label columns
    on the SAME rows (e.g. the n2 profit_strict pair, for a real
    horizon="n2" exercise) can reuse the identical formula instead of
    duplicating/risking drift from it.
    """
    up_positions = list(range(0, n_rows, 3))[:200]
    dn_positions = list(range(1, n_rows, 3))[:200]
    return up_positions, dn_positions


def _engineered_slim(seed: int = 0, days: int = 12) -> pd.DataFrame:
    """``make_slim()`` with tf=15 engineered to have >=60 usable long/short
    points on BOTH sides (task-5-brief.md's own recipe): 200 positions
    forced to class +2 ("up"), 200 disjoint positions forced to class -2
    ("dn"), each cycled through the long/short/both/neither pattern above --
    50 long + 50 short usable points per side (well above the 60 floor),
    perfectly alternating. tf=60/240 are left completely untouched: at
    seed=0/days=12 they are naturally far too sparse (long+short in the
    10-30 range, verified against the real fixture) to ever clear the
    60-point floor, which is exactly the skip path this suite needs to
    exercise -- for free, in the same fixture used by every other test that
    needs a realistic slim.
    """
    df = make_slim(seed=seed, days=days)
    up_positions, dn_positions = _strong_point_positions(len(df))
    _force_strong_points_both_sides(df, 15, up_positions, dn_positions)
    _assign_mixed_labels(df, 15, up_positions)
    _assign_mixed_labels(df, 15, dn_positions)
    return df


def _assign_skewed_labels_n2(df: pd.DataFrame, tf: int, positions: list) -> None:
    """Writes the n2 profit_strict pair DIRECTLY by column name
    (``config.LABEL_COLS[tf]["pslong_n2"]``/``["psshort_n2"]``) -- unlike
    ``_assign_mixed_labels`` above, this can't go through ``set_labels()``:
    that helper only ever touches the n1 pair (see its own docstring).
    Cycles a DELIBERATELY different pattern than ``_assign_mixed_labels``'s
    50/50/50/50 n1 split -- [long, long, short, both] -- so 200 positions
    become 100 long / 50 short / 50 both / 0 neither. The mismatch against
    n1's counts is the whole point: it is what lets a real end-to-end test
    catch a horizon="n2" bug that silently keeps reading n1 columns (the
    counts would come back n1's 50/50/50/50 instead of this pattern's
    100/50/50/0).
    """
    long_col = LABEL_COLS[tf]["pslong_n2"]
    short_col = LABEL_COLS[tf]["psshort_n2"]
    long_loc = df.columns.get_loc(long_col)
    short_loc = df.columns.get_loc(short_col)
    pattern = [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]  # long, long, short, both
    for i, (long_val, short_val) in enumerate(pattern):
        subset = positions[i::4]
        if subset:
            df.iloc[subset, long_loc] = long_val
            df.iloc[subset, short_loc] = short_val


@pytest.fixture
def loop_artifacts_dir(tmp_path, monkeypatch):
    """Patch tdlib.loop's own ``artifacts_dir`` binding (the name it
    imported from tdlib.config) to a tmp_path location. Mirrors
    test_l1_extract.py's ``slim_paths`` fixture: monkeypatch the CONSUMING
    module's own bound name, not tdlib.config's."""
    out_dir = tmp_path / "trend_detection"
    monkeypatch.setattr("tdlib.loop.artifacts_dir", lambda: str(out_dir))
    return out_dir


def _fake_metrics_dict(roc_auc: float) -> dict:
    """A complete, well-formed eval_classifier-shaped dict (all 8 keys) --
    write_iter_report/write_summary index into every one of these keys, so a
    hand-built ComboResult used to test THEM must supply the full shape even
    though a given test only cares about one or two of the values."""
    return {
        "roc_auc": roc_auc, "acc": 0.5, "base_rate": 0.5, "n": 20,
        "prec_top_decile": 0.5, "prec_bottom_decile": 0.5, "lift_long": 0.0, "lift_short": 0.0,
    }


def _fake_combo_result(gbc_test_auc: float, marker: str = "topfeat") -> ComboResult:
    """A hand-built, non-skipped ComboResult whose gbc test roc_auc is
    exactly ``gbc_test_auc`` -- used to script IterResult.mean_test_auc
    without running any real model fit."""
    metrics = {
        "logistic": {"train": _fake_metrics_dict(0.5), "test": _fake_metrics_dict(0.5)},
        "gbc": {"train": _fake_metrics_dict(0.5), "test": _fake_metrics_dict(gbc_test_auc)},
    }
    importance_top = pd.DataFrame({"feature": [marker], "imp_mean": [1.0], "imp_std": [0.0]})
    return ComboResult(
        tf=15, side=2,
        counts={"long": 100, "short": 100, "both": 0, "neither": 0, "nan": 0},
        robustness={"long_fwd1": 0.6, "long_fwd4": 0.6, "short_fwd1": 0.4, "short_fwd4": 0.4},
        n_features=1, screen_top=pd.DataFrame(), metrics=metrics, importance_top=importance_top,
        chart_paths=[], skipped=False, skip_reason="",
    )


# --- chrono_split --------------------------------------------------------------


def test_chrono_split_positional_no_shuffle():
    n = 100
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    y = pd.Series(np.arange(n) % 2, dtype=np.int8)

    X_tr, X_te, y_tr, y_te = chrono_split(X, y, frac=0.7)

    assert len(X_tr) == 70
    assert len(X_te) == 30
    assert len(y_tr) == 70
    assert len(y_te) == 30
    assert X_tr.index[-1] < X_te.index[0]  # chronological: no shuffle
    assert list(X_tr["a"]) == list(range(70))
    assert list(X_te["a"]) == list(range(70, 100))
    assert list(y_tr.index) == list(X_tr.index)
    assert list(y_te.index) == list(X_te.index)


# --- apply_transform -------------------------------------------------------------


def test_apply_transform_baseline_identity():
    X_tr = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    X_te = pd.DataFrame({"a": [5.0], "b": [6.0]})

    out_tr, out_te = apply_transform(X_tr, X_te, "baseline", None)

    pd.testing.assert_frame_equal(out_tr, X_tr)
    pd.testing.assert_frame_equal(out_te, X_te)


def test_apply_transform_horizon_n2_leaves_columns_unchanged():
    """horizon_n2's own effect is entirely in run_iteration's horizon= kwarg
    to feature_matrix -- apply_transform itself treats it as a no-op,
    identical to baseline."""
    X_tr = pd.DataFrame({"a": [1.0, 2.0]})
    X_te = pd.DataFrame({"a": [3.0]})

    out_tr, out_te = apply_transform(X_tr, X_te, "horizon_n2", None)

    pd.testing.assert_frame_equal(out_tr, X_tr)
    pd.testing.assert_frame_equal(out_te, X_te)


def test_apply_transform_unknown_transform_raises_value_error():
    X_tr = pd.DataFrame({"a": [1.0]})
    X_te = pd.DataFrame({"a": [2.0]})

    with pytest.raises(ValueError):
        apply_transform(X_tr, X_te, "not_a_real_transform", None)


def test_apply_transform_prune_top40_keeps_exactly_top_40_by_imp_mean_desc():
    n = 5
    cols = [f"f{i}" for i in range(50)]
    X_tr = pd.DataFrame({c: [float(i)] * n for i, c in enumerate(cols)})
    X_te = pd.DataFrame({c: [float(i)] * 2 for i, c in enumerate(cols)})
    # imp_mean == i -> descending order is f49, f48, ..., f0; top 40 = f49..f10
    importance_prev = pd.DataFrame({
        "feature": cols,
        "imp_mean": [float(i) for i in range(50)],
        "imp_std": [0.01] * 50,
    })

    out_tr, out_te = apply_transform(X_tr, X_te, "prune_top40", importance_prev)

    expected_top40 = [f"f{i}" for i in range(49, 9, -1)]
    assert list(out_tr.columns) == expected_top40
    assert list(out_te.columns) == expected_top40
    assert len(out_tr.columns) == 40


def test_apply_transform_prune_top40_keeps_all_when_fewer_than_40_available():
    cols = ["a", "b", "c"]
    X_tr = pd.DataFrame({c: [1.0, 2.0] for c in cols})
    X_te = pd.DataFrame({c: [3.0] for c in cols})
    importance_prev = pd.DataFrame({"feature": cols, "imp_mean": [0.3, 0.1, 0.2], "imp_std": [0.0] * 3})

    out_tr, out_te = apply_transform(X_tr, X_te, "prune_top40", importance_prev)

    assert set(out_tr.columns) == set(cols)
    assert len(out_tr.columns) == 3
    assert set(out_te.columns) == set(cols)


def test_apply_transform_prune_top40_no_importance_prev_is_identity():
    X_tr = pd.DataFrame({"a": [1.0], "b": [2.0]})
    X_te = pd.DataFrame({"a": [3.0], "b": [4.0]})

    out_tr, out_te = apply_transform(X_tr, X_te, "prune_top40", None)

    pd.testing.assert_frame_equal(out_tr, X_tr)
    pd.testing.assert_frame_equal(out_te, X_te)


def test_apply_transform_interact_time_left_adds_expected_products_no_nan_explosion():
    X_tr = pd.DataFrame({
        "time_left_60": [0.5, 1.0, 0.25],
        "time_left_240": [0.1, 0.2, 0.3],  # present but NOT the lowest htf -- must not be used
        "featA": [2.0, 4.0, np.nan],
        "featB": [10.0, 20.0, 30.0],
        "time_in_candle_60": [0.5, 0.0, 0.75],  # a time_* column -- must be excluded from base_cols
    })
    X_te = pd.DataFrame({
        "time_left_60": [0.6],
        "time_left_240": [0.15],
        "featA": [3.0],
        "featB": [5.0],
        "time_in_candle_60": [0.4],
    })
    importance_prev = pd.DataFrame({
        "feature": ["featA", "featB", "time_in_candle_60", "time_left_60"],
        "imp_mean": [0.9, 0.8, 0.95, 0.99],  # time_* columns rank HIGHEST but must still be excluded
        "imp_std": [0.01] * 4,
    })

    out_tr, out_te = apply_transform(X_tr, X_te, "interact_time_left", importance_prev)

    assert "featA__x__time_left_60" in out_tr.columns
    assert "featB__x__time_left_60" in out_tr.columns
    assert "time_in_candle_60__x__time_left_60" not in out_tr.columns  # time_* excluded from base_cols
    assert "featA__x__time_left_240" not in out_tr.columns  # 240 is present but not the LOWEST htf

    expected_tr = X_tr["featA"] * X_tr["time_left_60"]
    pd.testing.assert_series_equal(out_tr["featA__x__time_left_60"], expected_tr, check_names=False)
    assert math.isnan(out_tr["featA__x__time_left_60"].iloc[2])  # NaN * x -> NaN, no crash/explosion

    expected_te = X_te["featB"] * X_te["time_left_60"]
    pd.testing.assert_series_equal(out_te["featB__x__time_left_60"], expected_te, check_names=False)

    for c in X_tr.columns:  # originals untouched, still present
        assert c in out_tr.columns
    for c in X_te.columns:
        assert c in out_te.columns


def test_apply_transform_interact_time_left_no_time_left_col_is_identity():
    X_tr = pd.DataFrame({"a": [1.0], "b": [2.0]})
    X_te = pd.DataFrame({"a": [3.0], "b": [4.0]})
    importance_prev = pd.DataFrame({"feature": ["a", "b"], "imp_mean": [0.9, 0.1], "imp_std": [0.0, 0.0]})

    out_tr, out_te = apply_transform(X_tr, X_te, "interact_time_left", importance_prev)

    pd.testing.assert_frame_equal(out_tr, X_tr)
    pd.testing.assert_frame_equal(out_te, X_te)


# --- run_iteration: skip path --------------------------------------------------


def test_run_iteration_skip_path_no_files_and_ignored_by_mean(loop_artifacts_dir):
    df = _engineered_slim()
    cfg = IterConfig(iter_no=1, transform="baseline", horizon="n1")

    result = run_iteration(cfg, df)

    # tf=15 both sides: engineered to clear the 60-point floor comfortably.
    assert result.combos[(15, 2)].skipped is False
    assert result.combos[(15, -2)].skipped is False

    # tf=60/240: left untouched -- naturally far too sparse at this seed to
    # ever clear the floor.
    for tf in (60, 240):
        for side in SIDES:
            combo = result.combos[(tf, side)]
            assert combo.skipped is True
            assert combo.skip_reason == "points<60"
            assert combo.counts["long"] + combo.counts["short"] < 60
            combo_dir = loop_artifacts_dir / "iter_01" / f"{tf}_{'up' if side > 0 else 'dn'}"
            assert not combo_dir.exists()  # no files at all for a skipped combo

    # non-skipped combo dirs DO have every expected file.
    for tf, side in ((15, 2), (15, -2)):
        combo_dir = loop_artifacts_dir / "iter_01" / f"{tf}_{'up' if side > 0 else 'dn'}"
        for fname in ("screen.csv", "metrics.json", "importance.csv", "selected_features.json"):
            assert (combo_dir / fname).exists()
        for fname in ("freeze.json", "logistic.joblib", "gbc.joblib", "feature_order.json"):
            assert (combo_dir / "bundle" / fname).exists()

    # mean_test_auc reflects ONLY the two non-skipped combos.
    auc_up = result.combos[(15, 2)].metrics["gbc"]["test"]["roc_auc"]
    auc_dn = result.combos[(15, -2)].metrics["gbc"]["test"]["roc_auc"]
    assert result.mean_test_auc == pytest.approx((auc_up + auc_dn) / 2.0)


# --- run_iteration: importance must be computed on TRAIN, never test ----------


def test_run_iteration_importance_table_called_on_train_split_not_test(loop_artifacts_dir, monkeypatch):
    """CRITICAL fix (fix pass 2): the importance table that feeds
    importance_prev (and gets persisted to importance.csv / stored in
    ComboResult.importance_top) MUST be computed on the TRAIN split, never
    the test split. chrono_split is a deterministic positional split, so if
    a later iteration picked features using THIS iteration's own test-set
    importance ranking, that later iteration's model would then be
    re-evaluated on the very same test rows its features were chosen to
    fit -- selection-on-test-set leakage that inflates apparent
    iteration-over-iteration gains. Pins the fix directly: monkeypatch
    tdlib.loop.importance_table to record the row count of whatever X it
    was actually called with, and assert it is the TRAIN split's row count
    (70), never the test split's (30).
    """
    df = _engineered_slim()  # tf=15 both sides: 100 usable points each -> chrono 70/30 -> 70 train, 30 test
    cfg = IterConfig(iter_no=1, transform="baseline", horizon="n1")

    calls = []

    def fake_importance_table(bundle, X, y, n_repeats=5):
        calls.append(len(X))
        cols = list(X.columns)
        return pd.DataFrame({
            "feature": cols,
            "imp_mean": list(np.linspace(1.0, 0.0, len(cols))),
            "imp_std": [0.0] * len(cols),
        })

    monkeypatch.setattr("tdlib.loop.importance_table", fake_importance_table)

    result = run_iteration(cfg, df)

    assert len(calls) == 2  # both non-skipped combos: (15, 2) and (15, -2)
    for n in calls:
        assert n == 70  # len(X_tr) -- the TRAIN split's row count
        assert n != 30  # explicitly NOT len(X_te) -- the test split's row count

    for tf, side in ((15, 2), (15, -2)):
        assert result.combos[(tf, side)].importance_top is not None


# --- run_iteration: horizon="n2" end-to-end, no monkeypatching of truth -------


def test_run_iteration_horizon_n2_reads_n2_labels_end_to_end(loop_artifacts_dir):
    """IMPORTANT fix (fix pass 2): REAL, non-monkeypatched exercise of
    horizon="n2" -- prior coverage of "n2" only ever ran through a
    monkeypatched run_iteration (the keep/reject-rule test), which proves
    cfg.horizon gets SET to "n2" but nothing about whether mark_truth
    actually reads the n2 columns when it gets there.

    set_labels() only ever writes the n1 profit_strict pair (see its own
    docstring), so this test writes the n2 pair DIRECTLY by column name
    (_assign_skewed_labels_n2), at the SAME tf=15 strong-point positions
    _engineered_slim already forces, with a DELIBERATELY different
    long/short/both/neither mix than n1's own 50/50/50/50
    (_assign_mixed_labels) -- 100 long / 50 short / 50 both / 0 neither.

    First confirms directly off mark_truth/truth_counts (no model fitting
    needed for this part) that the two horizons' labels genuinely differ at
    both non-skipped combos. Then runs run_iteration(horizon="n2") for real
    and checks it reproduces EXACTLY the n2-specific counts -- a bug that
    silently kept reading n1 columns under horizon="n2" would reproduce
    n1's 50/50/50/50 counts instead and fail this test outright.
    """
    df = _engineered_slim()  # tf=15 both sides forced; n1 labels: 50/50/50/50 per side
    up_positions, dn_positions = _strong_point_positions(len(df))
    _assign_skewed_labels_n2(df, 15, up_positions)
    _assign_skewed_labels_n2(df, 15, dn_positions)

    expected_n1_counts = {"long": 50, "short": 50, "both": 50, "neither": 50, "nan": 0}
    expected_n2_counts = {"long": 100, "short": 50, "both": 50, "neither": 0, "nan": 0}

    # sanity: the two horizons' labels really do differ at every non-skipped
    # combo, verified directly off mark_truth/truth_counts (cheap, no model
    # fitting) before trusting run_iteration's own end-to-end result below.
    for side in SIDES:
        pts = strong_points(df, 15, side)
        counts_n1 = truth_counts(mark_truth(pts, 15, "n1"))
        counts_n2 = truth_counts(mark_truth(pts, 15, "n2"))
        assert counts_n1 == expected_n1_counts
        assert counts_n2 == expected_n2_counts
        assert counts_n1 != counts_n2

    cfg = IterConfig(iter_no=1, transform="baseline", horizon="n2")
    result = run_iteration(cfg, df)

    for tf, side in ((15, 2), (15, -2)):
        combo = result.combos[(tf, side)]
        assert combo.skipped is False
        assert combo.counts == expected_n2_counts  # NOT n1's 50/50/50/50
        for model in ("logistic", "gbc"):
            for split in ("train", "test"):
                assert math.isfinite(combo.metrics[model][split]["roc_auc"])


# --- improvement_loop: keep/reject rule + importance_prev + horizon_n2 --------


def test_improvement_loop_keep_reject_rule_importance_prev_and_horizon_scheduling(loop_artifacts_dir, monkeypatch):
    scripted_aucs = [0.60, 0.62, 0.615, 0.61]
    calls = []

    def fake_run_iteration(cfg, slim, importance_prev=None):
        calls.append((cfg.iter_no, cfg.transform, cfg.horizon, importance_prev))
        auc = scripted_aucs[cfg.iter_no - 1]
        combo = _fake_combo_result(auc, marker=f"iter{cfg.iter_no}_feature")
        return IterResult(cfg=cfg, combos={(15, 2): combo})

    monkeypatch.setattr("tdlib.loop.run_iteration", fake_run_iteration)

    placeholder_slim = make_slim(seed=0, days=2)  # contents irrelevant -- run_iteration is stubbed
    results = improvement_loop(placeholder_slim, max_iters=4, eps_auc=0.005)

    assert [r.cfg.iter_no for r in results] == [1, 2, 3, 4]
    assert [r.kept for r in results] == [True, True, False, False]
    assert len(calls) == 4

    # horizon_n2 scheduling: iteration 4 == IMPROVEMENTS[3] == "horizon_n2".
    assert results[3].cfg.transform == "horizon_n2"
    assert results[3].cfg.horizon == "n2"
    for r in results[:3]:
        assert r.cfg.horizon == "n1"

    # importance_prev propagation: iter1 starts from {}; iter2 is fed iter1's
    # own importance; iter3 AND iter4 are BOTH fed iter2's (iter3 rejected,
    # so it never overwrites iter2's as the feed for the next call).
    iter1_importance = results[0].combos[(15, 2)].importance_top
    iter2_importance = results[1].combos[(15, 2)].importance_top

    assert calls[0][3] == {}
    assert calls[1][3][(15, 2)] is iter1_importance
    assert calls[2][3][(15, 2)] is iter2_importance
    assert calls[3][3][(15, 2)] is iter2_importance


def test_improvement_loop_stops_after_two_consecutive_rejections_before_exhaustion(loop_artifacts_dir, monkeypatch):
    """Proves the early-stop check fires on its own merits, not merely
    because the scripted sequence also happened to exhaust IMPROVEMENTS/
    max_iters at the same iteration (test above): 6 fake transforms and
    max_iters=6 are both available, but the loop must still stop right after
    the SECOND consecutive rejection (iterations 3, 4)."""
    monkeypatch.setattr("tdlib.loop.IMPROVEMENTS", ["baseline", "t2", "t3", "t4", "t5", "t6"])
    scripted_aucs = [0.60, 0.62, 0.615, 0.61, 0.90, 0.95]  # 5/6 would be great -- loop must never reach them

    def fake_run_iteration(cfg, slim, importance_prev=None):
        auc = scripted_aucs[cfg.iter_no - 1]
        return IterResult(cfg=cfg, combos={(15, 2): _fake_combo_result(auc)})

    monkeypatch.setattr("tdlib.loop.run_iteration", fake_run_iteration)

    results = improvement_loop(make_slim(seed=0, days=2), max_iters=6, eps_auc=0.005)

    assert [r.cfg.iter_no for r in results] == [1, 2, 3, 4]


# --- write_iter_report -----------------------------------------------------------


def test_write_iter_report_contains_counts_metrics_and_verdict(tmp_path):
    combo_ok = _fake_combo_result(0.75, marker="topfeat")
    cfg = IterConfig(iter_no=2, transform="prune_top40", horizon="n1")
    res = IterResult(cfg=cfg, combos={(15, 2): combo_ok}, kept=True, best_before=0.60, eps_auc=0.005)

    path = write_iter_report(res, str(tmp_path))

    assert path == str(tmp_path / "iter_02" / "report.md")
    text = Path(path).read_text()

    assert "15_up" in text
    assert "prune_top40" in text
    assert "0.7500" in text  # gbc test roc_auc, formatted to 4dp
    assert "KEPT" in text
    assert "100" in text  # n_long/n_short from combo_ok's counts
    assert "topfeat" in text  # top-importance feature name


def test_write_iter_report_rejected_verdict_and_skip_row(tmp_path):
    combo_ok = _fake_combo_result(0.55, marker="f")
    combo_skip = ComboResult(
        tf=60, side=2, counts={"long": 3, "short": 2, "both": 0, "neither": 1, "nan": 0},
        robustness={}, n_features=0, screen_top=pd.DataFrame(), metrics={}, importance_top=None,
        chart_paths=[], skipped=True, skip_reason="points<60",
    )
    cfg = IterConfig(iter_no=3, transform="interact_time_left", horizon="n1")
    res = IterResult(cfg=cfg, combos={(15, 2): combo_ok, (60, 2): combo_skip}, kept=False, best_before=0.62, eps_auc=0.005)

    path = write_iter_report(res, str(tmp_path))
    text = Path(path).read_text()

    assert "60_up" in text
    assert "REJECTED" in text
    assert "True" in text  # skip combo's own "skipped" column cell


def test_write_iter_report_source_has_no_wallclock_read():
    import tdlib.loop as loop_module

    source = inspect.getsource(loop_module)
    assert "datetime.now" not in source
    assert "time.time()" not in source


# --- select_best (public: shared by write_summary AND run_loop.py's console print) ---


def test_select_best_picks_highest_mean_test_auc_among_kept():
    r1 = IterResult(cfg=IterConfig(1, "baseline", "n1"), combos={(15, 2): _fake_combo_result(0.60)}, kept=True)
    r2 = IterResult(cfg=IterConfig(2, "prune_top40", "n1"), combos={(15, 2): _fake_combo_result(0.62)}, kept=True)
    r3 = IterResult(cfg=IterConfig(3, "interact_time_left", "n1"), combos={(15, 2): _fake_combo_result(0.615)}, kept=False)

    assert select_best([r1, r2, r3]) is r2


def test_select_best_falls_back_to_last_iteration_when_none_kept():
    r1 = IterResult(cfg=IterConfig(1, "baseline", "n1"), combos={(15, 2): _fake_combo_result(0.4)}, kept=False)
    r2 = IterResult(cfg=IterConfig(2, "prune_top40", "n1"), combos={(15, 2): _fake_combo_result(0.3)}, kept=False)

    assert select_best([r1, r2]) is r2  # last iteration overall, even though never "kept"


def test_select_best_empty_list_returns_none():
    assert select_best([]) is None


# --- write_summary / best.json --------------------------------------------------


def test_write_summary_best_json_identifies_best_iteration_and_relative_paths(tmp_path):
    combo1 = _fake_combo_result(0.60, marker="f1")
    combo2 = _fake_combo_result(0.62, marker="f2")
    combo3 = _fake_combo_result(0.615, marker="f3")

    r1 = IterResult(cfg=IterConfig(1, "baseline", "n1"), combos={(15, 2): combo1},
                     kept=True, best_before=float("-inf"), eps_auc=0.005)
    r2 = IterResult(cfg=IterConfig(2, "prune_top40", "n1"), combos={(15, 2): combo2},
                     kept=True, best_before=0.60, eps_auc=0.005)
    r3 = IterResult(cfg=IterConfig(3, "interact_time_left", "n1"), combos={(15, 2): combo3},
                     kept=False, best_before=0.62, eps_auc=0.005)

    path = write_summary([r1, r2, r3], str(tmp_path))

    assert path == str(tmp_path / "summary.md")
    assert (tmp_path / "summary.md").exists()
    summary_text = (tmp_path / "summary.md").read_text()
    assert "rejected" in summary_text
    assert "kept" in summary_text

    with open(tmp_path / "best.json") as f:
        best = json.load(f)

    assert best["best_iter"] == 2  # r2 has the highest mean_test_auc among the KEPT iterations
    assert best["transform"] == "prune_top40"
    assert best["horizon"] == "n1"
    assert best["mean_test_auc"] == pytest.approx(0.62)
    assert set(best["combos"].keys()) == {"15_up"}
    assert best["combos"]["15_up"]["bundle_dir"] == "iter_02/15_up/bundle"
    assert best["combos"]["15_up"]["features"] == "iter_02/15_up/selected_features.json"
    assert best["combos"]["15_up"]["n_test"] == 20


def test_write_summary_no_kept_iteration_falls_back_without_crash(tmp_path):
    combo = _fake_combo_result(0.4, marker="f")
    r1 = IterResult(cfg=IterConfig(1, "baseline", "n1"), combos={(15, 2): combo},
                     kept=False, best_before=float("-inf"), eps_auc=0.005)

    path = write_summary([r1], str(tmp_path))

    assert Path(path).exists()
    with open(tmp_path / "best.json") as f:
        best = json.load(f)
    assert best["best_iter"] == 1  # fallback: last iteration overall, even though never "kept"


# --- determinism -----------------------------------------------------------------


def test_improvement_loop_determinism_byte_identical_metrics_json(tmp_path, monkeypatch):
    df = _engineered_slim()

    run1_dir = tmp_path / "run1"
    monkeypatch.setattr("tdlib.loop.artifacts_dir", lambda: str(run1_dir))
    improvement_loop(df, max_iters=1)

    run2_dir = tmp_path / "run2"
    monkeypatch.setattr("tdlib.loop.artifacts_dir", lambda: str(run2_dir))
    improvement_loop(df, max_iters=1)

    for combo_name in ("15_up", "15_dn"):
        p1 = run1_dir / "iter_01" / combo_name / "metrics.json"
        p2 = run2_dir / "iter_01" / combo_name / "metrics.json"
        assert p1.read_bytes() == p2.read_bytes()


# --- integration: L5 -> L6 boundary ----------------------------------------------


def test_loop_freezes_for_oos(loop_artifacts_dir):
    """improvement_loop(max_iters=1) freezes exactly what Layer 6 needs:
    per non-skipped combo, a full artifact set on disk, plus a best.json
    whose relative paths resolve from base_dir to a bundle that
    models.load_bundle can load and immediately predict_proba on X rebuilt
    from selected_features.json alone -- the exact contract task-6-brief.md
    depends on.
    """
    df = _engineered_slim()

    results = improvement_loop(df, max_iters=1)

    assert len(results) == 1
    base = loop_artifacts_dir
    assert (base / "iter_01" / "report.md").exists()

    non_skipped = [(15, 2), (15, -2)]
    for tf, side in non_skipped:
        combo_dir = base / "iter_01" / f"{tf}_{'up' if side > 0 else 'dn'}"
        for fname in ("screen.csv", "metrics.json", "importance.csv", "selected_features.json"):
            assert (combo_dir / fname).exists()
        for fname in ("freeze.json", "logistic.joblib", "gbc.joblib", "feature_order.json"):
            assert (combo_dir / "bundle" / fname).exists()

    best_json_path = base / "best.json"
    assert best_json_path.exists()
    with open(best_json_path) as f:
        best = json.load(f)

    assert best["best_iter"] == 1
    assert best["transform"] == "baseline"
    assert best["horizon"] == "n1"
    assert set(best["combos"].keys()) == {"15_up", "15_dn"}

    combo_info = best["combos"]["15_up"]
    bundle_dir = base / combo_info["bundle_dir"]
    features_path = base / combo_info["features"]
    assert bundle_dir.exists()
    assert features_path.exists()
    assert combo_info["bundle_dir"] == "iter_01/15_up/bundle"  # relative to base_dir, not absolute

    # L6 contract: rebuild X from selected_features.json alone, load_bundle,
    # predict_proba must just work.
    with open(features_path) as f:
        selected_features = json.load(f)

    X, y = feature_matrix(df, 15, 2)
    assert set(selected_features) <= set(X.columns)
    X_rebuilt = X[selected_features]

    bundle = load_bundle(bundle_dir)
    Z = bundle["freeze"].transform(X_rebuilt)
    proba = bundle["gbc"].predict_proba(Z)
    assert proba.shape == (len(X_rebuilt), 2)
    assert np.isfinite(proba).all()
