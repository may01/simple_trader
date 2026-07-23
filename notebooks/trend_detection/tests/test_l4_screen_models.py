"""Layer 4 tests -- tdlib.screen (univariate feature screening + charts) and
tdlib.models (classifier fit/eval/importance/persistence). See
task-4-brief.md.

Filename carries "l4" so `pytest -k l4` selects every test in this module
(mirrors test_l0_conftest.py / test_l1_extract.py / test_l2_points_truth.py /
test_l3_features.py's own "l0"/"l1"/"l2"/"l3" naming).

For most unit tests X/y are built directly (small, hand-controlled arrays --
faster and easier to hand-verify than round-tripping through make_slim +
feature_matrix). feature_matrix is used in exactly the one integration test
below, per task-4-brief.md's "Context you cannot infer" note -- that test
pins the real L3 -> L4/L5 contract (feature_matrix's X/y shape feeding
straight into fit_classifiers/eval_classifier/save_bundle/load_bundle).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from tdlib.config import MOVE_CUTS
from tdlib.features import FreezeStats, feature_matrix
from tdlib.models import eval_classifier, fit_classifiers, importance_table, load_bundle, save_bundle
from tdlib.screen import _mannwhitney_auc, screen_charts, univariate_screen

from .conftest import make_slim, set_labels

# matplotlib.pyplot is only usable AFTER something has forced the Agg
# backend -- tdlib.screen does that at import time (see its own module
# docstring), and it's already imported above, so this is safe here.
import matplotlib.pyplot as plt  # noqa: E402

# --- shared helpers ----------------------------------------------------------


def _force_strong_points(df: pd.DataFrame, tf: int, positions: list) -> None:
    """In place: neutralize ``{tf}_rsi_ma8_diff`` to 0.0 across the WHOLE
    frame, then force it to class +2 at exactly ``positions`` (which must
    already be ``{tf}_is_closed`` rows). So ``strong_points(df, tf, 2)``
    selects exactly ``positions`` -- nothing more, nothing less.

    Verbatim copy of test_l3_features.py's own helper of the same name
    (feature_matrix runs strong_points internally -- see features.py's
    module docstring -- so every feature_matrix-driven test needs its
    target rows to be genuine strong-point selections, not an arbitrary
    caller-supplied ``pts``). Duplicated here rather than imported cross-
    module since each test_l*.py file in this suite is self-contained,
    importing only from ``.conftest`` and ``tdlib`` (matching that file's
    own convention).
    """
    diff_col = df.columns.get_loc(f"{tf}_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0
    df.iloc[positions, diff_col] = MOVE_CUTS[tf][3] + 1.0


class _FixedProbaModel:
    """Stub classifier: ``predict_proba`` always returns the SAME pre-baked
    probabilities regardless of its input matrix -- lets a test hand-verify
    ``eval_classifier``'s decile/lift arithmetic exactly, without needing a
    real fitted model whose probabilities would depend on floating-point
    solver internals. "bundle-free" in the brief's sense: no FreezeStats-
    transformed matrix actually drives the prediction, only ``X``'s ROW
    COUNT matters (``bundle["freeze"]`` is still a real, fitted FreezeStats
    -- eval_classifier always calls ``.transform()`` -- but what it computes
    is irrelevant to this stub).
    """

    def __init__(self, proba_pos) -> None:
        self._proba_pos = np.asarray(proba_pos, dtype=float)

    def predict_proba(self, Z) -> np.ndarray:
        pos = self._proba_pos
        return np.column_stack([1.0 - pos, pos])


# --- integration: L4 -> L5 boundary -------------------------------------------


def test_models_bundle_roundtrip_feeds_loop(tmp_path):
    """A synthetic-but-realistic (make_slim + feature_matrix -- the real L3
    contract, not a hand-rolled X/y) separable dataset: fit a bundle, prove
    the gbc actually LEARNED something (roc_auc > 0.8 on a temporally
    held-out slice, not just memorized train), then round-trip the bundle
    through save_bundle/load_bundle and prove the reload is functionally
    IDENTICAL -- same predict_proba, same eval_classifier metrics. Pins the
    exact freeze/model persistence contract L5 (which will WRITE bundles)
    and L6 (which will READ them back for inference) both depend on.
    """
    df = make_slim(seed=200, days=10)
    n_points = 700
    # tf=15 -> 15_is_closed is True at EVERY row of the slim frame (see
    # conftest.py's make_slim docstring), so strong-point positions can be
    # any row index at all -- unlike tf=60/240 (spaced 4/16 rows apart).
    positions = list(range(n_points))
    _force_strong_points(df, 15, positions)

    # Interleaved (even/odd), NOT a contiguous block -- so a later
    # CHRONOLOGICAL train/holdout split still has both classes present in
    # the holdout tail, instead of a train-only-long/holdout-only-short
    # split that would make roc_auc undefined on the holdout.
    long_positions = positions[0::2]
    short_positions = positions[1::2]
    set_labels(df, 15, long_positions, long_val=1.0, short_val=0.0)
    set_labels(df, 15, short_positions, long_val=0.0, short_val=1.0)

    # A real (raw, non-blocklisted) slim column, engineered into a strong
    # long/short separator: N(2.0, 0.4) vs N(-2.0, 0.4) is ~10 sigma apart,
    # comfortably separable by a downstream classifier while still being
    # genuine per-row NOISE (not a hand-planted constant), so this is a
    # real generalization test, not memorization.
    rng = np.random.default_rng(201)
    sep_col = df.columns.get_loc("15_body_ratio")
    df.iloc[long_positions, sep_col] = rng.normal(2.0, 0.4, len(long_positions))
    df.iloc[short_positions, sep_col] = rng.normal(-2.0, 0.4, len(short_positions))

    X, y = feature_matrix(df, 15, 2)
    assert len(X) == n_points

    X_tr, y_tr = X.iloc[:560], y.iloc[:560]
    X_ho, y_ho = X.iloc[560:], y.iloc[560:]
    assert y_ho.nunique() == 2  # sanity: interleaving really does keep both classes in the holdout

    bundle = fit_classifiers(X_tr, y_tr)
    metrics = eval_classifier(bundle, "gbc", X_ho, y_ho)
    assert metrics["roc_auc"] > 0.8

    out_dir = tmp_path / "bundle"
    save_bundle(bundle, out_dir)
    loaded = load_bundle(out_dir)

    proba_orig = bundle["gbc"].predict_proba(bundle["freeze"].transform(X_ho))
    proba_loaded = loaded["gbc"].predict_proba(loaded["freeze"].transform(X_ho))
    np.testing.assert_allclose(proba_orig, proba_loaded)

    metrics_loaded = eval_classifier(loaded, "gbc", X_ho, y_ho)
    assert metrics_loaded == pytest.approx(metrics)


# --- univariate_screen: AUC formula (Mann-Whitney U convention) ---------------


def test_mannwhitney_auc_hand_checked_perfect_and_overlap():
    """Locks down scipy's U-statistic convention univariate_screen depends
    on: ``mannwhitneyu(x1, x0).statistic`` is U FOR x1 (the first sample) --
    confirmed empirically against scipy 1.18 (task-4-report.md), not merely
    assumed from the docs. x_long=[3,4,5] beats every x_short=[1,2] value
    (6 of 6 pairs) -> U1=6, auc=6/(3*2)=1.0. x_long=[1,3] vs x_short=[2]:
    only the (3,2) pair has long>short (1 of 2) -> U1=1, auc=1/(2*1)=0.5.
    """
    u1, auc = _mannwhitney_auc(np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0]))
    assert u1 == pytest.approx(6.0)
    assert auc == pytest.approx(1.0)

    u1b, aucb = _mannwhitney_auc(np.array([1.0, 3.0]), np.array([2.0]))
    assert u1b == pytest.approx(1.0)
    assert aucb == pytest.approx(0.5)


# --- univariate_screen: screening semantics -----------------------------------


def test_univariate_screen_result_index_and_columns_exact():
    rng = np.random.default_rng(30)
    n = 40
    y = pd.Series([1] * 20 + [0] * 20, dtype=np.int8)
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})

    scr = univariate_screen(X, y)

    assert scr.index.name == "feature"
    assert set(scr.index) == {"f1", "f2"}
    assert list(scr.columns) == [
        "n", "n_long", "n_short", "auc", "abs_auc_dev", "auc_se",
        "auc_low95", "auc_high95", "ks_stat", "mean_long", "mean_short", "note",
    ]


def test_univariate_screen_perfect_and_inverted_separators_top_two_by_abs_auc_dev():
    n = 100
    y = pd.Series([1] * n + [0] * n, dtype=np.int8)
    perfect_vals = np.concatenate([np.linspace(10.0, 20.0, n), np.linspace(0.0, 9.0, n)])
    inverted_vals = np.concatenate([np.linspace(0.0, 9.0, n), np.linspace(10.0, 20.0, n)])
    rng = np.random.default_rng(1)
    X = pd.DataFrame({
        "perfect": perfect_vals,
        "inverted": inverted_vals,
        "noise": rng.normal(0.0, 1.0, size=2 * n),
    })

    scr = univariate_screen(X, y)

    assert scr.loc["perfect", "auc"] == pytest.approx(1.0)
    assert scr.loc["inverted", "auc"] == pytest.approx(0.0)
    assert scr.loc["perfect", "abs_auc_dev"] == pytest.approx(0.5)
    assert scr.loc["inverted", "abs_auc_dev"] == pytest.approx(0.5)
    assert set(scr.index[:2]) == {"perfect", "inverted"}
    assert pd.isna(scr.loc["perfect", "note"])  # a real, non-degenerate row: no note

    assert scr.loc["perfect", "mean_long"] == pytest.approx(np.linspace(10.0, 20.0, n).mean())
    assert scr.loc["perfect", "mean_short"] == pytest.approx(np.linspace(0.0, 9.0, n).mean())


def test_univariate_screen_seeded_noise_auc_near_half_ranks_below_strong_signal():
    rng = np.random.default_rng(2)
    n = 150
    y = pd.Series([1] * n + [0] * n, dtype=np.int8)
    noise = rng.normal(0.0, 1.0, size=2 * n)
    strong = np.concatenate([np.linspace(5.0, 15.0, n), np.linspace(-15.0, -5.0, n)])
    X = pd.DataFrame({"noise": noise, "strong": strong})

    scr = univariate_screen(X, y)

    assert scr.loc["noise", "auc"] == pytest.approx(0.5, abs=0.1)
    assert scr.index[-1] == "noise"  # lower abs_auc_dev than "strong" -> ranks last of the two


def test_univariate_screen_constant_column_is_degenerate_not_crash():
    n = 60
    y = pd.Series([1] * 30 + [0] * 30, dtype=np.int8)
    X = pd.DataFrame({"const": [5.0] * n})

    scr = univariate_screen(X, y)

    assert scr.loc["const", "note"] == "degenerate"
    assert math.isnan(scr.loc["const", "auc"])
    assert math.isnan(scr.loc["const", "auc_se"])


def test_univariate_screen_too_few_long_rows_noted_too_few():
    y = pd.Series([1] * 5 + [0] * 50, dtype=np.int8)  # only 5 long rows, below the 10 floor
    X = pd.DataFrame({"feat": np.arange(len(y), dtype=float)})

    scr = univariate_screen(X, y)

    assert scr.loc["feat", "note"] == "too_few"
    assert scr.loc["feat", "n_long"] == 5
    assert scr.loc["feat", "n_short"] == 50
    assert math.isnan(scr.loc["feat", "auc"])


def test_univariate_screen_too_few_checked_before_degenerate_when_both_apply():
    """A column that is BOTH too-few (5 long rows) AND fully constant must
    report "too_few", not "degenerate" -- there is not even enough data to
    ask the variance question, so the group-size gate wins."""
    y = pd.Series([1] * 5 + [0] * 50, dtype=np.int8)
    X = pd.DataFrame({"const_and_sparse": [7.0] * 55})

    scr = univariate_screen(X, y)

    assert scr.loc["const_and_sparse", "note"] == "too_few"


def test_univariate_screen_nan_holed_feature_n_reflects_drops():
    n = 100
    y = pd.Series([1] * n + [0] * n, dtype=np.int8)
    values = np.concatenate([np.linspace(10.0, 20.0, n), np.linspace(0.0, 9.0, n)])
    values[:15] = np.nan  # drop 15 long rows
    values[n : n + 10] = np.nan  # drop 10 short rows
    X = pd.DataFrame({"holey": values})

    scr = univariate_screen(X, y)

    assert scr.loc["holey", "n"] == (2 * n) - 15 - 10
    assert scr.loc["holey", "n_long"] == n - 15
    assert scr.loc["holey", "n_short"] == n - 10
    assert not math.isnan(scr.loc["holey", "auc"])  # still well above the n>=10 floor per side


def test_univariate_screen_sorted_desc_abs_auc_dev_with_nan_rows_last():
    n = 30
    y = pd.Series([1] * n + [0] * n, dtype=np.int8)
    strong = np.concatenate([np.linspace(5.0, 15.0, n), np.linspace(-15.0, -5.0, n)])
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, size=2 * n)
    const = np.full(2 * n, 1.0)
    X = pd.DataFrame({"strong": strong, "noise": noise, "const": const})

    scr = univariate_screen(X, y)

    assert scr.index[-1] == "const"  # the one NaN-stat row sinks to the bottom
    non_nan = scr.loc[scr["abs_auc_dev"].notna(), "abs_auc_dev"]
    assert non_nan.is_monotonic_decreasing


def test_univariate_screen_ks_stat_present_and_bounded_0_1():
    rng = np.random.default_rng(5)
    n = 60
    y = pd.Series([1] * 30 + [0] * 30, dtype=np.int8)
    X = pd.DataFrame({"feat": rng.normal(0.0, 1.0, n)})

    scr = univariate_screen(X, y)

    ks = scr.loc["feat", "ks_stat"]
    assert not math.isnan(ks)
    assert 0.0 <= ks <= 1.0


def test_univariate_screen_auc_ci_bounds_consistent_with_se():
    rng = np.random.default_rng(31)
    n = 100
    y = pd.Series([1] * n + [0] * n, dtype=np.int8)
    X = pd.DataFrame({"feat": rng.normal(0.0, 1.0, 2 * n)})

    scr = univariate_screen(X, y)
    row = scr.loc["feat"]

    assert row["auc_se"] > 0.0
    assert row["auc_low95"] < row["auc"] < row["auc_high95"]
    assert row["auc_low95"] == pytest.approx(row["auc"] - 1.96 * row["auc_se"])
    assert row["auc_high95"] == pytest.approx(row["auc"] + 1.96 * row["auc_se"])


# --- screen_charts -------------------------------------------------------------


def test_screen_charts_creates_files_returns_paths_and_closes_all_figures(tmp_path):
    rng = np.random.default_rng(6)
    n = 60
    y = pd.Series([1] * 30 + [0] * 30, dtype=np.int8)
    cols = {f"feat{i}": rng.normal(0.0 if i % 2 == 0 else 1.5, 1.0, n) for i in range(5)}
    X = pd.DataFrame(cols)
    scr = univariate_screen(X, y)

    out_dir = tmp_path / "nested" / "charts"  # does not exist yet, not even its parent
    assert not out_dir.exists()

    top_k = 3
    paths = screen_charts(scr, X, y, out_dir, top_k=top_k)

    assert out_dir.exists()
    assert len(paths) == 2 * top_k
    for p in paths:
        p_path = Path(p)
        assert p_path.exists()
        assert p_path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_screen_charts_skips_nan_stat_rows(tmp_path):
    n = 40
    y = pd.Series([1] * 20 + [0] * 20, dtype=np.int8)
    rng = np.random.default_rng(7)
    X = pd.DataFrame({
        "good1": rng.normal(0.0, 1.0, n),
        "good2": rng.normal(0.0, 1.0, n),
        "const": np.full(n, 1.0),
    })
    scr = univariate_screen(X, y)

    paths = screen_charts(scr, X, y, tmp_path / "out", top_k=10)  # top_k exceeds valid-row count

    assert len(paths) == 2 * 2  # only good1/good2 have real stats; const is skipped
    assert not any("const" in p for p in paths)
    assert plt.get_fignums() == []


# --- fit_classifiers -----------------------------------------------------------


def test_fit_classifiers_bundle_keys_shapes_and_freeze_fitted():
    rng = np.random.default_rng(20)
    n = 150
    X_tr = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)})
    y_tr = pd.Series(rng.integers(0, 2, n), dtype=np.int8)

    bundle = fit_classifiers(X_tr, y_tr)

    assert set(bundle.keys()) == {"freeze", "logistic", "gbc"}
    assert isinstance(bundle["freeze"], FreezeStats)
    assert set(bundle["freeze"].stats.keys()) == {"a", "b", "c"}  # really fitted, not left empty

    assert bundle["logistic"].coef_.shape == (1, 3)

    Z = bundle["freeze"].transform(X_tr)
    proba = bundle["gbc"].predict_proba(Z)  # only succeeds on a genuinely fitted estimator
    assert proba.shape == (n, 2)


def test_fit_classifiers_transforms_once_both_models_share_matrix(monkeypatch):
    """Proves "both models see the SAME transformed matrix" (task-4-brief)
    directly: FreezeStats.transform is called EXACTLY once during
    fit_classifiers, so logistic and gbc cannot possibly have been fit on
    two independently-rescaled matrices."""
    rng = np.random.default_rng(21)
    n = 80
    X_tr = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y_tr = pd.Series(rng.integers(0, 2, n), dtype=np.int8)

    call_count = {"n": 0}
    original_transform = FreezeStats.transform

    def counting_transform(self, X):
        call_count["n"] += 1
        return original_transform(self, X)

    monkeypatch.setattr(FreezeStats, "transform", counting_transform)

    fit_classifiers(X_tr, y_tr)

    assert call_count["n"] == 1


# --- eval_classifier -------------------------------------------------------------


def test_eval_classifier_hand_built_fixed_proba_exact_deciles_and_lifts():
    """n=20 -> k=ceil(20/10)=2 rows at each end. p is strictly increasing
    (no ties) so the top-2/bottom-2 rows are unambiguous: indices 18,19
    (p=.95/1.00, y=1,1) and 0,1 (p=.05/.10, y=0,0) respectively."""
    n = 20
    p = np.arange(1, n + 1) / 20.0  # 0.05 .. 1.00, strictly increasing
    y_vals = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    assert len(y_vals) == n
    y = pd.Series(y_vals, dtype=np.int8)
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    freeze = FreezeStats().fit(X)
    bundle = {"freeze": freeze, "fake": _FixedProbaModel(p)}
    freeze_stats_before = copy.deepcopy(bundle["freeze"].stats)

    result = eval_classifier(bundle, "fake", X, y)

    base_rate = float(y_vals.mean())  # 10/20 = 0.5
    assert result["n"] == 20
    assert result["base_rate"] == pytest.approx(base_rate)
    assert result["prec_top_decile"] == pytest.approx(1.0)  # top 2 (idx 18,19) are both y=1
    assert result["prec_bottom_decile"] == pytest.approx(1.0)  # bottom 2 (idx 0,1) are both y=0
    assert result["lift_long"] == pytest.approx(1.0 - base_rate)
    assert result["lift_short"] == pytest.approx(1.0 - (1.0 - base_rate))

    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_vals, p))
    assert result["acc"] == pytest.approx(accuracy_score(y_vals, (p >= 0.5).astype(int)))

    assert bundle["freeze"].stats == freeze_stats_before  # eval_classifier never refits


def test_eval_classifier_tie_break_is_stable_and_deterministic():
    n = 20
    p = np.full(n, 0.5)  # every row tied -- worst case for tie-break stability
    y_vals = np.array([1] * 10 + [0] * 10)
    y = pd.Series(y_vals, dtype=np.int8)
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    bundle = {"freeze": FreezeStats().fit(X), "fake": _FixedProbaModel(p)}

    r1 = eval_classifier(bundle, "fake", X, y)
    r2 = eval_classifier(bundle, "fake", X, y)

    assert r1 == r2  # deterministic given identical (fully tied) inputs
    # stable ascending sort of an all-tied array preserves original row
    # order -> bottom-2 = rows [0,1] (y=1,1), top-2 = rows [18,19] (y=0,0)
    assert r1["prec_top_decile"] == pytest.approx(0.0)
    assert r1["prec_bottom_decile"] == pytest.approx(0.0)


def test_eval_classifier_single_class_y_all_nan_except_n():
    n = 6
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})
    y = pd.Series([1] * n, dtype=np.int8)  # single class
    bundle = {"freeze": FreezeStats().fit(X), "fake": _FixedProbaModel(np.linspace(0.1, 0.9, n))}

    result = eval_classifier(bundle, "fake", X, y)

    assert result["n"] == n
    for key in ("roc_auc", "acc", "base_rate", "prec_top_decile", "prec_bottom_decile", "lift_long", "lift_short"):
        assert math.isnan(result[key])


# --- importance_table -------------------------------------------------------------


def test_importance_table_informative_col_ranks_above_noise_and_columns_exact():
    rng = np.random.default_rng(22)
    n = 400
    informative = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = pd.Series((informative > 0).astype(np.int8))
    X = pd.DataFrame({"informative": informative, "noise": noise})
    bundle = fit_classifiers(X, y)

    table = importance_table(bundle, X, y, n_repeats=5)

    assert list(table.columns) == ["feature", "imp_mean", "imp_std"]
    assert list(table["imp_mean"]) == sorted(table["imp_mean"], reverse=True)

    inf_imp = table.loc[table["feature"] == "informative", "imp_mean"].iloc[0]
    noise_imp = table.loc[table["feature"] == "noise", "imp_mean"].iloc[0]
    assert inf_imp > noise_imp


def test_importance_table_deterministic_across_two_calls():
    rng = np.random.default_rng(23)
    n = 250
    informative = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = pd.Series((informative > 0).astype(np.int8))
    X = pd.DataFrame({"informative": informative, "noise": noise})
    bundle = fit_classifiers(X, y)

    t1 = importance_table(bundle, X, y, n_repeats=5)
    t2 = importance_table(bundle, X, y, n_repeats=5)

    pd.testing.assert_frame_equal(t1, t2)


# --- save_bundle / load_bundle -----------------------------------------------------


def test_save_bundle_writes_expected_files_creating_dir(tmp_path):
    rng = np.random.default_rng(10)
    n = 80
    X = pd.DataFrame({"zzz": rng.normal(0, 1, n), "aaa": rng.normal(0, 1, n)})
    y = pd.Series(rng.integers(0, 2, n), dtype=np.int8)
    bundle = fit_classifiers(X, y)

    out_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not out_dir.exists()

    save_bundle(bundle, out_dir)

    for fname in ("freeze.json", "logistic.joblib", "gbc.joblib", "feature_order.json"):
        p = out_dir / fname
        assert p.exists()
        assert p.stat().st_size > 0


def test_load_bundle_empty_dir_raises_actionable_error_naming_all_missing(tmp_path):
    empty_dir = tmp_path / "empty_bundle"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_bundle(empty_dir)

    msg = str(exc_info.value)
    for fname in ("freeze.json", "logistic.joblib", "gbc.joblib", "feature_order.json"):
        assert fname in msg


def test_save_load_bundle_feature_order_round_trip_preserves_column_order(tmp_path):
    """freeze.json ALONE would lose this (FreezeStats.to_json sorts keys
    alphabetically) -- feature_order.json is what makes the round trip
    preserve the ORIGINAL fit order. Deliberately non-alphabetical column
    names so an alphabetization bug would be caught, not masked."""
    rng = np.random.default_rng(11)
    n = 80
    X = pd.DataFrame({"zzz": rng.normal(0, 1, n), "mmm": rng.normal(0, 1, n), "aaa": rng.normal(0, 1, n)})
    y = pd.Series(rng.integers(0, 2, n), dtype=np.int8)
    bundle = fit_classifiers(X, y)
    original_order = list(bundle["freeze"].stats.keys())
    assert original_order == ["zzz", "mmm", "aaa"]  # sanity: genuinely non-alphabetical

    out_dir = tmp_path / "bundle"
    save_bundle(bundle, out_dir)
    loaded = load_bundle(out_dir)

    assert list(loaded["freeze"].stats.keys()) == original_order


def test_save_load_bundle_predictions_identical_after_round_trip(tmp_path):
    rng = np.random.default_rng(12)
    n = 200
    X = pd.DataFrame({"zzz": rng.normal(0, 1, n), "mmm": rng.normal(0, 1, n), "aaa": rng.normal(0, 1, n)})
    y = pd.Series(rng.integers(0, 2, n), dtype=np.int8)
    bundle = fit_classifiers(X, y)

    out_dir = tmp_path / "bundle"
    save_bundle(bundle, out_dir)
    loaded = load_bundle(out_dir)

    Z_orig = bundle["freeze"].transform(X)
    Z_loaded = loaded["freeze"].transform(X)
    pd.testing.assert_frame_equal(Z_orig, Z_loaded)

    for name in ("logistic", "gbc"):
        proba_orig = bundle[name].predict_proba(Z_orig)
        proba_loaded = loaded[name].predict_proba(Z_loaded)
        np.testing.assert_allclose(proba_orig, proba_loaded)
