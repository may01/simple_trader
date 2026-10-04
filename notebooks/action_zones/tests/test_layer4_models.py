"""Layer 4 tests — azlib.models (regression + classification, design spec §9).

Covers: fit_regression/predict_reg (linear/poly2/gbr), fit_classification/
predict_clf (logistic/gbc), save_result/load_result (JSON+joblib
round-trip), groups (same-space attribute combinatorics), plot_1d/plot_2d
(azlib/models.py). See .superpowers/sdd/task-4-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §9.

Filename carries "layer4" so `pytest -k layer4` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.models existing, so — same
convention as test_layer2_space.py/test_layer3_indicators.py — the whole
module imports it at top level (only the Layer-5 forward integration test
below needs an additional importorskip, for azlib.infer, which is Task 5
and does not exist yet).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from azlib.models import (
    ClfResult,
    RegResult,
    fit_classification,
    fit_regression,
    groups,
    load_result,
    plot_1d,
    plot_2d,
    predict_clf,
    predict_reg,
    save_result,
)


# --- integration -> Layer 5 (azlib.infer, Task 5) ---------------------------


def test_regression_feeds_fusion(rng_matrix):
    """Brief's Layer-5 forward integration test, verbatim in spirit.

    azlib.infer (Task 5, fuse_inverse_variance) is not implemented yet in
    this repo, so — same decision as Task 1/2/3's forward-dep tests — the
    azlib.infer import boundary is guarded with importorskip rather than
    left as a permanent hard failure. Everything Layer 4 owns
    (fit_regression, predict_reg) is exercised unconditionally first.
    """
    X, y = rng_matrix
    r1 = fit_regression(X[:, :1], y, "linear")
    r2 = fit_regression(X[:, 1:2], y, "poly2")
    m1, s1 = predict_reg(r1, X[:, :1])
    m2, s2 = predict_reg(r2, X[:, 1:2])

    azlib_infer = pytest.importorskip(
        "azlib.infer",
        reason="Task 5 (azlib/infer.py, fuse_inverse_variance) is not implemented yet",
    )
    fuse_inverse_variance = azlib_infer.fuse_inverse_variance

    mean, std = fuse_inverse_variance(np.vstack([m1, m2]), np.vstack([s1, s2]))
    assert mean.shape == y.shape and (std >= 0).all()


# --- fixtures local to this module ------------------------------------------


@pytest.fixture
def separable_clf_data() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, mostly-separable-but-not-perfectly 2-class 1D data.

    Two well-separated clusters (class 0 around -3, class 1 around +3) plus
    two explicit overlap points placed so the classes are NOT perfectly
    linearly separable (class-0 point at +0.4 sits ABOVE the class-1 point
    at -0.4) -- this deliberately rules out sklearn's LogisticRegression
    "perfectly separable data" failure mode, where the true MLE has
    unbounded coefficients and lbfgs raises ConvergenceWarning regardless of
    max_iter (see task-4-report.md). With a well-defined finite optimum,
    logistic regression converges normally well within max_iter=1000, and
    the false-positive rate stays near 0 (only the 2 deliberately-placed
    borderline points are anywhere near the boundary).
    """
    rng = np.random.default_rng(7)
    neg = rng.normal(loc=-3.0, scale=1.0, size=(58, 1))
    pos = rng.normal(loc=3.0, scale=1.0, size=(58, 1))
    overlap_neg = np.array([[0.4]])
    overlap_pos = np.array([[-0.4]])
    X = np.vstack([neg, overlap_neg, pos, overlap_pos])
    label = np.array([0] * 59 + [1] * 59)
    return X, label


# --- fit_regression / predict_reg -------------------------------------------


def test_fit_regression_linear_perfect_line_r2_and_resid_std():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 7.0  # exact line, no noise

    res = fit_regression(X, y, "linear")

    assert isinstance(res, RegResult)
    assert res.kind == "linear"
    assert res.metrics["r2"] == pytest.approx(1.0, abs=1e-9)
    assert res.metrics["resid_std"] == pytest.approx(0.0, abs=1e-6)
    assert set(res.metrics) == {"r2", "rmse", "resid_std"}


def test_predict_reg_returns_resid_std_as_std_for_perfect_line():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    y = 3.0 * X.ravel() + 7.0

    res = fit_regression(X, y, "linear")
    mean, std = predict_reg(res, X)

    np.testing.assert_allclose(mean, y, atol=1e-6)
    assert mean.shape == y.shape
    assert std.shape == y.shape
    np.testing.assert_allclose(std, res.metrics["resid_std"])


@pytest.mark.parametrize("kind", ["poly2", "gbr"])
def test_fit_regression_poly2_gbr_fit_predict_without_error(rng_matrix, kind):
    X, y = rng_matrix

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit_regression(X, y, kind)

    assert res.kind == kind
    assert set(res.metrics) == {"r2", "rmse", "resid_std"}
    assert res.metrics["resid_std"] >= 0.0

    mean, std = predict_reg(res, X)
    assert mean.shape == y.shape
    assert std.shape == y.shape
    assert (std >= 0).all()


def test_fit_regression_rejects_unknown_kind(rng_matrix):
    X, y = rng_matrix
    with pytest.raises(ValueError):
        fit_regression(X[:, :1], y, "bogus")


@pytest.fixture
def region_varying_noise_data() -> tuple[np.ndarray, np.ndarray]:
    """1D data where residual noise is much larger in one prediction
    region than another -- the real regression guard for the gbr binned-
    residual-std fix (task-4-brief.md / plan line 329: "residual std by
    binned prediction"). ``y`` tracks ``X`` closely (mean = X), so gbr's
    own in-sample prediction is itself close to ``X`` -- binning by
    PREDICTED value (what predict_reg actually does) therefore reproduces
    this X-region split. Quiet region (X < 5): noise std 0.05. Noisy
    region (X >= 5): noise std 2.0, a 40x difference -- a single GLOBAL
    residual-std scalar (the pre-fix behavior) could not possibly reflect
    this; a per-bin std must.
    """
    rng = np.random.default_rng(11)
    n = 300
    X = rng.uniform(0.0, 10.0, size=(n, 1))
    noise_std = np.where(X[:, 0] < 5.0, 0.05, 2.0)
    y = X[:, 0] + rng.normal(scale=noise_std)
    return X, y


def test_predict_reg_gbr_std_varies_by_prediction_region(region_varying_noise_data):
    X, y = region_varying_noise_data
    res = fit_regression(X, y, "gbr")

    quiet_X = np.array([[1.0], [2.0], [3.0]])
    noisy_X = np.array([[7.0], [8.0], [9.0]])
    _, quiet_std = predict_reg(res, quiet_X)
    _, noisy_std = predict_reg(res, noisy_X)

    # The real guard: std must not be one constant scalar across regions,
    # and must be clearly higher where the training noise actually was.
    assert not np.allclose(quiet_std, noisy_std)
    assert noisy_std.mean() > quiet_std.mean() * 2


def test_save_load_result_gbr_round_trip_identical_predictions_and_std(
    region_varying_noise_data, tmp_path
):
    X, y = region_varying_noise_data
    res = fit_regression(X, y, "gbr")
    path = str(tmp_path / "gbr_result.json")

    save_result(res, path)
    loaded = load_result(path)

    assert loaded.params["bin_edges"] == res.params["bin_edges"]
    assert loaded.params["bin_std"] == res.params["bin_std"]

    mean_orig, std_orig = predict_reg(res, X)
    mean_loaded, std_loaded = predict_reg(loaded, X)
    np.testing.assert_array_equal(mean_orig, mean_loaded)
    np.testing.assert_array_equal(std_orig, std_loaded)


# --- fit_classification / predict_clf ---------------------------------------


@pytest.mark.parametrize("kind", ["logistic", "gbc"])
def test_fit_classification_separable_low_false_positive_rate(separable_clf_data, kind):
    X, label = separable_clf_data

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit_classification(X, label, kind)

    assert isinstance(res, ClfResult)
    assert res.kind == kind
    assert set(res.metrics) == {"false_pos_rate", "accuracy", "auc"}
    assert res.metrics["false_pos_rate"] == pytest.approx(0.0, abs=0.1)
    assert res.metrics["accuracy"] > 0.9


@pytest.mark.parametrize("kind", ["logistic", "gbc"])
def test_predict_clf_in_unit_interval(separable_clf_data, kind):
    X, label = separable_clf_data
    res = fit_classification(X, label, kind)

    proba = predict_clf(res, X)

    assert proba.shape == label.shape
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()


def test_fit_classification_rejects_unknown_kind(separable_clf_data):
    X, label = separable_clf_data
    with pytest.raises(ValueError):
        fit_classification(X, label, "bogus")


# --- groups ------------------------------------------------------------------


def test_groups_rsi_1d_three_singletons():
    assert groups("rsi", 1) == [("position",), ("slope",), ("distance",)]


def test_groups_rsi_2d_three_pairs():
    result = groups("rsi", 2)
    assert len(result) == 3
    assert set(result) == {
        ("position", "slope"),
        ("position", "distance"),
        ("slope", "distance"),
    }


def test_groups_rsi_3d_one_triple():
    assert groups("rsi", 3) == [("position", "slope", "distance")]


def test_groups_ma_3d_empty():
    # ma has only 2 attrs (position, slope) -- no triple exists.
    assert groups("ma", 3) == []


def test_groups_ma_2d_one_pair():
    assert groups("ma", 2) == [("position", "slope")]


def test_groups_ma_1d_two_singletons():
    assert groups("ma", 1) == [("position",), ("slope",)]


# --- save_result / load_result ------------------------------------------------


def test_save_load_result_regression_round_trip(rng_matrix, tmp_path):
    X, y = rng_matrix
    res = fit_regression(X[:, :1], y, "linear")
    path = str(tmp_path / "reg_result.json")

    save_result(res, path)
    loaded = load_result(path)

    assert isinstance(loaded, RegResult)
    assert loaded.kind == res.kind
    assert loaded.params == res.params
    assert loaded.metrics == pytest.approx(res.metrics)

    mean_orig, std_orig = predict_reg(res, X[:, :1])
    mean_loaded, std_loaded = predict_reg(loaded, X[:, :1])
    np.testing.assert_array_equal(mean_orig, mean_loaded)
    np.testing.assert_array_equal(std_orig, std_loaded)


def test_save_load_result_classification_round_trip(separable_clf_data, tmp_path):
    X, label = separable_clf_data
    res = fit_classification(X, label, "logistic")
    path = str(tmp_path / "clf_result.json")

    save_result(res, path)
    loaded = load_result(path)

    assert isinstance(loaded, ClfResult)
    assert loaded.kind == res.kind
    assert loaded.params == res.params
    assert loaded.metrics == pytest.approx(res.metrics)

    proba_orig = predict_clf(res, X)
    proba_loaded = predict_clf(loaded, X)
    np.testing.assert_array_equal(proba_orig, proba_loaded)


# --- plot_1d / plot_2d --------------------------------------------------------


def test_plot_1d_writes_nonempty_png(rng_matrix, tmp_path):
    X, y = rng_matrix
    res = fit_regression(X[:, :1], y, "linear")
    out_path = tmp_path / "plot_1d.png"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_1d(X[:, 0], y, res, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_1d_with_classification_result_writes_nonempty_png(separable_clf_data, tmp_path):
    X, label = separable_clf_data
    res = fit_classification(X, label, "logistic")
    out_path = tmp_path / "plot_1d_clf.png"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_1d(X[:, 0], label.astype(float), res, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_2d_writes_nonempty_png(rng_matrix, tmp_path):
    X, y = rng_matrix
    res = fit_regression(X, y, "poly2")
    out_path = tmp_path / "plot_2d.png"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_2d(X[:, 0], X[:, 1], y, res, str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
