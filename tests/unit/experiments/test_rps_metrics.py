import math
import warnings

import numpy as np
import pandas as pd
import pytest

from experiments.rsi_params_selection.metrics import (
    evaluate, eta_squared, mutual_information, spearman,
)


def test_mi_zero_when_independent_and_positive_when_dependent():
    rng = np.random.default_rng(2)
    cls = rng.integers(-2, 3, 20_000)
    y_ind = rng.integers(0, 2, 20_000).astype(float)
    y_dep = (cls > 0).astype(float)
    assert mutual_information(cls, y_ind) == pytest.approx(0.0, abs=0.001)
    assert mutual_information(cls, y_dep) > 0.5


def test_eta_squared_bounds():
    cls = np.array([-1, -1, 0, 0, 1, 1])
    assert eta_squared(cls, np.array([0, 0, 0, 1, 1, 1.0])) > 0.5
    assert eta_squared(cls, np.array([1, 1, 1, 1, 1, 1.0])) == 0.0


def test_spearman_perfect_and_inverse():
    a = np.array([1.0, 2, 3, 4, 5])
    assert spearman(a, a * 10) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def _labels_frame(cls):
    """Long fires on positive classes, short on negative — clean separation."""
    rng = np.random.default_rng(3)
    n = len(cls)
    frame = {}
    for kind in ("plain", "strict"):
        for hn in (1, 2):
            p_long = np.where(cls > 0, 0.6, 0.1)
            p_short = np.where(cls < 0, 0.6, 0.1)
            frame[f"{kind}_n{hn}_long"] = (rng.random(n) < p_long).astype(float)
            frame[f"{kind}_n{hn}_short"] = (rng.random(n) < p_short).astype(float)
    return pd.DataFrame(frame)


def test_evaluate_shape_and_separation():
    rng = np.random.default_rng(4)
    cls = rng.integers(-2, 3, 50_000)
    res = evaluate(cls, _labels_frame(cls), 5)
    assert res["n_rows"] == 50_000
    assert sum(res["population"].values()) == 50_000
    lab = res["labels"]["strict_n1_long"]
    assert lab["rate"]["2"] > lab["rate"]["-2"]
    assert lab["lift"]["2"] > 1.0 > lab["lift"]["-2"]
    assert lab["mi"] > 0.05
    # n_classes = 5, K x 2 contingency table -> (K-1)/(2 N ln 2) bits
    expected_floor = 4 / (2 * lab["n"] * math.log(2))
    assert lab["mi_null_floor"] == pytest.approx(expected_floor)
    for other in res["labels"].values():
        assert "mi_null_floor" in other
    pair = res["pairs"]["strict_n1"]
    assert pair["spread"]["2"] > 0 > pair["spread"]["-2"]
    assert pair["monotonicity_rho"] > 0.8
    assert pair["flip"] is None  # 5-class


def test_evaluate_flip_detection_7class():
    rng = np.random.default_rng(5)
    cls = rng.integers(-3, 4, 80_000)
    n = len(cls)
    # continuation up to |2|, reversal in |3| (extra) classes
    p_long = np.select([cls == 3, cls > 0, cls == -3], [0.1, 0.6, 0.6], 0.1)
    p_short = np.select([cls == -3, cls < 0, cls == 3], [0.1, 0.6, 0.6], 0.1)
    frame = {}
    for kind in ("plain", "strict"):
        for hn in (1, 2):
            frame[f"{kind}_n{hn}_long"] = (rng.random(n) < p_long).astype(float)
            frame[f"{kind}_n{hn}_short"] = (rng.random(n) < p_short).astype(float)
    res = evaluate(cls, pd.DataFrame(frame), 7)
    flip = res["pairs"]["strict_n1"]["flip"]
    assert flip["pos"]["sign_flip"] is True   # +2 long-dominant, +3 short-dominant
    assert flip["neg"]["sign_flip"] is True
    assert flip["pos"]["strong"] > 0 > flip["pos"]["extra"]


def test_evaluate_handles_nan_labels_and_empty_class():
    cls = np.zeros(100, dtype=int)  # only neutral class populated
    frame = _labels_frame(cls)
    frame.iloc[:50, frame.columns.get_loc("strict_n1_long")] = np.nan
    res = evaluate(cls, frame, 5)
    assert res["labels"]["strict_n1_long"]["n"] == 50
    assert res["labels"]["strict_n1_long"]["rate"]["2"] is None  # empty class
    assert res["population"]["2"] == 0
    # Regression: monotonicity_rho must be None (not NaN) when degenerate
    assert res["pairs"]["strict_n1"]["monotonicity_rho"] is None
    expected_floor = 4 / (2 * 50 * math.log(2))
    assert res["labels"]["strict_n1_long"]["mi_null_floor"] == pytest.approx(expected_floor)


def test_mi_null_floor_none_when_label_fully_nan():
    cls = np.zeros(100, dtype=int)
    frame = _labels_frame(cls)
    frame["strict_n1_long"] = np.nan
    # mi()/eta_squared() on the resulting empty (0-row) arrays hit pandas'/
    # numpy's own empty-slice RuntimeWarnings (pre-existing, unrelated to
    # mi_null_floor) — not under test here, so silence them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = evaluate(cls, frame, 5)
    lab = res["labels"]["strict_n1_long"]
    assert lab["n"] == 0
    assert lab["mi_null_floor"] is None
