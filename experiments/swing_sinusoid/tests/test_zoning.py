import numpy as np
import pandas as pd

from experiments.swing_sinusoid.zoning import classify, time_left_bucket, score_classes
from experiments.swing_sinusoid.apex import label_series


def test_classify_rules():
    pred = pd.DataFrame({
        "ok": [True, True, True, False, True],
        "amp": [2.0, 2.0, 0.5, 2.0, 2.0],
        "resid_std": [1.0, 1.0, 1.0, 1.0, 1.0],
        "period_fit": [20, 20, 20, 20, 100],
        "sign_pred": [1, -1, 1, 1, -1],
    })
    cls = classify(pred, a_min=1.0, p_lo=10, p_hi=40)
    assert list(cls) == [1, -1, 0, 0, 0]


def test_time_left_bucket():
    assert list(time_left_bucket(np.array([0.3, 1.5, 2.0, 4.0, 9.0]))) == ["1", "1", "2", "3-4", "5+"]


def test_score_classes_perfect_predictor_has_lift_above_one():
    n = 400
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 20)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    frame = pd.DataFrame({"s": y, "close": 100 + y}, index=idx)
    labels, _ = label_series(frame, "s", 0.5)
    # oracle: class = direction of the series over the next 2 candles
    fdir = labels["fwd_dir_2"].to_numpy()
    rows = np.arange(n - 5)
    pred = pd.DataFrame({"row": rows, "ok": True, "amp": 1.0, "resid_std": 0.1, "period_fit": 20.0,
                         "sign_pred": fdir[rows], "dt_pred": 3.0})
    cls = classify(pred, 1.0, 10, 40)
    tab = score_classes(pred, cls, labels, frame, 15, horizons=(2,))
    up = tab[(tab["class"] == "up") & (tab["k"] == 2)].iloc[0]
    assert up["ind_hit"] == 1.0 and up["ind_lift"] > 1.5 and up["n"] > 100
    assert up["ind_hit"] > up["ind_perm95"]
