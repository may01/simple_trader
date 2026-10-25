import numpy as np
import pandas as pd

from experiments.cb_class_bounds import replay as R


def test_candidates_long_zone_and_overshoot():
    idx = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    df = pd.DataFrame({"1_low": [10.0, 9.0, 9.04, 8.0, 9.5], "1_high": [11.0] * 5}, index=idx)
    b = {"cb_high": pd.Series(10.0, index=idx), "cb_low": pd.Series(9.0, index=idx),
         "std_low": pd.Series(0.5, index=idx), "std_high": pd.Series(0.5, index=idx)}
    c = R.candidates(df, b, 15, "long")
    # zone_long = 9 + 0.05 = 9.05: rows 1, 2, 3 in zone; row 3 overshoots (1.0 / 0.5 = 2 > thr) → vetoed
    assert c.tolist() == [False, True, True, False, False]


def test_paired_bootstrap_sign():
    v = {1: 5.0, 2: 5.0, 3: 5.0}
    u = {1: 1.0, 2: 1.0, 4: 1.0}
    s, lo, hi, p = R.paired(v, u)
    assert s == 12.0 and p < 0.05
