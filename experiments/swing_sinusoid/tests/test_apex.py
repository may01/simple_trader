import numpy as np
import pandas as pd

from experiments.swing_sinusoid.apex import find_apexes, label_rows, label_series


def _sine(n=400, period=20.0, noise=0.0, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sin(2 * np.pi * t / period) + noise * rng.normal(size=n)


def test_find_apexes_clean_sine_alternates_and_hits_true_extrema():
    y = _sine()
    idx, sgn = find_apexes(y, prominence=0.5, distance=2)
    assert np.all(np.diff(sgn) != 0)              # alternating
    assert np.all(np.abs(np.diff(idx) - 10) <= 1)  # half period = 10
    assert sgn[0] == 1 and idx[0] == 5           # first high at t=5


def test_find_apexes_noisy_sine_keeps_one_extremum_per_half_swing():
    y = _sine(noise=0.15)
    idx, sgn = find_apexes(y, prominence=0.5 * np.std(y), distance=2)
    assert np.all(np.diff(sgn) != 0)
    # 400 / 10 = 40 half swings, allow edge loss
    assert 34 <= len(idx) <= 44  # ≤ 10 % spurious pairs from noise at prominence 0.5 σ


def test_label_rows_next_apex_and_forward_targets():
    y = _sine()
    close = 100 + np.arange(len(y)) * 0.1
    idx, sgn = find_apexes(y, prominence=0.5)
    lab = label_rows(y, close, idx, sgn, horizons=(1, 3))
    # row 0: next apex is the high at 5
    assert lab.loc[0, "next_apex_dt"] == 5 and lab.loc[0, "next_apex_sign"] == 1
    assert np.isclose(lab.loc[0, "next_apex_val"], y[5])
    # row 5 (the high itself): next apex is the low at 15
    assert lab.loc[5, "next_apex_dt"] == 10 and lab.loc[5, "next_apex_sign"] == -1
    assert lab.loc[5, "prev_apex_dt"] == 0 and lab.loc[5, "prev_apex_sign"] == 1
    assert np.isnan(lab.loc[0, "prev_apex_dt"])
    # forward direction: rising at t=0, falling right after the high
    assert lab.loc[0, "fwd_dir_1"] == 1 and lab.loc[6, "fwd_dir_1"] == -1
    assert np.isclose(lab.loc[0, "fwd_ret_3"], close[3] / close[0] - 1)
    assert np.isnan(lab.loc[len(y) - 1, "fwd_dir_1"])
    assert np.isnan(lab.loc[len(y) - 1, "next_apex_dt"])


def test_label_series_meta():
    y = _sine()
    frame = pd.DataFrame({"s": y, "close": 100 + 0 * y},
                         index=pd.date_range("2024-01-01", periods=len(y), freq="15min", tz="UTC"))
    lab, meta = label_series(frame, "s", prominence_frac=0.5)
    assert len(lab) == len(frame) and lab.index.equals(frame.index)
    assert meta["n_apex"] >= 38 and abs(meta["half_swing_median"] - 10) <= 1
