"""Unit tests for scripts/nn_ps5_materialize.py — the one-shot TF5
profit_strict label column materializer.

add_profit_strict_labels(wide_df, tf=5, ...) reads 1_high/1_low (the 1-minute
high/low arrays used as the entry/forward-window source) and the precomputed
{tf}_atr_{atr_period}_ma_{ma_length} column (here 5_atr_14_ma_5, the defaults)
for sizing — NOT 5_close/5_high/5_low. See indicators/labels.py::_entry_state
and _forward_labels.
"""
import numpy as np
import pandas as pd

from scripts.nn_ps5_materialize import add_ps5_labels

EXPECTED = [
    "5_pslong_n1_m1_x0p3_l15_y0p2", "5_psshort_n1_m1_x0p3_l15_y0p2",
    "5_pslong_n2_m1_x0p3_l15_y0p2", "5_psshort_n2_m1_x0p3_l15_y0p2",
]


def _synthetic_wide(rows=400):
    # add_profit_strict_labels(tf=5) reads 1_high/1_low (1-minute entry/forward
    # source) and 5_atr_14_ma_5 (precomputed volatility sizing column) —
    # NOT 5_close/5_high/5_low.
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.1, rows))
    df = pd.DataFrame(index=idx)
    df["1_high"] = mid + np.abs(rng.normal(0, 0.05, rows)) + 0.05
    df["1_low"] = mid - np.abs(rng.normal(0, 0.05, rows)) - 0.05
    df["5_atr_14_ma_5"] = 0.5
    return df


def test_add_ps5_labels_adds_four_columns():
    df = _synthetic_wide()
    added = add_ps5_labels(df)
    assert set(EXPECTED).issubset(df.columns)
    assert set(added) == set(EXPECTED)


def test_add_ps5_labels_idempotent():
    df = _synthetic_wide()
    add_ps5_labels(df)
    cols_before = list(df.columns)
    add_ps5_labels(df)  # second call must not duplicate or error
    assert list(df.columns) == cols_before
