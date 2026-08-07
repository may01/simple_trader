import numpy as np
import pandas as pd
import pytest

from experiments.rsi_params_selection import config
from experiments.rsi_params_selection.data_loading import (
    add_labels_for_tfs, extract_closed,
)


def test_label_cols_names_match_baked_columns():
    cols = config.label_cols(60)
    assert cols["plain_n1_long"] == "60_plong_n1_m1_x0p2"
    assert cols["strict_n2_short"] == "60_psshort_n2_m1_x0p2_l15_y0p1"
    cols15 = config.label_cols(15)
    assert cols15["strict_n1_long"] == "15_pslong_n1_m1_x0p3_l15_y0p2"
    cols240 = config.label_cols(240)
    assert cols240["plain_n2_short"] == "240_pshort_n2_m1_x0p1"
    assert len(cols) == 8


def _tiny_wide(n=2000, tf=15):
    """Minimal wide frame labels.py + extract_closed can run on."""
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame(index=idx)
    df["1_high"] = close + 0.05
    df["1_low"] = close - 0.05
    df[f"{tf}_atr_14_ma_5"] = 1.0
    df[f"{tf}_is_closed"] = (np.arange(n) % tf) == (tf - 1)
    for w in config.WINDOWS:
        df[f"{tf}_rsi_ma{w}_diff"] = rng.normal(0, 1, n)
    return df


def test_add_labels_for_tfs_appends_all_8_columns():
    df = _tiny_wide(tf=15)
    add_labels_for_tfs(df, [15], config_path="configs/indicators_config.yaml")
    for col in config.label_cols(15).values():
        assert col in df.columns, col
    lab = df["15_plong_n1_m1_x0p3"]
    vals = lab.dropna().unique()
    assert set(vals).issubset({0.0, 1.0})
    assert lab.tail(5).isna().all()  # future window incomplete at frame end


def test_add_labels_for_tfs_skips_other_tfs():
    df = _tiny_wide(tf=15)
    add_labels_for_tfs(df, [15], config_path="configs/indicators_config.yaml")
    assert not any(c.startswith("60_p") for c in df.columns)


def test_extract_closed_filters_and_renames():
    df = _tiny_wide(n=600, tf=15)
    add_labels_for_tfs(df, [15], config_path="configs/indicators_config.yaml")
    out = extract_closed(df, 15)
    assert len(out) == int(df["15_is_closed"].sum())
    assert set(out.columns) == {"diff8", "diff12", "diff24",
                                *config.label_cols(15).keys()}
    closed_ts = df.index[df["15_is_closed"].astype(bool)]
    assert out.index.equals(closed_ts)
    got = out["diff12"].to_numpy()
    want = df.loc[closed_ts, "15_rsi_ma12_diff"].to_numpy()
    assert np.array_equal(got, want)
