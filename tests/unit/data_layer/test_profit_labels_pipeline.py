"""Integration tests: labels config → DataPreparer._compute_profit_labels →
label columns on the wide frame matching direct labels.py calls."""

import numpy as np
import pandas as pd
import pytest


def _make_wide_df(n_minutes: int = 16 * 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_minutes, freq="1min", tz="UTC")
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n_minutes))
    raw = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": np.full(n_minutes, 100.0),
            "taker_base_vol": np.full(n_minutes, 50.0),
        },
        index=idx,
    )
    from data import _build_wide_df
    return _build_wide_df(raw)


def _write_labels_yaml(tmp_path, body: str) -> str:
    path = tmp_path / "indicators_config.yaml"
    path.write_text("fields: []\n" + body)
    return str(path)


def _make_preparer(config_path: str):
    from training.data_preparer import DataPreparer
    dp = DataPreparer(config_path, "unused.pkl", "unused_attrs.pkl")
    dp.num_workers = 1
    return dp


def test_compute_profit_labels_appends_matching_columns(tmp_path):
    """profit spec → plong/pshort columns equal to direct labels.py output."""
    config = _write_labels_yaml(tmp_path, """
labels:
  - type: profit
    tfs: [15]
    n: 1
    m: 1
    x: 0.4
    atr_period: 2
""")
    df = _make_wide_df()
    expected_df = df.copy()

    dp = _make_preparer(config)
    dp._compute_profit_labels(df)

    assert "15_plong_n1_m1_x0p4" in df.columns
    assert "15_pshort_n1_m1_x0p4" in df.columns

    from indicators.labels import profit_long, profit_short
    pd.testing.assert_series_equal(
        df["15_plong_n1_m1_x0p4"],
        profit_long(expected_df, 15, 1, 1, 0.4, atr_period=2),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        df["15_pshort_n1_m1_x0p4"],
        profit_short(expected_df, 15, 1, 1, 0.4, atr_period=2),
        check_names=False,
    )
    # Sanity: at least one labeled row
    assert df["15_plong_n1_m1_x0p4"].notna().any()


def test_compute_profit_labels_strict_spec(tmp_path):
    """profit_strict spec → pslong/psshort columns appended."""
    config = _write_labels_yaml(tmp_path, """
labels:
  - type: profit_strict
    tfs: [15]
    n: 1
    m: 1
    x: 0.4
    l: 15
    y: 0.4
    atr_period: 2
""")
    df = _make_wide_df()
    expected_df = df.copy()

    dp = _make_preparer(config)
    dp._compute_profit_labels(df)

    assert "15_pslong_n1_m1_x0p4_l15_y0p4" in df.columns
    assert "15_psshort_n1_m1_x0p4_l15_y0p4" in df.columns

    from indicators.labels import profit_strict_long
    pd.testing.assert_series_equal(
        df["15_pslong_n1_m1_x0p4_l15_y0p4"],
        profit_strict_long(expected_df, 15, 1, 1, 0.4, 15, 0.4, atr_period=2),
        check_names=False,
    )


def test_compute_profit_labels_multiple_tfs(tmp_path):
    """One spec with several tfs labels each of them."""
    config = _write_labels_yaml(tmp_path, """
labels:
  - type: profit
    tfs: [15, 60]
    n: 1
    m: 1
    x: 0.4
    atr_period: 2
""")
    df = _make_wide_df()
    dp = _make_preparer(config)
    dp._compute_profit_labels(df)

    assert "15_plong_n1_m1_x0p4" in df.columns
    assert "60_plong_n1_m1_x0p4" in df.columns


def test_compute_profit_labels_no_section_noop(tmp_path):
    """Empty labels config → frame unchanged."""
    config = _write_labels_yaml(tmp_path, "")
    df = _make_wide_df()
    cols_before = list(df.columns)

    dp = _make_preparer(config)
    dp._compute_profit_labels(df)

    assert list(df.columns) == cols_before
