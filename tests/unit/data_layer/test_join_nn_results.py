"""Unit tests for data.join_nn_results — consumer-side left-join of df_with_nn.pkl.

The NN inference batch writes {dataset_dir}/df_with_nn.pkl (timeframe-agnostic
nn_res_* columns on the 1-min index). Consumers left-join it at construction;
the canonical df_with_indicators.pkl is never mutated, and absence is a no-op.
"""

import os

import numpy as np
import pandas as pd
import pytest

from data import join_nn_results, _build_wide_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_df():
    np.random.seed(7)
    idx = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    raw = pd.DataFrame(
        {
            "open": np.random.uniform(100, 110, 120),
            "high": np.random.uniform(110, 120, 120),
            "low": np.random.uniform(90, 100, 120),
            "close": np.random.uniform(100, 110, 120),
            "volume": np.random.uniform(1, 100, 120),
            "taker_base_vol": np.random.uniform(0.5, 50, 120),
        },
        index=idx,
    )
    return _build_wide_df(raw)


def _nn_df(idx):
    return pd.DataFrame(
        {
            "nn_res_dir15_prob_up": np.linspace(0.0, 1.0, len(idx)),
            "nn_res_dir15_prob_down": np.linspace(1.0, 0.0, len(idx)),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_present_adds_nn_res_columns(tmp_path, wide_df):
    """When df_with_nn.pkl exists, its nn_res_* columns appear after the join."""
    nn = _nn_df(wide_df.index)
    nn.to_pickle(str(tmp_path / "df_with_nn.pkl"))

    out = join_nn_results(wide_df, str(tmp_path))

    assert "nn_res_dir15_prob_up" in out.columns
    assert "nn_res_dir15_prob_down" in out.columns
    # Values are index-aligned (left-join, every original row kept)
    assert len(out) == len(wide_df)
    assert out["nn_res_dir15_prob_up"].iloc[0] == pytest.approx(0.0)
    assert out["nn_res_dir15_prob_up"].iloc[-1] == pytest.approx(1.0)


def test_absent_is_noop(tmp_path, wide_df):
    """When df_with_nn.pkl is absent, df is returned unchanged (no nn_res_* cols)."""
    out = join_nn_results(wide_df, str(tmp_path))
    assert not any(c.startswith("nn_res_") for c in out.columns)
    assert list(out.columns) == list(wide_df.columns)


def test_join_does_not_overwrite_existing_columns(tmp_path, wide_df):
    """The join must not clobber existing df columns even if the pkl carries one."""
    nn = _nn_df(wide_df.index)
    # Sneak in a column that collides with an existing wide-df column.
    nn["1_close"] = -999.0
    nn.to_pickle(str(tmp_path / "df_with_nn.pkl"))

    original_close = wide_df["1_close"].copy()
    out = join_nn_results(wide_df, str(tmp_path))

    pd.testing.assert_series_equal(out["1_close"], original_close)
    assert "nn_res_dir15_prob_up" in out.columns


def test_does_not_mutate_indicators_pkl_on_disk(tmp_path, wide_df):
    """Constructing the join never rewrites df_with_indicators.pkl (mtime stable)."""
    ind_path = tmp_path / "df_with_indicators.pkl"
    wide_df.to_pickle(str(ind_path))
    mtime_before = os.path.getmtime(ind_path)

    nn = _nn_df(wide_df.index)
    nn.to_pickle(str(tmp_path / "df_with_nn.pkl"))

    loaded = pd.read_pickle(str(ind_path))
    _ = join_nn_results(loaded, str(tmp_path))

    assert os.path.getmtime(ind_path) == mtime_before, (
        "join_nn_results must not rewrite df_with_indicators.pkl"
    )
