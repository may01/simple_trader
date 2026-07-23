"""Smoke tests for tests/conftest.py's make_slim()/synthetic_slim_df fixture
(Task 0 of the trend_detection experiment).

Exercises the sys.path wiring (bare ``import tdlib``), the SLIM frame's row
count / determinism / is_closed boundary fractions / label-column presence
and NaN share, and the ``set_labels()`` test helper. See conftest.py's own
docstring and ``make_slim()``'s docstring for the full column/semantics spec
every later trend_detection task's tests build on.

The expected label-column names below are copied VERBATIM from the Task 0
brief (not re-derived via conftest.py's own private naming helpers), so this
file independently verifies conftest.py's output matches the brief's spec
rather than only checking internal self-consistency.
"""

from __future__ import annotations

import pandas as pd

import tdlib  # proves conftest.py's sys.path wiring resolves the bare top-level import

from .conftest import make_slim, set_labels

_EXPECTED_LABEL_COLUMNS = [
    # tf=15
    "15_pslong_n1_m1_x0p3_l15_y0p2",
    "15_psshort_n1_m1_x0p3_l15_y0p2",
    "15_pslong_n2_m1_x0p3_l15_y0p2",
    "15_psshort_n2_m1_x0p3_l15_y0p2",
    "15_plong_n1_m1_x0p3",
    "15_pshort_n1_m1_x0p3",
    # tf=60
    "60_pslong_n1_m1_x0p2_l15_y0p1",
    "60_psshort_n1_m1_x0p2_l15_y0p1",
    "60_pslong_n2_m1_x0p2_l15_y0p1",
    "60_psshort_n2_m1_x0p2_l15_y0p1",
    "60_plong_n1_m1_x0p2",
    "60_pshort_n1_m1_x0p2",
    # tf=240
    "240_pslong_n1_m1_x0p1_l15_y0p1",
    "240_psshort_n1_m1_x0p1_l15_y0p1",
    "240_pslong_n2_m1_x0p1_l15_y0p1",
    "240_psshort_n2_m1_x0p1_l15_y0p1",
    "240_plong_n1_m1_x0p1",
    "240_pshort_n1_m1_x0p1",
]


def test_tdlib_bare_import_works():
    """`import tdlib` (bare, no `trend_detection.` prefix) must resolve --
    this is the whole point of conftest.py's sys.path wiring."""
    assert tdlib.__name__ == "tdlib"


def test_fixture_builds(synthetic_slim_df):
    assert len(synthetic_slim_df) > 0


def test_synthetic_slim_df_matches_make_slim_defaults(synthetic_slim_df):
    pd.testing.assert_frame_equal(synthetic_slim_df, make_slim())


def test_row_count_is_days_times_96():
    df = make_slim(seed=0, days=12)
    assert len(df) == 12 * 96 == 1152


def test_row_count_scales_with_days():
    df = make_slim(seed=0, days=3)
    assert len(df) == 3 * 96


def test_make_slim_is_deterministic():
    df1 = make_slim(0)
    df2 = make_slim(0)
    pd.testing.assert_frame_equal(df1, df2)


def test_is_closed_counts():
    df = make_slim(seed=0, days=12)
    n = len(df)
    assert df["15_is_closed"].sum() == n  # every row IS a 15m close, by construction
    assert df["60_is_closed"].sum() == n // 4
    assert df["240_is_closed"].sum() == n // 16
    assert df["1440_is_closed"].sum() == n // 96


def test_rsi_ma8_diff_has_both_strong_tails():
    df = make_slim(seed=0, days=12)
    diff = df["15_rsi_ma8_diff"]
    assert (diff < -1.36).any()
    assert (diff > 1.36).any()


def test_label_columns_exist_with_plausible_nan_share():
    df = make_slim(seed=0, days=12)
    for col in _EXPECTED_LABEL_COLUMNS:
        assert col in df.columns, col
        nan_share = df[col].isna().mean()
        assert 0.01 <= nan_share <= 0.15, (col, nan_share)


def test_set_labels_round_trip():
    df = make_slim(seed=0, days=12)
    positions = [0, 1, 2]

    set_labels(df, 15, positions, long_val=1.0, short_val=0.0)
    assert (df.iloc[positions]["15_pslong_n1_m1_x0p3_l15_y0p2"] == 1.0).all()
    assert (df.iloc[positions]["15_psshort_n1_m1_x0p3_l15_y0p2"] == 0.0).all()

    # Nothing else changed: compare the whole frame against a pristine,
    # independently-built reference with only the same two cells patched in.
    # (A spot-check like "the n2 column isn't all 1.0" would be a weak,
    # luck-dependent assertion -- roughly a 1-in-9 chance three random 0/1
    # draws are all 1.0 -- so this instead proves equality everywhere except
    # the two intentionally-mutated columns.)
    pristine = make_slim(seed=0, days=12)
    pristine.iloc[positions, pristine.columns.get_loc("15_pslong_n1_m1_x0p3_l15_y0p2")] = 1.0
    pristine.iloc[positions, pristine.columns.get_loc("15_psshort_n1_m1_x0p3_l15_y0p2")] = 0.0
    pd.testing.assert_frame_equal(df, pristine)
