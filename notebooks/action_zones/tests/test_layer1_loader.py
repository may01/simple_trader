"""Layer 1 tests — azlib.loader (read-only wide-df access + profit labels).

Covers: load_wide_df, LabelParams, add_labels, label_col (azlib/loader.py)
and default_label_params (azlib/config.py, Task 1.2). See
.superpowers/sdd/task-1-brief.md.

Filename carries "layer1" so `pytest -k layer1` selects every test in this
module (pytest -k matches against each item's full ancestry of names,
which includes the containing module's basename).
"""
from __future__ import annotations

import pandas as pd
import pytest

from azlib.config import default_label_params
from azlib.loader import LabelParams, add_labels, label_col, load_wide_df


# --- load_wide_df -------------------------------------------------------


def test_load_wide_df_returns_expected_columns(synthetic_wide_df, tmp_path, monkeypatch):
    pkl_path = tmp_path / "df_with_indicators.pkl"
    synthetic_wide_df.to_pickle(pkl_path)
    # loader.py imports wide_df_path with `from helpers import wide_df_path`,
    # so the name to patch is the one bound in azlib.loader's own namespace,
    # not helpers.wide_df_path itself (already-bound reference).
    monkeypatch.setattr("azlib.loader.wide_df_path", lambda: str(pkl_path))

    df = load_wide_df()

    assert isinstance(df, pd.DataFrame)
    for col in ("1_high", "1_low", "1_close"):
        assert col in df.columns
    for tf in (15, 60, 240):
        assert f"{tf}_high" in df.columns
        assert f"{tf}_low" in df.columns
    # "returns the wide df unchanged" — round-tripping through pickle must
    # not alter values/dtypes/index.
    pd.testing.assert_frame_equal(df, synthetic_wide_df)


# --- add_labels -----------------------------------------------------------


def test_add_labels_adds_all_four_columns(synthetic_wide_df):
    p = LabelParams(tf=15, n=1, m=2.0, x=2.0, l=15, y=1.0)
    add_labels(synthetic_wide_df, p)

    expected = {
        label_col(p, "long", strict=True),
        label_col(p, "short", strict=True),
        label_col(p, "long", strict=False),
        label_col(p, "short", strict=False),
    }
    assert len(expected) == 4  # sanity: all four names are distinct
    assert expected <= set(synthetic_wide_df.columns)


def test_add_labels_strict_positives_are_subset_of_non_strict(synthetic_wide_df):
    # Loose m/x so the ~400-row synthetic random walk actually produces some
    # positives in both the strict and non-strict columns for both
    # directions — otherwise the subset assertion below would be vacuously
    # true and this test wouldn't catch a broken implementation.
    p = LabelParams(tf=15, n=1, m=0.3, x=0.3, l=15, y=1.0)
    add_labels(synthetic_wide_df, p)

    saw_a_positive = False
    for direction in ("long", "short"):
        strict = synthetic_wide_df[label_col(p, direction, strict=True)]
        plain = synthetic_wide_df[label_col(p, direction, strict=False)]
        strict_positive = strict == 1.0
        saw_a_positive = saw_a_positive or bool(strict_positive.any())
        # Every strict-positive row must also be positive in the
        # corresponding non-strict column (strict ⊆ non-strict).
        assert (plain[strict_positive] == 1.0).all()

    assert saw_a_positive, (
        "fixture/params produced zero strict positives in either direction — "
        "subset assertion above was vacuous, tighten m/x or the fixture"
    )


def test_add_labels_raises_keyerror_when_atr_column_missing(synthetic_wide_df):
    p = LabelParams(tf=15, n=1, m=2.0, x=2.0, l=15, y=1.0)
    synthetic_wide_df.drop(columns=["15_atr_14_ma_5"], inplace=True)

    with pytest.raises(KeyError, match="15_atr_14_ma_5"):
        add_labels(synthetic_wide_df, p)


# --- label_col --------------------------------------------------------------


def test_label_col_round_trips_suffix_convention():
    # m=2.0 is integer-valued -> "m2"; x=1.5 is not -> "x1p5" (brief's
    # required examples). n/l are plain ints, y=1.0 is integer-valued.
    p = LabelParams(tf=15, n=1, m=2.0, x=1.5, l=15, y=1.0)

    assert label_col(p, "long", strict=True) == "15_pslong_n1_m2_x1p5_l15_y1"
    assert label_col(p, "short", strict=True) == "15_psshort_n1_m2_x1p5_l15_y1"
    assert label_col(p, "long", strict=False) == "15_plong_n1_m2_x1p5"
    assert label_col(p, "short", strict=False) == "15_pshort_n1_m2_x1p5"


def test_label_col_matches_columns_actually_written_by_add_labels(synthetic_wide_df):
    p = LabelParams(tf=60, n=2, m=1.5, x=0.5, l=30, y=1.0)
    before = set(synthetic_wide_df.columns)
    add_labels(synthetic_wide_df, p)
    new_cols = set(synthetic_wide_df.columns) - before

    computed = {
        label_col(p, "long", strict=True),
        label_col(p, "short", strict=True),
        label_col(p, "long", strict=False),
        label_col(p, "short", strict=False),
    }
    assert computed == new_cols


def test_label_col_rejects_bad_direction():
    p = LabelParams(tf=15, n=1, m=2.0, x=2.0, l=15, y=1.0)
    with pytest.raises(ValueError):
        label_col(p, "sideways", strict=True)


# --- azlib.config.default_label_params (Task 1.2) --------------------------


def test_default_label_params_matches_documented_defaults():
    p = default_label_params(tf=60)

    assert p == LabelParams(tf=60, n=1, m=2.0, x=2.0, l=60, y=1.0)
    assert p.atr_period == 14
    assert p.ma_length == 5


# --- integration -> Layer 2 (azlib.space, Task 2) --------------------------


def test_labels_feed_action_space(synthetic_wide_df):
    """Layer 1 -> Layer 2 integration.

    add_labels()'s columns must be consumable by azlib.space.label_coeff
    (Task 2). Task 2 is not built yet in this repo (this task only
    implements azlib/loader.py + azlib/config.py), so this test exercises
    everything Layer 1 owns unconditionally and then skips — rather than
    fails — at the azlib.space import boundary. See task-1-report.md for
    the skip-vs-reduce decision recorded there.
    """
    p = LabelParams(tf=15, n=1, m=2.0, x=2.0, l=15, y=1.0)
    add_labels(synthetic_wide_df, p)
    assert label_col(p, "long", strict=True) in synthetic_wide_df.columns

    azlib_space = pytest.importorskip(
        "azlib.space",
        reason="Task 2 (azlib/space.py, label_coeff) is not implemented yet",
    )
    label_coeff = azlib_space.label_coeff

    lc = label_coeff(synthetic_wide_df, tf=15, direction="long")
    labeled = synthetic_wide_df[label_col(p, "long", strict=True)] == 1.0
    assert lc[labeled].between(0, 1).all()
