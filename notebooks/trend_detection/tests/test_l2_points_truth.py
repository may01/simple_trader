"""Layer 2 tests -- tdlib.points (strong-point selection on the SLIM frame)
and tdlib.truth (profit_strict ground-truth labeling + forward-return
robustness). See task-2-brief.md.

Filename carries "l2" so `pytest -k l2` selects every test in this module
(mirrors test_l0_conftest.py / test_l1_extract.py's own "l0"/"l1" naming).

Cut values are always read from `tdlib.config.MOVE_CUTS` (never retyped as
literals) -- see task-1-report.md: MOVE_CUTS was corrected in Task 1 to the
verified source-file values, which diverge from any older brief text
starting around the 6th decimal digit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tdlib.config import LABEL_COLS, MOVE_CUTS
from tdlib.points import crosscheck_baked, strong_points, sym0_class
from tdlib.truth import fwd_log_return, mark_truth, robustness_agreement, truth_counts

from .conftest import make_slim, set_labels

# --- integration: L1 -> L2 -> L3 boundary -----------------------------------


def test_points_truth_feed_features():
    """strong_points(+2) + strong_points(-2) + mark_truth chained together on
    6 fully-engineered tf=60 rows (3 forced to class +2, 3 forced to class
    -2), labeled to a known 2 long / 2 short / 1 both / 1 neither
    distribution. Pins the L2 side of the L2->L3 boundary: L3's
    feature_matrix (next task) is expected to keep exactly the "long"/
    "short" marked rows -- 4 of these 6.
    """
    df = make_slim(seed=21, days=6)
    cuts = MOVE_CUTS[60]

    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    plus_positions = closed_positions[:3].tolist()
    minus_positions = closed_positions[3:6].tolist()

    # Neutralize the WHOLE column first (well within every tf's cuts) so no
    # other row -- make_slim's random rsi_ma8_diff has real mass beyond
    # +-1.36 by design (test_l0_conftest.py) -- contaminates the
    # exact-row-identity assertions below.
    diff_col = df.columns.get_loc("60_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0
    df.iloc[plus_positions, diff_col] = cuts[3] + 1.0  # class +2
    df.iloc[minus_positions, diff_col] = cuts[0] - 1.0  # class -2

    p_long1, p_long2, p_both = plus_positions
    p_short1, p_short2, p_neither = minus_positions

    set_labels(df, 60, [p_long1, p_long2], long_val=1.0, short_val=0.0)
    set_labels(df, 60, [p_both], long_val=1.0, short_val=1.0)
    set_labels(df, 60, [p_short1, p_short2], long_val=0.0, short_val=1.0)
    set_labels(df, 60, [p_neither], long_val=0.0, short_val=0.0)

    pts_plus = strong_points(df, 60, 2)
    pts_minus = strong_points(df, 60, -2)

    assert list(pts_plus.index) == list(df.index[plus_positions])
    assert list(pts_minus.index) == list(df.index[minus_positions])

    pts = pd.concat([pts_plus, pts_minus])
    marked = mark_truth(pts, 60)

    assert marked.loc[df.index[p_long1]] == "long"
    assert marked.loc[df.index[p_long2]] == "long"
    assert marked.loc[df.index[p_both]] == "both"
    assert marked.loc[df.index[p_short1]] == "short"
    assert marked.loc[df.index[p_short2]] == "short"
    assert marked.loc[df.index[p_neither]] == "neither"

    assert truth_counts(marked) == {"long": 2, "short": 2, "both": 1, "neither": 1, "nan": 0}

    # X-side contract for L3 (feature_matrix lands next task): the "long"/
    # "short" marked rows are exactly what downstream keeps.
    assert marked.isin(["long", "short"]).sum() == 4


# --- sym0_class ---------------------------------------------------------------


def test_sym0_class_boundaries_nan_and_dtype():
    cuts = MOVE_CUTS[60]
    diff = pd.Series(
        {
            "at_upper_cut": cuts[3],  # right=False: boundary lands upper tier -> +2
            "beyond_upper": cuts[3] + 1.0,  # -> +2
            "at_lower_cut": cuts[0],  # lands index 1 (not 0) -> class -1
            "beyond_lower": cuts[0] - 1.0,  # -> -2
            "center": 0.0,  # dead center -> 0
            "missing": float("nan"),  # NaN -> 0 (never the top bin)
        }
    )

    result = sym0_class(diff, cuts)

    assert result.dtype == np.int8
    assert list(result.index) == list(diff.index)
    assert result["at_upper_cut"] == 2
    assert result["beyond_upper"] == 2
    assert result["at_lower_cut"] == -1
    assert result["beyond_lower"] == -2
    assert result["center"] == 0
    assert result["missing"] == 0


def test_sym0_class_symmetric_minus2_case():
    cuts = MOVE_CUTS[15]
    diff = pd.Series([cuts[0] - 5.0])

    result = sym0_class(diff, cuts)

    assert result.iloc[0] == -2


# --- strong_points --------------------------------------------------------------


def test_strong_points_excludes_non_closed_rows_even_if_diff_extreme():
    df = make_slim(seed=22, days=6)
    cuts = MOVE_CUTS[60]
    diff_col = df.columns.get_loc("60_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0

    closed_pos = int(np.flatnonzero(df["60_is_closed"].to_numpy())[0])
    not_closed_pos = int(np.flatnonzero(~df["60_is_closed"].to_numpy())[0])

    df.iloc[closed_pos, diff_col] = cuts[3] + 1.0
    df.iloc[not_closed_pos, diff_col] = cuts[3] + 1.0  # extreme, but tf's candle isn't closed

    result = strong_points(df, 60, 2)

    assert list(result.index) == [df.index[closed_pos]]
    assert df.index[not_closed_pos] not in result.index


def test_strong_points_side_filter_is_exact():
    df = make_slim(seed=23, days=6)
    cuts = MOVE_CUTS[60]
    diff_col = df.columns.get_loc("60_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0

    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    plus_pos, minus_pos = int(closed_positions[0]), int(closed_positions[1])
    df.iloc[plus_pos, diff_col] = cuts[3] + 1.0  # class +2
    df.iloc[minus_pos, diff_col] = cuts[0] - 1.0  # class -2

    result = strong_points(df, 60, 2)

    assert list(result.index) == [df.index[plus_pos]]
    assert df.index[minus_pos] not in result.index


def test_strong_points_invalid_side_raises_value_error(synthetic_slim_df):
    with pytest.raises(ValueError):
        strong_points(synthetic_slim_df, 60, 1)


def test_strong_points_unknown_tf_raises_key_error(synthetic_slim_df):
    with pytest.raises(KeyError):
        strong_points(synthetic_slim_df, 999, 2)


def test_strong_points_empty_selection_warns_and_returns_empty_frame_with_columns():
    df = make_slim(seed=24, days=6)
    diff_col = df.columns.get_loc("60_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0  # nothing anywhere crosses the cuts

    with pytest.warns(UserWarning) as record:
        result = strong_points(df, 60, 2)

    assert result.empty
    assert list(result.columns) == list(df.columns)
    messages = [str(w.message) for w in record]
    assert any("60" in m and "2" in m for m in messages)


# --- crosscheck_baked ------------------------------------------------------------


def test_crosscheck_baked_returns_nan_when_column_absent():
    df = make_slim(seed=25, days=6)
    assert "60_move_class_sym0" not in df.columns

    result = crosscheck_baked(df, 60)

    assert math.isnan(result)


def test_crosscheck_baked_is_1_when_baked_equals_computed():
    df = make_slim(seed=25, days=6)
    df["60_move_class_sym0"] = sym0_class(df["60_rsi_ma8_diff"], MOVE_CUTS[60])

    result = crosscheck_baked(df, 60)

    assert result == 1.0


def test_crosscheck_baked_flipping_one_closed_row_gives_n_minus_1_over_n():
    df = make_slim(seed=25, days=6)
    df["60_move_class_sym0"] = sym0_class(df["60_rsi_ma8_diff"], MOVE_CUTS[60])

    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    n = len(closed_positions)
    flip_pos = int(closed_positions[0])

    baked_col = df.columns.get_loc("60_move_class_sym0")
    original = df.iloc[flip_pos, baked_col]
    df.iloc[flip_pos, baked_col] = (original + 1) if original < 2 else (original - 1)

    result = crosscheck_baked(df, 60)

    assert result == pytest.approx((n - 1) / n)


# --- mark_truth --------------------------------------------------------------------


def test_mark_truth_all_five_categories():
    df = make_slim(seed=26, days=2)

    set_labels(df, 60, [0], long_val=1.0, short_val=0.0)  # long
    set_labels(df, 60, [1], long_val=0.0, short_val=1.0)  # short
    set_labels(df, 60, [2], long_val=1.0, short_val=1.0)  # both
    set_labels(df, 60, [3], long_val=0.0, short_val=0.0)  # neither
    set_labels(df, 60, [4], long_val=float("nan"), short_val=0.0)  # nan (long missing)

    pts = df.iloc[[0, 1, 2, 3, 4]]
    marked = mark_truth(pts, 60)

    assert marked.tolist() == ["long", "short", "both", "neither", "nan"]
    assert list(marked.index) == list(pts.index)


def test_mark_truth_horizon_n2_reads_n2_columns_not_n1():
    df = make_slim(seed=27, days=2)
    n2_long_col = LABEL_COLS[60]["pslong_n2"]
    n2_short_col = LABEL_COLS[60]["psshort_n2"]

    df.iloc[0, df.columns.get_loc(n2_long_col)] = 1.0
    df.iloc[0, df.columns.get_loc(n2_short_col)] = 0.0  # n2 pair says "long"

    set_labels(df, 60, [0], long_val=0.0, short_val=1.0)  # n1 pair says "short"

    pts = df.iloc[[0]]

    assert mark_truth(pts, 60, horizon="n1").iloc[0] == "short"
    assert mark_truth(pts, 60, horizon="n2").iloc[0] == "long"


def test_mark_truth_unknown_horizon_raises_key_error(synthetic_slim_df):
    with pytest.raises(KeyError):
        mark_truth(synthetic_slim_df, 60, horizon="zz")


# --- truth_counts --------------------------------------------------------------------


def test_truth_counts_all_five_keys_zero_filled():
    marked = pd.Series(["long", "long", "short"])

    assert truth_counts(marked) == {"long": 2, "short": 1, "both": 0, "neither": 0, "nan": 0}


def test_truth_counts_empty_series_all_zero():
    marked = pd.Series([], dtype=object)

    assert truth_counts(marked) == {"long": 0, "short": 0, "both": 0, "neither": 0, "nan": 0}


# --- fwd_log_return --------------------------------------------------------------------


def _tiny_close_frame() -> pd.DataFrame:
    """8 rows, 240_is_closed True at [0,2,3,5,6,7] (6 closed rows) / False
    at [1,4], 240_close hand-picked so consecutive CLOSED rows are each
    +10% over the previous closed row -- log(1.1) at every bars=1 gap
    between consecutive closed rows, by construction. Non-closed rows carry
    an obviously-irrelevant placeholder close (never read by fwd_log_return,
    since those rows are excluded from the closed subseries entirely)."""
    is_closed = [True, False, True, True, False, True, True, True]
    closes = [100.0, 12345.0, 110.0, 121.0, 12345.0, 133.1, 146.41, 161.051]
    return pd.DataFrame({"240_is_closed": is_closed, "240_close": closes}, index=pd.RangeIndex(8))


def test_fwd_log_return_bars1_hand_computed():
    df = _tiny_close_frame()

    fwd1 = fwd_log_return(df, 240, bars=1)

    assert list(fwd1.index) == list(df.index)
    assert fwd1.loc[0] == pytest.approx(np.log(110.0 / 100.0))  # first closed row: 100 -> 110
    assert math.isnan(fwd1.loc[1])  # non-closed row
    assert math.isnan(fwd1.loc[4])  # non-closed row
    assert math.isnan(fwd1.loc[7])  # last closed row: nothing ahead of it


def test_fwd_log_return_bars4_tail_nan_count():
    df = _tiny_close_frame()

    fwd4 = fwd_log_return(df, 240, bars=4)

    closed_positions = df.index[df["240_is_closed"]]  # 6 closed rows
    closed_values = fwd4.loc[closed_positions]

    assert closed_values.isna().sum() == 4
    assert closed_values.notna().sum() == 2


# --- robustness_agreement --------------------------------------------------------------------


def test_robustness_agreement_exact_fractions():
    marked = pd.Series(["long", "long", "long", "short", "short"], index=[0, 1, 2, 3, 4])
    fwd1 = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=[0, 1, 2, 3, 4])
    fwd4 = pd.Series([0.05, 0.04, -0.01, -0.03, -0.02], index=[0, 1, 2, 3, 4])

    result = robustness_agreement(marked, fwd1, fwd4)

    assert result["long_fwd1"] == pytest.approx(2 / 3)  # [0.01, -0.02, 0.03] -> 2 positive
    assert result["long_fwd4"] == pytest.approx(2 / 3)  # [0.05, 0.04, -0.01] -> 2 positive
    assert result["short_fwd1"] == pytest.approx(1 / 2)  # [-0.01, 0.02] -> 1 negative
    assert result["short_fwd4"] == pytest.approx(1.0)  # [-0.03, -0.02] -> 2 negative
    assert result["n_long"] == 3
    assert result["n_short"] == 2


def test_robustness_agreement_excludes_nan_fwd_rows_from_fraction_but_not_n():
    marked = pd.Series(["long", "long", "long"], index=[0, 1, 2])
    fwd1 = pd.Series([0.01, float("nan"), -0.01], index=[0, 1, 2])
    fwd4 = pd.Series([0.01, 0.02, 0.03], index=[0, 1, 2])

    result = robustness_agreement(marked, fwd1, fwd4)

    # row 1's fwd1 is NaN -> excluded from both numerator and denominator:
    # remaining fwd1 values [0.01, -0.01] -> 1/2 positive.
    assert result["long_fwd1"] == pytest.approx(1 / 2)
    # n_long is the unconditional class size, not tied to fwd1's own NaN pattern.
    assert result["n_long"] == 3


def test_robustness_agreement_empty_class_is_nan_not_crash():
    marked = pd.Series(["long", "long"], index=[0, 1])
    fwd1 = pd.Series([0.01, 0.02], index=[0, 1])
    fwd4 = pd.Series([0.01, 0.02], index=[0, 1])

    result = robustness_agreement(marked, fwd1, fwd4)

    assert math.isnan(result["short_fwd1"])
    assert math.isnan(result["short_fwd4"])
    assert result["n_short"] == 0
