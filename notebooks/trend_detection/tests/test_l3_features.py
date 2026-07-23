"""Layer 3 tests -- tdlib.features (feature engineering: engineered_features,
default_feature_cols, feature_matrix, FreezeStats). See task-3-brief.md.

Filename carries "l3" so `pytest -k l3` selects every test in this module
(mirrors test_l0_conftest.py / test_l1_extract.py / test_l2_points_truth.py's
own "l0"/"l1"/"l2" naming).

Column-name literals that ALSO exist as named constants elsewhere in tdlib
(MOVE_CUTS, LABEL_COLS) are always read from those constants, never retyped
-- same discipline test_l2_points_truth.py documents for MOVE_CUTS.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tdlib.config import LABEL_COLS, MOVE_CUTS
from tdlib.features import FreezeStats, default_feature_cols, engineered_features, feature_matrix

from .conftest import make_slim, set_labels

# --- shared helpers -----------------------------------------------------------


def _pairwise_separation(values: pd.Series, y: pd.Series) -> float:
    """Mann-Whitney U statistic normalized to [0, 1] (a.k.a. AUC): fraction
    of (y==1, y==0) pairs where the y==1 member's value is strictly greater
    than the y==0 member's value. 1.0 = perfect separation (every positive
    ranks above every negative), 0.5 = no separation (coin flip)."""
    pos = values[y == 1].to_numpy()
    neg = values[y == 0].to_numpy()
    comparisons = pos[:, None] > neg[None, :]
    return float(comparisons.mean())


def _force_strong_points(df: pd.DataFrame, tf: int, positions: list) -> None:
    """In place: neutralize ``{tf}_rsi_ma8_diff`` to 0.0 across the WHOLE
    frame, then force it to class +2 at exactly ``positions`` (which must
    already be ``{tf}_is_closed`` rows). So ``strong_points(df, tf, 2)``
    selects exactly ``positions`` -- nothing more, nothing less.

    feature_matrix now takes the full slim frame and runs strong_points
    internally (task-3 fix pass -- see features.py's module docstring), so
    every feature_matrix test needs its target rows to be genuine
    strong-point selections rather than an arbitrary caller-supplied
    ``pts``. This is the shared setup for that."""
    diff_col = df.columns.get_loc(f"{tf}_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0
    df.iloc[positions, diff_col] = MOVE_CUTS[tf][3] + 1.0


# --- integration: L3 -> L4 boundary -------------------------------------------


def test_features_feed_screen_contract():
    """bb_pos_60 engineered to near-perfectly separate a hand-marked
    long/short group on strong tf=60 +2 points; a plain random slim column
    (never touched by this engineering) must NOT separate. Pins the X/y
    contract L4's univariate_screen (next task) will consume: alignment
    between X and y, no leak columns in X, and y's long=1/short=0 encoding.
    """
    df = make_slim(seed=100, days=10)

    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    plus_positions = closed_positions[:20].tolist()
    _force_strong_points(df, 60, plus_positions)  # class +2 (strong up-move)

    long_positions = plus_positions[:10]
    short_positions = plus_positions[10:]

    set_labels(df, 60, long_positions, long_val=1.0, short_val=0.0)
    set_labels(df, 60, short_positions, long_val=0.0, short_val=1.0)

    up_col = df.columns.get_loc("60_bb_upper_20_2")
    lo_col = df.columns.get_loc("60_bb_lower_20_2")
    close_col = df.columns.get_loc("60_close")
    df.iloc[plus_positions, up_col] = 110.0
    df.iloc[plus_positions, lo_col] = 100.0
    df.iloc[long_positions, close_col] = 109.0  # bb_pos_60 = 0.9
    df.iloc[short_positions, close_col] = 101.0  # bb_pos_60 = 0.1

    X, y = feature_matrix(df, 60, 2)

    assert "bb_pos_60" in X.columns
    assert list(X.index) == list(y.index)
    assert set(y.unique().tolist()) <= {0, 1}
    assert len(y) == 20

    sep_signal = _pairwise_separation(X["bb_pos_60"], y)
    assert sep_signal == pytest.approx(1.0)

    noise_col = "60_body_ratio"  # random uniform in the fixture, untouched above
    assert noise_col in X.columns
    sep_noise = _pairwise_separation(X[noise_col], y)
    assert abs(sep_noise - 0.5) < 0.35  # nowhere near the perfect-separation extremes


# --- engineered_features: validation -----------------------------------------


def test_engineered_features_invalid_side_raises_value_error(synthetic_slim_df):
    with pytest.raises(ValueError):
        engineered_features(synthetic_slim_df, 60, 1)


def test_engineered_features_does_not_mutate_input_slim():
    df = make_slim(seed=131, days=2)
    before = df.copy(deep=True)

    engineered_features(df, 60, 2)

    pd.testing.assert_frame_equal(df, before)


# --- bb_pos --------------------------------------------------------------------


def test_bb_pos_degenerate_band_is_nan_not_inf():
    df = make_slim(seed=101, days=2)
    up_col = df.columns.get_loc("60_bb_upper_20_2")
    lo_col = df.columns.get_loc("60_bb_lower_20_2")
    df.iloc[0, up_col] = 100.0
    df.iloc[0, lo_col] = 100.0  # degenerate: upper == lower

    feats = engineered_features(df, 60, 2)

    assert math.isnan(feats["bb_pos_60"].iloc[0])


def test_bb_pos_at_lower_and_upper_bounds():
    df = make_slim(seed=102, days=2)
    up_col = df.columns.get_loc("60_bb_upper_20_2")
    lo_col = df.columns.get_loc("60_bb_lower_20_2")
    close_col = df.columns.get_loc("60_close")

    df.iloc[0, up_col] = 110.0
    df.iloc[0, lo_col] = 100.0
    df.iloc[0, close_col] = 100.0  # == lower -> 0

    df.iloc[1, up_col] = 110.0
    df.iloc[1, lo_col] = 100.0
    df.iloc[1, close_col] = 110.0  # == upper -> 1

    feats = engineered_features(df, 60, 2)

    assert feats["bb_pos_60"].iloc[0] == pytest.approx(0.0)
    assert feats["bb_pos_60"].iloc[1] == pytest.approx(1.0)


# --- oob_* -----------------------------------------------------------------------


def test_oob_strict_inequality_20_2_family():
    df = make_slim(seed=103, days=2)
    up_col = df.columns.get_loc("60_bb_upper_20_2")
    lo_col = df.columns.get_loc("60_bb_lower_20_2")
    close_col = df.columns.get_loc("60_close")

    df.iloc[0, up_col] = 110.0
    df.iloc[0, close_col] = 110.0  # == upper -> not oob
    df.iloc[1, up_col] = 110.0
    df.iloc[1, close_col] = 110.01  # > upper -> oob

    df.iloc[2, lo_col] = 90.0
    df.iloc[2, close_col] = 90.0  # == lower -> not oob
    df.iloc[3, lo_col] = 90.0
    df.iloc[3, close_col] = 89.99  # < lower -> oob

    feats = engineered_features(df, 60, 2)

    assert feats["oob_up2_60"].iloc[0] == 0
    assert feats["oob_up2_60"].iloc[1] == 1
    assert feats["oob_dn2_60"].iloc[2] == 0
    assert feats["oob_dn2_60"].iloc[3] == 1
    assert feats["oob_up2_60"].dtype == np.int8
    assert feats["oob_dn2_60"].dtype == np.int8


def test_oob_strict_inequality_20_3_family():
    df = make_slim(seed=104, days=2)
    up3_col = df.columns.get_loc("60_bb_upper_20_3")
    lo3_col = df.columns.get_loc("60_bb_lower_20_3")
    close_col = df.columns.get_loc("60_close")

    df.iloc[0, up3_col] = 120.0
    df.iloc[0, close_col] = 120.0  # == upper_3 -> not oob
    df.iloc[1, up3_col] = 120.0
    df.iloc[1, close_col] = 120.01  # > upper_3 -> oob

    df.iloc[2, lo3_col] = 80.0
    df.iloc[2, close_col] = 80.0  # == lower_3 -> not oob
    df.iloc[3, lo3_col] = 80.0
    df.iloc[3, close_col] = 79.99  # < lower_3 -> oob

    feats = engineered_features(df, 60, 2)

    assert feats["oob_up3_60"].iloc[0] == 0
    assert feats["oob_up3_60"].iloc[1] == 1
    assert feats["oob_dn3_60"].iloc[2] == 0
    assert feats["oob_dn3_60"].iloc[3] == 1
    assert feats["oob_up3_60"].dtype == np.int8
    assert feats["oob_dn3_60"].dtype == np.int8


# --- bb_dist_* -------------------------------------------------------------------


def test_bb_dist_atr_nan_or_zero_is_nan():
    df = make_slim(seed=105, days=2)
    atr_col = df.columns.get_loc("60_atr_14")
    df.iloc[0, atr_col] = 0.0
    df.iloc[1, atr_col] = float("nan")

    feats = engineered_features(df, 60, 2)

    assert math.isnan(feats["bb_dist_up_60"].iloc[0])
    assert math.isnan(feats["bb_dist_dn_60"].iloc[0])
    assert math.isnan(feats["bb_dist_up_60"].iloc[1])
    assert math.isnan(feats["bb_dist_dn_60"].iloc[1])


def test_bb_dist_hand_value():
    df = make_slim(seed=106, days=2)
    up_col = df.columns.get_loc("60_bb_upper_20_2")
    lo_col = df.columns.get_loc("60_bb_lower_20_2")
    close_col = df.columns.get_loc("60_close")
    atr_col = df.columns.get_loc("60_atr_14")

    df.iloc[0, up_col] = 110.0
    df.iloc[0, lo_col] = 95.0
    df.iloc[0, close_col] = 100.0
    df.iloc[0, atr_col] = 5.0

    feats = engineered_features(df, 60, 2)

    assert feats["bb_dist_up_60"].iloc[0] == pytest.approx((110.0 - 100.0) / 5.0)
    assert feats["bb_dist_dn_60"].iloc[0] == pytest.approx((100.0 - 95.0) / 5.0)


# --- swing_dist_* ----------------------------------------------------------------


def test_swing_dist_shift_excludes_current_bar_new_high_is_negative():
    df = make_slim(seed=107, days=8)
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())
    assert len(closed240) == 48

    high_col = df.columns.get_loc("240_high")
    close_col = df.columns.get_loc("240_close")
    atr_col = df.columns.get_loc("240_atr_14")

    baseline_positions = closed240[:47].tolist()
    spike_position = int(closed240[47])

    df.iloc[baseline_positions, high_col] = 100.0
    df.iloc[spike_position, high_col] = 200.0  # brand-new all-time high, AT this row
    df.iloc[spike_position, close_col] = 200.0
    df.iloc[spike_position, atr_col] = 10.0

    feats = engineered_features(df, 60, 2)

    dist_hi = feats["swing_dist_hi_240_20"].iloc[spike_position]
    # prior swing high (excluding this row) is still the 100.0 baseline
    assert dist_hi == pytest.approx((100.0 - 200.0) / 10.0)
    assert dist_hi < 0


def test_swing_dist_warmup_nan_then_first_valid_value():
    df = make_slim(seed=108, days=8)
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())

    feats = engineered_features(df, 60, 2)

    warmup_pos = int(closed240[19])  # 20th closed row: only 19 PRIOR closed bars -> NaN
    first_valid_pos = int(closed240[20])  # 21st closed row: exactly 20 prior closed bars

    assert math.isnan(feats["swing_dist_hi_240_20"].iloc[warmup_pos])
    assert math.isnan(feats["swing_dist_lo_240_20"].iloc[warmup_pos])
    assert not math.isnan(feats["swing_dist_hi_240_20"].iloc[first_valid_pos])
    assert not math.isnan(feats["swing_dist_lo_240_20"].iloc[first_valid_pos])


def test_swing_dist_ffill_between_ctf_closes():
    df = make_slim(seed=109, days=8)
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())
    p = int(closed240[25])  # well past warmup; closed240 spacing is 16 rows

    feats = engineered_features(df, 60, 2)
    expected = feats["swing_dist_hi_240_20"].iloc[p]
    assert not math.isnan(expected)

    for offset in (1, 5, 15):  # all strictly before the NEXT 240 close (p+16)
        assert feats["swing_dist_hi_240_20"].iloc[p + offset] == pytest.approx(expected)


def test_swing_dist_no_lookahead_future_spike_does_not_change_past_rows():
    df = make_slim(seed=110, days=8)
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())

    feats_before = engineered_features(df, 60, 2)

    df2 = df.copy()
    late_pos = int(closed240[40])
    high_col = df2.columns.get_loc("240_high")
    df2.iloc[late_pos, high_col] = df2.iloc[late_pos, high_col] + 10_000.0  # huge future spike

    feats_after = engineered_features(df2, 60, 2)

    before_vals = feats_before["swing_dist_hi_240_20"].iloc[:late_pos]
    after_vals = feats_after["swing_dist_hi_240_20"].iloc[:late_pos]
    pd.testing.assert_series_equal(before_vals, after_vals)


# --- near_level ------------------------------------------------------------------


def test_near_level_within_1atr_is_1():
    df = make_slim(seed=111, days=8)
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())

    high_col = df.columns.get_loc("240_high")
    low_col = df.columns.get_loc("240_low")
    close_col = df.columns.get_loc("240_close")
    atr_col = df.columns.get_loc("240_atr_14")

    baseline_positions = closed240[:47].tolist()
    target_pos = int(closed240[47])

    df.iloc[baseline_positions, high_col] = 100.0
    df.iloc[baseline_positions, low_col] = 50.0
    df.iloc[target_pos, close_col] = 100.5  # 0.5 atr from prior swing high (100.0)
    df.iloc[target_pos, atr_col] = 1.0

    feats = engineered_features(df, 60, 2)

    assert feats["near_level_240"].iloc[target_pos] == 1
    assert feats["near_level_240"].dtype == np.int8


def test_near_level_far_from_both_levels_is_0():
    df = make_slim(seed=112, days=8)
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())

    high_col = df.columns.get_loc("240_high")
    low_col = df.columns.get_loc("240_low")
    close_col = df.columns.get_loc("240_close")
    atr_col = df.columns.get_loc("240_atr_14")

    baseline_positions = closed240[:47].tolist()
    target_pos = int(closed240[47])

    df.iloc[baseline_positions, high_col] = 100.0
    df.iloc[baseline_positions, low_col] = 50.0
    df.iloc[target_pos, close_col] = 75.0  # equidistant, 25 atr from both levels
    df.iloc[target_pos, atr_col] = 1.0

    feats = engineered_features(df, 60, 2)

    assert feats["near_level_240"].iloc[target_pos] == 0


def test_near_level_nan_swing_dist_is_0_not_nan():
    df = make_slim(seed=113, days=2)  # only 12 closed-240 rows: entirely within w=20 warmup
    closed240 = np.flatnonzero(df["240_is_closed"].to_numpy())
    pos = int(closed240[0])

    feats = engineered_features(df, 60, 2)

    assert math.isnan(feats["swing_dist_hi_240_20"].iloc[pos])
    assert feats["near_level_240"].iloc[pos] == 0


# --- time_in_candle / time_left -------------------------------------------------


def test_time_in_candle_at_open_floor_is_0_time_left_is_1():
    df = make_slim(seed=114, days=2)
    row_pos = 0
    open_idx_col = df.columns.get_loc("240_open_index")
    df.iloc[row_pos, open_idx_col] = df.index[row_pos]  # elapsed == 0

    feats = engineered_features(df, 60, 2)

    assert feats["time_in_candle_240"].iloc[row_pos] == pytest.approx(0.0)
    assert feats["time_left_240"].iloc[row_pos] == pytest.approx(1.0)


def test_time_in_candle_hand_value_one_row_before_close():
    df = make_slim(seed=115, days=2)
    row_pos = 10
    open_idx_col = df.columns.get_loc("240_open_index")
    df.iloc[row_pos, open_idx_col] = df.index[row_pos] - pd.Timedelta(minutes=225)

    feats = engineered_features(df, 60, 2)

    assert feats["time_in_candle_240"].iloc[row_pos] == pytest.approx(225 / 240)
    assert feats["time_left_240"].iloc[row_pos] == pytest.approx(15 / 240)


def test_time_in_candle_range_check_0_to_1_exclusive():
    df = make_slim(seed=116, days=5)  # unmodified: real floor()-derived open_index throughout

    feats = engineered_features(df, 60, 2)

    values = feats["time_in_candle_240"]
    assert values.notna().all()
    assert (values >= 0).all()
    assert (values < 1).all()


# --- need_speed --------------------------------------------------------------------


def test_need_speed_denominator_clamped_no_explosion():
    df = make_slim(seed=117, days=2)
    row_pos = 5

    open_idx_col = df.columns.get_loc("240_open_index")
    up_col = df.columns.get_loc("240_bb_upper_20_2")
    close_col = df.columns.get_loc("240_close")
    atr_col = df.columns.get_loc("240_atr_14")

    # elapsed 237.6 / 240 -> time_left = 0.01, well below the 0.05 floor
    df.iloc[row_pos, open_idx_col] = df.index[row_pos] - pd.Timedelta(minutes=237.6)
    df.iloc[row_pos, up_col] = 110.0
    df.iloc[row_pos, close_col] = 100.0
    df.iloc[row_pos, atr_col] = 5.0

    feats = engineered_features(df, 60, 2)

    time_left = feats["time_left_240"].iloc[row_pos]
    assert time_left < 0.05

    bb_dist_up = feats["bb_dist_up_240"].iloc[row_pos]
    assert bb_dist_up == pytest.approx((110.0 - 100.0) / 5.0)

    expected = bb_dist_up / 0.05  # clamped denominator, not the raw (near-zero) time_left
    got = feats["need_speed_up_240"].iloc[row_pos]
    assert got == pytest.approx(expected)
    assert math.isfinite(got)


def test_need_speed_hand_value_no_clamp():
    df = make_slim(seed=118, days=2)
    row_pos = 5

    open_idx_col = df.columns.get_loc("240_open_index")
    up_col = df.columns.get_loc("240_bb_upper_20_2")
    close_col = df.columns.get_loc("240_close")
    atr_col = df.columns.get_loc("240_atr_14")

    df.iloc[row_pos, open_idx_col] = df.index[row_pos] - pd.Timedelta(minutes=120)  # time_left = 0.5
    df.iloc[row_pos, up_col] = 110.0
    df.iloc[row_pos, close_col] = 100.0
    df.iloc[row_pos, atr_col] = 5.0

    feats = engineered_features(df, 60, 2)

    assert feats["time_left_240"].iloc[row_pos] == pytest.approx(0.5)
    assert feats["need_speed_up_240"].iloc[row_pos] == pytest.approx(2.0 / 0.5)


# --- htf_move_1440 / htf_diff_sign / htf_move_agree -----------------------------


def test_htf_move_1440_uses_240_cuts_fallback():
    assert 1440 not in MOVE_CUTS  # sanity: this really exercises CUTS_FALLBACK
    df = make_slim(seed=119, days=2)
    diff_col = df.columns.get_loc("1440_rsi_ma8_diff")
    cuts_240 = MOVE_CUTS[240]
    df.iloc[0, diff_col] = cuts_240[3] + 1.0  # beyond 240's upper cut
    df.iloc[1, diff_col] = cuts_240[0] - 1.0  # beyond 240's lower cut

    feats = engineered_features(df, 240, 2)

    assert feats["htf_move_1440"].iloc[0] == 2
    assert feats["htf_move_1440"].iloc[1] == -2
    assert feats["htf_move_1440"].dtype == np.int8


def test_htf_move_agree_matches_side_sign_symmetric():
    df = make_slim(seed=120, days=2)
    diff_col = df.columns.get_loc("1440_rsi_ma8_diff")
    df.iloc[0, diff_col] = 5.0  # positive
    df.iloc[1, diff_col] = -5.0  # negative

    feats_plus = engineered_features(df, 240, 2)
    feats_minus = engineered_features(df, 240, -2)

    assert feats_plus["htf_move_agree_1440"].iloc[0] == 1  # side +2, diff + -> agree
    assert feats_plus["htf_move_agree_1440"].iloc[1] == 0  # side +2, diff - -> disagree
    assert feats_minus["htf_move_agree_1440"].iloc[0] == 0  # side -2, diff + -> disagree
    assert feats_minus["htf_move_agree_1440"].iloc[1] == 1  # side -2, diff - -> agree
    assert feats_plus["htf_move_agree_1440"].dtype == np.int8


# --- default_feature_cols --------------------------------------------------------


def test_default_feature_cols_includes_lower_and_higher_tf_for_60():
    df = make_slim(seed=121, days=2)

    cols = default_feature_cols(list(df.columns), 60)

    assert any(c.startswith("5_") for c in cols)  # lower tf (LOWER_TF[60])
    assert any(c.startswith("15_") for c in cols)  # lower tf (LOWER_TF[60])
    assert any(c.startswith("240_") for c in cols)  # higher tf (HIGHER_TF[60])
    assert any(c.startswith("1440_") for c in cols)  # higher tf (HIGHER_TF[60])
    assert any(c.startswith("60_") for c in cols)  # own tf


# sar_002_02/vol_ma_20/vol_ma_20_minus_volume/tgt_long are real per-tf slim
# suffixes (config.py's _CCI_SAR_SUFFIXES/_VOLUME_SUFFIXES/_TARGET_SUFFIXES)
# that the SYNTHETIC fixture just doesn't happen to synthesize (see
# conftest.py's _tf_value_columns -- it generates a representative subset,
# not literally every whitelist suffix). default_feature_cols is a pure
# name-filter over `slim_cols: list[str]` -- it needs no real column DATA --
# so these cases are exercised by appending the candidate NAMES directly
# rather than requiring the fixture to have generated real data for them.
_EXTRA_CANDIDATE_COLS = ["60_tgt_long", "60_sar_002_02", "60_vol_ma_20", "60_vol_ma_20_minus_volume"]


@pytest.mark.parametrize(
    "forbidden",
    [
        LABEL_COLS[60]["pslong_n1"],
        "60_close",
        "240_open_index",
        "60_tgt_long",
        "1440_bb_upper_20_2",
        "60_sar_002_02",
        "60_vol_ma_20",
    ],
)
def test_default_feature_cols_excludes_every_blocklisted_family(forbidden):
    df = make_slim(seed=122, days=2)
    candidate_cols = list(df.columns) + _EXTRA_CANDIDATE_COLS
    assert forbidden in candidate_cols  # sanity: the case is actually exercised

    cols = default_feature_cols(candidate_cols, 60)

    assert forbidden not in cols


@pytest.mark.parametrize("kept", ["60_vol_ma_20_minus_volume", "60_rsi_ma8_diff", "5_rsi_ma8"])
def test_default_feature_cols_keeps_expected_relative_columns(kept):
    df = make_slim(seed=123, days=2)
    candidate_cols = list(df.columns) + _EXTRA_CANDIDATE_COLS
    assert kept in candidate_cols

    cols = default_feature_cols(candidate_cols, 60)

    assert kept in cols


# --- feature_matrix ----------------------------------------------------------------


def test_feature_matrix_y_encoding_and_both_neither_nan_dropped():
    df = make_slim(seed=124, days=2)
    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed_positions[:5].tolist()
    _force_strong_points(df, 60, positions)

    set_labels(df, 60, [positions[0]], long_val=1.0, short_val=0.0)  # long
    set_labels(df, 60, [positions[1]], long_val=0.0, short_val=1.0)  # short
    set_labels(df, 60, [positions[2]], long_val=1.0, short_val=1.0)  # both -> dropped
    set_labels(df, 60, [positions[3]], long_val=0.0, short_val=0.0)  # neither -> dropped
    set_labels(df, 60, [positions[4]], long_val=float("nan"), short_val=0.0)  # nan -> dropped

    X, y = feature_matrix(df, 60, 2)

    expected_index = [df.index[positions[0]], df.index[positions[1]]]
    assert list(y.index) == expected_index
    assert y.tolist() == [1, 0]
    assert y.dtype == np.int8
    assert list(X.index) == list(y.index)


def test_feature_matrix_planted_leak_col_in_feature_cols_raises_value_error():
    df = make_slim(seed=125, days=2)
    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed_positions[:2].tolist()
    _force_strong_points(df, 60, positions)
    set_labels(df, 60, positions, long_val=1.0, short_val=0.0)

    leak_col = LABEL_COLS[60]["pslong_n1"]

    with pytest.raises(ValueError):
        feature_matrix(df, 60, 2, feature_cols=[leak_col, "60_rsi_ma8_diff"])


def test_feature_matrix_drops_all_nan_column_on_kept_rows():
    df = make_slim(seed=126, days=2)
    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed_positions[:2].tolist()
    _force_strong_points(df, 60, positions)
    set_labels(df, 60, positions, long_val=1.0, short_val=0.0)

    # NaN out a DIFFERENT column than 60_rsi_ma8_diff on the kept rows --
    # rsi_ma8_diff itself must stay real, it's what _force_strong_points
    # used to make these rows strong points in the first place.
    allnan_col = "60_natr_14"
    df.iloc[positions, df.columns.get_loc(allnan_col)] = float("nan")

    X, y = feature_matrix(df, 60, 2, feature_cols=[allnan_col, "60_body_ratio"])

    assert allnan_col not in X.columns
    assert "60_body_ratio" in X.columns


def test_feature_matrix_row_order_matches_slim_chronological_order():
    """feature_matrix's row order now comes from strong_points' scan of the
    FULL slim frame (always chronological -- boolean masking never
    reorders), not from a caller-supplied ``pts`` order. Labels are written
    in a scrambled CALL order here specifically to confirm that has no
    effect: set_labels always writes to absolute row positions, never
    appends, so the output must still land in slim's own chronological
    order regardless of what order the labels happened to be assigned in.
    """
    df = make_slim(seed=127, days=3)
    closed = np.flatnonzero(df["60_is_closed"].to_numpy())
    picked = closed[:6].tolist()  # already chronological (flatnonzero is ascending)
    _force_strong_points(df, 60, picked)

    set_labels(df, 60, [picked[3]], long_val=1.0, short_val=0.0)
    set_labels(df, 60, [picked[0]], long_val=1.0, short_val=0.0)
    set_labels(df, 60, [picked[5]], long_val=0.0, short_val=1.0)
    set_labels(df, 60, [picked[1]], long_val=0.0, short_val=1.0)
    set_labels(df, 60, [picked[4]], long_val=1.0, short_val=0.0)
    set_labels(df, 60, [picked[2]], long_val=0.0, short_val=1.0)

    X, y = feature_matrix(df, 60, 2)

    expected_index = [df.index[p] for p in picked]
    assert list(X.index) == expected_index
    assert list(y.index) == expected_index


def test_feature_matrix_does_not_mutate_input_slim():
    df = make_slim(seed=128, days=2)
    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed_positions[:2].tolist()
    _force_strong_points(df, 60, positions)
    set_labels(df, 60, positions, long_val=1.0, short_val=0.0)

    before = df.copy(deep=True)

    feature_matrix(df, 60, 2)

    pd.testing.assert_frame_equal(df, before)


def test_feature_matrix_boolean_column_cast_to_int8():
    df = make_slim(seed=129, days=2)
    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed_positions[:2].tolist()
    _force_strong_points(df, 60, positions)
    set_labels(df, 60, positions, long_val=1.0, short_val=0.0)

    df["60_trend_up_50"] = False
    df.iloc[positions[0], df.columns.get_loc("60_trend_up_50")] = True

    X, y = feature_matrix(df, 60, 2, feature_cols=["60_trend_up_50", "60_rsi_ma8_diff"])

    assert X["60_trend_up_50"].dtype == np.int8
    assert X["60_trend_up_50"].tolist() == [1, 0]


def test_feature_matrix_non_numeric_column_dropped():
    df = make_slim(seed=130, days=2)
    closed_positions = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed_positions[:2].tolist()
    _force_strong_points(df, 60, positions)
    set_labels(df, 60, positions, long_val=1.0, short_val=0.0)

    df["60_weird_text_col"] = "x"

    X, y = feature_matrix(df, 60, 2, feature_cols=["60_weird_text_col", "60_rsi_ma8_diff"])

    assert "60_weird_text_col" not in X.columns
    assert "60_rsi_ma8_diff" in X.columns


def test_feature_matrix_swing_dist_uses_consecutive_closed_bars_not_strong_points_only():
    """Regression for the fix pass: feature_matrix must run
    engineered_features on the FULL slim frame, then row-subset the
    RESULT -- never engineered_features on an already-strong-points-
    filtered frame. Proof: with only 2 strong +2 points total (indices far
    apart), a rolling(20, min_periods=20) window computed over "prior
    strong points only" could NEVER reach 20 rows (there are only 2 total)
    -- swing_dist_hi_60_20 at the second point would ALWAYS be NaN under
    that (buggy, pre-fix) subset-based computation, no matter what the
    real market structure looked like. The fix must instead reflect the
    true prior-20-CONSECUTIVE-closed-bar high, which is made up almost
    entirely of ordinary (non-strong) closed bars between the two points --
    proven here by engineering that true window to a KNOWN baseline while
    planting a wildly different value at the first (far-away) strong
    point, which must NOT leak into the second point's swing high.
    """
    df = make_slim(seed=140, days=10)
    closed60 = np.flatnonzero(df["60_is_closed"].to_numpy())
    assert len(closed60) >= 101

    point_a_idx = 5  # first strong point -- far outside point B's own w=20 lookback
    point_b_idx = 100  # second strong point

    point_a_pos = int(closed60[point_a_idx])
    point_b_pos = int(closed60[point_b_idx])
    _force_strong_points(df, 60, [point_a_pos, point_b_pos])
    set_labels(df, 60, [point_a_pos, point_b_pos], long_val=1.0, short_val=0.0)

    # true prior-20-consecutive-closed-bar window for point B: closed60[80:100]
    # (rolling(20).max().shift(1) at closed-position p uses window [p-20, p-1]).
    window_positions = closed60[point_b_idx - 20 : point_b_idx].tolist()
    high_col = df.columns.get_loc("60_high")
    df.iloc[window_positions, high_col] = 100.0  # known baseline swing high
    df.iloc[point_a_pos, high_col] = 9_999.0  # wildly outside point B's true window
    df.iloc[point_b_pos, df.columns.get_loc("60_close")] = 90.0
    df.iloc[point_b_pos, df.columns.get_loc("60_atr_14")] = 2.0

    X, y = feature_matrix(df, 60, 2)

    b_ts = df.index[point_b_pos]
    assert b_ts in X.index

    dist_hi = X.loc[b_ts, "swing_dist_hi_60_20"]
    # would be NaN under the old strong-points-only bug (only 2 rows total, needs 20)
    assert not math.isnan(dist_hi)
    # true consecutive-window high (100.0), NOT the 9999 planted at the far-away point A
    assert dist_hi == pytest.approx((100.0 - 90.0) / 2.0)


# --- FreezeStats -----------------------------------------------------------------


def test_freeze_stats_transform_train_is_zero_median_unit_ish_scale():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(10.0, 2.0, 200), "b": rng.normal(-5.0, 1.0, 200)})

    fs = FreezeStats().fit(X)
    Z = fs.transform(X)

    for col in ("a", "b"):
        assert Z[col].median() == pytest.approx(0.0, abs=1e-9)
        iqr = Z[col].quantile(0.75) - Z[col].quantile(0.25)
        assert iqr == pytest.approx(1.349, abs=1e-6)  # (raw_iqr/1.349) scale -> transformed iqr == 1.349


def test_freeze_stats_nan_cell_becomes_zero_after_transform():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, float("nan"), 5.0]})
    fs = FreezeStats().fit(X)

    Z = fs.transform(X)

    nan_pos = 3
    assert math.isnan(X["a"].iloc[nan_pos])  # sanity: really NaN
    assert Z["a"].iloc[nan_pos] == pytest.approx(0.0)


def test_freeze_stats_json_round_trip(tmp_path):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 10.0, 10.0, 10.0, 10.0]})
    fs = FreezeStats().fit(X)

    path = str(tmp_path / "freeze_stats.json")
    fs.to_json(path)
    fs2 = FreezeStats.from_json(path)

    assert fs2.stats == fs.stats


def test_freeze_stats_transform_never_refits():
    X1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X2 = pd.DataFrame({"a": [1000.0, 2000.0, 3000.0]})

    fs = FreezeStats().fit(X1)
    stats_before = {k: dict(v) for k, v in fs.stats.items()}

    fs.transform(X2)

    assert fs.stats == stats_before


def test_freeze_stats_degenerate_constant_column_scale_floor_no_div_by_zero():
    X = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0]})
    fs = FreezeStats().fit(X)

    assert fs.stats["a"]["scale"] >= 1e-9
    assert fs.stats["a"]["scale"] == pytest.approx(1.0)  # IQR==0 -> std==0 -> flat 1.0 fallback

    Z = fs.transform(X)
    assert Z["a"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_freeze_stats_unseen_column_at_transform_raises_value_error():
    X_fit = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_other = pd.DataFrame({"a": [1.0, 2.0], "b": [9.0, 9.0]})

    fs = FreezeStats().fit(X_fit)

    with pytest.raises(ValueError):
        fs.transform(X_other)


def test_freeze_stats_column_missing_at_transform_treated_as_median():
    X_fit = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    fs = FreezeStats().fit(X_fit)

    X_missing = pd.DataFrame(index=[0, 1, 2])  # no "a" column at all

    Z = fs.transform(X_missing)

    assert "a" in Z.columns
    assert Z["a"].tolist() == pytest.approx([0.0, 0.0, 0.0])
