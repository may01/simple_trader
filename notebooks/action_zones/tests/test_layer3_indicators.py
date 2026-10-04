"""Layer 3 tests — azlib.indicators (indicator attributes + freeze stats).

Covers: raw_attribute, FreezeStats (to_json/from_json), fit_stats,
zscore_clamp, attribute_frame (azlib/indicators.py). See
.superpowers/sdd/task-3-brief.md.

Granularity (task-3-brief.md, "GRANULARITY DECISION"): indicator attributes
are POINT-IN-TIME per 1-minute row — the wide-df indicator columns are read
AS-IS at each row (already closed-candle-based + partial-current). This is
intentionally DIFFERENT from Task 2's completed-candle held-constant levels
(azlib.space.price_levels) — no {tf}_is_closed reduction and no
held-constant broadcast happens anywhere in this module. `slope` is a plain
`.diff()` over 1-minute rows; `distance` is the documented column
difference, per row.

Filename carries "layer3" so `pytest -k layer3` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.indicators existing, so — same
convention as test_layer2_space.py — the whole module imports it at top
level (only the Layer-4 forward integration test below needs an additional
importorskip, for azlib.models, which is Task 4 and does not exist yet).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from azlib.config import MACD_COL, MACD_HIST_COL, MA_COL, RSI_COL, RSI_MA_COL
from azlib.indicators import (
    ATTRS,
    INDICATORS,
    FreezeStats,
    attribute_frame,
    fit_stats,
    raw_attribute,
    zscore_clamp,
)


# --- integration -> Layer 4 (azlib.models, Task 4) --------------------------


def test_attributes_feed_regression(synthetic_wide_df):
    """Brief's Layer-4 forward integration test, verbatim in spirit.

    azlib.models (Task 4, fit_regression) is not implemented yet in this
    repo, so — same decision as Task 1's test_labels_feed_action_space and
    Task 2's test_label_coeff_feeds_indicator_join — the azlib.models import
    boundary is guarded with importorskip rather than left as a permanent
    hard failure. Everything Layer 3 owns (fit_stats, attribute_frame,
    label_coeff-join) is exercised unconditionally first.
    """
    from azlib.space import label_coeff

    stats = fit_stats(synthetic_wide_df, tf=15)
    X = attribute_frame(synthetic_wide_df, 15, "rsi", stats)[["position"]]
    y = label_coeff(synthetic_wide_df, 15, "long")
    df = X.join(y.rename("y")).dropna()
    assert not df.empty

    azlib_models = pytest.importorskip(
        "azlib.models",
        reason="Task 4 (azlib/models.py, fit_regression) is not implemented yet",
    )
    fit_regression = azlib_models.fit_regression

    res = fit_regression(df[["position"]].to_numpy(), df["y"].to_numpy(), kind="linear")
    assert res.metrics["r2"] is not None


# --- module-level constants -------------------------------------------------


def test_indicators_and_attrs_shape():
    assert INDICATORS == ("rsi", "macd", "ma")
    assert ATTRS["rsi"] == ("position", "slope", "distance")
    assert ATTRS["macd"] == ("position", "slope", "distance")
    # MA has NO distance attribute (brief, explicit).
    assert ATTRS["ma"] == ("position", "slope")
    assert "distance" not in ATTRS["ma"]


# --- raw_attribute: rsi -----------------------------------------------------


def test_raw_attribute_rsi_position_is_rsi_ma8_column(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=15, indicator="rsi", attr="position")
    expected = synthetic_wide_df[f"15_{RSI_MA_COL}"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_raw_attribute_rsi_slope_is_diff_of_rsi_ma8(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=15, indicator="rsi", attr="slope")
    expected = synthetic_wide_df[f"15_{RSI_MA_COL}"].diff()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_raw_attribute_rsi_distance_is_rsi14_minus_rsi_ma8(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=15, indicator="rsi", attr="distance")
    expected = synthetic_wide_df[f"15_{RSI_COL}"] - synthetic_wide_df[f"15_{RSI_MA_COL}"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


# --- raw_attribute: macd ----------------------------------------------------


def test_raw_attribute_macd_position_is_macd_column(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=60, indicator="macd", attr="position")
    expected = synthetic_wide_df[f"60_{MACD_COL}"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_raw_attribute_macd_slope_is_diff_of_macd(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=60, indicator="macd", attr="slope")
    expected = synthetic_wide_df[f"60_{MACD_COL}"].diff()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_raw_attribute_macd_distance_is_macd_hist_column(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=60, indicator="macd", attr="distance")
    expected = synthetic_wide_df[f"60_{MACD_HIST_COL}"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


# --- raw_attribute: ma ------------------------------------------------------


def test_raw_attribute_ma_position_is_close_ema_pct(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=240, indicator="ma", attr="position")
    close = synthetic_wide_df["240_close"]
    ema = synthetic_wide_df[f"240_{MA_COL}"]
    expected = (close - ema) / close * 100.0
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_raw_attribute_ma_slope_is_diff_of_ema(synthetic_wide_df):
    result = raw_attribute(synthetic_wide_df, tf=240, indicator="ma", attr="slope")
    expected = synthetic_wide_df[f"240_{MA_COL}"].diff()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_raw_attribute_ma_rejects_distance():
    with pytest.raises((KeyError, ValueError)):
        raw_attribute(pd.DataFrame(), tf=15, indicator="ma", attr="distance")


def test_raw_attribute_rejects_unknown_indicator(synthetic_wide_df):
    with pytest.raises(ValueError):
        raw_attribute(synthetic_wide_df, tf=15, indicator="bogus", attr="position")


def test_raw_attribute_rejects_unknown_attr(synthetic_wide_df):
    with pytest.raises(ValueError):
        raw_attribute(synthetic_wide_df, tf=15, indicator="rsi", attr="bogus")


# --- raw_attribute: point-in-time granularity (no completed-candle reduction) --


def test_raw_attribute_is_not_held_constant_within_a_forming_candle(synthetic_wide_df):
    """Distinguishing property vs. Task 2's price_levels: raw_attribute reads
    the wide-df indicator column AS-IS per 1-minute row -- it must vary
    minute to minute (matching the underlying column exactly), never be
    broadcast-held-constant across a tf-bucket the way price_levels is.
    """
    result = raw_attribute(synthetic_wide_df, tf=240, indicator="rsi", attr="position")
    # the raw 240_rsi_ma8 column itself is not held constant across its own
    # forming buckets (it's a per-minute EWM-derived value in the fixture) --
    # confirm attribute output tracks it exactly, row for row, including
    # within a single still-open bucket.
    tail = result.tail(50)
    assert tail.nunique(dropna=True) > 1


# --- zscore_clamp -----------------------------------------------------------


def test_zscore_clamp_value_at_mean_is_zero():
    s = pd.Series([10.0, 10.0, 10.0])
    result = zscore_clamp(s, mean=10.0, std=2.0)
    np.testing.assert_allclose(result.to_numpy(), [0.0, 0.0, 0.0])


def test_zscore_clamp_mean_plus_3std_is_3():
    s = pd.Series([10.0 + 3 * 2.0])
    result = zscore_clamp(s, mean=10.0, std=2.0)
    np.testing.assert_allclose(result.to_numpy(), [3.0])


def test_zscore_clamp_mean_plus_5std_is_clamped_to_3():
    s = pd.Series([10.0 + 5 * 2.0])
    result = zscore_clamp(s, mean=10.0, std=2.0)
    np.testing.assert_allclose(result.to_numpy(), [3.0])


def test_zscore_clamp_mean_minus_5std_is_clamped_to_neg3():
    s = pd.Series([10.0 - 5 * 2.0])
    result = zscore_clamp(s, mean=10.0, std=2.0)
    np.testing.assert_allclose(result.to_numpy(), [-3.0])


def test_zscore_clamp_std_zero_returns_zero_no_warning():
    s = pd.Series([10.0, 20.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = zscore_clamp(s, mean=10.0, std=0.0)
    np.testing.assert_array_equal(result.to_numpy(), [0.0, 0.0, 0.0])


def test_zscore_clamp_custom_lo_hi():
    s = pd.Series([10.0 + 10 * 2.0])
    result = zscore_clamp(s, mean=10.0, std=2.0, lo=-1.0, hi=1.0)
    np.testing.assert_allclose(result.to_numpy(), [1.0])


# --- fit_stats / FreezeStats -------------------------------------------------


def test_fit_stats_covers_every_tf_indicator_attr_combo(synthetic_wide_df):
    stats = fit_stats(synthetic_wide_df, tf=15)
    for indicator in INDICATORS:
        for attr in ATTRS[indicator]:
            assert (15, indicator, attr) in stats.stats


def test_fit_stats_mean_std_match_hand_computed_nonnan_stats(synthetic_wide_df):
    stats = fit_stats(synthetic_wide_df, tf=60)
    raw = raw_attribute(synthetic_wide_df, tf=60, indicator="rsi", attr="position").dropna()

    mean, std = stats.stats[(60, "rsi", "position")]
    assert mean == pytest.approx(raw.mean())
    assert std == pytest.approx(raw.std())


def test_fit_stats_to_json_from_json_round_trips(synthetic_wide_df, tmp_path):
    stats = fit_stats(synthetic_wide_df, tf=15)
    path = tmp_path / "freeze_stats.json"
    stats.to_json(str(path))

    assert path.exists()
    loaded = FreezeStats.from_json(str(path))

    assert set(loaded.stats.keys()) == set(stats.stats.keys())
    for key, (mean, std) in stats.stats.items():
        loaded_mean, loaded_std = loaded.stats[key]
        assert loaded_mean == pytest.approx(mean)
        assert loaded_std == pytest.approx(std)


def test_freeze_stats_to_json_writes_valid_json(synthetic_wide_df, tmp_path):
    stats = fit_stats(synthetic_wide_df, tf=15)
    path = tmp_path / "freeze_stats.json"
    stats.to_json(str(path))

    with open(path) as f:
        raw_json = json.load(f)  # must not raise
    assert raw_json  # non-empty


# --- attribute_frame ---------------------------------------------------------


def test_attribute_frame_raw_when_stats_none(synthetic_wide_df):
    frame = attribute_frame(synthetic_wide_df, tf=15, indicator="rsi")

    assert list(frame.columns) == list(ATTRS["rsi"])
    for attr in ATTRS["rsi"]:
        expected = raw_attribute(synthetic_wide_df, tf=15, indicator="rsi", attr=attr)
        pd.testing.assert_series_equal(frame[attr], expected, check_names=False)


def test_attribute_frame_ma_has_no_distance_column(synthetic_wide_df):
    frame = attribute_frame(synthetic_wide_df, tf=15, indicator="ma")
    assert list(frame.columns) == ["position", "slope"]
    assert "distance" not in frame.columns


def test_attribute_frame_zscored_and_clamped_when_stats_given(synthetic_wide_df):
    stats = fit_stats(synthetic_wide_df, tf=15)
    frame = attribute_frame(synthetic_wide_df, tf=15, indicator="rsi", stats=stats)

    for attr in ATTRS["rsi"]:
        mean, std = stats.stats[(15, "rsi", attr)]
        raw = raw_attribute(synthetic_wide_df, tf=15, indicator="rsi", attr=attr)
        expected = zscore_clamp(raw, mean, std)
        pd.testing.assert_series_equal(frame[attr], expected, check_names=False)

    # clamped range invariant, over non-NaN values
    non_nan = frame.dropna()
    assert (non_nan >= -3.0).all().all()
    assert (non_nan <= 3.0).all().all()


def test_attribute_frame_frozen_stats_not_recomputed_on_different_frame(synthetic_wide_df):
    """OOS application must use the FROZEN train stats, never recompute from
    the frame being scored -- the core freeze/no-look-ahead guarantee this
    task exists to provide (task-3-brief.md's "Normalization / freezing").
    """
    train_stats = fit_stats(synthetic_wide_df, tf=15)

    # A different frame (shifted/rescaled indicator values) with a
    # deliberately different distribution -- if attribute_frame recomputed
    # stats from THIS frame instead of using the frozen train stats, the
    # z-scored output would differ from applying train_stats by hand.
    oos_df = synthetic_wide_df.copy()
    oos_df["15_rsi_ma8"] = oos_df["15_rsi_ma8"] + 1000.0  # shift far from train mean

    result = attribute_frame(oos_df, tf=15, indicator="rsi", stats=train_stats)

    train_mean, train_std = train_stats.stats[(15, "rsi", "position")]
    raw_oos_position = raw_attribute(oos_df, tf=15, indicator="rsi", attr="position")
    expected = zscore_clamp(raw_oos_position, train_mean, train_std)

    pd.testing.assert_series_equal(result["position"], expected, check_names=False)
    # With a +1000 shift far beyond train's std, every non-NaN value must be
    # clamped at the +3 ceiling -- proof the OOS frame's own (very different)
    # distribution was NOT used to (re)compute mean/std.
    non_nan = result["position"].dropna()
    assert not non_nan.empty
    assert non_nan.to_numpy() == pytest.approx(3.0)
