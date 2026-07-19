"""Layer 2 tests — azlib.space (action space + label_coeff).

Covers: diff_prc, diff_prc_ma, diff_prc_std, price_levels, coeff, label_coeff
(azlib/space.py). See .superpowers/sdd/task-2-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §1-2.

Filename carries "layer2" so `pytest -k layer2` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.space existing, so — unlike
test_layer1_loader.py, where only one forward-looking test needed a guard —
the whole module imports it at top level. Before azlib/space.py exists this
produces a single collection error (RED), which is the expected/desired
signal per the brief's TDD requirement.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from azlib.space import coeff, diff_prc, diff_prc_ma, diff_prc_std, label_coeff, price_levels


# --- integration -> Layer 3 (azlib.indicators, Task 3) ----------------------


def test_label_coeff_feeds_indicator_join(synthetic_wide_df):
    """Brief's Layer-3 forward integration test, verbatim in spirit.

    azlib.indicators (Task 3) is not implemented yet in this repo, so — same
    decision as Task 1's test_labels_feed_action_space (see
    task-1-report.md) — the azlib.indicators import boundary is guarded with
    importorskip rather than left as a permanent hard failure. Everything
    Layer 2 owns (label_coeff itself) is exercised unconditionally first.
    """
    lc = label_coeff(synthetic_wide_df, tf=15, direction="long")
    assert lc.between(0, 1).dropna().shape[0] > 0

    azlib_indicators = pytest.importorskip(
        "azlib.indicators",
        reason="Task 3 (azlib/indicators.py, attribute_frame) is not implemented yet",
    )
    attribute_frame = azlib_indicators.attribute_frame

    attrs = attribute_frame(synthetic_wide_df, tf=15, indicator="rsi")
    joined = attrs.join(lc.rename("label_coeff")).dropna()
    assert not joined.empty
    assert joined["label_coeff"].between(0, 1).all()


# --- diff_prc -----------------------------------------------------------


def test_diff_prc_known_series_exact_percentages_first_row_nan():
    s = pd.Series([100.0, 110.0, 99.0, 99.0])
    result = diff_prc(s)

    assert pd.isna(result.iloc[0])
    expected = pd.Series([np.nan, 10.0, -10.0, 0.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


# --- diff_prc_ma ----------------------------------------------------------


def test_diff_prc_ma_rolling_mean_matches_hand_computed_values():
    diff = pd.Series([10.0, -10.0, 0.0, 20.0, -20.0, 30.0])
    ma = diff_prc_ma(diff, window=3)

    assert pd.isna(ma.iloc[0]) and pd.isna(ma.iloc[1])
    assert ma.iloc[2] == pytest.approx(0.0)  # mean(10, -10, 0)
    assert ma.iloc[3] == pytest.approx(10.0 / 3.0)  # mean(-10, 0, 20)
    assert ma.iloc[4] == pytest.approx(0.0)  # mean(0, 20, -20)
    assert ma.iloc[5] == pytest.approx(10.0)  # mean(20, -20, 30)


def test_diff_prc_ma_default_window_is_6():
    diff = pd.Series(np.arange(8, dtype=float))
    ma = diff_prc_ma(diff)  # no window kwarg -> must default to 6
    pd.testing.assert_series_equal(ma, diff.rolling(6).mean(), check_names=False)


# --- diff_prc_std -----------------------------------------------------------


def test_diff_prc_std_plain_rolling_std_matches_hand_computed_value():
    diff = pd.Series([10.0, -10.0, 0.0, 20.0, -20.0, 30.0])
    std = diff_prc_std(diff, window=3)

    assert pd.isna(std.iloc[0]) and pd.isna(std.iloc[1])
    # std([10, -10, 0], ddof=1) = sqrt((100+100+0)/2) = 10.0 exactly
    assert std.iloc[2] == pytest.approx(10.0)
    # cross-check the rest against numpy's own ddof=1 std (independent of
    # pandas' internal rolling implementation, confirms "plain", un-sided std)
    assert std.iloc[3] == pytest.approx(np.std([-10.0, 0.0, 20.0], ddof=1))
    assert std.iloc[5] == pytest.approx(np.std([20.0, -20.0, 30.0], ddof=1))


def test_diff_prc_std_is_plain_not_sided():
    # A window straddling both sides of its own mean must use ALL values,
    # not a one-sided (_std_above/_std_below) subset -- i.e. it must differ
    # from the one-sided std of just the above-mean values (a single value,
    # [100.0]; price_derivatives.py's own convention is 0.0 for a
    # single-element sided subset -- see its module docstring).
    diff = pd.Series([1.0, 2.0, 3.0, 100.0])  # mean=26.5, heavily skewed
    std = diff_prc_std(diff, window=4)
    plain = float(np.std([1.0, 2.0, 3.0, 100.0], ddof=1))
    above_only_sided_std = 0.0  # single-element subset -> 0.0, per price_derivatives.py convention
    assert std.iloc[3] == pytest.approx(plain)
    assert std.iloc[3] != pytest.approx(above_only_sided_std)


# --- price_levels -----------------------------------------------------------


def test_price_levels_recomputes_from_raw_when_diff_columns_absent():
    """No {tf}_*_diff_prc / *_rm_* columns at all -> full raw-recompute path."""
    tf = 15
    window = 2
    x = 2.0
    # high: +10% then +10% again -> diff_prc = [nan, 10.0, 10.0] -> std=0 (both equal)
    high = pd.Series([100.0, 110.0, 121.0])
    # low: -10% then -10% again -> diff_prc = [nan, -10.0, -10.0] -> std=0
    low = pd.Series([50.0, 45.0, 40.5])
    wide_df = pd.DataFrame({f"{tf}_high": high, f"{tf}_low": low})
    assert f"{tf}_high_diff_prc" not in wide_df.columns  # sanity: no reuse path available

    price_high_level, price_low_level = price_levels(wide_df, tf=tf, window=window, x=x)

    # high_level = ma(10,10) + x*std(=0) = 10.0 -> price = prev_high(110.0)*1.10 = 121.0
    assert price_high_level.iloc[2] == pytest.approx(121.0)
    # low_level = ma(-10,-10) - x*std(=0) = -10.0 -> price = prev_low(45.0)*0.90 = 40.5
    assert price_low_level.iloc[2] == pytest.approx(40.5)

    # positive x -> high_level pushes price_high_level above prev_high, and
    # low_level pushes price_low_level below prev_low
    assert price_high_level.iloc[2] > high.iloc[1]
    assert price_low_level.iloc[2] < low.iloc[1]


def test_price_levels_uses_existing_diff_prc_when_rm_column_absent():
    """{tf}_*_diff_prc present but no *_rm_{window} column -> ma/std computed
    from the *existing* diff_prc column (not recomputed from raw high/low).
    """
    tf = 15
    window = 3
    x = 2.0
    n = 4
    high = pd.Series([100.0, 100.0, 150.0, 100.0])
    low = pd.Series([50.0, 50.0, 80.0, 50.0])
    high_diff = pd.Series([np.nan, -10.0, 0.0, 10.0])
    low_diff = pd.Series([np.nan, 5.0, 0.0, -5.0])
    wide_df = pd.DataFrame(
        {
            f"{tf}_high": high,
            f"{tf}_low": low,
            f"{tf}_high_diff_prc": high_diff,
            f"{tf}_low_diff_prc": low_diff,
        },
        index=pd.RangeIndex(n),
    )
    assert f"{tf}_high_diff_prc_rm_{window}" not in wide_df.columns

    price_high_level, price_low_level = price_levels(wide_df, tf=tf, window=window, x=x)

    # window=[-10,0,10] -> mean=0, std(ddof=1)=10 -> high_level = 0+2*10=20
    # price_high_level = prev_high(150.0) * 1.20 = 180.0
    assert price_high_level.iloc[3] == pytest.approx(180.0)
    # window=[5,0,-5] -> mean=0, std(ddof=1)=5 -> low_level = 0-2*5=-10
    # price_low_level = prev_low(80.0) * 0.90 = 72.0
    assert price_low_level.iloc[3] == pytest.approx(72.0)

    assert price_high_level.iloc[3] > high.iloc[2]  # prev_high
    assert price_low_level.iloc[3] < low.iloc[2]  # prev_low


def test_price_levels_prefers_existing_columns_over_recompute():
    """{tf}_*_diff_prc AND *_rm_{window} both present -> used as-is, proven
    by making raw high/low inconsistent with the injected diff columns (a
    constant raw series would recompute to diff_prc==0.0 everywhere, which
    would NOT match the nonzero injected diff below if recompute happened).
    """
    tf = 60
    window = 6
    x = 2.0
    n = 8
    high = pd.Series([300.0] * n)  # constant -> raw pct_change would be 0.0
    low = pd.Series([200.0] * n)
    high_diff = pd.Series([2.0] * n)  # constant, non-zero, inconsistent w/ raw
    low_diff = pd.Series([-3.0] * n)
    wide_df = pd.DataFrame(
        {
            f"{tf}_high": high,
            f"{tf}_low": low,
            f"{tf}_high_diff_prc": high_diff,
            f"{tf}_low_diff_prc": low_diff,
            f"{tf}_high_diff_prc_rm_{window}": high_diff.rolling(window).mean(),
            f"{tf}_low_diff_prc_rm_{window}": low_diff.rolling(window).mean(),
        }
    )

    price_high_level, price_low_level = price_levels(wide_df, tf=tf, window=window, x=x)

    # std of a constant series is 0 -> level == ma exactly == injected diff value
    row = window
    expected_high = 300.0 * (1 + 2.0 / 100.0)  # 306.0, NOT 300.0 (raw recompute value)
    expected_low = 200.0 * (1 + -3.0 / 100.0)  # 194.0, NOT 200.0
    assert price_high_level.iloc[row] == pytest.approx(expected_high)
    assert price_low_level.iloc[row] == pytest.approx(expected_low)


def test_price_levels_default_window_and_x():
    tf = 15
    wide_df = pd.DataFrame(
        {f"{tf}_high": pd.Series(np.linspace(100, 120, 20)), f"{tf}_low": pd.Series(np.linspace(90, 110, 20))}
    )
    high6, low6 = price_levels(wide_df, tf=tf)  # defaults: window=6, x=2.0
    high6_explicit, low6_explicit = price_levels(wide_df, tf=tf, window=6, x=2.0)
    pd.testing.assert_series_equal(high6, high6_explicit)
    pd.testing.assert_series_equal(low6, low6_explicit)


# --- coeff --------------------------------------------------------------


def test_coeff_endpoints_and_midpoint():
    low = np.array([100.0, 100.0, 100.0])
    high = np.array([200.0, 200.0, 200.0])
    price = np.array([100.0, 200.0, 150.0])

    result = coeff(price, low, high)

    np.testing.assert_allclose(result, [0.0, 1.0, 0.5])


def test_coeff_below_and_above_range_are_clamped():
    low = np.array([100.0, 100.0])
    high = np.array([200.0, 200.0])
    price = np.array([50.0, 250.0])  # below low_level, above high_level

    result = coeff(price, low, high)

    np.testing.assert_allclose(result, [0.0, 1.0])


def test_coeff_degenerate_equal_levels_returns_zero_no_warning():
    low = np.array([100.0, 100.0])
    high = np.array([100.0, 100.0])  # high_level == low_level (degenerate)
    price = np.array([100.0, 150.0])

    with warnings.catch_warnings():
        # Belt-and-suspenders: this specific test asserts no warning fires
        # regardless of how the surrounding suite is invoked (the project's
        # own -W error verification run is the other half of that guarantee).
        warnings.simplefilter("error")
        result = coeff(price, low, high)

    np.testing.assert_array_equal(result, [0.0, 0.0])


# --- label_coeff --------------------------------------------------------


def _tiny_levels_frame():
    """4-row frame reproducing test_price_levels_uses_existing_diff_prc_...'s
    setup (price_high_level=180.0, price_low_level=72.0 at row 3), plus
    1_low/1_high entry-extreme columns for label_coeff.
    """
    tf = 15
    high = pd.Series([100.0, 100.0, 150.0, 100.0])
    low = pd.Series([50.0, 50.0, 80.0, 50.0])
    high_diff = pd.Series([np.nan, -10.0, 0.0, 10.0])
    low_diff = pd.Series([np.nan, 5.0, 0.0, -5.0])
    entry_low = pd.Series([np.nan, np.nan, np.nan, 100.0])
    entry_high = pd.Series([np.nan, np.nan, np.nan, 150.0])
    wide_df = pd.DataFrame(
        {
            f"{tf}_high": high,
            f"{tf}_low": low,
            f"{tf}_high_diff_prc": high_diff,
            f"{tf}_low_diff_prc": low_diff,
            "1_low": entry_low,
            "1_high": entry_high,
        }
    )
    return tf, wide_df


def test_label_coeff_long_uses_entry_low_not_high():
    tf, wide_df = _tiny_levels_frame()
    lc = label_coeff(wide_df, tf=tf, direction="long", window=3, x=2.0)

    # price_low_level=72.0, price_high_level=180.0 (see test_price_levels_
    # uses_existing_diff_prc_when_rm_column_absent for the derivation);
    # long uses 1_low=100.0 -> coeff = (100-72)/(180-72)
    assert lc.iloc[3] == pytest.approx((100.0 - 72.0) / (180.0 - 72.0))


def test_label_coeff_short_uses_entry_high_not_low():
    tf, wide_df = _tiny_levels_frame()
    lc = label_coeff(wide_df, tf=tf, direction="short", window=3, x=2.0)

    # short uses 1_high=150.0 -> coeff = (150-72)/(180-72)
    assert lc.iloc[3] == pytest.approx((150.0 - 72.0) / (180.0 - 72.0))


def test_label_coeff_long_and_short_differ_when_low_ne_high():
    tf, wide_df = _tiny_levels_frame()
    long_lc = label_coeff(wide_df, tf=tf, direction="long", window=3, x=2.0)
    short_lc = label_coeff(wide_df, tf=tf, direction="short", window=3, x=2.0)

    assert long_lc.iloc[3] != pytest.approx(short_lc.iloc[3])


def test_label_coeff_rejects_bad_direction():
    tf, wide_df = _tiny_levels_frame()
    with pytest.raises(ValueError):
        label_coeff(wide_df, tf=tf, direction="sideways")


@pytest.mark.parametrize("tf", [15, 60, 240])
def test_label_coeff_bounded_0_1_on_synthetic_fixture(synthetic_wide_df, tf):
    for direction in ("long", "short"):
        lc = label_coeff(synthetic_wide_df, tf=tf, direction=direction)
        assert lc.dropna().between(0, 1).all()
        assert lc.index.equals(synthetic_wide_df.index)
