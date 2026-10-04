"""Layer 2 tests — azlib.space (action space + label_coeff).

Covers: diff_prc, diff_prc_ma, diff_prc_std, price_levels, coeff, label_coeff
(azlib/space.py). See .superpowers/sdd/task-2-brief.md and
external/docs/superpowers/specs/2026-07-18-zone-selection-design.md §1-2.

This is a REWORK of the module (see task-2-report.md): the original
implementation (commit 40b5fe7) computed levels via a plain ``.shift(1)`` on
the 1-minute *forming* ``{tf}_high``/``{tf}_low`` columns — a minute-to-minute
forming increment, not the candle-to-candle change the design spec calls
for, and not held-constant/look-ahead-free. The ``price_levels``/
``label_coeff`` tests below were rewritten from scratch against a hand-built
tiny frame that spans several COMPLETED ``{tf}``-candles (marked via
``{tf}_is_closed``) so every number can be independently hand-verified — the
now-invalid tests that asserted the old "prefer existing 1-minute diff_prc
columns" reuse behavior were deleted (that reuse path no longer exists:
``price_levels`` always computes its own diff/ma/std from the
completed-candle reduction, never from any wide-df diff_prc column — see
task-2-brief.md's explicit "Do NOT reuse ... those are forming per-minute"
constraint).

Filename carries "layer2" so `pytest -k layer2` selects every test in this
module (pytest -k matches against each item's full ancestry of names, which
includes the containing module's basename).

Every test in this module depends on azlib.space existing, so — unlike
test_layer1_loader.py, where only one forward-looking test needed a guard —
the whole module imports it at top level.
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
    Layer 2 owns (label_coeff itself) is exercised unconditionally first,
    with window=2 (see the tf=240 note on
    test_label_coeff_bounded_0_1_on_synthetic_fixture below for why the
    default window=6 would not warm up within the fixture for every tf; here
    only tf=15 is used, which warms up fine even at the default, but window
    is pinned explicitly anyway so this test's warm-up assumption doesn't
    silently depend on the fixture's row count).
    """
    lc = label_coeff(synthetic_wide_df, tf=15, direction="long", window=2)
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


# --- price_levels: hand-crafted completed-candle frame -----------------------
#
# 4 completed tf=15 candles (indices 0-3, one {tf}_is_closed=True row each,
# at minute 14 of its own 15-row bucket) followed by a 5th, still-forming
# candle (index 4, {tf}_is_closed all False). window=2 keeps the warm-up
# region short enough that all 3 warm-up cases (no c-1 at all; c-1 exists
# but its own window isn't full; first fully-warmed level) show up within
# just 4 completed candles, and the forming candle's OWN levels are
# reachable by hand.
#
# Candle-final highs/lows are chosen so each step is exactly +/-10%:
#   high: 100.0 -> 110.0 (+10%) -> 99.0 (-10%) -> 108.9 (+10%)
#   low:   50.0 ->  45.0 (-10%) -> 49.5 (+10%) ->  44.55 (-10%)
# so diff_prc/ma/std are exact, round, hand-checkable numbers.

_TF = 15
_WINDOW = 2
_CANDLE_HIGH = [100.0, 110.0, 99.0, 108.9]
_CANDLE_LOW = [50.0, 45.0, 49.5, 44.55]
_N_FORMING_ROWS = 5


def _candle_frame(mutate_forming_high_row: int | None = None, mutate_forming_high_value: float = 99999.0):
    """Build the 1-min hand-crafted frame described above.

    ``{tf}_high``/``{tf}_low`` are held constant at each candle's own final
    value across every one of that candle's rows (the intra-candle forming
    SHAPE doesn't matter here — only the value at the closing row is ever
    consumed by ``price_levels``, which is exactly what the no-look-ahead
    test below exploits). The forming candle (index 4) gets
    ``_N_FORMING_ROWS`` rows of arbitrary (row-varying) high/low, optionally
    with one row's high overwritten via ``mutate_forming_high_row`` — used
    by the no-look-ahead test to prove that mutating a forming candle's OWN
    data never changes that candle's OWN broadcast levels.
    """
    high_col: list[float] = []
    low_col: list[float] = []
    is_closed: list[bool] = []

    for h, low in zip(_CANDLE_HIGH, _CANDLE_LOW):
        for m in range(_TF):
            high_col.append(h)
            low_col.append(low)
            is_closed.append(m == _TF - 1)

    forming_start_row = len(high_col)
    for i in range(_N_FORMING_ROWS):
        high_col.append(_CANDLE_HIGH[-1] + 1.0 + i)  # arbitrary, row-varying forming values
        low_col.append(_CANDLE_LOW[-1] - 1.0 - i * 0.1)
        is_closed.append(False)

    if mutate_forming_high_row is not None:
        high_col[forming_start_row + mutate_forming_high_row] = mutate_forming_high_value

    n = len(high_col)
    wide_df = pd.DataFrame(
        {
            f"{_TF}_high": high_col,
            f"{_TF}_low": low_col,
            f"{_TF}_is_closed": is_closed,
        },
        index=pd.RangeIndex(n),
    )
    return wide_df, forming_start_row


def _expected_level(prev_high_seq, prev_low_seq, window, x):
    """Hand-derive (price_high_level, price_low_level) straight from the
    design spec's formulas via plain numpy — independent of price_levels'
    own pandas-rolling implementation.

    ``prev_high_seq``/``prev_low_seq`` = chronological completed-candle
    highs/lows up to AND INCLUDING the reference candle c-1 (its own value
    is the last element).
    """
    high = np.array(prev_high_seq, dtype=float)
    low = np.array(prev_low_seq, dtype=float)
    high_diff = (high[1:] - high[:-1]) / high[:-1] * 100.0
    low_diff = (low[1:] - low[:-1]) / low[:-1] * 100.0
    h_window = high_diff[-window:]
    l_window = low_diff[-window:]
    high_ma = h_window.mean()
    low_ma = l_window.mean()
    high_std = h_window.std(ddof=1)
    low_std = l_window.std(ddof=1)
    price_high_level = high[-1] * (1.0 + (high_ma + x * high_std) / 100.0)
    price_low_level = low[-1] * (1.0 + (low_ma - x * low_std) / 100.0)
    return price_high_level, price_low_level


def test_price_levels_warmup_nan_before_window_completed_candles():
    """Candles 0, 1, 2's own buckets must ALL be NaN:
      - candle 0's bucket: no c-1 at all (it's the very first candle).
      - candle 1's bucket: c-1=candle 0, but candle 0 alone has zero
        completed diff_prc values (there's no candle -1) -> ma/std undefined.
      - candle 2's bucket: c-1=candle 1, but window=2 needs 2 diff_prc
        values ending at candle 1 (diff[0], diff[1]); diff[0] is itself
        undefined (candle 0 has no predecessor) -> still not enough.
    """
    wide_df, _ = _candle_frame()
    price_high_level, price_low_level = price_levels(wide_df, tf=_TF, window=_WINDOW, x=2.0)

    for candle_idx in (0, 1, 2):
        bucket = slice(candle_idx * _TF, (candle_idx + 1) * _TF)
        assert price_high_level.iloc[bucket].isna().all(), f"candle {candle_idx} high_level should be all-NaN"
        assert price_low_level.iloc[bucket].isna().all(), f"candle {candle_idx} low_level should be all-NaN"


def test_price_levels_hand_computed_from_previous_completed_candle():
    """Candle 3's bucket is the first with a fully-warmed window=2: its
    reference is candle 2 (c-1), whose own window=2 diff_prc's are
    diff[1] (candle0->1) and diff[2] (candle1->2) -- both defined. Every
    1-min row of candle 3's bucket must equal the SAME hand-computed value
    derived only from candles 0, 1, 2.
    """
    wide_df, _ = _candle_frame()
    price_high_level, price_low_level = price_levels(wide_df, tf=_TF, window=_WINDOW, x=2.0)

    expected_high, expected_low = _expected_level(_CANDLE_HIGH[:3], _CANDLE_LOW[:3], window=_WINDOW, x=2.0)

    bucket = slice(3 * _TF, 4 * _TF)
    assert price_high_level.iloc[bucket].to_numpy() == pytest.approx(expected_high)
    assert price_low_level.iloc[bucket].to_numpy() == pytest.approx(expected_low)

    # positive x -> the high bound reaches above the reference high, the low
    # bound reaches below the reference low (design spec §1's asymmetry)
    assert expected_high > _CANDLE_HIGH[2]
    assert expected_low < _CANDLE_LOW[2]


def test_price_levels_held_constant_across_forming_candle_rows():
    """Every 1-min row of the still-forming candle 4 must share the exact
    same (price_high_level, price_low_level) pair -- computed from candle 3
    (c-1), never varying minute to minute within the bucket.
    """
    wide_df, forming_start_row = _candle_frame()
    price_high_level, price_low_level = price_levels(wide_df, tf=_TF, window=_WINDOW, x=2.0)

    forming_high = price_high_level.iloc[forming_start_row : forming_start_row + _N_FORMING_ROWS]
    forming_low = price_low_level.iloc[forming_start_row : forming_start_row + _N_FORMING_ROWS]

    assert forming_high.nunique(dropna=False) == 1
    assert forming_low.nunique(dropna=False) == 1
    assert not forming_high.isna().any()  # candle 3 (c-1) is fully warmed -> non-NaN

    expected_high, expected_low = _expected_level(_CANDLE_HIGH, _CANDLE_LOW, window=_WINDOW, x=2.0)
    assert forming_high.iloc[0] == pytest.approx(expected_high)
    assert forming_low.iloc[0] == pytest.approx(expected_low)
    assert expected_high > _CANDLE_HIGH[3]
    assert expected_low < _CANDLE_LOW[3]


def test_price_levels_no_look_ahead_from_own_forming_candle():
    """Mutating the still-forming candle 4's OWN {tf}_high at one of its own
    rows must NOT change candle 4's broadcast levels -- they depend only on
    candles <= c-1 (candle 3 and earlier), never on candle 4's own
    (necessarily incomplete, still-changing) data.

    This is exactly the property the OLD (pre-rework) implementation
    violated: it computed ``prev_high = wide_df[f"{tf}_high"].shift(1)`` on
    the raw 1-minute forming column directly, so a later row within the
    SAME forming bucket could see an EARLIER row's mutated value from that
    very same bucket -- a form of look-ahead/self-reference this test would
    have caught.
    """
    baseline_df, forming_start_row = _candle_frame()
    mutated_df, _ = _candle_frame(mutate_forming_high_row=2, mutate_forming_high_value=99999.0)

    # sanity: the mutation actually changed the forming candle's own high
    assert (
        baseline_df[f"{_TF}_high"].iloc[forming_start_row + 2]
        != mutated_df[f"{_TF}_high"].iloc[forming_start_row + 2]
    )

    baseline_high, baseline_low = price_levels(baseline_df, tf=_TF, window=_WINDOW, x=2.0)
    mutated_high, mutated_low = price_levels(mutated_df, tf=_TF, window=_WINDOW, x=2.0)

    forming = slice(forming_start_row, forming_start_row + _N_FORMING_ROWS)
    pd.testing.assert_series_equal(
        baseline_high.iloc[forming], mutated_high.iloc[forming], check_names=False
    )
    pd.testing.assert_series_equal(
        baseline_low.iloc[forming], mutated_low.iloc[forming], check_names=False
    )


def test_price_levels_default_window_and_x():
    wide_df, _ = _candle_frame()
    high_default, low_default = price_levels(wide_df, tf=_TF)  # defaults: window=6, x=2.0
    high_explicit, low_explicit = price_levels(wide_df, tf=_TF, window=6, x=2.0)
    pd.testing.assert_series_equal(high_default, high_explicit)
    pd.testing.assert_series_equal(low_default, low_explicit)


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


def _labeled_candle_frame():
    """Extend the shared hand-crafted candle frame with 1_low/1_high entry
    columns on the forming candle's rows, for label_coeff's long/short
    entry-extreme dispatch.
    """
    wide_df, forming_start_row = _candle_frame()
    n = len(wide_df)
    entry_low = pd.Series(np.nan, index=wide_df.index)
    entry_high = pd.Series(np.nan, index=wide_df.index)
    # forming candle 4's rows: distinct, known low/high entry extremes
    entry_low.iloc[forming_start_row : forming_start_row + _N_FORMING_ROWS] = 100.0
    entry_high.iloc[forming_start_row : forming_start_row + _N_FORMING_ROWS] = 150.0
    wide_df["1_low"] = entry_low
    wide_df["1_high"] = entry_high
    return wide_df, forming_start_row


def test_label_coeff_long_uses_entry_low_not_high():
    wide_df, forming_start_row = _labeled_candle_frame()
    lc = label_coeff(wide_df, tf=_TF, direction="long", window=_WINDOW, x=2.0)

    expected_high, expected_low = _expected_level(_CANDLE_HIGH, _CANDLE_LOW, window=_WINDOW, x=2.0)
    expected_coeff = coeff(np.array([100.0]), np.array([expected_low]), np.array([expected_high]))[0]

    assert lc.iloc[forming_start_row] == pytest.approx(expected_coeff)


def test_label_coeff_short_uses_entry_high_not_low():
    wide_df, forming_start_row = _labeled_candle_frame()
    lc = label_coeff(wide_df, tf=_TF, direction="short", window=_WINDOW, x=2.0)

    expected_high, expected_low = _expected_level(_CANDLE_HIGH, _CANDLE_LOW, window=_WINDOW, x=2.0)
    expected_coeff = coeff(np.array([150.0]), np.array([expected_low]), np.array([expected_high]))[0]

    assert lc.iloc[forming_start_row] == pytest.approx(expected_coeff)


def test_label_coeff_long_and_short_differ_when_low_ne_high():
    wide_df, forming_start_row = _labeled_candle_frame()
    long_lc = label_coeff(wide_df, tf=_TF, direction="long", window=_WINDOW, x=2.0)
    short_lc = label_coeff(wide_df, tf=_TF, direction="short", window=_WINDOW, x=2.0)

    assert long_lc.iloc[forming_start_row] != pytest.approx(short_lc.iloc[forming_start_row])


def test_label_coeff_held_constant_across_forming_candle_rows():
    wide_df, forming_start_row = _labeled_candle_frame()
    lc = label_coeff(wide_df, tf=_TF, direction="long", window=_WINDOW, x=2.0)

    forming = lc.iloc[forming_start_row : forming_start_row + _N_FORMING_ROWS]
    assert forming.nunique(dropna=False) == 1
    assert not forming.isna().any()


def test_label_coeff_rejects_bad_direction():
    wide_df, _ = _labeled_candle_frame()
    with pytest.raises(ValueError):
        label_coeff(wide_df, tf=_TF, direction="sideways")


@pytest.mark.parametrize("tf", [15, 60, 240])
def test_label_coeff_bounded_0_1_on_synthetic_fixture(synthetic_wide_df, tf):
    # window=2 (not the function's default 6) is pinned explicitly here so
    # tf=240 gets a non-vacuous (non-empty-after-dropna) warm-up within the
    # fixture's 750 rows: 750 // 240 == 3 completed candles, which is
    # exactly enough for window=2's first valid level (see
    # conftest.py::_N_ROWS's docstring for the full arithmetic). The
    # default-window algorithmic correctness itself is proven exactly by
    # the dedicated hand-crafted-frame tests above; this test is a
    # tf-parameterized smoke check on realistic (non-hand-crafted) data.
    for direction in ("long", "short"):
        lc = label_coeff(synthetic_wide_df, tf=tf, direction=direction, window=2)
        non_nan = lc.dropna()
        assert non_nan.shape[0] > 0, f"tf={tf} direction={direction} produced zero non-NaN rows"
        assert non_nan.between(0, 1).all()
        assert lc.index.equals(synthetic_wide_df.index)
