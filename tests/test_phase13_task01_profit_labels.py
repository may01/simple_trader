"""Tests for lookahead profit labels (phase 13 task 01).

Contains a naive reference implementation (impl B) used as the oracle for the
vectorized production implementation in indicators/labels.py (impl A).
"""

import numpy as np
import pandas as pd
import pytest
import talib

from data import _build_wide_df
from indicators.labels import (
    add_profit_labels,
    add_profit_strict_labels,
    profit_long,
    profit_short,
    profit_strict_long,
    profit_strict_short,
)


# ---------------------------------------------------------------------------
# Reference implementation (impl B) — naive per-entry loop, trivially auditable
# ---------------------------------------------------------------------------


def ref_labels(wide_df, tf, n, m, x, atr_period=14, direction="long",
               l=None, y=None):
    """Oracle: walk 1-minute bars per entry. Strict variant when l/y given."""
    n_rows = len(wide_df)
    closed = wide_df[f"{tf}_is_closed"].to_numpy(dtype=bool)
    high = wide_df[f"{tf}_high"].to_numpy(dtype=float)
    low = wide_df[f"{tf}_low"].to_numpy(dtype=float)
    close = wide_df[f"{tf}_close"].to_numpy(dtype=float)
    one_high = wide_df["1_high"].to_numpy(dtype=float)
    one_low = wide_df["1_low"].to_numpy(dtype=float)

    pos = np.flatnonzero(closed)
    atr_closed = talib.ATR(high[pos], low[pos], close[pos],
                           timeperiod=atr_period)
    out = np.full(n_rows, np.nan)
    win = n * tf

    for k, p in enumerate(pos):
        atr = atr_closed[k]
        if np.isnan(atr):
            continue  # ATR warmup → NaN
        if p + win > n_rows - 1:
            continue  # incomplete future window → NaN
        if l is not None and p + 1 < l:
            continue  # strict: fewer than l rows of history → NaN

        entry = close[p]
        if direction == "long":
            target, stop = entry + m * atr, entry - x * atr
        else:
            target, stop = entry - m * atr, entry + x * atr

        label = 0.0
        for j in range(p + 1, p + win + 1):
            if direction == "long":
                hit_target = one_high[j] > target
                hit_stop = one_low[j] < stop
            else:
                hit_target = one_low[j] < target
                hit_stop = one_high[j] > stop
            if hit_stop:
                label = 0.0  # stop first or same minute → pessimistic 0
                break
            if hit_target:
                label = 1.0
                break

        if l is not None and label == 1.0:
            past_thresh = entry - y * atr if direction == "long" else entry + y * atr
            for j in range(p - l + 1, p + 1):  # entry row inclusive
                if direction == "long":
                    if one_low[j] < past_thresh:
                        label = 0.0
                        break
                else:
                    if one_high[j] > past_thresh:
                        label = 0.0
                        break

        out[p] = label

    return pd.Series(out, index=wide_df.index)


# ---------------------------------------------------------------------------
# Wide-df builders for hand-crafted paths
# ---------------------------------------------------------------------------


def make_wide(high, low, close, tf):
    """Minimal wide df from 1-min h/l/c arrays for a single tf (+ tf=1 cols)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(high), freq="1min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["1_high"] = high
    df["1_low"] = low
    open_index = idx.floor(f"{tf}min")
    df[f"{tf}_high"] = pd.Series(high, index=idx).groupby(open_index).cummax()
    df[f"{tf}_low"] = pd.Series(low, index=idx).groupby(open_index).cummin()
    df[f"{tf}_close"] = close
    df[f"{tf}_is_closed"] = (
        (idx + pd.Timedelta(minutes=1)).floor(f"{tf}min") != open_index
    )
    return df


def baseline_arrays(n_rows):
    """Flat market: every 1-min bar high=101 low=99 close=100 → tf ATR == 2."""
    return (np.full(n_rows, 101.0), np.full(n_rows, 99.0),
            np.full(n_rows, 100.0))


# Baseline geometry used by hand tests: tf=5, atr_period=2, n=2.
# Closed rows at minutes 4, 9, 14, ... — ATR valid from minute 14 on.
# Entry under test: minute 24 (entry=100, atr=2). Window: minutes 25..34.
# Long: m=2 → target 104, x=1 → stop 98.
TF, N, M, X, ATR_P = 5, 2, 2.0, 1.0, 2
ENTRY = 24
ROWS = 40


def run_long(high, low, close):
    return profit_long(make_wide(high, low, close, TF), TF, N, M, X,
                       atr_period=ATR_P)


def run_short(high, low, close):
    return profit_short(make_wide(high, low, close, TF), TF, N, M, X,
                        atr_period=ATR_P)


# ---------------------------------------------------------------------------
# Unit tests — plain labels, long
# ---------------------------------------------------------------------------


class TestProfitLong:
    def test_target_touched_stop_never(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 105.0  # > target 104
        assert run_long(h, l, c).iloc[ENTRY] == 1.0

    def test_stop_first_target_later(self):
        h, l, c = baseline_arrays(ROWS)
        l[26] = 97.5   # < stop 98 first
        h[28] = 105.0  # target later
        assert run_long(h, l, c).iloc[ENTRY] == 0.0

    def test_both_same_minute_pessimistic(self):
        h, l, c = baseline_arrays(ROWS)
        h[27] = 105.0
        l[27] = 97.5
        assert run_long(h, l, c).iloc[ENTRY] == 0.0

    def test_neither_touched(self):
        h, l, c = baseline_arrays(ROWS)
        assert run_long(h, l, c).iloc[ENTRY] == 0.0

    def test_touch_requires_strict_inequality(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 104.0  # == target, not >
        assert run_long(h, l, c).iloc[ENTRY] == 0.0

    def test_window_truncated_is_nan(self):
        h, l, c = baseline_arrays(ROWS)
        s = run_long(h, l, c)
        # Entry at minute 34 needs rows 35..44; only 35..39 exist.
        assert np.isnan(s.iloc[34])

    def test_atr_warmup_is_nan(self):
        h, l, c = baseline_arrays(ROWS)
        s = run_long(h, l, c)
        # talib ATR(2) is NaN for the first 2 closed candles (minutes 4, 9).
        assert np.isnan(s.iloc[4]) and np.isnan(s.iloc[9])
        assert not np.isnan(s.iloc[14])

    def test_values_only_at_closed_rows(self):
        h, l, c = baseline_arrays(ROWS)
        wide = make_wide(h, l, c, TF)
        s = profit_long(wide, TF, N, M, X, atr_period=ATR_P)
        assert s.index.equals(wide.index)
        assert s[~wide[f"{TF}_is_closed"].astype(bool)].isna().all()


# ---------------------------------------------------------------------------
# Unit tests — plain labels, short mirror (target 96, stop 102)
# ---------------------------------------------------------------------------


class TestProfitShort:
    def test_target_touched_stop_never(self):
        h, l, c = baseline_arrays(ROWS)
        l[28] = 95.0  # < target 96
        assert run_short(h, l, c).iloc[ENTRY] == 1.0

    def test_stop_first_target_later(self):
        h, l, c = baseline_arrays(ROWS)
        h[26] = 102.5  # > stop 102 first
        l[28] = 95.0
        assert run_short(h, l, c).iloc[ENTRY] == 0.0

    def test_both_same_minute_pessimistic(self):
        h, l, c = baseline_arrays(ROWS)
        h[27] = 102.5
        l[27] = 95.0
        assert run_short(h, l, c).iloc[ENTRY] == 0.0

    def test_neither_touched(self):
        h, l, c = baseline_arrays(ROWS)
        assert run_short(h, l, c).iloc[ENTRY] == 0.0

    def test_window_truncated_is_nan(self):
        h, l, c = baseline_arrays(ROWS)
        assert np.isnan(run_short(h, l, c).iloc[34])

    def test_atr_warmup_is_nan(self):
        h, l, c = baseline_arrays(ROWS)
        s = run_short(h, l, c)
        assert np.isnan(s.iloc[4]) and np.isnan(s.iloc[9])


# ---------------------------------------------------------------------------
# Unit tests — strict variant
# ---------------------------------------------------------------------------
# Deep dips/spikes (90 / 110) shift ATR via the candle they land in; magnitudes
# are chosen so the past condition outcome is unambiguous regardless, and the
# forward touch (120 / 80) clears any resulting target.

L, Y = 8, 0.5  # past window: rows 17..24 inclusive for entry 24


def run_strict_long(high, low, close, l=L, y=Y):
    return profit_strict_long(make_wide(high, low, close, TF), TF, N, M, X,
                              l=l, y=y, atr_period=ATR_P)


def run_strict_short(high, low, close, l=L, y=Y):
    return profit_strict_short(make_wide(high, low, close, TF), TF, N, M, X,
                               l=l, y=y, atr_period=ATR_P)


class TestProfitStrictLong:
    def test_bottom_forward_profit_past_clean(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 120.0
        assert run_strict_long(h, l, c).iloc[ENTRY] == 1.0

    def test_rising_slope_past_dip_kills_label(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 120.0
        l[19] = 90.0  # deep dip inside past window
        assert run_strict_long(h, l, c).iloc[ENTRY] == 0.0

    def test_dip_at_window_edge_counts(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 120.0
        l[17] = 90.0  # oldest row of the 8-row window
        assert run_strict_long(h, l, c).iloc[ENTRY] == 0.0

    def test_dip_one_row_before_window_ignored(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 120.0
        l[16] = 90.0  # just outside the window
        assert run_strict_long(h, l, c).iloc[ENTRY] == 1.0

    def test_entry_minute_own_wick_counts(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 120.0
        l[24] = 90.0  # entry row itself, window inclusive
        assert run_strict_long(h, l, c).iloc[ENTRY] == 0.0

    def test_short_history_is_nan(self):
        h, l, c = baseline_arrays(ROWS)
        h[18] = 120.0
        s = run_strict_long(h, l, c, l=16)
        # Entry minute 14 has only 15 rows of history (0..14) < 16.
        assert np.isnan(s.iloc[14])

    def test_exactly_l_rows_of_history_is_valid(self):
        h, l, c = baseline_arrays(ROWS)
        h[18] = 120.0
        s = run_strict_long(h, l, c, l=15)
        assert not np.isnan(s.iloc[14])

    def test_forward_not_profitable_stays_zero(self):
        h, l, c = baseline_arrays(ROWS)  # clean past, no forward touch
        assert run_strict_long(h, l, c).iloc[ENTRY] == 0.0


class TestProfitStrictShort:
    def test_top_forward_profit_past_clean(self):
        h, l, c = baseline_arrays(ROWS)
        l[28] = 80.0
        assert run_strict_short(h, l, c).iloc[ENTRY] == 1.0

    def test_falling_slope_past_spike_kills_label(self):
        h, l, c = baseline_arrays(ROWS)
        l[28] = 80.0
        h[19] = 110.0
        assert run_strict_short(h, l, c).iloc[ENTRY] == 0.0

    def test_spike_at_window_edge_counts(self):
        h, l, c = baseline_arrays(ROWS)
        l[28] = 80.0
        h[17] = 110.0
        assert run_strict_short(h, l, c).iloc[ENTRY] == 0.0

    def test_spike_one_row_before_window_ignored(self):
        h, l, c = baseline_arrays(ROWS)
        l[28] = 80.0
        h[16] = 110.0
        assert run_strict_short(h, l, c).iloc[ENTRY] == 1.0

    def test_entry_minute_own_wick_counts(self):
        h, l, c = baseline_arrays(ROWS)
        l[28] = 80.0
        h[24] = 110.0
        assert run_strict_short(h, l, c).iloc[ENTRY] == 0.0

    def test_short_history_is_nan(self):
        h, l, c = baseline_arrays(ROWS)
        l[18] = 80.0
        s = run_strict_short(h, l, c, l=16)
        assert np.isnan(s.iloc[14])


# ---------------------------------------------------------------------------
# Property test — vectorized impl A ≡ reference impl B on random walks
# ---------------------------------------------------------------------------


def random_walk_arrays(n_rows, seed):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n_rows))
    spread = np.abs(rng.normal(0, 0.2, n_rows))
    return close + spread, close - spread, close


@pytest.mark.parametrize("tf,n,m,x,l,y", [
    (15, 4, 2.0, 1.0, 8, 0.5),
    (15, 2, 1.5, 0.5, 30, 1.0),
    (60, 3, 2.0, 1.0, 60, 0.5),
    (60, 1, 1.0, 1.5, 10, 2.0),
])
def test_vectorized_matches_reference(tf, n, m, x, l, y):
    h, lo, c = random_walk_arrays(4000, seed=tf * 1000 + n)
    wide = make_wide(h, lo, c, tf)
    atr_p = 3

    pd.testing.assert_series_equal(
        profit_long(wide, tf, n, m, x, atr_period=atr_p),
        ref_labels(wide, tf, n, m, x, atr_period=atr_p, direction="long"),
        check_names=False)
    pd.testing.assert_series_equal(
        profit_short(wide, tf, n, m, x, atr_period=atr_p),
        ref_labels(wide, tf, n, m, x, atr_period=atr_p, direction="short"),
        check_names=False)
    pd.testing.assert_series_equal(
        profit_strict_long(wide, tf, n, m, x, l=l, y=y, atr_period=atr_p),
        ref_labels(wide, tf, n, m, x, atr_period=atr_p, direction="long",
                   l=l, y=y),
        check_names=False)
    pd.testing.assert_series_equal(
        profit_strict_short(wide, tf, n, m, x, l=l, y=y, atr_period=atr_p),
        ref_labels(wide, tf, n, m, x, atr_period=atr_p, direction="short",
                   l=l, y=y),
        check_names=False)


def test_labels_not_all_trivial_on_random_walk():
    """Sanity: the property test exercises both 0s and 1s."""
    h, lo, c = random_walk_arrays(4000, seed=42)
    wide = make_wide(h, lo, c, 15)
    s = profit_long(wide, 15, 2, 1.0, 1.0, atr_period=3).dropna()
    assert (s == 1.0).any() and (s == 0.0).any()


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------


class TestAddLabels:
    def test_add_profit_labels_naming(self):
        h, l, c = baseline_arrays(ROWS)
        wide = make_wide(h, l, c, 5)
        add_profit_labels(wide, 5, n=4, m=2.0, x=1.5, atr_period=ATR_P)
        assert "5_plong_n4_m2_x1p5" in wide.columns
        assert "5_pshort_n4_m2_x1p5" in wide.columns

    def test_add_profit_strict_labels_naming(self):
        h, l, c = baseline_arrays(ROWS)
        wide = make_wide(h, l, c, 5)
        add_profit_strict_labels(wide, 5, n=4, m=2.0, x=1.5, l=8, y=0.5,
                                 atr_period=ATR_P)
        assert "5_pslong_n4_m2_x1p5_l8_y0p5" in wide.columns
        assert "5_psshort_n4_m2_x1p5_l8_y0p5" in wide.columns

    def test_add_matches_function_values(self):
        h, l, c = baseline_arrays(ROWS)
        h[28] = 105.0
        wide = make_wide(h, l, c, TF)
        expected = profit_long(wide, TF, N, M, X, atr_period=ATR_P)
        add_profit_labels(wide, TF, n=N, m=M, x=X, atr_period=ATR_P)
        pd.testing.assert_series_equal(
            wide["5_plong_n2_m2_x1"], expected, check_names=False)


# ---------------------------------------------------------------------------
# Integration — real wide-df builder from data.py
# ---------------------------------------------------------------------------


def test_labels_on_wide_df():
    n_rows = 2880  # 2 days of 1-min bars
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1min", tz="UTC")
    h, lo, c = random_walk_arrays(n_rows, seed=7)
    synthetic_ohlcv_df = pd.DataFrame(
        {
            "open": c,
            "high": h,
            "low": lo,
            "close": c,
            "volume": np.ones(n_rows),
            "taker_base_vol": np.ones(n_rows) * 0.5,
        },
        index=idx,
    )

    wide = _build_wide_df(synthetic_ohlcv_df)
    s = profit_long(wide, tf=15, n=4, m=2.0, x=1.0)
    assert s.index.equals(wide.index)
    assert s[~wide["15_is_closed"].astype(bool)].isna().all()
    assert s.dropna().isin([0, 1]).all()
