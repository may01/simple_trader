"""Unit tests for data.py — DataPoint, LiveDataPoint, WideDataPoint."""

import math
import numpy as np
import pandas as pd
import pytest

from data import DataPoint, LiveDataPoint, WideDataPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_live_ohlc(tf: int, n: int = 5, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal per-TF DataFrame with columns {tf}_open, {tf}_close, etc."""
    idx = pd.date_range(start, periods=n, freq=f"{tf}min", tz="UTC")
    return pd.DataFrame(
        {
            f"{tf}_open":  [float(i + 1) for i in range(n)],
            f"{tf}_close": [float(i + 10) for i in range(n)],
            f"{tf}_high":  [float(i + 11) for i in range(n)],
            f"{tf}_low":   [float(i + 0.5) for i in range(n)],
        },
        index=idx,
    )


def _make_wide_df() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build a wide DataFrame that covers tf=1 and tf=5 with is_closed columns."""
    # 10 rows at 1-minute intervals
    idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    data = {}

    # columns built below via direct dict assignment

    # tf=1: every row is a new 1-min candle — all closed except the last
    data["1_open"]      = [float(i + 1) for i in range(10)]
    data["1_close"]     = [float(i + 10) for i in range(10)]
    data["1_high"]      = [float(i + 11) for i in range(10)]
    data["1_low"]       = [float(i + 0.5) for i in range(10)]
    # rows 0..8 are closed; row 9 is the partial (current) candle
    data["1_is_closed"] = [True] * 9 + [False]

    # tf=5: a new 5-min candle closes every 5 rows (row 4 and row 9)
    data["5_open"]      = [float(i + 100) for i in range(10)]
    data["5_close"]     = [float(i + 200) for i in range(10)]
    data["5_high"]      = [float(i + 201) for i in range(10)]
    data["5_low"]       = [float(i + 99) for i in range(10)]
    # closed at row index 4 and row index 9
    is_closed_5 = [False] * 10
    is_closed_5[4] = True
    is_closed_5[9] = True
    data["5_is_closed"] = is_closed_5

    df = pd.DataFrame(data, index=idx)
    ts = idx[-1]  # timestamp = last row
    return df, ts


# ---------------------------------------------------------------------------
# LiveDataPoint tests
# ---------------------------------------------------------------------------

class TestLiveDataPoint:
    def test_get_shift0(self):
        """get() with shift=0 returns the last row's value."""
        ohlc = {1: _make_live_ohlc(1)}
        ldp = LiveDataPoint(ohlc)
        # last row (index 4) for tf=1: open = 5.0
        assert ldp.get("open", tf=1, shift=0) == 5.0

    def test_get_shift1(self):
        """get() with shift=1 returns the second-to-last row."""
        ohlc = {1: _make_live_ohlc(1)}
        ldp = LiveDataPoint(ohlc)
        # second-to-last row (index 3): open = 4.0
        assert ldp.get("open", tf=1, shift=1) == 4.0

    def test_get_shift_default_is_zero(self):
        """shift defaults to 0."""
        ohlc = {1: _make_live_ohlc(1)}
        ldp = LiveDataPoint(ohlc)
        assert ldp.get("close", tf=1) == ldp.get("close", tf=1, shift=0)

    def test_get_multiple_tfs(self):
        """get() works for different timeframes."""
        ohlc = {
            1: _make_live_ohlc(1, n=5),
            5: _make_live_ohlc(5, n=5),
        }
        ldp = LiveDataPoint(ohlc)
        # tf=5 last row: open = 5.0 (same formula: i+1, last i=4)
        assert ldp.get("open", tf=5, shift=0) == 5.0
        assert ldp.get("open", tf=1, shift=0) == 5.0

    def test_timestamp_returns_last_index_of_tf1(self):
        """timestamp property returns the last index entry of the tf=1 DataFrame."""
        ohlc = {1: _make_live_ohlc(1, n=5, start="2024-06-01")}
        ldp = LiveDataPoint(ohlc)
        expected = pd.Timestamp("2024-06-01 00:04:00", tz="UTC")
        assert ldp.timestamp == expected

    def test_get_df_returns_underlying_dataframe(self):
        """get_df(tf) returns the exact same DataFrame object passed in."""
        df_1 = _make_live_ohlc(1)
        ohlc = {1: df_1}
        ldp = LiveDataPoint(ohlc)
        assert ldp.get_df(1) is df_1

    def test_cur_price_uses_tf1(self):
        """cur_price() returns get(price_type, tf=1)."""
        ohlc = {1: _make_live_ohlc(1, n=3)}
        ldp = LiveDataPoint(ohlc)
        # last row (index 2): open = 3.0, close = 12.0
        assert ldp.cur_price("open") == 3.0
        assert ldp.cur_price("close") == 12.0

    def test_cur_price_equals_get_tf1(self):
        """cur_price(x) == get(x, tf=1)."""
        ohlc = {1: _make_live_ohlc(1, n=4)}
        ldp = LiveDataPoint(ohlc)
        for col in ("open", "close", "high", "low"):
            assert ldp.cur_price(col) == ldp.get(col, tf=1)

    def test_get_shift_exceeds_len_returns_nan(self):
        """get() returns float('nan') when shift >= len(df) (not raise)."""
        ohlc = {1: _make_live_ohlc(1, n=3)}
        ldp = LiveDataPoint(ohlc)
        # n=3 rows; shift=3 is exactly out-of-bounds
        result = ldp.get("open", tf=1, shift=3)
        assert math.isnan(result)
        # shift well beyond length also returns nan
        result2 = ldp.get("open", tf=1, shift=100)
        assert math.isnan(result2)

    def test_get_missing_tf_raises_key_error(self):
        """get() raises KeyError with descriptive message when tf is absent."""
        ohlc = {1: _make_live_ohlc(1)}
        ldp = LiveDataPoint(ohlc)
        with pytest.raises(KeyError, match="tf=5 not in LiveDataPoint"):
            ldp.get("open", tf=5)


# ---------------------------------------------------------------------------
# DataPoint ABC enforcement
# ---------------------------------------------------------------------------

class TestDataPointABC:
    def test_cannot_instantiate_abstract_base(self):
        """Instantiating DataPoint directly raises TypeError."""
        with pytest.raises(TypeError):
            DataPoint()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# WideDataPoint tests
# ---------------------------------------------------------------------------

class TestWideDataPoint:
    @pytest.fixture
    def wide(self):
        df, ts = _make_wide_df()
        return WideDataPoint(df, ts)

    @pytest.fixture
    def wide_df(self):
        df, _ = _make_wide_df()
        return df

    def test_timestamp_property(self, wide):
        """timestamp returns the ts passed to __init__."""
        _, ts = _make_wide_df()
        assert wide.timestamp == ts

    def test_get_shift0_returns_current_row_value(self, wide, wide_df):
        """shift=0 returns df.loc[ts, col] directly."""
        df, ts = _make_wide_df()
        # At ts (last row, index 9): 1_close = 9 + 10 = 19.0
        assert wide.get("close", tf=1, shift=0) == df.loc[ts, "1_close"]

    def test_get_shift0_partial_candle(self, wide, wide_df):
        """shift=0 can return a partial-candle value (1_is_closed=False at ts)."""
        df, ts = _make_wide_df()
        # Row 9 is partial for tf=1 (1_is_closed=False), but shift=0 still returns it
        val = wide.get("open", tf=1, shift=0)
        assert val == df.loc[ts, "1_open"]

    def test_get_shift1_looks_at_is_closed_rows(self, wide, wide_df):
        """shift=1 returns the last {tf}_is_closed=True row's value, at or before ts."""
        df, ts = _make_wide_df()
        # For tf=1: closed rows are index 0..8 (is_closed=True); last closed row is index 8
        # 1_close at index 8 = 8 + 10 = 18.0
        expected = df.loc[df.index[8], "1_close"]
        assert wide.get("close", tf=1, shift=1) == expected

    def test_get_shift1_tf5_returns_last_closed(self, wide, wide_df):
        """shift=1 for tf=5 returns the most recent closed-5m candle value."""
        df, ts = _make_wide_df()
        # For tf=5: closed rows are index 4 and 9; last closed row at or before ts=9 is index 9
        # 5_close at index 9 = 9 + 200 = 209.0
        expected = df.loc[df.index[9], "5_close"]
        assert wide.get("close", tf=5, shift=1) == expected

    def test_get_shift2_tf5_returns_second_last_closed(self, wide, wide_df):
        """shift=2 for tf=5 returns the second-to-last closed candle."""
        df, ts = _make_wide_df()
        # For tf=5: closed rows are index 4 and 9; 2nd-last = index 4
        # 5_close at index 4 = 4 + 200 = 204.0
        expected = df.loc[df.index[4], "5_close"]
        assert wide.get("close", tf=5, shift=2) == expected

    def test_get_returns_nan_when_shift_exceeds_history(self, wide):
        """Returns float('nan') when shift > number of closed rows."""
        # For tf=5 there are only 2 closed rows (indices 4 and 9)
        result = wide.get("close", tf=5, shift=3)
        assert math.isnan(result)

    def test_get_shift0_uses_exact_column_name(self, wide, wide_df):
        """Column lookup uses '{tf}_{col}' format."""
        df, ts = _make_wide_df()
        assert wide.get("high", tf=1, shift=0) == df.loc[ts, "1_high"]

    def test_cur_price_uses_tf1(self, wide, wide_df):
        """cur_price(price_type) calls get(price_type, tf=1)."""
        df, ts = _make_wide_df()
        assert wide.cur_price("open") == wide.get("open", tf=1)
        assert wide.cur_price("close") == wide.get("close", tf=1)

    def test_get_df_returns_slice_up_to_ts(self, wide, wide_df):
        """get_df(tf) returns a slice of the wide df up to ts (placeholder impl)."""
        df, ts = _make_wide_df()
        result = wide.get_df(1)
        # The result must be a DataFrame and its last index must be ts
        assert isinstance(result, pd.DataFrame)
        assert result.index[-1] == ts

    def test_wide_data_point_earlier_ts(self):
        """WideDataPoint at an earlier ts uses rows up to that ts."""
        df, _ = _make_wide_df()
        # Use ts = index[6] (7th row, 0-indexed)
        ts = df.index[6]
        wdp = WideDataPoint(df, ts)
        # shift=0: 1_open at row 6 = 6+1 = 7.0
        assert wdp.get("open", tf=1, shift=0) == df.loc[ts, "1_open"]
        # is_closed for tf=1: rows 0..8 are True, row 9 is False.
        # closed rows up to ts (index 6) = rows 0..6 (all True, since row 6 < 9).
        # shift=1 → last closed row = row 6 → 1_close at row 6 = 6+10 = 16.0
        assert wdp.get("close", tf=1, shift=1) == df.loc[df.index[6], "1_close"]
        # shift=2 → second-to-last closed row = row 5 → 1_close = 5+10 = 15.0
        assert wdp.get("close", tf=1, shift=2) == df.loc[df.index[5], "1_close"]

    def test_get_shift0_ts_not_in_index_raises(self):
        """get(shift=0) raises KeyError when ts is not in the DataFrame index.

        This documents the contract: ts must always be present in the index.
        A missing ts is a programming error (SimulationData only uses timestamps
        from the index), so the raw KeyError from .loc is intentional.
        """
        df, _ = _make_wide_df()
        bad_ts = pd.Timestamp("2099-01-01", tz="UTC")
        wdp = WideDataPoint(df, bad_ts)
        with pytest.raises(KeyError):
            wdp.get("open", tf=1, shift=0)

    def test_get_nn_res_resolves_bare_column(self):
        """nn_res_* columns carry no {tf}_ prefix — looked up by the bare name.

        The same value is returned for any tf (timeframe-agnostic), unlike
        ordinary indicators which resolve as '{tf}_{col}'.
        """
        df, ts = _make_wide_df()
        df["nn_res_dir15_prob_up"] = [float(i) / 10 for i in range(len(df))]
        wdp = WideDataPoint(df, ts)

        # Bare column name resolution (NOT '15_nn_res_dir15_prob_up').
        assert wdp.get("nn_res_dir15_prob_up", tf=15) == df.loc[ts, "nn_res_dir15_prob_up"]
        # Same value for any tf — column is shared across timeframes.
        assert wdp.get("nn_res_dir15_prob_up", tf=60) == df.loc[ts, "nn_res_dir15_prob_up"]

    def test_get_nn_res_shift_resolves_bare_column(self):
        """shift>0 for an nn_res_* column also resolves the bare name."""
        df, ts = _make_wide_df()
        df["nn_res_dir15_prob_up"] = [float(i) for i in range(len(df))]
        wdp = WideDataPoint(df, ts)
        # tf=1 closed rows are indices 0..8; shift=1 → last closed row (index 8).
        assert wdp.get("nn_res_dir15_prob_up", tf=1, shift=1) == df["nn_res_dir15_prob_up"].iloc[8]
