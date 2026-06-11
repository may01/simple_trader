"""Unit tests for IndicatorField subclasses (Task 04).

Tests verify compute() outputs for all indicator field classes.
"""

from __future__ import annotations

import json
import math
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from config_loader import IndicatorFieldConfig
from data import LiveDataPoint
from indicators import (
    # Momentum
    RSI14Field,
    RSI_MAField,
    RSI_MA_DiffField,
    # Trend
    EMAField,
    MACDField,
    MACDSignalField,
    MACDHistField,
    MACDFastField,
    MACDFastSignalField,
    # Oscillators / Volatility
    SARField,
    ATR14Field,
    NATR14Field,
    ATR_MAField,
    BollingerUpperField,
    BollingerMiddleField,
    BollingerLowerField,
    BollingerFastUpperField,
    BollingerFastLowerField,
    BollingerWideUpperField,
    BollingerWideLowerField,
    CCI14Field,
    CCI_MAField,
    # Volume
    VolMAField,
    VolBuyMAField,
    VolSellMAField,
    # Price derivatives
    CloseDiffPrcField,
    CloseDiffPrcRMField,
    CloseDiffPrcRMMeanAboveField,
    CloseDiffPrcRMMeanBelowField,
    HighDiffPrcField,
    HighDiffPrcRMField,
    HighDiffPrcRMMeanAboveField,
    HighDiffPrcRMMeanBelowField,
    LowDiffPrcField,
    LowDiffPrcRMField,
    LowDiffPrcRMMeanAboveField,
    LowDiffPrcRMMeanBelowField,
    # Classification
    MoveClassField,
    ZoneClassField,
    OverLowField,
    OverHighField,
    # Targets
    TgtLongField,
    SLLongField,
    TgtShortField,
    SLShortField,
    ZBField,
    ZSField,
    # Trend flags
    TrendUpField,
    TrendDownField,
    # NN features
    NNRSINormField,
    NNCloseDiffATRField,
)


def make_cfg(name: str, group: str, applies_to: list, depends_on: list) -> IndicatorFieldConfig:
    """Build a minimal IndicatorFieldConfig for testing."""
    return IndicatorFieldConfig(
        name=name,
        group=group,
        applies_to=applies_to,
        depends_on=depends_on,
        library=None,
        params={},
    )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_indicator_df(tf: int = 5, n: int = 120) -> pd.DataFrame:
    """Build synthetic closed-candle DataFrame with {tf}_* columns."""
    np.random.seed(42)
    idx = pd.date_range("2023-09-01", periods=n, freq=f"{tf}min", tz="UTC")
    close = np.cumsum(np.random.normal(0, 0.1, n)) + 20.0
    close = np.abs(close)
    high = close * np.random.uniform(1.001, 1.01, n)
    low = close * np.random.uniform(0.99, 0.999, n)
    df = pd.DataFrame(
        {
            f"{tf}_open": close * np.random.uniform(0.999, 1.001, n),
            f"{tf}_high": high,
            f"{tf}_low": low,
            f"{tf}_close": close,
            f"{tf}_volume": np.random.uniform(100, 200, n),
            f"{tf}_buy_volume": np.random.uniform(50, 100, n),
            f"{tf}_is_closed": True,
            f"{tf}_open_index": idx,
        },
        index=idx,
    )
    return df


def make_dp(tf: int = 5, n: int = 120) -> tuple[LiveDataPoint, pd.DataFrame]:
    """Return (LiveDataPoint, df) for given tf."""
    df = make_indicator_df(tf=tf, n=n)
    dp = LiveDataPoint({tf: df})
    return dp, df


# ---------------------------------------------------------------------------
# Helper to compute and write a field to df, then return the series
# ---------------------------------------------------------------------------

def run_field(dp: LiveDataPoint, df: pd.DataFrame, field, tf: int = 5) -> pd.Series:
    """Run field.compute(dp, tf), write result to df, return series."""
    series = field.compute(dp, tf)
    df[f"{tf}_{field.name}"] = series
    return series


# ---------------------------------------------------------------------------
# 1. RSI14 output range
# ---------------------------------------------------------------------------

class TestRSI14:
    def test_rsi14_output_range(self):
        """RSI14 values between 0 and 100 (ignoring NaN warmup)."""
        tf = 5
        dp, df = make_dp(tf=tf)
        field = RSI14Field()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

        valid = series.dropna()
        assert len(valid) > 0, "Expected non-NaN RSI values"
        assert (valid >= 0).all(), "RSI values must be >= 0"
        assert (valid <= 100).all(), "RSI values must be <= 100"


# ---------------------------------------------------------------------------
# 2. RSI MA8 depends on rsi_14
# ---------------------------------------------------------------------------

class TestRSIMA:
    def test_rsi_ma8_depends_on_rsi14(self):
        """Compute RSI14 first, write column, then RSI_MA8 produces non-NaN output."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        # Step 1: compute RSI14 and write it to df
        rsi14 = RSI14Field()
        run_field(dp, df, rsi14, tf)

        # Step 2: compute rsi_ma8 — it reads df[f"{tf}_rsi_14"]
        rsi_ma8 = RSI_MAField(source="rsi_14", length=8, name="rsi_ma8")
        series = run_field(dp, df, rsi_ma8, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)
        valid = series.dropna()
        assert len(valid) > 0, "Expected non-NaN values after warmup"

    def test_rsi_ma_diff(self):
        """RSI_MA_DiffField returns series of differences."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        rsi14 = RSI14Field()
        run_field(dp, df, rsi14, tf)

        rsi_ma8 = RSI_MAField(source="rsi_14", length=8, name="rsi_ma8")
        run_field(dp, df, rsi_ma8, tf)

        diff_field = RSI_MA_DiffField(source_ma="rsi_ma8", name="rsi_ma8_diff")
        series = run_field(dp, df, diff_field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)


# ---------------------------------------------------------------------------
# 3. EMA50 output length and NaN warmup
# ---------------------------------------------------------------------------

class TestEMAField:
    def test_ema50_output_length(self):
        """EMA50 Series same length as input, NaN for first 49 rows."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = EMAField(length=50, name="ema_50")
        series = run_field(dp, df, field, tf)

        assert len(series) == len(df)
        # First 49 should be NaN (talib EMA warmup)
        assert all(math.isnan(v) for v in series.iloc[:49])
        # Row 50 onward should have values
        valid = series.dropna()
        assert len(valid) > 0

    def test_ema7_fewer_warmup_rows(self):
        """EMA7 has fewer NaN warmup rows than EMA50."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        ema7 = EMAField(length=7, name="ema_7")
        series = run_field(dp, df, ema7, tf)
        valid_ema7 = series.dropna()

        ema50 = EMAField(length=50, name="ema_50")
        series50 = run_field(dp, df, ema50, tf)
        valid_ema50 = series50.dropna()

        assert len(valid_ema7) > len(valid_ema50)


# ---------------------------------------------------------------------------
# 4. MACD produces Series
# ---------------------------------------------------------------------------

class TestMACDFields:
    def test_macd_produces_series(self):
        """MACD output is Series, same length as input."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = MACDField()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

    def test_macd_signal_produces_series(self):
        """MACDSignalField returns Series of same length."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = MACDSignalField()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

    def test_macd_hist_produces_series(self):
        """MACDHistField returns Series of same length."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = MACDHistField()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

    def test_macd_fast_produces_series(self):
        """MACDFastField returns Series of same length."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = MACDFastField()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)


# ---------------------------------------------------------------------------
# 5. ATR14 positive values
# ---------------------------------------------------------------------------

class TestATRFields:
    def test_atr14_positive_values(self):
        """ATR14 values >= 0 (ignoring NaN)."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = ATR14Field()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        valid = series.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_natr14_produces_series(self):
        """NATR14 returns Series of same length."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = NATR14Field()
        series = run_field(dp, df, field, tf)
        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

    def test_atr_ma_depends_on_atr14(self):
        """ATR_MAField produces non-NaN output after atr_14 is computed."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        atr14 = ATR14Field()
        run_field(dp, df, atr14, tf)

        atr_14_ma_20 = ATR_MAField()
        series = run_field(dp, df, atr_14_ma_20, tf)

        valid = series.dropna()
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# 6. Bollinger bands: upper > lower
# ---------------------------------------------------------------------------

class TestBollingerFields:
    def test_bollinger_upper_above_lower(self):
        """BB upper > BB lower for non-NaN rows."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        upper_field = BollingerUpperField()
        middle_field = BollingerMiddleField()
        lower_field = BollingerLowerField()

        upper = run_field(dp, df, upper_field, tf)
        middle = run_field(dp, df, middle_field, tf)
        lower = run_field(dp, df, lower_field, tf)

        # Align by dropping NaN
        mask = upper.notna() & lower.notna()
        assert mask.any(), "Expected non-NaN rows"
        assert (upper[mask] > lower[mask]).all(), "Upper band must be > lower band"
        assert (upper[mask] >= middle[mask]).all(), "Upper >= middle"
        assert (middle[mask] >= lower[mask]).all(), "Middle >= lower"

    def test_bollinger_fast_fields(self):
        """BollingerFastUpper and BollingerFastLower return Series."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        fast_upper = run_field(dp, df, BollingerFastUpperField(), tf)
        fast_lower = run_field(dp, df, BollingerFastLowerField(), tf)

        mask = fast_upper.notna() & fast_lower.notna()
        assert mask.any()
        assert (fast_upper[mask] > fast_lower[mask]).all()

    def test_bollinger_wide_fields(self):
        """BollingerWideUpper and BollingerWideLower return Series."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        wide_upper = run_field(dp, df, BollingerWideUpperField(), tf)
        wide_lower = run_field(dp, df, BollingerWideLowerField(), tf)

        mask = wide_upper.notna() & wide_lower.notna()
        assert mask.any()
        assert (wide_upper[mask] > wide_lower[mask]).all()


# ---------------------------------------------------------------------------
# 7. CCI14 produces Series
# ---------------------------------------------------------------------------

class TestCCI14:
    def test_cci14_produces_series(self):
        """CCI14 returns Series same length."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = CCI14Field()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

    def test_cci_ma_depends_on_cci14(self):
        """CCI_MAField produces non-NaN output after cci_14 is computed."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        cci14 = CCI14Field()
        run_field(dp, df, cci14, tf)

        cci_14_ma_20 = CCI_MAField()
        series = run_field(dp, df, cci_14_ma_20, tf)
        valid = series.dropna()
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# 8. VolMA uses volume column
# ---------------------------------------------------------------------------

class TestVolumeFields:
    def test_vol_ma_uses_volume_column(self):
        """VolMA returns rolling mean, check last non-NaN value."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = VolMAField()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        valid = series.dropna()
        assert len(valid) > 0

        # Last value should equal rolling(20).mean() of volume
        expected_last = df[f"{tf}_volume"].rolling(20).mean().iloc[-1]
        assert abs(series.iloc[-1] - expected_last) < 1e-10

    def test_vol_buy_ma(self):
        """VolBuyMA returns rolling mean of buy_volume."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = VolBuyMAField()
        series = run_field(dp, df, field, tf)

        expected_last = df[f"{tf}_buy_volume"].rolling(20).mean().iloc[-1]
        assert abs(series.iloc[-1] - expected_last) < 1e-10

    def test_vol_sell_ma_depends_on_vol_ma_and_buy_ma(self):
        """VolSellMA = vol_ma_20 - vol_buy_ma_20."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        vol_ma_field = VolMAField()
        run_field(dp, df, vol_ma_field, tf)

        vol_buy_ma_field = VolBuyMAField()
        run_field(dp, df, vol_buy_ma_field, tf)

        vol_sell = VolSellMAField()
        series = run_field(dp, df, vol_sell, tf)

        # spot-check: last valid value
        expected_last = df[f"{tf}_vol_ma_20"].iloc[-1] - df[f"{tf}_vol_buy_ma_20"].iloc[-1]
        assert abs(series.iloc[-1] - expected_last) < 1e-10


# ---------------------------------------------------------------------------
# 9. CloseDiffPrc formula
# ---------------------------------------------------------------------------

class TestPriceDerivatives:
    def test_close_diff_prc_formula(self):
        """Verify formula: (close[i] - close[i-1]) / close[i-1] * 100."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = CloseDiffPrcField()
        series = run_field(dp, df, field, tf)

        close = df[f"{tf}_close"]
        for i in range(1, min(10, len(df))):
            expected = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1] * 100
            assert abs(series.iloc[i] - expected) < 1e-10, (
                f"Row {i}: expected {expected}, got {series.iloc[i]}"
            )

        # First row should be NaN
        assert math.isnan(series.iloc[0])

    def test_close_diff_prc_rm(self):
        """CloseDiffPrcRM is a rolling mean of close_diff_prc."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        cdp = CloseDiffPrcField()
        run_field(dp, df, cdp, tf)

        cdp_rm = CloseDiffPrcRMField()
        series = run_field(dp, df, cdp_rm, tf)

        valid = series.dropna()
        assert len(valid) > 0

    def test_close_diff_prc_rm_mean_above_non_negative(self):
        """CloseDiffPrcRMMeanAbove returns non-negative values (or 0 if none positive)."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        for fld in [CloseDiffPrcField(), CloseDiffPrcRMField(), CloseDiffPrcRMMeanAboveField()]:
            run_field(dp, df, fld, tf)

        series = df[f"{tf}_close_diff_prc_rm_20_mean_above"]
        valid = series.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_close_diff_prc_rm_mean_below_non_positive(self):
        """CloseDiffPrcRMMeanBelow returns non-positive values (or 0 if none negative)."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        for fld in [CloseDiffPrcField(), CloseDiffPrcRMField(), CloseDiffPrcRMMeanBelowField()]:
            run_field(dp, df, fld, tf)

        series = df[f"{tf}_close_diff_prc_rm_20_mean_below"]
        valid = series.dropna()
        assert len(valid) > 0
        assert (valid <= 0).all()

    def test_high_diff_prc_fields(self):
        """High diff prc family produces valid Series."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        for fld in [
            HighDiffPrcField(),
            HighDiffPrcRMField(),
            HighDiffPrcRMMeanAboveField(),
            HighDiffPrcRMMeanBelowField(),
        ]:
            series = run_field(dp, df, fld, tf)
            assert isinstance(series, pd.Series)
            assert len(series) == len(df)

    def test_low_diff_prc_fields(self):
        """Low diff prc family produces valid Series."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        for fld in [
            LowDiffPrcField(),
            LowDiffPrcRMField(),
            LowDiffPrcRMMeanAboveField(),
            LowDiffPrcRMMeanBelowField(),
        ]:
            series = run_field(dp, df, fld, tf)
            assert isinstance(series, pd.Series)
            assert len(series) == len(df)


# ---------------------------------------------------------------------------
# 10. TrendUp boolean
# ---------------------------------------------------------------------------

class TestTrendFlags:
    def test_trend_up_boolean(self):
        """TrendUpField returns boolean Series after ema_50 is computed."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        ema50 = EMAField(length=50, name="ema_50")
        run_field(dp, df, ema50, tf)

        trend_up_50 = TrendUpField()
        series = run_field(dp, df, trend_up_50, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)
        # Valid values must be boolean/bool-like
        valid = series.dropna()
        assert len(valid) > 0
        assert valid.dtype == bool or valid.isin([True, False]).all()

    def test_trend_down_boolean(self):
        """TrendDownField returns boolean Series after ema_50 is computed."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        ema50 = EMAField(length=50, name="ema_50")
        run_field(dp, df, ema50, tf)

        trend_down_50 = TrendDownField()
        series = run_field(dp, df, trend_down_50, tf)

        assert isinstance(series, pd.Series)
        valid = series.dropna()
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# 11. SAR edge case: short DataFrame
# ---------------------------------------------------------------------------

class TestSARField:
    def test_sar_edge_case_short_df(self):
        """With len=1, SAR returns Series of NaN (no crash)."""
        tf = 5
        dp, df = make_dp(tf=tf, n=1)
        field = SARField()
        series = field.compute(dp, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == 1
        assert math.isnan(series.iloc[0])

    def test_sar_normal_case(self):
        """SAR returns Series of same length for normal DataFrame."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = SARField()
        series = run_field(dp, df, field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)


# ---------------------------------------------------------------------------
# 12. NNRSINorm no NaN after warmup
# ---------------------------------------------------------------------------

class TestNNFeatureFields:
    def test_nn_rsi_norm_no_nan_after_warmup(self):
        """NNRSINormField produces non-NaN after 20 rows warmup."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        rsi14 = RSI14Field()
        run_field(dp, df, rsi14, tf)

        rsi_ma8 = RSI_MAField(source="rsi_14", length=8, name="rsi_ma8")
        run_field(dp, df, rsi_ma8, tf)

        nn_rsi = NNRSINormField()
        series = run_field(dp, df, nn_rsi, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)

        # After 20-row warmup (plus RSI warmup), should have valid values near the end
        valid = series.dropna()
        assert len(valid) > 0

    def test_nn_close_diff_atr_ma(self):
        """NNCloseDiffATRField produces non-NaN after warmup."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)

        # Need atr_14, atr_14_ma_20, close_diff_prc
        atr14 = ATR14Field()
        run_field(dp, df, atr14, tf)

        atr_14_ma_20 = ATR_MAField()
        run_field(dp, df, atr_14_ma_20, tf)

        cdp = CloseDiffPrcField()
        run_field(dp, df, cdp, tf)

        nn_field = NNCloseDiffATRField()
        series = run_field(dp, df, nn_field, tf)

        assert isinstance(series, pd.Series)
        assert len(series) == len(df)
        valid = series.dropna()
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# 13. Classification fields with monkeypatched _load_rsi_classification
# ---------------------------------------------------------------------------

class TestClassificationFields:
    def _make_rsi_df(self, tf: int = 15, n: int = 50) -> tuple:
        """Return (dp, df) with rsi_ma8 column pre-populated."""
        df = make_indicator_df(tf=tf, n=n)
        df[f"{tf}_rsi_ma8"] = 52.0
        dp = LiveDataPoint({tf: df})
        return dp, df

    def test_move_class_with_stats(self, monkeypatch):
        """MoveClassField computes correctly with monkeypatched loader."""
        import indicators.library.classification
        stats = {"15": {"mean": 50.0, "std": 14.0}}
        monkeypatch.setenv("DATA_ROOT", "dataset")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setattr(indicators.library.classification, "_load_rsi_classification", lambda path: stats)
        field = MoveClassField()
        dp, df = self._make_rsi_df(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.notna().any()

    def test_zone_class_with_stats(self, monkeypatch):
        """ZoneClassField computes correctly with monkeypatched loader."""
        import indicators.library.classification
        import indicators.library.targets
        stats = {"15": {"mean": 50.0, "std": 14.0}}
        monkeypatch.setenv("DATA_ROOT", "dataset")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setattr(indicators.library.classification, "_load_rsi_classification", lambda path: stats)
        field = ZoneClassField()
        dp, df = self._make_rsi_df(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        assert result.notna().any()

    def test_over_low_with_stats(self, monkeypatch):
        """OverLowField computes correctly with monkeypatched loader."""
        import indicators.library.classification
        import indicators.library.targets
        stats = {"15": {"mean": 50.0, "std": 14.0}}
        monkeypatch.setenv("DATA_ROOT", "dataset")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setattr(indicators.library.classification, "_load_rsi_classification", lambda path: stats)
        field = OverLowField()
        dp, df = self._make_rsi_df(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        # rsi_ma8=52 > mean-std=36, so all True
        assert result.all()

    def test_over_high_with_stats(self, monkeypatch):
        """OverHighField computes correctly with monkeypatched loader."""
        import indicators.library.classification
        import indicators.library.targets
        stats = {"15": {"mean": 50.0, "std": 14.0}}
        monkeypatch.setenv("DATA_ROOT", "dataset")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setattr(indicators.library.classification, "_load_rsi_classification", lambda path: stats)
        field = OverHighField()
        dp, df = self._make_rsi_df(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50
        # rsi_ma8=52 < mean+std=64, so all False
        assert not result.any()


# ---------------------------------------------------------------------------
# 14. Targets fields with monkeypatched _load_diff_stats
# ---------------------------------------------------------------------------

class TestTargetsFields:
    def _make_close_dp(self, tf: int = 15, n: int = 50) -> tuple:
        df = make_indicator_df(tf=tf, n=n)
        dp = LiveDataPoint({tf: df})
        return dp, df

    def _patch_diff_stats(self, monkeypatch, tf: int = 15):
        import indicators.library.classification
        import indicators.library.targets
        diff_stats = {str(tf): {
            "mean_long": 0.02,
            "mean_long_sl": 0.01,
            "mean_short": 0.02,
            "mean_short_sl": 0.01,
            "zb_threshold": 1.0,
            "zs_threshold": 100.0,
        }}
        monkeypatch.setenv("DATA_ROOT", "dataset")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setattr(indicators.library.targets, "_load_diff_stats", lambda path: diff_stats)

    def test_tgt_long_with_stats(self, monkeypatch):
        """TgtLongField computes close * (1 + mean_long)."""
        self._patch_diff_stats(monkeypatch)
        field = TgtLongField()
        dp, df = self._make_close_dp(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_sl_long_with_stats(self, monkeypatch):
        """SLLongField computes close * (1 - mean_long_sl)."""
        self._patch_diff_stats(monkeypatch)
        field = SLLongField()
        dp, df = self._make_close_dp(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_tgt_short_with_stats(self, monkeypatch):
        """TgtShortField computes close * (1 - mean_short)."""
        self._patch_diff_stats(monkeypatch)
        field = TgtShortField()
        dp, df = self._make_close_dp(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_sl_short_with_stats(self, monkeypatch):
        """SLShortField computes close * (1 + mean_short_sl)."""
        self._patch_diff_stats(monkeypatch)
        field = SLShortField()
        dp, df = self._make_close_dp(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_zb_with_stats(self, monkeypatch):
        """ZBField returns integer Series."""
        self._patch_diff_stats(monkeypatch)
        field = ZBField()
        dp, df = self._make_close_dp(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_zs_with_stats(self, monkeypatch):
        """ZSField returns integer Series."""
        self._patch_diff_stats(monkeypatch)
        field = ZSField()
        dp, df = self._make_close_dp(tf=15)
        result = field.compute(dp, 15)
        assert isinstance(result, pd.Series)
        assert len(result) == 50


# ---------------------------------------------------------------------------
# 15. MACDFastSignalField
# ---------------------------------------------------------------------------

class TestMACDFastSignalField:
    def test_macd_fast_signal_produces_series(self):
        """MACDFastSignalField returns Series of same length."""
        tf = 5
        dp, df = make_dp(tf=tf, n=120)
        field = MACDFastSignalField()
        series = run_field(dp, df, field, tf)
        assert isinstance(series, pd.Series)
        assert len(series) == len(df)


# ---------------------------------------------------------------------------
# 16. SAR empty df edge case
# ---------------------------------------------------------------------------

class TestSAREdgeCases:
    def test_sar_edge_case_empty_df(self):
        """With n=0, SAR returns an empty Series without crashing."""
        tf = 5
        dp, df = make_dp(tf=tf, n=0)
        field = SARField()
        series = field.compute(dp, tf)
        assert isinstance(series, pd.Series)
        assert len(series) == 0


# ---------------------------------------------------------------------------
# 17. Registry completeness
# ---------------------------------------------------------------------------

def test_registry_contains_all_field_names():
    """Every field name in the config must have a registry entry."""
    from indicators import _FIELD_REGISTRY, load_indicators_config
    configs = load_indicators_config()
    missing = [cfg.name for cfg in configs if cfg.name not in _FIELD_REGISTRY]
    assert missing == [], f"Missing from registry: {missing}"
