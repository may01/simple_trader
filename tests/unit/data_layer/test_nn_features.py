"""Unit tests for Task 03 NN feature engineering fields + robust normalisation stats.

Covers the new IndicatorField subclasses in indicators/library/nn_features.py
(NNDiffField, NNSlopeField, orthogonal engineered fields, NNCrossTFAlignField)
and the winsorised z-score stats in indicators/attributes.py.

Tests assert:
- column name == {tf}_{name}
- compute correctness on small synthetic dfs
- no look-ahead (slope at row t uses only rows <= t)
- NaN on warmup (slope needs `window` trailing rows)
- EMA-pair direction (shorter - longer)
- cross-TF align sign agreement
- robust winsorised stats {q01, q99, mean, std} + the [-4, 4] apply clamp
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data import LiveDataPoint
from indicators import DataAttributes
from indicators.framework import Indicators
from indicators.library.nn_features import (
    NNDiffField,
    NNSlopeField,
    NNLogRetField,
    NNRangeATRField,
    NNBodyRatioField,
    NNWickUpField,
    NNWickDnField,
    NNVolRegimeField,
    NNSinTodField,
    NNCosTodField,
    NNSinDowField,
    NNCosDowField,
    NNCrossTFAlignField,
    _rolling_ols_slope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(tf: int, n: int, start: str = "2023-09-01", cols: dict | None = None) -> pd.DataFrame:
    """Build a {tf}_*-prefixed df with a UTC DatetimeIndex and is_closed=True."""
    idx = pd.date_range(start, periods=n, freq=f"{tf}min", tz="UTC")
    data = {f"{tf}_is_closed": True}
    if cols:
        for name, values in cols.items():
            data[f"{tf}_{name}"] = values
    return pd.DataFrame(data, index=idx)


def run(field, df: pd.DataFrame, tf: int) -> pd.Series:
    """Run field.compute against a LiveDataPoint wrapping df, write + return series."""
    dp = LiveDataPoint({tf: df})
    series = field.compute(dp, tf)
    df[f"{tf}_{field.name}"] = series
    return series


# ---------------------------------------------------------------------------
# Group 2 — NNDiffField
# ---------------------------------------------------------------------------

class TestNNDiffField:
    def test_name_and_dependencies(self):
        f = NNDiffField(left="ema_7", right="close")
        assert f.name == "ema_7_minus_close"
        assert f.group == "nn_features"
        assert set(f.dependencies) == {"ema_7", "close"}

    def test_column_name_is_tf_prefixed(self):
        tf = 15
        df = make_df(tf, 5, cols={"ema_7": [1, 2, 3, 4, 5], "close": [0, 0, 0, 0, 0]})
        f = NNDiffField(left="ema_7", right="close")
        run(f, df, tf)
        assert f"{tf}_ema_7_minus_close" in df.columns

    def test_diff_is_left_minus_right(self):
        tf = 60
        df = make_df(tf, 4, cols={"ema_7": [10.0, 11.0, 12.0, 13.0],
                                  "close": [9.0, 9.0, 15.0, 13.0]})
        f = NNDiffField(left="ema_7", right="close")
        s = run(f, df, tf)
        np.testing.assert_allclose(s.values, [1.0, 2.0, -3.0, 0.0])

    def test_ema_pair_direction_shorter_minus_longer(self):
        """Positive ema_7 - ema_50 ⇒ fast above slow ⇒ up-momentum."""
        tf = 15
        df = make_df(tf, 3, cols={"ema_7": [100.0, 101.0, 102.0],
                                  "ema_50": [99.0, 102.0, 102.0]})
        f = NNDiffField(left="ema_7", right="ema_50")
        s = run(f, df, tf)
        # row0: fast above slow (+), row1: fast below slow (-), row2: equal (0)
        assert s.iloc[0] > 0
        assert s.iloc[1] < 0
        assert s.iloc[2] == 0.0

    def test_atr_resolves_to_atr_14(self):
        f = NNDiffField(left="atr_14_ma_5", right="atr_14")
        assert f.name == "atr_14_ma_5_minus_atr_14"
        assert "atr_14" in f.dependencies


# ---------------------------------------------------------------------------
# Group 3 — NNSlopeField
# ---------------------------------------------------------------------------

class TestNNSlopeField:
    def test_name_window_dependencies(self):
        f = NNSlopeField(source="ema_50", window=5)
        assert f.name == "ema_50_slope"
        assert f.dependencies == ["ema_50"]
        assert f.params.get("window") == 5

    def test_slope_of_linear_series_equals_per_bar_step(self):
        """A perfectly linear series of step k has OLS slope == k everywhere
        once the trailing window is full."""
        tf = 15
        k = 3.0
        vals = [10.0 + k * i for i in range(20)]
        df = make_df(tf, 20, cols={"ema_50": vals})
        f = NNSlopeField(source="ema_50", window=5)
        s = run(f, df, tf)
        valid = s.dropna()
        np.testing.assert_allclose(valid.values, k, atol=1e-9)

    def test_slope_sign_tracks_direction(self):
        tf = 15
        up = list(range(0, 10))           # increasing
        down = list(range(20, 10, -1))    # decreasing
        df_up = make_df(tf, 10, cols={"ema_7": [float(x) for x in up]})
        df_dn = make_df(tf, 10, cols={"ema_7": [float(x) for x in down]})
        f = NNSlopeField(source="ema_7", window=5)
        s_up = run(f, df_up, tf).dropna()
        s_dn = run(f, df_dn, tf).dropna()
        assert (s_up > 0).all()
        assert (s_dn < 0).all()

    def test_warmup_rows_are_nan(self):
        """First window-1 rows lack a full trailing window ⇒ NaN."""
        tf = 15
        df = make_df(tf, 10, cols={"ema_7": [float(i) for i in range(10)]})
        f = NNSlopeField(source="ema_7", window=5)
        s = run(f, df, tf)
        assert s.iloc[:4].isna().all()
        assert not math.isnan(s.iloc[4])

    def test_no_look_ahead_slope_uses_only_trailing_rows(self):
        """Slope at row t is unchanged if any row > t is mutated to garbage."""
        tf = 15
        base = [float(i) * 2.0 for i in range(15)]
        df_a = make_df(tf, 15, cols={"ema_25": list(base)})
        df_b = make_df(tf, 15, cols={"ema_25": list(base)})
        # Corrupt the FUTURE (rows after index 9) in df_b
        col = f"{tf}_ema_25"
        df_b.loc[df_b.index[10:], col] = -999.0
        f = NNSlopeField(source="ema_25", window=5)
        s_a = run(f, df_a, tf)
        s_b = run(f, df_b, tf)
        # slope at row 9 (window covers rows 5..9) must be identical
        assert s_a.iloc[9] == pytest.approx(s_b.iloc[9])


# ---------------------------------------------------------------------------
# Vectorised OLS slope — closed-form correctness vs np.polyfit reference
# ---------------------------------------------------------------------------

class TestVectorisedOLSSlope:
    """Verify the vectorised _rolling_ols_slope matches np.polyfit per-window."""

    def _polyfit_slope(self, y: np.ndarray, w: int) -> np.ndarray:
        """Reference: np.polyfit slope for every valid trailing window."""
        n = len(y)
        out = np.full(n, np.nan)
        for t in range(w - 1, n):
            out[t] = np.polyfit(np.arange(w, dtype=float), y[t - w + 1 : t + 1], 1)[0]
        return out

    def test_linear_series_slope_equals_step(self):
        """y = 3x + 2 → slope should equal 3.0 at every valid row."""
        y = np.array([2.0 + 3.0 * i for i in range(30)])
        s = pd.Series(y)
        result = _rolling_ols_slope(s, window=7)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 3.0, atol=1e-9)

    def test_matches_polyfit_on_curved_series(self):
        """Curved (quadratic) series: vectorised result matches np.polyfit per window."""
        rng = np.random.RandomState(42)
        y = rng.randn(100).cumsum()
        w = 8
        s = pd.Series(y)
        result = _rolling_ols_slope(s, window=w).values
        reference = self._polyfit_slope(y, w)
        # Both arrays should be NaN for the first w-1 rows
        np.testing.assert_array_equal(np.isnan(result[: w - 1]), True)
        np.testing.assert_allclose(result[w - 1 :], reference[w - 1 :], atol=1e-9)

    def test_warmup_rows_are_nan(self):
        """First window-1 rows must be NaN; row window-1 must be finite."""
        y = pd.Series(np.arange(20, dtype=float))
        w = 5
        result = _rolling_ols_slope(y, window=w)
        assert result.iloc[: w - 1].isna().all()
        assert np.isfinite(result.iloc[w - 1])

    def test_no_look_ahead(self):
        """Corrupting rows after index t must not change the slope at t."""
        rng = np.random.RandomState(7)
        y = rng.randn(50).cumsum()
        w = 6
        t = 20  # row whose slope we will verify
        s_clean = pd.Series(y.copy())
        s_dirty = pd.Series(y.copy())
        # corrupt all rows after t
        s_dirty.iloc[t + 1 :] = 1e9
        clean_slope = _rolling_ols_slope(s_clean, window=w).iloc[t]
        dirty_slope = _rolling_ols_slope(s_dirty, window=w).iloc[t]
        assert clean_slope == pytest.approx(dirty_slope, abs=1e-9)

    def test_window_less_than_2_returns_all_nan(self):
        """A window of 1 is degenerate — no slope can be defined."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = _rolling_ols_slope(y, window=1)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Orthogonal fields
# ---------------------------------------------------------------------------

class TestOrthogonalFields:
    def test_logret(self):
        tf = 15
        close = [100.0, 110.0, 99.0]
        df = make_df(tf, 3, cols={"close": close})
        s = run(NNLogRetField(), df, tf)
        assert math.isnan(s.iloc[0])  # no prior close
        assert s.iloc[1] == pytest.approx(math.log(110.0 / 100.0))
        assert s.iloc[2] == pytest.approx(math.log(99.0 / 110.0))

    def test_range_atr(self):
        tf = 15
        df = make_df(tf, 2, cols={"high": [12.0, 20.0], "low": [10.0, 10.0],
                                  "atr_14": [2.0, 0.0]})
        f = NNRangeATRField()
        assert f.name == "range_atr"
        s = run(f, df, tf)
        assert s.iloc[0] == pytest.approx((12.0 - 10.0) / 2.0)
        # atr clipped at 1e-8, so finite (large) value, not inf
        assert np.isfinite(s.iloc[1])

    def test_body_ratio(self):
        tf = 15
        df = make_df(tf, 1, cols={"open": [10.0], "close": [12.0],
                                  "high": [13.0], "low": [9.0]})
        s = run(NNBodyRatioField(), df, tf)
        assert s.iloc[0] == pytest.approx((12.0 - 10.0) / (13.0 - 9.0))

    def test_wick_up_and_dn(self):
        tf = 15
        df = make_df(tf, 1, cols={"open": [10.0], "close": [12.0],
                                  "high": [15.0], "low": [8.0]})
        up = run(NNWickUpField(), df, tf)
        dn = run(NNWickDnField(), df, tf)
        rng = 15.0 - 8.0
        assert up.iloc[0] == pytest.approx((15.0 - max(10.0, 12.0)) / rng)
        assert dn.iloc[0] == pytest.approx((min(10.0, 12.0) - 8.0) / rng)

    def test_vol_regime_buckets_in_range(self):
        tf = 15
        n = 150
        rng = np.random.RandomState(0)
        atr = np.abs(rng.normal(5.0, 2.0, n)) + 0.1
        df = make_df(tf, n, cols={"atr_14": atr})
        f = NNVolRegimeField(atr_col="atr_14", window=100, buckets=3)
        assert f.name == "vol_regime"
        s = run(f, df, tf)
        valid = s.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 2  # buckets-1

    def test_vol_regime_window_exceeds_indicator_window_rows_raises(self):
        """window > INDICATOR_WINDOW_ROWS (105) must raise ValueError at construction."""
        with pytest.raises(ValueError, match="vol_regime window=200 exceeds INDICATOR_WINDOW_ROWS"):
            NNVolRegimeField(window=200)

    def test_vol_regime_valid_window_produces_non_nan_tail(self):
        """150-row series with window=100 → tail rows (≥99) are non-NaN and in [0, buckets-1]."""
        tf = 15
        n = 150
        rng = np.random.RandomState(42)
        atr = np.abs(rng.normal(5.0, 2.0, n)) + 0.1
        df = make_df(tf, n, cols={"atr_14": atr})
        f = NNVolRegimeField(atr_col="atr_14", window=100, buckets=3)
        s = run(f, df, tf)
        # First 99 rows are warmup (NaN); rows 100..149 must be finite and bucketed
        tail = s.iloc[99:]
        assert tail.notna().all(), "tail rows should be non-NaN with a full window"
        assert tail.min() >= 0
        assert tail.max() <= 2  # buckets - 1

    def test_cyclical_time_from_index(self):
        tf = 15
        # midnight UTC start → seconds_since_midnight==0 at row 0
        idx = pd.date_range("2023-09-04 00:00", periods=4, freq="6h", tz="UTC")
        df = pd.DataFrame({f"{tf}_is_closed": True}, index=idx)
        sin_tod = run(NNSinTodField(), df, tf)
        cos_tod = run(NNCosTodField(), df, tf)
        # 00:00 → sin=0, cos=1
        assert sin_tod.iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert cos_tod.iloc[0] == pytest.approx(1.0, abs=1e-9)
        # 06:00 → quarter day → sin=1, cos=0
        assert sin_tod.iloc[1] == pytest.approx(1.0, abs=1e-9)
        assert cos_tod.iloc[1] == pytest.approx(0.0, abs=1e-9)

    def test_cyclical_dow_from_index(self):
        tf = 1440
        # 2023-09-04 is a Monday (dayofweek == 0)
        idx = pd.date_range("2023-09-04", periods=7, freq="1D", tz="UTC")
        df = pd.DataFrame({f"{tf}_is_closed": True}, index=idx)
        sin_dow = run(NNSinDowField(), df, tf)
        cos_dow = run(NNCosDowField(), df, tf)
        assert sin_dow.iloc[0] == pytest.approx(math.sin(0.0), abs=1e-9)
        assert cos_dow.iloc[0] == pytest.approx(math.cos(0.0), abs=1e-9)
        assert sin_dow.iloc[1] == pytest.approx(math.sin(2 * math.pi * 1 / 7), abs=1e-9)


# ---------------------------------------------------------------------------
# Cross-TF alignment
# ---------------------------------------------------------------------------

class TestNNCrossTFAlignField:
    def test_name_and_other_tf(self):
        f = NNCrossTFAlignField(other_tf=60, trend_col="ema_50")
        assert f.name == "align_60"

    def test_sign_agreement_plus_one(self):
        """Both this-TF slope and higher-TF slope rising ⇒ +1."""
        tf = 15
        other = 60
        n = 20
        # this-TF ema_50 strictly increasing → positive slope
        this_vals = [100.0 + i for i in range(n)]
        # higher-TF ema_50 also increasing (present on the same index, ffilled)
        other_vals = [100.0 + 2 * i for i in range(n)]
        df = make_df(tf, n, cols={"ema_50": this_vals})
        df[f"{other}_ema_50"] = other_vals
        f = NNCrossTFAlignField(other_tf=other, trend_col="ema_50")
        s = run(f, df, tf)
        valid = s.dropna()
        assert (valid == 1.0).all()

    def test_sign_disagreement_minus_one(self):
        tf = 15
        other = 60
        n = 20
        this_vals = [100.0 + i for i in range(n)]       # rising
        other_vals = [200.0 - i for i in range(n)]      # falling
        df = make_df(tf, n, cols={"ema_50": this_vals})
        df[f"{other}_ema_50"] = other_vals
        f = NNCrossTFAlignField(other_tf=other, trend_col="ema_50")
        s = run(f, df, tf)
        valid = s.dropna()
        assert (valid == -1.0).all()

    def test_missing_higher_tf_column_emits_nan_not_crash(self):
        """Live path: per-TF frame lacks the higher-TF column ⇒ all-NaN series
        (dropped at NNDataset build), never a KeyError."""
        tf = 15
        n = 20
        df = make_df(tf, n, cols={"ema_50": [100.0 + i for i in range(n)]})
        # NOTE: no 60_ema_50 column present (as in the live LiveDataPoint path)
        f = NNCrossTFAlignField(other_tf=60, trend_col="ema_50")
        s = run(f, df, tf)
        assert len(s) == n
        assert s.isna().all()

    def test_no_look_ahead_forward_fill_only(self):
        """The higher-TF column is read as-is on this-TF index (last closed value),
        never reaching into a future row — corrupting future rows leaves an
        earlier alignment value unchanged."""
        tf = 15
        other = 60
        n = 20
        this_vals = [100.0 + i for i in range(n)]
        other_vals = [100.0 + 2 * i for i in range(n)]
        df_a = make_df(tf, n, cols={"ema_50": list(this_vals)})
        df_a[f"{other}_ema_50"] = list(other_vals)
        df_b = make_df(tf, n, cols={"ema_50": list(this_vals)})
        df_b[f"{other}_ema_50"] = list(other_vals)
        # corrupt future rows (>=12) of df_b inputs
        df_b.loc[df_b.index[12:], f"{tf}_ema_50"] = -999.0
        df_b.loc[df_b.index[12:], f"{other}_ema_50"] = -999.0
        f = NNCrossTFAlignField(other_tf=other, trend_col="ema_50")
        s_a = run(f, df_a, tf)
        s_b = run(f, df_b, tf)
        # alignment at row 10 must be identical (slope window ends at row 10)
        assert s_a.iloc[10] == s_b.iloc[10]


# ---------------------------------------------------------------------------
# Scheduling: align fields must only appear for their designated TF
# ---------------------------------------------------------------------------

class TestAlignFieldScheduling:
    """Verify the cross-TF align ladder: each align_{other_tf} is scheduled only
    on the next-lower CANDLE. Ladder: 1→align_5, 5→align_15, 15→align_60,
    60→align_240, 240→align_1440; tf=1440 (top) has no align.

    Uses Indicators._sorted_fields — the same entry point the rest of the system
    uses to decide which fields to run on a given TF.
    """

    def setup_method(self):
        # Reset the class-level registry cache so each test starts fresh and
        # picks up the current (post-fix) registry state.
        Indicators._registry = None

    def _align_names_for_tf(self, tf: int) -> set[str]:
        """Return the set of field names in group nn_features for the given TF."""
        fields = Indicators._sorted_fields(tf, groups=["nn_features"], check_resources=False)
        return {f.name for f in fields}

    def test_align_60_scheduled_only_on_tf_15(self):
        """align_60 must appear for tf=15 and NOT for tf=60 or tf=240."""
        assert "align_60" in self._align_names_for_tf(15), \
            "align_60 missing from tf=15 schedule"
        assert "align_60" not in self._align_names_for_tf(60), \
            "align_60 wrongly scheduled on tf=60 (self-alignment)"
        assert "align_60" not in self._align_names_for_tf(240), \
            "align_60 wrongly scheduled on tf=240 (forward-alignment)"

    def test_align_240_scheduled_only_on_tf_60(self):
        """align_240 must appear for tf=60 and NOT for tf=15 or tf=240."""
        assert "align_240" in self._align_names_for_tf(60), \
            "align_240 missing from tf=60 schedule"
        assert "align_240" not in self._align_names_for_tf(15), \
            "align_240 wrongly scheduled on tf=15 (forward-alignment)"
        assert "align_240" not in self._align_names_for_tf(240), \
            "align_240 wrongly scheduled on tf=240 (self-alignment)"

    def test_align_5_scheduled_only_on_tf_1(self):
        """align_5 must appear for tf=1 (1→5) and NOT for tf=5 (self-alignment)."""
        assert "align_5" in self._align_names_for_tf(1), \
            "align_5 missing from tf=1 schedule"
        assert "align_5" not in self._align_names_for_tf(5), \
            "align_5 wrongly scheduled on tf=5 (self-alignment)"

    def test_align_15_scheduled_only_on_tf_5(self):
        """align_15 must appear for tf=5 (5→15) and NOT for tf=1 or tf=15."""
        assert "align_15" in self._align_names_for_tf(5), \
            "align_15 missing from tf=5 schedule"
        assert "align_15" not in self._align_names_for_tf(1), \
            "align_15 wrongly scheduled on tf=1 (forward-alignment)"
        assert "align_15" not in self._align_names_for_tf(15), \
            "align_15 wrongly scheduled on tf=15 (self-alignment)"

    def test_align_1440_scheduled_only_on_tf_240(self):
        """align_1440 must appear for tf=240 (240→1440) and NOT for tf=1440."""
        assert "align_1440" in self._align_names_for_tf(240), \
            "align_1440 missing from tf=240 schedule"
        assert "align_1440" not in self._align_names_for_tf(1440), \
            "align_1440 wrongly scheduled on tf=1440 (self-alignment)"

    def test_no_align_on_tf_1440(self):
        """tf=1440 is the top of the ladder — no align field of any kind."""
        names = self._align_names_for_tf(1440)
        assert not {n for n in names if n.startswith("align_")}, \
            f"tf=1440 must have no align fields, got {names}"


# ---------------------------------------------------------------------------
# Non-align nn_features scheduling: config applies_to governs all nn_features
# ---------------------------------------------------------------------------

class TestNNFeaturesScheduling:
    """Verify nn_features are scheduled on every CANDLE (D11 supersedes D9).

    Config now sets applies_to: [1, 5, 15, 60, 240, 1440] for every non-align
    nn_features entry, EXCEPT sin_tod/cos_tod which omit 1440 (daily bars share
    one wall-clock open → constant). The scheduler must honour this:
    - tf in {1, 5, 15, 60, 240}: full non-align set + sin_tod/cos_tod + one align.
    - tf=1440: full non-align set MINUS sin_tod/cos_tod, no align (top of ladder).
    """

    # Per-tf nn_features field counts (incl. align where applicable).
    EXPECTED_COUNT = {1: 48, 5: 48, 15: 48, 60: 48, 240: 48, 1440: 45}

    def setup_method(self):
        Indicators._registry = None

    def _nn_names_for_tf(self, tf: int) -> set[str]:
        fields = Indicators._sorted_fields(tf, groups=["nn_features"], check_resources=False)
        return {f.name for f in fields}

    @pytest.mark.parametrize("tf", [1, 5, 15, 60, 240, 1440])
    def test_core_nn_features_scheduled_on_every_tf(self, tf):
        """logret + macd_12_26_9_slope must appear on every CANDLE."""
        names = self._nn_names_for_tf(tf)
        assert "logret" in names, f"logret missing from tf={tf} schedule"
        assert "macd_12_26_9_slope" in names, \
            f"macd_12_26_9_slope missing from tf={tf} schedule"

    @pytest.mark.parametrize("tf,count", EXPECTED_COUNT.items())
    def test_nn_features_count_per_tf(self, tf, count):
        """Each CANDLE schedules the expected number of nn_features fields."""
        names = self._nn_names_for_tf(tf)
        assert len(names) == count, \
            f"Expected {count} nn_features on tf={tf} but got {len(names)}: {sorted(names)}"

    def test_time_of_day_present_below_1440(self):
        """sin_tod/cos_tod scheduled on tf=1 (and below 1440) — intraday time varies."""
        names = self._nn_names_for_tf(1)
        assert {"sin_tod", "cos_tod"} <= names, \
            f"sin_tod/cos_tod missing from tf=1 schedule: {sorted(names)}"

    def test_time_of_day_absent_on_1440(self):
        """sin_tod/cos_tod NOT scheduled on tf=1440 (daily bars → constant)."""
        names = self._nn_names_for_tf(1440)
        assert "sin_tod" not in names and "cos_tod" not in names, \
            f"sin_tod/cos_tod wrongly scheduled on tf=1440: {sorted(names)}"
        # day-of-week still varies daily and must remain.
        assert {"sin_dow", "cos_dow"} <= names, \
            f"sin_dow/cos_dow missing from tf=1440 schedule: {sorted(names)}"
