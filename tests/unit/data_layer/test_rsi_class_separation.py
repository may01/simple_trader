"""Tests for move_class/zone_class separation.

move_class classifies rsi_ma8_diff (momentum) against FROZEN sym0 cuts
(0 ± {0.3,1,2}·diff_std, 2y fit — MOVE_CLASS_CUTS, 7 classes -3..3; RSI
params-selection experiment winner). zone_class classifies rsi_ma8 (level)
against mean/std from rsi_classification.json (five tiers, spec §5.7).
Zone stats fall back to the nearest available TF entry; move cuts fall back
to the nearest fitted TF.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from data import LiveDataPoint
from indicators import MoveClassField, ZoneClassField
from indicators.attributes import DataAttributes


STATS = {
    "15": {"mean": 50.0, "std": 10.0, "diff_mean": 0.0, "diff_std": 2.0},
}


@pytest.fixture
def patched_stats(monkeypatch):
    import indicators.library.classification as cls_mod
    monkeypatch.setenv("DATA_ROOT", "dataset")
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setattr(
        cls_mod, "_load_rsi_classification", lambda path: STATS
    )


def _dp(tf: int, rsi_ma8, rsi_ma8_diff):
    n = len(rsi_ma8)
    idx = pd.date_range("2023-09-01", periods=n, freq=f"{tf}min", tz="UTC")
    df = pd.DataFrame(
        {
            f"{tf}_close": np.linspace(20, 21, n),
            f"{tf}_rsi_ma8": rsi_ma8,
            f"{tf}_rsi_ma8_diff": rsi_ma8_diff,
            f"{tf}_is_closed": True,
        },
        index=idx,
    )
    return LiveDataPoint({tf: df})


# ---------------------------------------------------------------------------
# move_class: rsi_ma8_diff momentum, sym0 7 classes (-3..3, frozen cuts)
# ---------------------------------------------------------------------------

class TestMoveClass:
    def test_seven_sym0_classes_on_diff(self, patched_stats):
        # TF15 frozen cuts ±(0.409, 1.364, 2.729)
        diffs = [-3.0, -1.5, -1.0, 0.0, 1.0, 1.5, 3.0]
        dp = _dp(15, [50.0] * 7, diffs)
        out = MoveClassField().compute(dp, 15)
        assert list(out) == [-3, -2, -1, 0, 1, 2, 3]

    def test_cuts_ignore_stale_stats_json(self, patched_stats):
        # patched stats say diff_std=2.0; frozen cuts must win: 0.5 is inside
        # the neutral band (±0.409·1.364σ → cut at 0.409), so 0.5 → class 1,
        # while stale-stats 5-tier logic would have put 0.5 in class 0.
        dp = _dp(15, [50.0], [0.5])
        assert list(MoveClassField().compute(dp, 15)) == [1]

    def test_uses_diff_not_level(self, patched_stats):
        # Level far above mean but diff neutral → class 0
        dp = _dp(15, [90.0], [0.0])
        assert list(MoveClassField().compute(dp, 15)) == [0]

    def test_nan_diff_maps_to_neutral(self, patched_stats):
        dp = _dp(15, [50.0], [np.nan])
        assert list(MoveClassField().compute(dp, 15)) == [0]

    def test_declares_diff_dependency(self):
        assert "rsi_ma8_diff" in MoveClassField().dependencies


# ---------------------------------------------------------------------------
# zone_class: rsi_ma8 level tiers (0..4)
# ---------------------------------------------------------------------------

class TestZoneClass:
    def test_five_tiers_on_level(self, patched_stats):
        # mean=50, std=10 → boundaries at 40, 45, 55, 60
        levels = [35.0, 42.0, 50.0, 57.0, 65.0]
        dp = _dp(15, levels, [0.0] * 5)
        out = ZoneClassField().compute(dp, 15)
        assert list(out) == [0, 1, 2, 3, 4]

    def test_uses_level_not_diff(self, patched_stats):
        # Diff extreme but level at mean → middle zone
        dp = _dp(15, [50.0], [99.0])
        assert list(ZoneClassField().compute(dp, 15)) == [2]

    def test_nan_level_maps_to_middle(self, patched_stats):
        dp = _dp(15, [np.nan], [0.0])
        assert list(ZoneClassField().compute(dp, 15)) == [2]

    def test_differs_from_move_class(self, patched_stats):
        # Rising diff with low level: move says steep up (3), zone says low (0)
        dp = _dp(15, [35.0], [3.0])
        assert list(MoveClassField().compute(dp, 15)) == [3]
        assert list(ZoneClassField().compute(dp, 15)) == [0]


# ---------------------------------------------------------------------------
# TF fallback when stats entry missing
# ---------------------------------------------------------------------------

class TestTfFallback:
    def test_falls_back_to_nearest_lower_tf(self, patched_stats):
        dp = _dp(1440, [65.0], [3.0])
        # zone: 1440 missing from STATS → falls back to the 15 entry.
        assert list(ZoneClassField().compute(dp, 1440)) == [4]
        # move: 1440 has no frozen fit → nearest lower fitted TF (240);
        # 3.0 > 2.919 (outer cut) → class 3.
        assert list(MoveClassField().compute(dp, 1440)) == [3]


# ---------------------------------------------------------------------------
# DataAttributes: stats file contents
# ---------------------------------------------------------------------------

@pytest.fixture
def stats_dir(tmp_path, monkeypatch):
    import helpers
    d = str(tmp_path) + "/"
    monkeypatch.setattr(helpers, "stats_folder", lambda: d)
    return d


def _wide_df(n=300):
    np.random.seed(7)
    idx = pd.date_range("2023-09-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(index=idx)
    for tf in (15, 60):
        df[f"{tf}_rsi_ma8"] = np.random.uniform(30, 70, n)
        df[f"{tf}_rsi_ma8_diff"] = np.random.normal(0, 1, n)
        df[f"{tf}_close"] = 20.0
        df[f"{tf}_close_diff_prc"] = np.random.normal(0, 0.1, n)
        df[f"{tf}_is_closed"] = True
    # 1440: columns exist but all-NaN (not enough history)
    df["1440_rsi_ma8"] = np.nan
    df["1440_rsi_ma8_diff"] = np.nan
    df["1440_close"] = 20.0
    df["1440_close_diff_prc"] = np.nan
    df["1440_is_closed"] = True
    return df


class TestRsiClassificationFile:
    def test_entries_carry_level_and_diff_stats(self, stats_dir):
        DataAttributes().compute(_wide_df())
        with open(stats_dir + "rsi_classification.json") as fh:
            data = json.load(fh)
        for tf in ("15", "60"):
            entry = data[tf]
            assert set(entry) >= {"mean", "std", "diff_mean", "diff_std"}

    def test_insufficient_tf_skipped_no_nan(self, stats_dir):
        DataAttributes().compute(_wide_df())
        raw = open(stats_dir + "rsi_classification.json").read()
        assert "NaN" not in raw
        data = json.loads(raw)
        assert "1440" not in data

    def test_over_fields_fall_back_too(self, patched_stats):
        from indicators import OverLowField, OverHighField
        dp = _dp(1440, [65.0], [0.0])
        assert list(OverLowField().compute(dp, 1440)) == [True]
        assert list(OverHighField().compute(dp, 1440)) == [True]
