"""Tests for move_class/zone_class separation (spec §5.7 indicators-class).

move_class classifies rsi_ma8_diff (momentum) against diff_mean/diff_std;
zone_class classifies rsi_ma8 (level) against mean/std. Five tiers each.
rsi_classification.json carries both stat pairs per TF and never contains
NaN; classification fields fall back to the nearest available TF entry.
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
    "15": {
        "mean": 50.0, "std": 10.0, "diff_mean": 0.0, "diff_std": 2.0,
        "zone_cuts": [40.0, 45.0, 55.0, 60.0],
        "move_cuts": [-2.0, -0.6, 0.6, 2.0],
    },
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
# move_class: rsi_ma8_diff momentum tiers (-2..2)
# ---------------------------------------------------------------------------

class TestMoveClass:
    def test_five_tiers_on_diff(self, patched_stats):
        # diff_mean=0, diff_std=2 → boundaries at -2, -1, 1, 2
        diffs = [-3.0, -1.5, 0.0, 1.5, 3.0]
        dp = _dp(15, [50.0] * 5, diffs)
        out = MoveClassField().compute(dp, 15)
        assert list(out) == [-2, -1, 0, 1, 2]

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
        # Rising diff with low level: move says up (2), zone says low (0)
        dp = _dp(15, [35.0], [3.0])
        assert list(MoveClassField().compute(dp, 15)) == [2]
        assert list(ZoneClassField().compute(dp, 15)) == [0]


# ---------------------------------------------------------------------------
# TF fallback when stats entry missing
# ---------------------------------------------------------------------------

class TestTfFallback:
    def test_falls_back_to_nearest_lower_tf(self, patched_stats):
        dp = _dp(1440, [65.0], [3.0])
        # 1440 missing from STATS → falls back to 15 entry
        assert list(ZoneClassField().compute(dp, 1440)) == [4]
        assert list(MoveClassField().compute(dp, 1440)) == [2]


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


# ---------------------------------------------------------------------------
# task-02: additive quantile/sym0 cuts + side-stats companion
# ---------------------------------------------------------------------------

def _wide_df_trend(direction: int, n=300):
    """Wide df whose {tf}_close moves monotonically (+1 up / -1 down)."""
    df = _wide_df(n)
    for tf in (15, 60):
        df[f"{tf}_close"] = np.linspace(20.0, 20.0 + direction, n)
    return df


class TestRsiClassificationCuts:
    def test_zone_and_move_cuts_present_ascending(self, stats_dir):
        DataAttributes().compute(_wide_df())
        data = json.load(open(stats_dir + "rsi_classification.json"))
        for tf in ("15", "60"):
            e = data[tf]
            assert len(e["zone_cuts"]) == 4 and e["zone_cuts"] == sorted(e["zone_cuts"])
            assert len(e["move_cuts"]) == 4 and e["move_cuts"] == sorted(e["move_cuts"])

    def test_move_cuts_symmetric_about_zero(self, stats_dir):
        DataAttributes().compute(_wide_df())
        e = json.load(open(stats_dir + "rsi_classification.json"))["15"]
        mc, s = e["move_cuts"], e["diff_std"]
        assert mc[0] == pytest.approx(-mc[3])
        assert mc[1] == pytest.approx(-mc[2])
        assert mc[3] == pytest.approx(1.0 * s, rel=1e-6)
        assert mc[2] == pytest.approx(0.3 * s, rel=1e-6)

    def test_zone_cuts_are_percentiles_in_range(self, stats_dir):
        DataAttributes().compute(_wide_df())
        zc = json.load(open(stats_dir + "rsi_classification.json"))["15"]["zone_cuts"]
        assert 30.0 <= zc[0] <= zc[3] <= 70.0

    def test_legacy_keys_unchanged(self, stats_dir):
        DataAttributes().compute(_wide_df())
        e = json.load(open(stats_dir + "rsi_classification.json"))["15"]
        assert set(e) >= {"mean", "std", "diff_mean", "diff_std", "zone_cuts", "move_cuts"}

    def test_no_nan_and_1440_skipped(self, stats_dir):
        DataAttributes().compute(_wide_df())
        raw = open(stats_dir + "rsi_classification.json").read()
        assert "NaN" not in raw
        assert "1440" not in json.loads(raw)


class TestRsiSideStats:
    def test_file_created_with_structure(self, stats_dir):
        DataAttributes().compute(_wide_df())
        path = stats_dir + "rsi_side_stats.json"
        assert os.path.exists(path)
        raw = open(path).read()
        assert "NaN" not in raw
        e = json.loads(raw)["15"]
        for field in ("zone_class_q", "move_class_sym0"):
            assert field in e and len(e[field]) == 5
            some = next(iter(e[field].values()))
            assert set(some) >= {
                "long_winrate", "short_winrate", "long_lift", "short_lift", "base_long",
            }

    def test_zone_keys_0_to_4_move_keys_neg2_to_2(self, stats_dir):
        DataAttributes().compute(_wide_df())
        e = json.load(open(stats_dir + "rsi_side_stats.json"))["15"]
        assert set(e["zone_class_q"]) == {"0", "1", "2", "3", "4"}
        assert set(e["move_class_sym0"]) == {"-2", "-1", "0", "1", "2"}

    def test_all_up_trend_gives_long_winrate_one(self, stats_dir):
        DataAttributes().compute(_wide_df_trend(+1))
        e = json.load(open(stats_dir + "rsi_side_stats.json"))["15"]["zone_class_q"]
        pops = [c for c in e.values() if c["n"] > 0]
        assert pops and all(c["long_winrate"] == 1.0 for c in pops)
        assert all(c["base_long"] == 1.0 for c in pops)
        assert all(c["long_lift"] == 0.0 for c in pops)

    def test_all_down_trend_gives_short_winrate_one(self, stats_dir):
        DataAttributes().compute(_wide_df_trend(-1))
        e = json.load(open(stats_dir + "rsi_side_stats.json"))["15"]["move_class_sym0"]
        pops = [c for c in e.values() if c["n"] > 0]
        assert pops and all(c["short_winrate"] == 1.0 for c in pops)

    def test_side_stats_absent_only_recompute(self, stats_dir):
        DataAttributes().compute(_wide_df())
        path = stats_dir + "rsi_side_stats.json"
        open(path, "w").write('{"sentinel": true}')
        DataAttributes().compute(_wide_df())  # already present → not overwritten
        assert json.load(open(path)) == {"sentinel": True}


# ---------------------------------------------------------------------------
# task-03: new coexisting fields — zone_class_q (quantile), move_class_sym0
# ---------------------------------------------------------------------------

from indicators.library.classification import (  # noqa: E402
    ZoneClassQField, MoveClassSym0Field, _apply_cuts,
)


class TestApplyCuts:
    def test_tiers_and_base(self):
        x = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])
        # cuts [2,4,6,8] → digitize 0..4
        assert list(_apply_cuts(x, [2, 4, 6, 8], 0)) == [0, 1, 2, 3, 4]
        assert list(_apply_cuts(x, [2, 4, 6, 8], -2)) == [-2, -1, 0, 1, 2]

    def test_nan_to_middle(self):
        x = pd.Series([np.nan])
        assert list(_apply_cuts(x, [2, 4, 6, 8], 0)) == [2]     # middle tier idx 2
        assert list(_apply_cuts(x, [2, 4, 6, 8], -2)) == [0]    # move neutral

    def test_boundary_lands_upper_tier(self):
        # value exactly on a cut → goes to the upper tier (right=False)
        assert list(_apply_cuts(pd.Series([4.0]), [2, 4, 6, 8], 0)) == [2]


class TestZoneClassQ:
    def test_quantile_tiers(self, patched_stats):
        levels = [35.0, 42.0, 50.0, 57.0, 65.0]  # cuts [40,45,55,60]
        out = ZoneClassQField().compute(_dp(15, levels, [0.0] * 5), 15)
        assert list(out) == [0, 1, 2, 3, 4]

    def test_uses_level_not_diff(self, patched_stats):
        assert list(ZoneClassQField().compute(_dp(15, [50.0], [99.0]), 15)) == [2]

    def test_nan_to_middle(self, patched_stats):
        assert list(ZoneClassQField().compute(_dp(15, [np.nan], [0.0]), 15)) == [2]

    def test_falls_back_across_tf(self, patched_stats):
        # 1440 missing → uses 15's zone_cuts
        assert list(ZoneClassQField().compute(_dp(1440, [65.0], [0.0]), 1440)) == [4]


class TestMoveClassSym0:
    def test_symmetric_tiers(self, patched_stats):
        diffs = [-3.0, -1.5, 0.0, 1.5, 3.0]  # cuts [-2,-0.6,0.6,2]
        out = MoveClassSym0Field().compute(_dp(15, [50.0] * 5, diffs), 15)
        assert list(out) == [-2, -1, 0, 1, 2]

    def test_neutral_brackets_zero(self, patched_stats):
        # small |diff| < 0.6 → neutral 0
        assert list(MoveClassSym0Field().compute(_dp(15, [50.0], [0.1]), 15)) == [0]

    def test_nan_to_neutral(self, patched_stats):
        assert list(MoveClassSym0Field().compute(_dp(15, [50.0], [np.nan]), 15)) == [0]

    def test_uses_diff_not_level(self, patched_stats):
        assert list(MoveClassSym0Field().compute(_dp(15, [90.0], [0.0]), 15)) == [0]


class TestLegacyFieldsUnchanged:
    def test_legacy_zone_and_move_still_five_tiers(self, patched_stats):
        # legacy path uses mean/std (not cuts) — unchanged by the new fields
        assert list(ZoneClassField().compute(_dp(15, [50.0], [0.0]), 15)) == [2]
        assert list(MoveClassField().compute(_dp(15, [50.0], [0.0]), 15)) == [0]


# ---------------------------------------------------------------------------
# task-04: wiring — registry + config expose the new fields to the pipeline
# ---------------------------------------------------------------------------

class TestWiring:
    def test_registry_builds_new_fields(self):
        from indicators.registry import _FIELD_REGISTRY
        assert isinstance(_FIELD_REGISTRY["zone_class_q"](None), ZoneClassQField)
        assert isinstance(_FIELD_REGISTRY["move_class_sym0"](None), MoveClassSym0Field)

    def test_config_lists_new_fields_with_deps(self):
        import yaml
        with open("configs/indicators_config.yaml") as fh:
            cfg = yaml.safe_load(fh)
        entries = {e["name"]: e for e in cfg["fields"]}
        assert {"zone_class_q", "move_class_sym0"} <= set(entries)
        assert entries["zone_class_q"]["depends_on"] == ["rsi_ma8"]
        assert entries["move_class_sym0"]["depends_on"] == ["rsi_ma8", "rsi_ma8_diff"]
        assert entries["zone_class_q"]["group"] == "classification"
