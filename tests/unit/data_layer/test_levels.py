"""Unit tests for levels.py — Levels class."""

import os
import pytest

from constants import (
    LI_NAME, LI_TIME1, LI_VAL1, LI_TIME2, LI_VAL2,
    LI_ACTIVE, LI_TYPE, LI_HAD_CONTACT,
    LEVEL_TYPE_LONG_SUPPORT, LEVEL_TYPE_LONG_RESISTANCE,
    LEVEL_TYPE_SHORT_SUPPORT, LEVEL_TYPE_SHORT_RESISTANCE,
    LEVEL_TYPE_TARGET, LEVEL_TYPE_AUTO_SUPPORT, LEVEL_TYPE_AUTO_RESISTANCE,
)
from levels import Levels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_level(
    name="s1",
    t1=1000,
    v1=20.0,
    t2=2000,
    v2=21.0,
    ltype=LEVEL_TYPE_LONG_SUPPORT,
    active=True,
):
    return {
        LI_NAME: name,
        LI_TIME1: t1,
        LI_VAL1: v1,
        LI_TIME2: t2,
        LI_VAL2: v2,
        LI_ACTIVE: active,
        LI_TYPE: ltype,
        LI_HAD_CONTACT: False,
    }


# ---------------------------------------------------------------------------
# get_level_value — static method, no file I/O
# ---------------------------------------------------------------------------

class TestGetLevelValue:
    def test_get_level_value_at_start(self):
        """Value at TIME1 returns VAL1."""
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=21.0)
        assert Levels.get_level_value(level, 1000) == pytest.approx(20.0)

    def test_get_level_value_at_end(self):
        """Value at TIME2 returns VAL2."""
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=21.0)
        assert Levels.get_level_value(level, 2000) == pytest.approx(21.0)

    def test_get_level_value_midpoint(self):
        """Linear interpolation at midpoint returns (VAL1+VAL2)/2."""
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=22.0)
        assert Levels.get_level_value(level, 1500) == pytest.approx(21.0)

    def test_get_level_value_extrapolates_before_start(self):
        """cur_time < TIME1 extrapolates correctly (negative direction)."""
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=22.0)
        # slope = (22-20)/(2000-1000) = 0.002 per sec
        # at t=500: 20.0 + 0.002*(500-1000) = 20.0 - 1.0 = 19.0
        assert Levels.get_level_value(level, 500) == pytest.approx(19.0)


# ---------------------------------------------------------------------------
# get_active_levels
# ---------------------------------------------------------------------------

class TestGetActiveLevels:
    def test_get_active_levels_returns_matching(self):
        """Only active levels of the right type within time range are returned."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = [
            make_level(name="a", t1=1000, t2=2000, ltype=LEVEL_TYPE_LONG_SUPPORT, active=True),
            make_level(name="b", t1=1000, t2=2000, ltype=LEVEL_TYPE_LONG_RESISTANCE, active=True),
        ]
        result = levels_obj.get_active_levels(LEVEL_TYPE_LONG_SUPPORT, 1500)
        assert len(result) == 1
        assert result[0][LI_NAME] == "a"

    def test_get_active_levels_excludes_inactive(self):
        """Inactive levels are excluded."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = [
            make_level(name="active", t1=1000, t2=2000, active=True),
            make_level(name="inactive", t1=1000, t2=2000, active=False),
        ]
        result = levels_obj.get_active_levels(LEVEL_TYPE_LONG_SUPPORT, 1500)
        assert len(result) == 1
        assert result[0][LI_NAME] == "active"

    def test_get_active_levels_excludes_out_of_range(self):
        """Levels outside the time range [TIME1, TIME2] are excluded."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = [
            make_level(name="in_range", t1=1000, t2=2000, active=True),
            make_level(name="future", t1=3000, t2=4000, active=True),
            make_level(name="past", t1=100, t2=500, active=True),
        ]
        result = levels_obj.get_active_levels(LEVEL_TYPE_LONG_SUPPORT, 1500)
        assert len(result) == 1
        assert result[0][LI_NAME] == "in_range"

    def test_get_active_levels_includes_boundary_times(self):
        """cur_time exactly equal to TIME1 or TIME2 is included."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = [
            make_level(name="at_start", t1=1000, t2=2000, active=True),
        ]
        assert len(levels_obj.get_active_levels(LEVEL_TYPE_LONG_SUPPORT, 1000)) == 1
        assert len(levels_obj.get_active_levels(LEVEL_TYPE_LONG_SUPPORT, 2000)) == 1


# ---------------------------------------------------------------------------
# Proximity helpers
# ---------------------------------------------------------------------------

class TestProximityHelpers:
    def test_is_price_near_level(self):
        """Price within ATR of level value returns True."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = []
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=20.0)  # flat level at 20.0
        assert levels_obj.is_price_near_level(20.3, level, atr=0.5, cur_time=1500) is True

    def test_is_price_not_near_level(self):
        """Price outside ATR of level value returns False."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = []
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=20.0)
        assert levels_obj.is_price_near_level(21.0, level, atr=0.5, cur_time=1500) is False

    def test_is_price_near_level_exact_boundary(self):
        """Price exactly ATR away is still near (<=)."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = []
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=20.0)
        assert levels_obj.is_price_near_level(20.5, level, atr=0.5, cur_time=1500) is True

    def test_is_price_over_level(self):
        """price > level value returns True."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = []
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=20.0)
        assert levels_obj.is_price_over_level(21.0, level, cur_time=1500) is True
        assert levels_obj.is_price_over_level(19.0, level, cur_time=1500) is False

    def test_is_price_under_level(self):
        """price < level value returns True."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = []
        level = make_level(t1=1000, v1=20.0, t2=2000, v2=20.0)
        assert levels_obj.is_price_under_level(19.0, level, cur_time=1500) is True
        assert levels_obj.is_price_under_level(21.0, level, cur_time=1500) is False


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

class TestLoadFromFile:
    def test_load_from_file(self, tmp_path, monkeypatch):
        """Writes levels.txt to tmp_path and verifies levels are loaded correctly."""
        levels_txt = tmp_path / "levels.txt"
        levels_txt.write_text(
            "NAME TIME1 VAL1 TIME2 VAL2 TYPE ACTIVE\n"
            "support_1 1693526400 20.50 1694736000 21.00 1 1\n"
            "resistance_1 1693526400 25.00 1694736000 25.50 2 0\n"
        )

        import helpers
        monkeypatch.setattr(helpers, "shared_folder", lambda: str(tmp_path) + "/")

        levels = Levels.load()
        assert len(levels) == 2

        s = levels[0]
        assert s[LI_NAME] == "support_1"
        assert s[LI_TIME1] == 1693526400
        assert s[LI_VAL1] == pytest.approx(20.50)
        assert s[LI_TIME2] == 1694736000
        assert s[LI_VAL2] == pytest.approx(21.00)
        assert s[LI_TYPE] == LEVEL_TYPE_LONG_SUPPORT
        assert s[LI_ACTIVE] is True
        assert s[LI_HAD_CONTACT] is False

        r = levels[1]
        assert r[LI_NAME] == "resistance_1"
        assert r[LI_ACTIVE] is False

    def test_load_absent_file_returns_empty(self, tmp_path, monkeypatch):
        """Non-existent levels.txt returns empty list without raising."""
        import helpers
        monkeypatch.setattr(helpers, "shared_folder", lambda: str(tmp_path) + "/nonexistent/")

        levels = Levels.load()
        assert levels == []

    def test_load_empty_file_returns_empty(self, tmp_path, monkeypatch):
        """Empty levels.txt (only header) returns empty list."""
        levels_txt = tmp_path / "levels.txt"
        levels_txt.write_text("NAME TIME1 VAL1 TIME2 VAL2 TYPE ACTIVE\n")

        import helpers
        monkeypatch.setattr(helpers, "shared_folder", lambda: str(tmp_path) + "/")

        levels = Levels.load()
        assert levels == []


# ---------------------------------------------------------------------------
# to_signal_levels
# ---------------------------------------------------------------------------

class TestToSignalLevels:
    def _make_data_point(self, unix_ts: float):
        """Minimal fake data_point with a .timestamp that has .timestamp() method."""
        import pandas as pd
        ts = pd.Timestamp(unix_ts, unit="s", tz="UTC")

        class FakeDataPoint:
            timestamp = ts

        return FakeDataPoint()

    def test_to_signal_levels_returns_dict(self):
        """Returns a dict with all LEVEL_TYPE_* integers as keys."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = []

        dp = self._make_data_point(1500.0)
        result = levels_obj.to_signal_levels(dp)

        assert isinstance(result, dict)
        all_types = [1, 2, 3, 4, 5, 6, 7]
        for lt in all_types:
            assert lt in result
            assert isinstance(result[lt], list)

    def test_to_signal_levels_with_active_level(self):
        """Active level appears in results at the correct type key with interpolated value."""
        levels_obj = Levels.__new__(Levels)
        # flat level at 20.0 for type LONG_SUPPORT (1), active during [1000, 2000]
        levels_obj._levels = [
            make_level(
                name="flat",
                t1=1000,
                v1=20.0,
                t2=2000,
                v2=20.0,
                ltype=LEVEL_TYPE_LONG_SUPPORT,
                active=True,
            )
        ]

        dp = self._make_data_point(1500.0)
        result = levels_obj.to_signal_levels(dp)

        assert result[LEVEL_TYPE_LONG_SUPPORT] == [pytest.approx(20.0)]
        # Other types should be empty
        assert result[LEVEL_TYPE_LONG_RESISTANCE] == []

    def test_to_signal_levels_inactive_level_excluded(self):
        """Inactive levels do not appear in to_signal_levels output."""
        levels_obj = Levels.__new__(Levels)
        levels_obj._levels = [
            make_level(name="inactive", t1=1000, t2=2000, active=False)
        ]
        dp = self._make_data_point(1500.0)
        result = levels_obj.to_signal_levels(dp)
        assert result[LEVEL_TYPE_LONG_SUPPORT] == []
