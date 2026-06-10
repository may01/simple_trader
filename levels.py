# levels.py — Support/resistance level loading, interpolation, and proximity checks.

from __future__ import annotations

from constants import (
    LI_NAME, LI_TIME1, LI_VAL1, LI_TIME2, LI_VAL2,
    LI_ACTIVE, LI_TYPE, LI_HAD_CONTACT,
    LEVEL_TYPE_LONG_SUPPORT, LEVEL_TYPE_LONG_RESISTANCE,
    LEVEL_TYPE_SHORT_SUPPORT, LEVEL_TYPE_SHORT_RESISTANCE,
    LEVEL_TYPE_TARGET, LEVEL_TYPE_AUTO_SUPPORT, LEVEL_TYPE_AUTO_RESISTANCE,
)

_ALL_LEVEL_TYPES = [
    LEVEL_TYPE_LONG_SUPPORT,
    LEVEL_TYPE_LONG_RESISTANCE,
    LEVEL_TYPE_SHORT_SUPPORT,
    LEVEL_TYPE_SHORT_RESISTANCE,
    LEVEL_TYPE_TARGET,
    LEVEL_TYPE_AUTO_SUPPORT,
    LEVEL_TYPE_AUTO_RESISTANCE,
]


class Levels:
    """Load and query support/resistance levels from levels.txt."""

    def __init__(self, thread_num: int = 0):
        """Load levels from shared/{pair}/levels.txt. Empty list if absent."""
        self._levels: list[dict] = self.load(thread_num)

    @staticmethod
    def load(thread_num: int = 0) -> list[dict]:
        """Load and parse levels.txt. Returns [] if file absent or empty."""
        from helpers import shared_folder

        path = shared_folder() + "levels.txt"
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []

        levels: list[dict] = []
        # First line is header — skip it
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            name = parts[0]
            t1 = int(parts[1])
            v1 = float(parts[2])
            t2 = int(parts[3])
            v2 = float(parts[4])
            ltype = int(parts[5])
            active = parts[6] == "1"
            levels.append({
                LI_NAME: name,
                LI_TIME1: t1,
                LI_VAL1: v1,
                LI_TIME2: t2,
                LI_VAL2: v2,
                LI_ACTIVE: active,
                LI_TYPE: ltype,
                LI_HAD_CONTACT: False,
            })
        return levels

    def get_active_levels(self, level_type: int, cur_time: float) -> list[dict]:
        """Return active levels of given type where TIME1 <= cur_time <= TIME2."""
        return [
            lvl for lvl in self._levels
            if (
                lvl[LI_TYPE] == level_type
                and lvl[LI_ACTIVE]
                and lvl[LI_TIME1] <= cur_time <= lvl[LI_TIME2]
            )
        ]

    @staticmethod
    def get_level_value(level_dict: dict, cur_time: float) -> float:
        """Linear interpolation (or extrapolation) of level value at cur_time.

        Formula: VAL1 + (VAL2 - VAL1) * (cur_time - TIME1) / (TIME2 - TIME1)
        Extrapolates beyond TIME1/TIME2 when needed.
        """
        t1: float = level_dict[LI_TIME1]
        v1: float = level_dict[LI_VAL1]
        t2: float = level_dict[LI_TIME2]
        v2: float = level_dict[LI_VAL2]
        return v1 + (v2 - v1) * (cur_time - t1) / (t2 - t1)

    def is_price_near_level(
        self, price: float, level: dict, atr: float, cur_time: float
    ) -> bool:
        """Return True if abs(price - level_value) <= atr."""
        return abs(price - self.get_level_value(level, cur_time)) <= atr

    def is_price_over_level(
        self, price: float, level: dict, cur_time: float
    ) -> bool:
        """Return True if price > level_value."""
        return price > self.get_level_value(level, cur_time)

    def is_price_under_level(
        self, price: float, level: dict, cur_time: float
    ) -> bool:
        """Return True if price < level_value."""
        return price < self.get_level_value(level, cur_time)

    def to_signal_levels(self, data_point) -> dict:
        """Return {level_type: [interpolated_prices...]} for all active levels.

        Called once per tick. cur_time is derived from data_point.timestamp.
        """
        cur_time: float = data_point.timestamp.timestamp()
        return {
            lt: [self.get_level_value(lvl, cur_time) for lvl in self.get_active_levels(lt, cur_time)]
            for lt in _ALL_LEVEL_TYPES
        }
