import pytest
from simple_trader.data.levels import Levels


def test_levels_initializes_empty():
    lvl = Levels()
    assert lvl.get_all() == []


def test_levels_add_stores_level():
    lvl = Levels()
    lvl.add(price=15.5, label="support")
    all_levels = lvl.get_all()
    assert len(all_levels) == 1
    assert all_levels[0]["price"] == 15.5


def test_levels_nearest_returns_closest_level():
    lvl = Levels()
    lvl.add(10.0, "support")
    lvl.add(20.0, "resistance")
    lvl.add(15.0, "support")
    nearest = lvl.nearest(price=14.0)
    assert nearest["price"] == 15.0


def test_levels_nearest_returns_none_when_empty():
    lvl = Levels()
    assert lvl.nearest(price=10.0) is None


def test_levels_load_from_list():
    lvl = Levels()
    lvl.load([{"price": 10.0, "label": "s"}, {"price": 20.0, "label": "r"}])
    assert len(lvl.get_all()) == 2


def test_levels_clear_removes_all():
    lvl = Levels()
    lvl.add(10.0, "support")
    lvl.clear()
    assert lvl.get_all() == []
