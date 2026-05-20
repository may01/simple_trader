import pytest
from simple_trader.signals.atomic import Less, Greater, CrossUp, CrossDown, Equal
from tests.signals.conftest import _make_live_point


def test_less_true_when_value_below_threshold():
    sig = Less("rsi_14", tf=1, threshold=60.0)
    point = _make_live_point({1: {"rsi_14": [50.0]}})
    assert sig.check(point) is True


def test_less_false_when_value_at_threshold():
    sig = Less("rsi_14", tf=1, threshold=50.0)
    point = _make_live_point({1: {"rsi_14": [50.0]}})
    assert sig.check(point) is False


def test_less_false_when_value_above_threshold():
    sig = Less("rsi_14", tf=1, threshold=40.0)
    point = _make_live_point({1: {"rsi_14": [50.0]}})
    assert sig.check(point) is False


def test_greater_true_when_value_above_threshold():
    sig = Greater("rsi_14", tf=1, threshold=40.0)
    point = _make_live_point({1: {"rsi_14": [50.0]}})
    assert sig.check(point) is True


def test_greater_false_at_threshold():
    sig = Greater("rsi_14", tf=1, threshold=50.0)
    point = _make_live_point({1: {"rsi_14": [50.0]}})
    assert sig.check(point) is False


def test_cross_up_true_when_crossing_above():
    sig = CrossUp("rsi_14", tf=1, threshold=50.0)
    point = _make_live_point({1: {"rsi_14": [40.0, 60.0]}})
    assert sig.check(point) is True


def test_cross_up_false_when_already_above():
    sig = CrossUp("rsi_14", tf=1, threshold=50.0)
    point = _make_live_point({1: {"rsi_14": [60.0, 70.0]}})
    assert sig.check(point) is False


def test_cross_down_true_when_crossing_below():
    sig = CrossDown("rsi_14", tf=1, threshold=50.0)
    point = _make_live_point({1: {"rsi_14": [60.0, 40.0]}})
    assert sig.check(point) is True


def test_cross_down_false_when_already_below():
    sig = CrossDown("rsi_14", tf=1, threshold=50.0)
    point = _make_live_point({1: {"rsi_14": [30.0, 40.0]}})
    assert sig.check(point) is False


def test_equal_true_within_tolerance():
    sig = Equal("rsi_14", tf=1, target=50.0, tolerance=1.0)
    point = _make_live_point({1: {"rsi_14": [50.5]}})
    assert sig.check(point) is True


def test_equal_false_outside_tolerance():
    sig = Equal("rsi_14", tf=1, target=50.0, tolerance=0.1)
    point = _make_live_point({1: {"rsi_14": [51.0]}})
    assert sig.check(point) is False
