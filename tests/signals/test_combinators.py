import pytest
from simple_trader.signals.atomic import Less, Greater
from simple_trader.signals.combinators import And, Or, Not, Continuous, History
from tests.signals.conftest import _make_live_point


def _pt(rsi: float) -> object:
    return _make_live_point({1: {"rsi_14": [rsi]}})


def test_and_true_when_all_true():
    sig = And([Less("rsi_14", 1, 70), Greater("rsi_14", 1, 30)])
    assert sig.check(_pt(50)) is True


def test_and_false_when_any_false():
    sig = And([Less("rsi_14", 1, 70), Greater("rsi_14", 1, 60)])
    assert sig.check(_pt(50)) is False


def test_or_true_when_any_true():
    sig = Or([Less("rsi_14", 1, 30), Greater("rsi_14", 1, 70)])
    assert sig.check(_pt(80)) is True


def test_or_false_when_all_false():
    sig = Or([Less("rsi_14", 1, 30), Greater("rsi_14", 1, 70)])
    assert sig.check(_pt(50)) is False


def test_not_inverts_result():
    sig = Not(Less("rsi_14", 1, 60))
    assert sig.check(_pt(50)) is False
    assert sig.check(_pt(70)) is True


def test_continuous_true_when_n_consecutive_true():
    sig = Continuous(Greater("rsi_14", 1, 30), n=3)
    points = [_pt(50), _pt(50), _pt(50)]
    results = [sig.check(p) for p in points]
    assert results == [False, False, True]


def test_continuous_resets_on_false():
    sig = Continuous(Greater("rsi_14", 1, 30), n=3)
    sig.check(_pt(50))
    sig.check(_pt(50))
    sig.check(_pt(10))
    sig.check(_pt(50))
    assert sig.check(_pt(50)) is False


def test_history_true_if_any_in_window_was_true():
    sig = History(Greater("rsi_14", 1, 60), window=3)
    sig.check(_pt(70))
    sig.check(_pt(40))
    assert sig.check(_pt(40)) is True


def test_history_false_after_window_expires():
    sig = History(Greater("rsi_14", 1, 60), window=2)
    sig.check(_pt(70))
    sig.check(_pt(40))
    assert sig.check(_pt(40)) is False
