import pytest
from simple_trader.signals.signal_chain import SignalChain
from simple_trader.signals.atomic import Greater, Less
from simple_trader.infrastructure.constants import StrategyAction
from tests.signals.conftest import _make_live_point


def _pt(rsi: float) -> object:
    return _make_live_point({1: {"rsi_14": [rsi]}})


def test_chain_starts_at_position_zero():
    chain = SignalChain("test", result_action=StrategyAction.OPEN_LONG, tf=1)
    chain.add(Greater("rsi_14", 1, 30))
    assert chain.cur_pos == 0


def test_chain_advances_when_signal_fires():
    chain = SignalChain("test", result_action=StrategyAction.OPEN_LONG, tf=1)
    chain.add(Greater("rsi_14", 1, 30))
    chain.add(Less("rsi_14", 1, 80))
    chain.check(_pt(50))
    assert chain.cur_pos == 1


def test_chain_does_not_advance_when_signal_false():
    chain = SignalChain("test", result_action=StrategyAction.OPEN_LONG, tf=1)
    chain.add(Greater("rsi_14", 1, 70))
    chain.check(_pt(50))
    assert chain.cur_pos == 0


def test_chain_complete_returns_true_after_all_signals():
    chain = SignalChain("test", result_action=StrategyAction.OPEN_LONG, tf=1)
    chain.add(Greater("rsi_14", 1, 30))
    chain.add(Less("rsi_14", 1, 80))
    chain.check(_pt(50))
    result = chain.check(_pt(50))
    assert result is True
    assert chain.is_complete


def test_chain_reset_restarts_from_zero():
    chain = SignalChain("test", result_action=StrategyAction.OPEN_LONG, tf=1)
    chain.add(Greater("rsi_14", 1, 30))
    chain.check(_pt(50))
    chain.reset()
    assert chain.cur_pos == 0
    assert not chain.is_complete


def test_chain_check_returns_false_when_not_complete():
    chain = SignalChain("test", result_action=StrategyAction.OPEN_LONG, tf=1)
    chain.add(Greater("rsi_14", 1, 30))
    chain.add(Less("rsi_14", 1, 80))
    result = chain.check(_pt(50))
    assert result is False


def test_chain_stores_result_action_and_tf():
    chain = SignalChain("entry", result_action=StrategyAction.OPEN_SHORT, tf=15)
    assert chain.result_action == StrategyAction.OPEN_SHORT
    assert chain.tf == 15
