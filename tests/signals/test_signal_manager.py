import pytest
from simple_trader.signals.signal_manager import SignalManager
from simple_trader.signals.signal_chain import SignalChain
from simple_trader.signals.atomic import Greater
from simple_trader.infrastructure.constants import StrategyAction
from tests.signals.conftest import _make_live_point


def _pt(rsi: float) -> object:
    return _make_live_point({1: {"rsi_14": [rsi]}})


def _make_chain(action: StrategyAction, threshold: float) -> SignalChain:
    chain = SignalChain(f"chain_{action.name}", result_action=action, tf=1)
    chain.add(Greater("rsi_14", 1, threshold))
    return chain


def test_manager_starts_empty():
    mgr = SignalManager()
    assert mgr.chains == []


def test_manager_add_chain():
    mgr = SignalManager()
    chain = _make_chain(StrategyAction.OPEN_LONG, 30)
    mgr.add(chain)
    assert len(mgr.chains) == 1


def test_manager_check_returns_completed_chains():
    mgr = SignalManager()
    chain = _make_chain(StrategyAction.OPEN_LONG, 30)
    mgr.add(chain)
    completed = mgr.check(_pt(50))
    assert len(completed) == 1
    assert completed[0].result_action == StrategyAction.OPEN_LONG


def test_manager_check_returns_empty_when_no_completion():
    mgr = SignalManager()
    chain = _make_chain(StrategyAction.OPEN_LONG, 70)
    mgr.add(chain)
    completed = mgr.check(_pt(50))
    assert completed == []


def test_manager_resets_completed_chains_after_check():
    mgr = SignalManager()
    chain = _make_chain(StrategyAction.OPEN_LONG, 30)
    mgr.add(chain)
    mgr.check(_pt(50))
    assert chain.cur_pos == 0


def test_manager_multiple_chains_can_complete_same_tick():
    mgr = SignalManager()
    chain_long  = _make_chain(StrategyAction.OPEN_LONG,  30)
    chain_short = _make_chain(StrategyAction.CLOSE_SHORT, 20)
    mgr.add(chain_long)
    mgr.add(chain_short)
    completed = mgr.check(_pt(50))
    assert len(completed) == 2
