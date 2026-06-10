"""Tests for SignalManager multi-chain container and evaluator (Phase 05, Task 02).

TDD: these tests are written BEFORE the implementation.
"""

import pytest
import pandas as pd

from data import LiveDataPoint
from signals_lib.common import Greater_Val_Signal, Less_Val_Signal
from signals_lib.base_signal import BaseSignal
from constants import (
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def make_dp(close_val: float, rsi: float = 50.0, tf: int = 15) -> LiveDataPoint:
    """Create a LiveDataPoint with 1-min and tf-min frames."""
    df_tf = pd.DataFrame({f"{tf}_rsi_14": [rsi], f"{tf}_close": [close_val]})
    df_1 = pd.DataFrame({"1_close": [close_val], "1_rsi_14": [rsi]})
    if tf == 1:
        return LiveDataPoint({1: df_1})
    return LiveDataPoint({tf: df_tf, 1: df_1})


class AlwaysFiresSignal(BaseSignal):
    """Signal that always fires."""

    def check(self, data_point, levels, action) -> bool:
        return True

    def get_marker_pos(self, data_point) -> float:
        return 1.0


class NeverFiresSignal(BaseSignal):
    """Signal that never fires."""

    def check(self, data_point, levels, action) -> bool:
        return False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSignalManagerConstruction:
    def test_init_creates_empty_chains_list(self):
        from signals_lib.signal_manager import SignalManager
        mgr = SignalManager()
        assert mgr.chains == []

    def test_chains_attribute_is_list(self):
        from signals_lib.signal_manager import SignalManager
        mgr = SignalManager()
        assert isinstance(mgr.chains, list)


# ---------------------------------------------------------------------------
# add_chain()
# ---------------------------------------------------------------------------

class TestSignalManagerAddChain:
    def test_add_chain_appends_to_chains(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        mgr.add_chain(chain)
        assert len(mgr.chains) == 1
        assert mgr.chains[0] is chain

    def test_add_chain_multiple_chains(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        c1 = SignalChain("c1", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        c2 = SignalChain("c2", STRATEGY_ACTION_OPEN_SHORT, tf=15, notify=False)
        c3 = SignalChain("c3", STRATEGY_ACTION_CLOSE_LONG, tf=5, notify=False)
        mgr.add_chain(c1)
        mgr.add_chain(c2)
        mgr.add_chain(c3)
        assert len(mgr.chains) == 3
        assert mgr.chains[1] is c2


# ---------------------------------------------------------------------------
# check() — empty and no-completion cases
# ---------------------------------------------------------------------------

class TestSignalManagerCheckEmpty:
    def test_check_returns_list(self):
        from signals_lib.signal_manager import SignalManager
        mgr = SignalManager()
        dp = make_dp(20.0)
        result = mgr.check(dp, {}, 1000.0, None)
        assert isinstance(result, list)

    def test_check_returns_empty_list_when_no_chains(self):
        from signals_lib.signal_manager import SignalManager
        mgr = SignalManager()
        dp = make_dp(20.0)
        result = mgr.check(dp, {}, 1000.0, None)
        assert result == []

    def test_check_returns_empty_list_when_no_chain_completes(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(NeverFiresSignal())
        mgr.add_chain(chain)
        dp = make_dp(20.0)
        result = mgr.check(dp, {}, 1000.0, None)
        assert result == []


# ---------------------------------------------------------------------------
# check() — completion and reset
# ---------------------------------------------------------------------------

class TestSignalManagerCheckCompletion:
    def test_one_chain_completes_returns_one_action(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        mgr.add_chain(chain)
        dp = make_dp(20.0)
        result = mgr.check(dp, {}, 1000.0, None)
        assert len(result) == 1

    def test_action_format_is_correct(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        mgr.add_chain(chain)
        dp = make_dp(42.0)
        result = mgr.check(dp, {}, 1000.0, None)
        action_tuple = result[0]
        assert len(action_tuple) == 4
        assert action_tuple[0] == STRATEGY_ACTION_OPEN_LONG
        assert action_tuple[1] == 15
        assert action_tuple[2] == 42.0  # 1-min close price
        assert "position_time" in action_tuple[3]
        assert action_tuple[3]["position_time"] == 15

    def test_price_uses_1min_close(self):
        """get_action() must receive data_point.get('close', 1, 0) — always 1-min."""
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        mgr.add_chain(chain)
        # 1-min close = 99.0; 15-min close = 42.0
        df_15 = pd.DataFrame({"15_rsi_14": [60.0], "15_close": [42.0]})
        df_1 = pd.DataFrame({"1_close": [99.0], "1_rsi_14": [60.0]})
        dp = LiveDataPoint({15: df_15, 1: df_1})
        result = mgr.check(dp, {}, 1000.0, None)
        assert result[0][2] == 99.0  # must be 1-min close, not 15-min

    def test_completed_chain_is_reset(self):
        """After a chain completes, it must be reset so it can fire again next tick."""
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        mgr.add_chain(chain)
        dp = make_dp(20.0)
        mgr.check(dp, {}, 1000.0, None)
        # Chain must be back at pos=0 after reset
        assert chain.cur_pos == 0

    def test_two_chains_both_complete_returns_two_actions(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        c1 = SignalChain("c1", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        c1.add(AlwaysFiresSignal())
        c2 = SignalChain("c2", STRATEGY_ACTION_OPEN_SHORT, tf=15, notify=False)
        c2.add(AlwaysFiresSignal())
        mgr.add_chain(c1)
        mgr.add_chain(c2)
        dp = make_dp(20.0)
        result = mgr.check(dp, {}, 1000.0, None)
        assert len(result) == 2

    def test_two_chains_only_one_completes(self):
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        c1 = SignalChain("c1", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        c1.add(AlwaysFiresSignal())
        c2 = SignalChain("c2", STRATEGY_ACTION_OPEN_SHORT, tf=15, notify=False)
        c2.add(NeverFiresSignal())
        mgr.add_chain(c1)
        mgr.add_chain(c2)
        dp = make_dp(20.0)
        result = mgr.check(dp, {}, 1000.0, None)
        assert len(result) == 1
        assert result[0][0] == STRATEGY_ACTION_OPEN_LONG

    def test_all_chains_evaluated_every_tick(self):
        """Even chains at cur_pos=0 must be evaluated each tick."""
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        # Two-signal chain — only first fires on tick 1
        c1 = SignalChain("c1", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        c1.add(Greater_Val_Signal(15, "rsi_14", 40))
        c1.add(Greater_Val_Signal(15, "rsi_14", 50))
        # One-signal chain — fires on any tick
        c2 = SignalChain("c2", STRATEGY_ACTION_OPEN_SHORT, tf=15, notify=False)
        c2.add(AlwaysFiresSignal())
        mgr.add_chain(c1)
        mgr.add_chain(c2)

        df_15 = pd.DataFrame({"15_rsi_14": [45.0], "15_close": [20.0]})
        df_1 = pd.DataFrame({"1_close": [20.0], "1_rsi_14": [45.0]})
        dp = LiveDataPoint({15: df_15, 1: df_1})

        # Tick 1: c1 advances to pos=1 (not done), c2 completes
        result = mgr.check(dp, {}, 1000.0, None)
        assert len(result) == 1
        assert result[0][0] == STRATEGY_ACTION_OPEN_SHORT
        assert c1.cur_pos == 1  # c1 advanced but not completed

    def test_reset_chain_can_complete_on_next_tick(self):
        """After a chain completes and resets, it should be able to complete again next tick."""
        from signals_lib.signal_manager import SignalManager, SignalChain
        mgr = SignalManager()
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        mgr.add_chain(chain)
        dp = make_dp(20.0)
        result1 = mgr.check(dp, {}, 1000.0, None)
        assert len(result1) == 1
        # Chain was reset — fires again on next tick
        result2 = mgr.check(dp, {}, 1001.0, None)
        assert len(result2) == 1


# ---------------------------------------------------------------------------
# Integration: verification snippet from task spec
# ---------------------------------------------------------------------------

class TestSignalManagerVerificationSnippet:
    def test_verification_snippet(self):
        """Reproduce the exact verification snippet from the task spec."""
        from signals_lib.signal_manager import SignalChain, SignalManager
        from signals_lib.common import Greater_Val_Signal

        mgr = SignalManager()
        chain1 = SignalChain("long", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain1.add(Greater_Val_Signal(15, "rsi_14", 50))
        chain2 = SignalChain("short", STRATEGY_ACTION_OPEN_SHORT, tf=15, notify=False)
        chain2.add(Greater_Val_Signal(15, "rsi_14", 70))
        mgr.add_chain(chain1)
        mgr.add_chain(chain2)

        df = pd.DataFrame({"15_rsi_14": [60.0], "1_close": [20.0]})
        pt = LiveDataPoint({15: df, 1: df})
        results = mgr.check(pt, {}, 1000.0, None)

        # chain1 fires (60 > 50), chain2 does not (60 < 70)
        assert len(results) == 1
        assert results[0][0] == STRATEGY_ACTION_OPEN_LONG
