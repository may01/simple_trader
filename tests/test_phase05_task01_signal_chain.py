"""Tests for SignalChain sequential state machine (Phase 05, Task 01).

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

def make_dp(tf: int, rsi: float) -> LiveDataPoint:
    """Create a LiveDataPoint with a single row for the given tf/rsi."""
    df = pd.DataFrame({f"{tf}_rsi_14": [rsi]})
    return LiveDataPoint({tf: df})


class MockAction:
    """Records add_multiply_action calls for assertion."""

    def __init__(self):
        self.calls = []

    def add_multiply_action(self, marker_pos, message):
        self.calls.append((marker_pos, message))


class AbortingSignal(BaseSignal):
    """Signal that always fires on check and always requests abort on reset."""

    def check(self, data_point, levels, action) -> bool:
        return True

    def reset(self) -> bool:
        return True  # always abort the chain


class NeverFiresSignal(BaseSignal):
    """Signal that never fires."""

    def check(self, data_point, levels, action) -> bool:
        return False


class AlwaysFiresSignal(BaseSignal):
    """Signal that always fires and provides custom get_data()."""

    def __init__(self, label: str = "always"):
        super().__init__()
        self.label = label

    def check(self, data_point, levels, action) -> bool:
        return True

    def get_data(self) -> dict:
        return {f"key_{self.label}": self.label}

    def get_marker_pos(self, data_point) -> float:
        return 99.0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSignalChainConstruction:
    def test_init_sets_attributes(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("my_chain", STRATEGY_ACTION_OPEN_LONG, tf=15)
        assert chain.name == "my_chain"
        assert chain.result_action == STRATEGY_ACTION_OPEN_LONG
        assert chain.tf == 15
        assert chain.notify is True
        assert chain.signals == []
        assert chain.cur_pos == 0
        assert chain.timer == 0

    def test_notify_false(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=5, notify=False)
        assert chain.notify is False

    def test_add_appends_signal(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15)
        s1 = AlwaysFiresSignal("a")
        s2 = AlwaysFiresSignal("b")
        chain.add(s1)
        chain.add(s2)
        assert chain.signals == [s1, s2]


# ---------------------------------------------------------------------------
# check() — state machine advancement
# ---------------------------------------------------------------------------

class TestSignalChainCheck:
    def test_no_fire_stays_at_pos0(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(NeverFiresSignal())
        dp = make_dp(15, 30.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.cur_pos == 0

    def test_first_signal_fires_advances_to_pos1(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(Greater_Val_Signal(15, "rsi_14", 40))
        chain.add(Greater_Val_Signal(15, "rsi_14", 50))
        dp = make_dp(15, 45.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.cur_pos == 1

    def test_timer_set_when_signal_fires(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1234.0, None)
        assert chain.timer == 1234.0

    def test_only_current_signal_evaluated(self):
        """After first fires, second is evaluated next tick, not on same tick."""
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        # Second signal never fires — if chain evaluated all signals it would complete
        chain.add(NeverFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        # cur_pos should be 1 (first fired), not 2 (second did not fire on same tick)
        assert chain.cur_pos == 1

    def test_second_signal_fires_on_next_tick(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(Greater_Val_Signal(15, "rsi_14", 40))
        chain.add(Greater_Val_Signal(15, "rsi_14", 50))
        dp1 = make_dp(15, 45.0)
        chain.check(dp1, {}, 1000.0, None)
        dp2 = make_dp(15, 55.0)
        chain.check(dp2, {}, 1001.0, None)
        assert chain.cur_pos == 2

    def test_check_no_op_when_no_signals(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        dp = make_dp(15, 50.0)
        # Should not raise
        chain.check(dp, {}, 1000.0, None)
        assert chain.cur_pos == 0

    def test_abort_on_new_signals_reset_true(self):
        """If new cur_pos signal's reset() returns True, chain resets to 0."""
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())  # pos 0 — fires immediately
        chain.add(AbortingSignal())     # pos 1 — reset() returns True → abort
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        # After first fires, new pos=1 signal's reset() returns True → reset to 0
        assert chain.cur_pos == 0
        assert chain.timer == 0

    def test_no_abort_when_reset_false(self):
        """If new cur_pos signal's reset() returns False, chain stays advanced."""
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        chain.add(NeverFiresSignal())  # reset() returns False (default)
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.cur_pos == 1


# ---------------------------------------------------------------------------
# completed()
# ---------------------------------------------------------------------------

class TestSignalChainCompleted:
    def test_not_completed_initially(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        assert chain.completed(dp, None) is False

    def test_completed_after_all_fire(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.completed(dp, None) is True

    def test_completed_false_when_mid_chain(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        chain.add(NeverFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.completed(dp, None) is False

    def test_completed_true_with_two_signals(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal("a"))
        chain.add(AlwaysFiresSignal("b"))
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        # After tick 1: pos=1 but not completed yet (only one per tick)
        assert chain.cur_pos == 1
        chain.check(dp, {}, 1001.0, None)
        assert chain.completed(dp, None) is True


# ---------------------------------------------------------------------------
# get_action()
# ---------------------------------------------------------------------------

class TestSignalChainGetAction:
    def test_get_action_returns_list_format(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal("x"))
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        result = chain.get_action(50.0, None)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_get_action_first_element_is_result_action(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_SHORT, tf=5, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(5, 50.0)
        chain.check(dp, {}, 1000.0, None)
        result = chain.get_action(50.0, None)
        assert result[0] == STRATEGY_ACTION_OPEN_SHORT

    def test_get_action_second_element_is_tf(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=60, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(60, 50.0)
        chain.check(dp, {}, 1000.0, None)
        result = chain.get_action(50.0, None)
        assert result[1] == 60

    def test_get_action_third_element_is_price(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        result = chain.get_action(42.5, None)
        assert result[2] == 42.5

    def test_get_action_data_dict_has_position_time(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        result = chain.get_action(50.0, None)
        assert "position_time" in result[3]
        assert result[3]["position_time"] == 15

    def test_get_action_merges_signal_get_data(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal("alpha"))
        chain.add(AlwaysFiresSignal("beta"))
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)  # pos -> 1
        chain.check(dp, {}, 1001.0, None)  # pos -> 2 (completed)
        result = chain.get_action(50.0, None)
        # Both signals' get_data() should be merged
        assert result[3]["key_alpha"] == "alpha"
        assert result[3]["key_beta"] == "beta"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestSignalChainReset:
    def test_reset_sets_cur_pos_to_0(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.cur_pos == 1
        chain.reset(None)
        assert chain.cur_pos == 0

    def test_reset_sets_timer_to_0(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 9999.0, None)
        chain.reset(None)
        assert chain.timer == 0


# ---------------------------------------------------------------------------
# notify / logging
# ---------------------------------------------------------------------------

class TestSignalChainNotify:
    def test_notify_true_calls_action_on_signal_fire(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("test_log", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=True)
        sig = AlwaysFiresSignal("z")
        chain.add(sig)
        chain.add(NeverFiresSignal())
        dp = make_dp(15, 50.0)
        mock_action = MockAction()
        chain.check(dp, {}, 1000.0, mock_action)
        assert len(mock_action.calls) == 1
        marker, msg = mock_action.calls[0]
        assert marker == 99.0  # AlwaysFiresSignal.get_marker_pos returns 99.0
        assert "test_log" in msg
        assert "0" in msg  # cur_pos before increment

    def test_notify_false_no_action_calls(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        chain.add(NeverFiresSignal())
        dp = make_dp(15, 50.0)
        mock_action = MockAction()
        chain.check(dp, {}, 1000.0, mock_action)
        assert len(mock_action.calls) == 0

    def test_notify_true_calls_action_on_completed(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=True)
        chain.add(AlwaysFiresSignal("z"))
        dp = make_dp(15, 50.0)
        mock_action = MockAction()
        chain.check(dp, {}, 1000.0, mock_action)  # fires signal, pos -> 1 = completed
        mock_action.calls.clear()
        result = chain.completed(dp, mock_action)
        assert result is True
        assert len(mock_action.calls) == 1
        marker, msg = mock_action.calls[0]
        assert "CPLTD" in msg
        assert "c" in msg

    def test_notify_false_no_action_on_completed(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal("z"))
        dp = make_dp(15, 50.0)
        mock_action = MockAction()
        chain.check(dp, {}, 1000.0, mock_action)
        result = chain.completed(dp, mock_action)
        assert result is True
        assert len(mock_action.calls) == 0

    def test_action_none_does_not_raise_when_notify_true(self):
        """action=None with notify=True must not raise (guard required)."""
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=True)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        # Must not raise even though action=None and notify=True
        chain.check(dp, {}, 1000.0, None)
        chain.completed(dp, None)


# ---------------------------------------------------------------------------
# Full integration scenario
# ---------------------------------------------------------------------------

class TestSignalChainIntegration:
    def test_two_signal_chain_full_flow(self):
        """Mirrors the docker verification snippet from the task spec."""
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("test_chain", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(Greater_Val_Signal(15, "rsi_14", 40))
        chain.add(Greater_Val_Signal(15, "rsi_14", 50))

        dp1 = make_dp(15, 45.0)
        chain.check(dp1, {}, 1000.0, None)
        assert chain.cur_pos == 1

        dp2 = make_dp(15, 55.0)
        chain.check(dp2, {}, 1001.0, None)
        assert chain.completed(dp2, None) is True

        result = chain.get_action(55.0, None)
        assert result[0] == STRATEGY_ACTION_OPEN_LONG
        assert result[1] == 15
        assert result[2] == 55.0
        assert result[3]["position_time"] == 15

    def test_three_signal_chain(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c3", STRATEGY_ACTION_CLOSE_LONG, tf=5, notify=False)
        chain.add(AlwaysFiresSignal("a"))
        chain.add(AlwaysFiresSignal("b"))
        chain.add(AlwaysFiresSignal("c"))
        dp = make_dp(5, 50.0)

        chain.check(dp, {}, 1000.0, None)
        assert chain.cur_pos == 1
        assert chain.completed(dp, None) is False

        chain.check(dp, {}, 1001.0, None)
        assert chain.cur_pos == 2
        assert chain.completed(dp, None) is False

        chain.check(dp, {}, 1002.0, None)
        assert chain.cur_pos == 3
        assert chain.completed(dp, None) is True

    def test_reset_allows_chain_to_restart(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(AlwaysFiresSignal())
        dp = make_dp(15, 50.0)
        chain.check(dp, {}, 1000.0, None)
        assert chain.completed(dp, None) is True
        chain.reset(None)
        assert chain.cur_pos == 0
        assert chain.completed(dp, None) is False

    def test_second_signal_does_not_fire_chain_stays_mid(self):
        from signals_lib.signal_manager import SignalChain
        chain = SignalChain("c", STRATEGY_ACTION_OPEN_LONG, tf=15, notify=False)
        chain.add(Greater_Val_Signal(15, "rsi_14", 40))
        chain.add(Greater_Val_Signal(15, "rsi_14", 60))

        dp1 = make_dp(15, 45.0)
        chain.check(dp1, {}, 1000.0, None)
        assert chain.cur_pos == 1  # first fired

        # Second signal condition not met (rsi=55 < 60)
        dp2 = make_dp(15, 55.0)
        chain.check(dp2, {}, 1001.0, None)
        assert chain.cur_pos == 1  # stuck waiting

        assert chain.completed(dp2, None) is False
