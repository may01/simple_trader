"""StrategyTest1Long — EMA-cross long test strategy (Phase 14, Task 01).

Open when ema_7 crosses above ema_14; close when ema_7 crosses back below.
Reuses existing crossover signal primitives — no new signal classes.
Default tf=5 (5_ema_7 / 5_ema_14 are present in the functional dataset).
"""

from strategies.strategy import Strategy
from signals_lib.signal_manager import SignalChain
from signals_lib.common import Cross_Up_Signal, Cross_Down_Signal
from constants import STRATEGY_ACTION_OPEN_LONG, STRATEGY_ACTION_CLOSE_LONG


class StrategyTest1Long(Strategy):
    """Long: EMA fast/slow crossover entry + exit."""

    def __init__(self, fee: float, tf: int = 5) -> None:
        self.tf: int = tf
        super().__init__(fee)

    def register_signals(self) -> None:
        entry = SignalChain("t1_long_entry", STRATEGY_ACTION_OPEN_LONG, tf=self.tf, notify=True)
        entry.add(Cross_Up_Signal(tf=self.tf, indi_1="ema_7", indi_2="ema_14"))
        self.signals.add_chain(entry)

        exit_ = SignalChain("t1_long_exit", STRATEGY_ACTION_CLOSE_LONG, tf=self.tf, notify=True)
        exit_.add(Cross_Down_Signal(tf=self.tf, indi_1="ema_7", indi_2="ema_14"))
        self.signals.add_chain(exit_)

    def check_conditions(self, data_point, position_state: int, action_msg) -> bool:
        return True
