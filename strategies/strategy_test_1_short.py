"""StrategyTest1Short — EMA-cross short test strategy (Phase 14, Task 01).

Mirror of StrategyTest1Long: open when ema_7 crosses below ema_14;
close when ema_7 crosses back above. Default tf=5.
"""

from strategies.strategy import Strategy
from signals_lib.signal_manager import SignalChain
from signals_lib.common import Cross_Up_Signal, Cross_Down_Signal
from constants import STRATEGY_ACTION_OPEN_SHORT, STRATEGY_ACTION_CLOSE_SHORT


class StrategyTest1Short(Strategy):
    """Short: EMA fast/slow crossover entry + exit (mirrored)."""

    def __init__(self, fee: float, tf: int = 5) -> None:
        self.tf: int = tf
        super().__init__(fee)

    def register_signals(self) -> None:
        entry = SignalChain("t1_short_entry", STRATEGY_ACTION_OPEN_SHORT, tf=self.tf, notify=True)
        entry.add(Cross_Down_Signal(tf=self.tf, indi_1="ema_7", indi_2="ema_14"))
        self.signals.add_chain(entry)

        exit_ = SignalChain("t1_short_exit", STRATEGY_ACTION_CLOSE_SHORT, tf=self.tf, notify=True)
        exit_.add(Cross_Up_Signal(tf=self.tf, indi_1="ema_7", indi_2="ema_14"))
        self.signals.add_chain(exit_)

    def check_conditions(self, data_point, position_state: int, action_msg) -> bool:
        return True
