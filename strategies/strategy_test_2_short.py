"""StrategyTest2Short — RSI zone_class short test strategy (Phase 14, Task 01).

Mirror of StrategyTest2Long: open when zone_class leaves overbought (falls below
the top tier); close when it enters oversold (reaches the bottom tier).
Default tf=15 (see StrategyTest2Long for the tf rationale).
"""

from strategies.strategy import Strategy
from signals_lib.signal_manager import SignalChain
from signals_lib.common import Cross_Down_Val_Signal
from constants import STRATEGY_ACTION_OPEN_SHORT, STRATEGY_ACTION_CLOSE_SHORT


class StrategyTest2Short(Strategy):
    """Short: RSI zone_class overbought-exit entry, oversold-entry exit."""

    def __init__(self, fee: float, tf: int = 15) -> None:
        self.tf: int = tf
        super().__init__(fee)

    def register_signals(self) -> None:
        # Leaves overbought: prev zone_class == 4, now <= 3  → crosses below 3.5
        entry = SignalChain("t2_short_entry", STRATEGY_ACTION_OPEN_SHORT, tf=self.tf, notify=True)
        entry.add(Cross_Down_Val_Signal(tf=self.tf, indi_1="zone_class", val=3.5))
        self.signals.add_chain(entry)

        # Enters oversold: now == 0  → crosses below 0.5
        exit_ = SignalChain("t2_short_exit", STRATEGY_ACTION_CLOSE_SHORT, tf=self.tf, notify=True)
        exit_.add(Cross_Down_Val_Signal(tf=self.tf, indi_1="zone_class", val=0.5))
        self.signals.add_chain(exit_)

    def check_conditions(self, data_point, position_state: int, action_msg) -> bool:
        return True
