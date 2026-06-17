"""StrategyTest2Long — RSI zone_class long test strategy (Phase 14, Task 01).

Open when zone_class leaves oversold (rises above the lowest tier); close when
it enters overbought (reaches the top tier). zone_class is the integer RSI tier
0..4 (0 = oversold, 4 = overbought), so half-integer thresholds express the
transitions with the existing value-crossover primitives.

Default tf=15: zone_class is computed for tf in {15,60,240,1440}, NOT tf=5, in
the functional dataset. (Spec named "5_rsi"; true 5m zone_class needs a dataset
regen — tracked as a follow-up.)
"""

from strategies.strategy import Strategy
from signals_lib.signal_manager import SignalChain
from signals_lib.common import Cross_Up_Val_Signal
from constants import STRATEGY_ACTION_OPEN_LONG, STRATEGY_ACTION_CLOSE_LONG


class StrategyTest2Long(Strategy):
    """Long: RSI zone_class oversold-exit entry, overbought-entry exit."""

    def __init__(self, fee: float, tf: int = 15) -> None:
        self.tf: int = tf
        super().__init__(fee)

    def register_signals(self) -> None:
        # Leaves oversold: prev zone_class == 0, now >= 1  → crosses above 0.5
        entry = SignalChain("t2_long_entry", STRATEGY_ACTION_OPEN_LONG, tf=self.tf, notify=True)
        entry.add(Cross_Up_Val_Signal(tf=self.tf, indi_1="zone_class", val=0.5))
        self.signals.add_chain(entry)

        # Enters overbought: now == 4  → crosses above 3.5
        exit_ = SignalChain("t2_long_exit", STRATEGY_ACTION_CLOSE_LONG, tf=self.tf, notify=True)
        exit_.add(Cross_Up_Val_Signal(tf=self.tf, indi_1="zone_class", val=3.5))
        self.signals.add_chain(exit_)

    def check_conditions(self, data_point, position_state: int, action_msg) -> bool:
        return True
