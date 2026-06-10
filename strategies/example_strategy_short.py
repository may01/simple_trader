"""ExampleStrategyShort — minimal reference implementation for pipeline validation (Phase 07, Task 02).

Purpose: Validate the full Strategy interface end-to-end without sophisticated trading logic.
Signal logic is intentionally trivial (CCI crossover).  Not for production use.
"""

from strategies.strategy import Strategy
from signals_lib.signal_manager import SignalChain
from signals_lib.common import Cross_Up_Val_Signal, Cross_Down_Val_Signal
from constants import (
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_SHORT,
)


class ExampleStrategyShort(Strategy):
    """Minimal short strategy that exercises the full Strategy interface.

    Signals:
    - Entry: CCI_14 crosses below 100 on tf=15 → OPEN_SHORT
    - Exit: CCI_14 crosses above -100 on tf=15 → CLOSE_SHORT

    Purpose: Validate the OPEN → fill → CLOSE → finalize pipeline end-to-end.
    """

    def register_signals(self) -> None:
        """Register two signal chains: short_entry and short_exit.

        Both operate on tf=15 (15-minute candles).
        Each has a single signal for intentional simplicity.
        """
        # Short entry: CCI crosses below 100
        short_entry = SignalChain(
            name="short_entry",
            res_action=STRATEGY_ACTION_OPEN_SHORT,
            tf=15,
            notify=True
        )
        short_entry.add(Cross_Down_Val_Signal(tf=15, indi_1="cci_14", val=100.0))
        self.signals.add_chain(short_entry)

        # Short exit: CCI crosses above -100
        short_exit = SignalChain(
            name="short_exit",
            res_action=STRATEGY_ACTION_CLOSE_SHORT,
            tf=15,
            notify=True
        )
        short_exit.add(Cross_Up_Val_Signal(tf=15, indi_1="cci_14", val=-100.0))
        self.signals.add_chain(short_exit)

    def check_conditions(self, data_point, position_state: int, action_msg) -> bool:
        """Always return True — incompatible action rejection is position's responsibility.

        Args:
            data_point: Current market data point.
            position_state: Current position state constant.
            action_msg: Action accumulator (may be None in tests).

        Returns:
            Always True to proceed with signal evaluation.
        """
        return True
