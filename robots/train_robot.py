"""TrainRobot — step-driven backtesting simulation robot (Phase 08, Task 01).

Drives a single simulation tick by tick: receives a DataPoint, asks the
StrategyManager what to do, then executes buy/sell/wait accordingly.
Fills are simulated immediately — no order-book latency.
"""

from constants import (
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
    STRATEGY_ACTION_DO_STOP_LOSS,
)
from logs import log_error
from position.position import Position

# Action sets for dispatch
_OPEN_ACTIONS = {STRATEGY_ACTION_OPEN_LONG, STRATEGY_ACTION_OPEN_SHORT}
_CLOSE_ACTIONS = {
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
    STRATEGY_ACTION_DO_STOP_LOSS,
}


class TrainRobot:
    """Step-driven backtesting robot with immediate fill simulation.

    Args:
        strategy_manager: Aggregated strategy that decides actions each tick.
        fee: Trading fee fraction forwarded to Position.
    """

    def __init__(self, strategy_manager, fee: float) -> None:
        self.strategy_manager = strategy_manager
        self.fee: float = fee
        self.position: Position = Position(thread_num=0, fee=fee)
        self.revenue_history: list[tuple[float, float]] = []
        self.trade_count: int = 0
        self._current_coin_amount: float = 0.0

    # ------------------------------------------------------------------
    # Public tick entry point
    # ------------------------------------------------------------------

    def step(self, data_point) -> None:
        """Main tick: call _do() and catch any exception without crashing.

        Args:
            data_point: Current market data point (NOT stored as attribute).
        """
        try:
            self._do(data_point)
        except Exception as exc:
            log_error(f"TrainRobot.step error: {exc}")

    # ------------------------------------------------------------------
    # Internal tick logic
    # ------------------------------------------------------------------

    def _do(self, data_point) -> None:
        """Execute one tick: query strategy and dispatch to buy/sell/wait.

        Args:
            data_point: Current market data point.
        """
        position_state = self.position.get_state()
        cur_time = data_point.timestamp

        strategy_action, open_prices, close_prices, stop_price, tf = (
            self.strategy_manager.check(data_point, position_state, cur_time, action_msg=None)
        )

        if strategy_action in _OPEN_ACTIONS:
            self.buy(data_point, strategy_action, open_prices, close_prices, stop_price, tf)
        elif strategy_action in _CLOSE_ACTIONS:
            self.sell(data_point, strategy_action, close_prices, stop_price, tf)
        else:
            self.wait(data_point)

    # ------------------------------------------------------------------
    # Trade actions
    # ------------------------------------------------------------------

    def buy(
        self,
        data_point,
        strategy_action: int,
        open_prices: list,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Open a new position and record the simulated entry fill.

        Args:
            data_point: Current market data point (not stored).
            strategy_action: OPEN_LONG or OPEN_SHORT constant.
            open_prices: Price list; open_prices[0] is the entry price.
            close_prices: Target exit prices.
            stop_price: Stop-loss price.
            tf: Time frame integer.
        """
        opened = self.position.open(
            strategy_action, open_prices, close_prices, stop_price, tf, None
        )
        if opened:
            self._current_coin_amount = self.position.full_position / open_prices[0]
            self.position.record_entry_fill(self._current_coin_amount, open_prices[0])

    def sell(
        self,
        data_point,
        strategy_action: int,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Close the open position and record the simulated exit fill.

        Args:
            data_point: Current market data point (not stored).
            strategy_action: CLOSE_* or DO_STOP_LOSS constant.
            close_prices: Price list; close_prices[0] is the exit price.
            stop_price: Stop-loss price at close time.
            tf: Time frame integer.
        """
        self.position.close(strategy_action, close_prices, stop_price, tf, None)
        self.position.record_exit_fill(self._current_coin_amount, close_prices[0])
        self._finalize()

    def wait(self, data_point) -> None:
        """Check passive exit conditions (stop-loss, timeout) and act if triggered.

        Args:
            data_point: Current market data point (not stored).
        """
        cur_price = data_point.cur_price('close')
        cur_time = data_point.timestamp

        if self.position.is_stop_loss_triggered(cur_price):
            self.sell(data_point, STRATEGY_ACTION_DO_STOP_LOSS, [cur_price], 0.0, 0)
            return

        if self.position.close_by_time(cur_time):
            self.sell(data_point, STRATEGY_ACTION_DO_STOP_LOSS, [cur_price], 0.0, 0)

    # ------------------------------------------------------------------
    # Post-trade bookkeeping
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        """Finalize the closed trade: record P&L and reset coin state."""
        revenue_pct, revenue_abs = self.position.finalize()
        self.revenue_history.append((revenue_pct, revenue_abs))
        self.trade_count += 1
        self._current_coin_amount = 0.0

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> dict:
        """Return aggregated simulation results.

        Returns:
            Dict with keys: total_trades, revenue_history,
            total_revenue_abs, avg_revenue_pct.
        """
        total_revenue_abs = sum(r[1] for r in self.revenue_history)
        avg_revenue_pct = (
            sum(r[0] for r in self.revenue_history) / self.trade_count
            if self.trade_count > 0
            else 0.0
        )
        return {
            'total_trades': self.trade_count,
            'revenue_history': self.revenue_history,
            'total_revenue_abs': total_revenue_abs,
            'avg_revenue_pct': avg_revenue_pct,
        }
