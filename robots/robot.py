"""robots/robot.py — Live trading robot with polling loop and crash recovery (Phase 10, Task 02).

Drives live trading: polls live_data for the current DataPoint each second,
asks StrategyManager what to do, and dispatches to order management methods
(stubs in Task 02; filled in Task 03).
"""

import time

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
from position.position import Position
from robots.live_order_tracker import LiveOrderTracker
from stocks.base_stock import StockInterface
from strategies.strategy_manager import StrategyManager

# Action dispatch sets
_OPEN_ACTIONS = {STRATEGY_ACTION_OPEN_LONG, STRATEGY_ACTION_OPEN_SHORT}
_CLOSE_ACTIONS = {
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
}


class Robot:
    """Live trading robot: polling loop with crash recovery and action dispatch.

    Args:
        strategy_manager: Aggregated strategy that decides actions each tick.
        live_data: Provides ``data_point`` attribute/property per tick.
        stock: Exchange interface for order operations.
        fee: Trading fee fraction.
        persist_path: File path for atomic JSON crash-recovery state.
    """

    def __init__(
        self,
        strategy_manager: StrategyManager,
        live_data,
        stock: StockInterface,
        fee: float,
        persist_path: str,
    ) -> None:
        self.strategy_manager: StrategyManager = strategy_manager
        self.live_data = live_data
        self.stock: StockInterface = stock
        self.fee: float = fee
        self.running: bool = False

        # Created internally
        self.position: Position = Position(thread_num=0, fee=fee)
        self.tracker: LiveOrderTracker = LiveOrderTracker(
            position=self.position,
            stock=stock,
            persist_path=persist_path,
        )

    # ------------------------------------------------------------------
    # Public loop entry point
    # ------------------------------------------------------------------

    def run_instantly(self) -> None:
        """Load crash-recovery state, then poll do() every second until stopped.

        Sets running=True before entering the loop.
        Handles KeyboardInterrupt gracefully by setting running=False.
        """
        self.tracker.load()
        self.running = True
        try:
            while self.running:
                self.do()
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False

    def stop(self) -> None:
        """Signal the polling loop to exit on its next iteration."""
        self.running = False

    # ------------------------------------------------------------------
    # Per-tick logic
    # ------------------------------------------------------------------

    def do(self) -> None:
        """Execute one tick: fetch data, query strategy, dispatch action.

        Gets a fresh DataPoint each call from live_data.data_point.
        """
        data_point = self.live_data.data_point
        position_state = self.position.get_state()
        cur_time = data_point.timestamp

        action, open_prices, close_prices, stop_price, tf = self.strategy_manager.check(
            data_point, position_state, cur_time, action_msg=None
        )

        if action in _OPEN_ACTIONS:
            self._open_position(data_point, action, open_prices, close_prices, stop_price, tf)
        elif action in _CLOSE_ACTIONS:
            self._close_position(data_point, action, close_prices, stop_price, tf)
        elif action == STRATEGY_ACTION_DO_STOP_LOSS:
            self._stop_loss(data_point, close_prices, stop_price, tf)
        else:
            self.wait(data_point)

    def wait(self, data_point) -> None:
        """Handle a NOTHING tick: check fills and stop-loss cancellations.

        Args:
            data_point: Current market data point.
        """
        self._process_executed_orders(data_point)
        self._stop_loss_cancel_actions(data_point)

    # ------------------------------------------------------------------
    # Order management stubs (Task 03 will implement these)
    # ------------------------------------------------------------------

    def _open_position(
        self,
        data_point,
        strategy_action: str,
        open_prices: list,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Place entry order to open a new position. (Task 03)"""
        pass

    def _close_position(
        self,
        data_point,
        strategy_action: str,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Place exit order to close the current position. (Task 03)"""
        pass

    def _stop_loss(
        self,
        data_point,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Execute stop-loss exit order. (Task 03)"""
        pass

    def _process_executed_orders(self, data_point) -> None:
        """Check for and process any filled orders. (Task 03)"""
        pass

    def _stop_loss_cancel_actions(self, data_point) -> None:
        """Cancel stale stop-loss orders if conditions changed. (Task 03)"""
        pass
