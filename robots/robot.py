"""robots/robot.py — Live trading robot with polling loop and crash recovery (Phase 10, Task 02).

Drives live trading: polls live_data for the current DataPoint each second,
asks StrategyManager what to do, and dispatches to order management methods
(implemented in Task 03).
"""

import logging
import time

from constants import (
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
    STATUS_FAIL,
    STATUS_SUCCESS,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
    STRATEGY_ACTION_DO_STOP_LOSS,
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    TRADE_BUY,
    TRADE_SELL,
)
from position.position import Position
from robots.live_order_tracker import LiveOrderTracker
from stocks.base_stock import StockInterface
from strategies.strategy_manager import StrategyManager

logger = logging.getLogger(__name__)

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
        try:
            self.tracker.load()
        except Exception:
            logger.exception("tracker.load() failed; starting fresh")
        self.running = True
        try:
            while self.running:
                try:
                    self.do()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception("do() raised an exception; continuing loop")
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
    # Order management (Task 03)
    # ------------------------------------------------------------------

    def _place_valid_order(self, trade_type: str, price: float, amount: float) -> str:
        """Place a LIMIT order if the entry price is still valid.

        Checks position.check_stop_open(price) first; returns "" if stale.
        Wraps stock.trade() in try/except — returns "" on any failure.

        Returns:
            order_id string on success, "" on any failure.
        """
        if not self.position.check_stop_open(price):
            return ""
        try:
            status, result = self.stock.trade(trade_type, price, amount)
            if status == STATUS_FAIL:
                return ""
            return result.get("order_id", "")
        except Exception:
            logger.exception("_place_valid_order: stock.trade raised")
            return ""

    def _open_position(
        self,
        data_point,
        strategy_action: str,
        open_prices: list,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Open a new position: create position, borrow (SHORT), place entry order."""
        if not open_prices:
            return
        opened = self.position.open(
            strategy_action, open_prices, close_prices, stop_price, tf, action_msg=None
        )
        if not opened:
            return

        price = open_prices[0]
        if price <= 0:
            logger.error("_open_position: price <= 0, aborting")
            return
        amount = self.position.full_position / price

        if strategy_action == STRATEGY_ACTION_OPEN_LONG:
            order_id = self._place_valid_order(TRADE_BUY, price, amount)
            if order_id == "":
                logger.error("_open_position: LONG order placement failed, skipping tracking")
                return
            self.tracker.set_buy_order(order_id)
        elif strategy_action == STRATEGY_ACTION_OPEN_SHORT:
            # Borrow coin before placing sell order
            try:
                self.stock.borrow(self.stock.coin, amount)
            except Exception:
                logger.exception("_open_position: stock.borrow raised")
                return
            order_id = self._place_valid_order(TRADE_SELL, price, amount)
            if order_id == "":
                logger.error("_open_position: SHORT order placement failed, skipping tracking")
                return
            self.tracker.set_sell_order(order_id)

    def _close_position(
        self,
        data_point,
        strategy_action: str,
        close_prices: list,
        stop_price: float,
        tf: int,
    ) -> None:
        """Close the current position: cancel existing order, place exit order."""
        if not self.position.close(strategy_action, close_prices, stop_price, tf, action_msg=None):
            return

        # Cancel any existing open order for this side
        if self.tracker.buy_id:
            self.tracker.cancel_buy()
        if self.tracker.sell_id:
            self.tracker.cancel_sell()

        if not close_prices:
            return
        price = close_prices[0]
        amount = self.position.full_position / price if price else 0.0

        if strategy_action == STRATEGY_ACTION_CLOSE_LONG or \
                strategy_action == STRATEGY_ACTION_CLOSE_LONG_PART:
            order_id = self._place_valid_order(TRADE_SELL, price, amount)
            self.tracker.set_sell_order(order_id)
        elif strategy_action == STRATEGY_ACTION_CLOSE_SHORT or \
                strategy_action == STRATEGY_ACTION_CLOSE_SHORT_PART:
            order_id = self._place_valid_order(TRADE_BUY, price, amount)
            self.tracker.set_buy_order(order_id)

    def _stop_loss(
        self,
        data_point,
        close_prices: list = None,
        stop_price: float = 0.0,
        tf: int = 0,
    ) -> None:
        """Cancel all open orders and place a LIMIT exit at current price."""
        if self.tracker.buy_id:
            self.tracker.cancel_buy()
        if self.tracker.sell_id:
            self.tracker.cancel_sell()

        cur_price = data_point.cur_price("close")
        amount = self.position.full_position / cur_price if cur_price else 0.0

        pos_impl = self.position.posImpl
        if pos_impl is not None and pos_impl.position_type == POSITION_TYPE_SHORT:
            order_id = self._place_valid_order(TRADE_BUY, cur_price, amount)
            self.tracker.set_buy_order(order_id)
        else:
            # Default: LONG or unknown — sell to exit
            order_id = self._place_valid_order(TRADE_SELL, cur_price, amount)
            self.tracker.set_sell_order(order_id)

    def _do_finalize_action(self) -> None:
        """Finalize the completed position: record P&L and clear tracker."""
        try:
            revenue_pct, revenue_abs = self.position.finalize()
        finally:
            self.tracker.clear()
        logger.info(
            "Trade finalized: revenue_pct=%.4f revenue_abs=%.2f",
            revenue_pct,
            revenue_abs,
        )

    def _process_executed_orders(self, data_point) -> None:
        """Poll for filled orders and record fills; finalize on full exit fill.

        Direction-aware routing:
          LONG:  buy_id  → entry_fill,  sell_id → exit_fill  (sell-to-close)
          SHORT: sell_id → entry_fill,  buy_id  → exit_fill  (buy-back-to-close)
        """
        pos_impl = self.position.posImpl
        is_short = (
            pos_impl is not None
            and pos_impl.position_type == POSITION_TYPE_SHORT
        )

        if is_short:
            # SHORT entry order is a sell; SHORT exit order is a buy-back
            entry_id = self.tracker.sell_id
            exit_id = self.tracker.buy_id
        else:
            # LONG (or no position yet): entry is a buy, exit is a sell
            entry_id = self.tracker.buy_id
            exit_id = self.tracker.sell_id

        if entry_id:
            status, fill = self.tracker.check_fill(entry_id)
            if status == STATUS_SUCCESS:
                coin_amount = fill["start_amount"] - fill["left_amount"]
                price = fill["rate"]
                self.position.record_entry_fill(coin_amount, price)

        if exit_id:
            status, fill = self.tracker.check_fill(exit_id)
            if status == STATUS_SUCCESS:
                coin_amount = fill["start_amount"] - fill["left_amount"]
                price = fill["rate"]
                self.position.record_exit_fill(coin_amount, price)
                if fill["left_amount"] == 0:
                    self._do_finalize_action()

    def _stop_loss_cancel_actions(self, data_point) -> None:
        """Trigger stop-loss or time-based close if conditions are met."""
        cur_price = data_point.cur_price("close")
        cur_time = data_point.timestamp

        if self.position.is_stop_loss_triggered(cur_price):
            self._stop_loss(data_point)
        elif self.position.close_by_time(cur_time):
            # Close at current price using CLOSE_LONG (direction resolved in _close_position)
            pos_impl = self.position.posImpl
            if pos_impl is not None and pos_impl.position_type == POSITION_TYPE_SHORT:
                action = STRATEGY_ACTION_CLOSE_SHORT
            else:
                action = STRATEGY_ACTION_CLOSE_LONG
            self._close_position(data_point, action, [cur_price], cur_price, 0)
