"""ShortPosition — direction-specific short trading position."""

import time

from constants import (
    POSITION_STATE_WAIT,
    POSITION_STATE_WAIT_BUY,
    POSITION_STATE_WAIT_SELL,
    POSITION_TYPE_SHORT,
)
from position.base_position import BasePosition
from position.coin import Coin


class ShortPosition(BasePosition):
    """Concrete short position: sell first, buy back later.

    Coin flow:
        Entry (sell): spend coin (borrowed) → receive USDT proceeds
        Exit  (buy):  spend USDT → receive coin (to repay borrow)

    Args:
        fee: Trading fee fraction (e.g. 0.001 for 0.1%).
        thread_num: Thread identifier for logging. Defaults to 0.
        full_position: Total capital allocated to this position. Defaults to 10000.0.
    """

    def __init__(self, fee: float, thread_num: int = 0, full_position: float = 10000.0):
        super().__init__(fee, thread_num, full_position)
        self.position_type = POSITION_TYPE_SHORT
        self.coinUse = Coin("coin")   # currency borrowed and sold to enter
        self.coinGet = Coin("usdt")   # USD proceeds from short sale

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, strategy_action, price_open, price_close, price_stop_loss,
             time_period, action_msg) -> bool:
        """Open a short position.

        Validates no open position exists, computes position size based on
        risk/reward, seeds coin balances, and transitions to WAIT_SELL state.

        Returns:
            True on success; False if already open or risk <= 0.
        """
        if self.state != POSITION_STATE_WAIT:
            return False

        avg_price = sum(price_open) / len(price_open)
        risk = price_stop_loss - avg_price
        if risk <= 0:
            return False

        size = min(self.full_position,
                   self.full_position * (self.risk_per_trade / (risk / avg_price)))

        self.coinUse.size = self.full_position
        self.coinUse.want_to_use = size
        self.coinUse.action_amount = size

        self.price_open = list(price_open)
        self.price_close = list(price_close)
        self.price_stop_loss = price_stop_loss
        self.safety_close_time = time_period * 60 * 4
        self.open_time = time.time()
        self.state = POSITION_STATE_WAIT_SELL
        self.action = strategy_action
        # OPEN action: executed=entry, target=take-profit (price_close[0]),
        # stop_loss read from self.price_stop_loss inside _record_change.
        self._record_change(
            "OPEN",
            target_price=self.price_close[0] if self.price_close else 0.0,
            executed_price=self.price_open[0],
        )
        return True

    def record_entry_fill(self, coin_amount: float, price: float) -> None:
        """Record a sell order fill for a short entry.

        Deducts coin from coinUse, credits USD proceeds to coinGet.

        Args:
            coin_amount: Number of coins sold.
            price: Execution price per coin.
        """
        usd_received = coin_amount * price * (1 - self.fee)
        self.coinUse.size -= coin_amount
        self.coinGet.size += usd_received
        self.executed_open.append(price)
        self.executed_open_amount.append(coin_amount * price)
        self.state = POSITION_STATE_WAIT_BUY

    def record_exit_fill(self, coin_amount: float, price: float) -> None:
        """Record a buy order fill for a short exit (buy back coin).

        Deducts USD from coinGet, credits coins back to coinUse.

        Args:
            coin_amount: Number of coins bought back.
            price: Execution price per coin.
        """
        usd_spent = coin_amount * price * (1 + self.fee)
        self.coinGet.size -= usd_spent
        self.coinUse.size += coin_amount
        self.executed_close.append(price)
        self.executed_close_amount.append(coin_amount)
        self.close_idx += 1

    def close(self, strategy_action, price_close, price_stop_loss,
              time_period, action_msg) -> None:
        """Update exit targets and force-set stop-loss.

        Does NOT change state — state is managed by fill callbacks.

        Args:
            strategy_action: New action constant.
            price_close: Updated list of exit price targets.
            price_stop_loss: New stop-loss price (applied with force).
            time_period: Signal timeframe in minutes (unused here).
            action_msg: Caller context string for logging.
        """
        self.price_close = price_close
        self.set_stop_loss(price_stop_loss, action_msg, force=True)
        self.action = strategy_action
        from constants import STRATEGY_ACTION_DO_STOP_LOSS
        self._record_change(
            "STOP_LOSS" if strategy_action == STRATEGY_ACTION_DO_STOP_LOSS else "CLOSE",
            target_price=price_close[0] if price_close else 0.0,
            executed_price=0.0,
            was_stop_loss=(strategy_action == STRATEGY_ACTION_DO_STOP_LOSS),
        )

    # ------------------------------------------------------------------
    # P&L helpers
    # ------------------------------------------------------------------

    def avg_price_open(self) -> float:
        """Compute weighted average entry price.

        Formula: total_USD / total_coins
            where total_coins = sum(amount_i / price_i)

        Returns:
            Average entry price, or 0.0 if no fills.
        """
        if not self.executed_open:
            return 0.0
        total_usd = sum(self.executed_open_amount)
        total_coins = sum(
            amt / px
            for amt, px in zip(self.executed_open_amount, self.executed_open)
        )
        return total_usd / total_coins

    def avg_price_close(self) -> float:
        """Compute weighted average exit price.

        Formula: sum(price_i * coin_amount_i) / sum(coin_amount_i)

        Returns:
            Average exit price, or 0.0 if no fills.
        """
        if not self.executed_close:
            return 0.0
        total_value = sum(
            px * amt
            for px, amt in zip(self.executed_close, self.executed_close_amount)
        )
        total_coins = sum(self.executed_close_amount)
        if total_coins == 0:
            return 0.0
        return total_value / total_coins

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    def direction_profit(self, val: float) -> float:
        """Short profit direction: down is positive — return -val."""
        return -val

    def direction_loss(self, val: float) -> float:
        """Short loss direction: up is positive — return val unchanged."""
        return val

    def first_in_profit(self, a: float, b: float) -> bool:
        """Short: lower buy-back price is more profitable."""
        return a < b

    def is_stop_loss_triggered(self, cur_price: float) -> bool:
        """Short stop-loss triggers when price rises to or above stop level."""
        return cur_price >= self.price_stop_loss
