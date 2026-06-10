"""LongPosition — direction-specific long trading position."""

import time

from constants import (
    POSITION_STATE_WAIT,
    POSITION_STATE_WAIT_BUY,
    POSITION_STATE_WAIT_SELL,
    POSITION_TYPE_LONG,
)
from position.base_position import BasePosition
from position.coin import Coin


class LongPosition(BasePosition):
    """Concrete long position: buy first, sell later.

    Coin flow:
        Entry (buy):  spend USDT → receive coin
        Exit  (sell): spend coin → receive USDT

    Args:
        fee: Trading fee fraction (e.g. 0.001 for 0.1%).
        thread_num: Thread identifier for logging. Defaults to 0.
        full_position: Total capital allocated to this position. Defaults to 10000.0.
    """

    def __init__(self, fee: float, thread_num: int = 0, full_position: float = 10000.0):
        super().__init__(fee, thread_num, full_position)
        self.position_type = POSITION_TYPE_LONG
        self.coinUse = Coin("usdt")   # currency spent to enter
        self.coinGet = Coin("coin")   # currency acquired

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, strategy_action, price_open, price_close, price_stop_loss,
             time_period, action_msg) -> bool:
        """Open a long position.

        Validates no open position exists, computes position size based on
        risk/reward, seeds coin balances, and transitions to WAIT_BUY state.

        Returns:
            True on success; False if already open or risk <= 0.
        """
        if self.state != POSITION_STATE_WAIT:
            return False

        avg_price = sum(price_open) / len(price_open)
        risk = avg_price - price_stop_loss
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
        self.state = POSITION_STATE_WAIT_BUY
        self.action = strategy_action
        return True

    def record_entry_fill(self, coin_amount: float, price: float) -> None:
        """Record a buy order fill for a long entry.

        Deducts USD from coinUse, credits coin to coinGet, records the fill.

        Args:
            coin_amount: Number of coins purchased.
            price: Execution price per coin.
        """
        usd_spent = coin_amount * price * (1 + self.fee)
        self.coinUse.size -= usd_spent
        self.coinUse.used += usd_spent
        self.coinGet.size += coin_amount
        self.executed_open.append(price)
        self.executed_open_amount.append(coin_amount * price)
        self.state = POSITION_STATE_WAIT_SELL

    def record_exit_fill(self, coin_amount: float, price: float) -> None:
        """Record a sell order fill for a long exit.

        Deducts coins from coinGet, credits USD to coinUse, records the fill.

        Args:
            coin_amount: Number of coins sold.
            price: Execution price per coin.
        """
        usd_received = coin_amount * price * (1 - self.fee)
        self.coinGet.size -= coin_amount
        self.coinUse.size += usd_received
        self.coinUse.returned += usd_received
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
        return total_value / total_coins

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    def direction_profit(self, val: float) -> float:
        """Long profit direction: up is positive — return val unchanged."""
        return val

    def direction_loss(self, val: float) -> float:
        """Long loss direction: down is negative — return -val."""
        return -val

    def first_in_profit(self, a: float, b: float) -> bool:
        """Long: higher stop-loss is more profitable (closer to current price)."""
        return a > b

    def is_stop_loss_triggered(self, cur_price: float) -> bool:
        """Long stop-loss triggers when price falls to or below stop level."""
        return cur_price <= self.price_stop_loss
