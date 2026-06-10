"""BasePosition — abstract trade lifecycle state machine."""

from abc import ABC, abstractmethod
from constants import (
    POSITION_STATE_WAIT,
    POSITION_STATE_WAIT_SAFETY_BUY,
    POSITION_STATE_WAIT_SAFETY_SELL,
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
    POSITION_STATE_WAIT_BUY,
    POSITION_STATE_WAIT_SELL,
    STRATEGY_ACTION_NOTHING,
)
from position.coin import Coin


class BasePosition(ABC):
    """Abstract base class for long and short position state machines.

    Tracks price targets, execution history, stop-loss enforcement,
    P&L calculation, and position lifecycle from WAIT → open → close → finalize.

    Subclasses must override all abstract methods to express direction-specific
    logic (long vs. short).

    Args:
        fee: Trading fee fraction (e.g. 0.001 for 0.1%). No default — must be injected.
        thread_num: Thread identifier for logging. Defaults to 0.
        full_position: Total capital allocated to this position. Defaults to 10000.0.
    """

    def __init__(self, fee: float, thread_num: int = 0, full_position: float = 10000.0):
        # State machine
        self.state: str = POSITION_STATE_WAIT
        self.action: str = STRATEGY_ACTION_NOTHING
        self.position_type: str = "POSITION_TYPE_UNKNOWN"  # overridden by subclass

        # Configuration
        self.fee: float = fee
        self.full_position: float = full_position
        self.risk_per_trade: float = 0.01
        self.thread_num: int = thread_num

        # Price targets (graduated entry/exit)
        self.price_open: list = []
        self.price_close: list = []
        self.price_stop_loss: float = 0.0

        # Execution history
        self.executed_open: list = []
        self.executed_open_amount: list = []
        self.executed_close: list = []
        self.executed_close_amount: list = []

        # Coin legs (set by subclasses)
        self.coinUse: Coin = Coin("unknown")
        self.coinGet: Coin = Coin("unknown")

        # Timing
        self.open_time: float = 0.0
        self.safety_close_time: int = 0

        # Exit target tracking
        self.close_idx: int = 0

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by LongPosition / ShortPosition
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self, strategy_action, price_open, price_close, price_stop_loss,
             time_period, action_msg) -> bool:
        """Open the position: set targets, state, and start timer."""

    @abstractmethod
    def close(self, strategy_action, price_close, price_stop_loss,
              time_period, action_msg) -> None:
        """Transition to closing state with updated targets."""

    @abstractmethod
    def record_entry_fill(self, coin_amount: float, price: float) -> None:
        """Record an executed entry order fill."""

    @abstractmethod
    def record_exit_fill(self, coin_amount: float, price: float) -> None:
        """Record an executed exit order fill."""

    @abstractmethod
    def avg_price_open(self) -> float:
        """Compute average entry fill price."""

    @abstractmethod
    def avg_price_close(self) -> float:
        """Compute average exit fill price."""

    @abstractmethod
    def direction_profit(self, val: float) -> float:
        """Transform a price delta into a signed profit direction."""

    @abstractmethod
    def direction_loss(self, val: float) -> float:
        """Transform a price delta into a signed loss direction."""

    @abstractmethod
    def first_in_profit(self, a: float, b: float) -> bool:
        """Return True if price *a* is more profitable than price *b* as a stop-loss.

        Long: higher is better (a > b).
        Short: lower is better (a < b).
        """

    @abstractmethod
    def is_stop_loss_triggered(self, cur_price: float) -> bool:
        """Return True if cur_price has crossed the stop-loss level.

        Long:  cur_price <= price_stop_loss
        Short: cur_price >= price_stop_loss
        """

    # ------------------------------------------------------------------
    # Concrete shared methods
    # ------------------------------------------------------------------

    def set_stop_loss(self, price: float, action_msg, force: bool = False) -> bool:
        """Move stop-loss only in the more-profitable direction.

        If force=False, the new price is accepted only when first_in_profit(price, current)
        is True (i.e., the new price is strictly better for the position holder).
        Returns True when the stop was updated, False when silently ignored.

        Args:
            price: Proposed new stop-loss price.
            action_msg: Caller context string (used for logging, not evaluated here).
            force: When True, override the directional guard and always set.

        Returns:
            True if price_stop_loss was updated, False otherwise.
        """
        if force or self.first_in_profit(price, self.price_stop_loss):
            self.price_stop_loss = price
            return True
        return False

    def check_stop_open(self, cur_price: float) -> bool:
        """Return True if price drifted more than 4×fee against the first entry target.

        This signals that an open order is unlikely to fill and should be cancelled.

        Long:  cur_price < price_open[0] * (1 - 4*fee)
        Short: cur_price > price_open[0] * (1 + 4*fee)

        Args:
            cur_price: Current market price.

        Returns:
            True if the order should be cancelled.
        """
        if not self.price_open:
            return False
        entry = self.price_open[0]
        if self.position_type == POSITION_TYPE_LONG:
            return cur_price < entry * (1 - 4 * self.fee)
        else:  # SHORT
            return cur_price > entry * (1 + 4 * self.fee)

    def close_by_time(self, cur_time: float) -> bool:
        """Return True if the position has been open longer than safety_close_time.

        A safety_close_time of 0 means the timeout is disabled (never triggers).

        Args:
            cur_time: Current Unix timestamp.

        Returns:
            True if elapsed time strictly exceeds safety_close_time.
        """
        if self.safety_close_time == 0:
            return False
        return (cur_time - self.open_time) > self.safety_close_time

    def get_action(self) -> str:
        """Return the current action intent.

        Returns:
            One of the STRATEGY_ACTION_* constants.
        """
        return self.action

    def get_action_amount(self, price: float) -> float:
        """Return the coin quantity for the next order.

        The calculation depends on position type and current state:
        - Long  + WAIT_BUY:  coinUse.action_amount / price  (USD → coins)
        - Long  + WAIT_SELL: coinGet.action_amount           (already coins)
        - Short + WAIT_SELL: coinUse.action_amount           (already coins)
        - Short + WAIT_BUY:  coinGet.action_amount / price   (USD → coins)

        WAIT_SAFETY_* states follow the same WAIT_SELL / WAIT_BUY logic.

        Args:
            price: Current market price for USD/coin conversion.

        Returns:
            Order quantity in coin units.
        """
        if self.position_type == POSITION_TYPE_LONG:
            if self.state in (POSITION_STATE_WAIT_BUY,):
                return self.coinUse.action_amount / price
            else:
                # WAIT_SELL, WAIT_SAFETY_SELL
                return self.coinGet.action_amount
        else:  # SHORT
            if self.state in (POSITION_STATE_WAIT_SELL,):
                return self.coinUse.action_amount
            else:
                # WAIT_BUY, WAIT_SAFETY_BUY
                return self.coinGet.action_amount / price

    def get_target(self) -> float:
        """Return the next exit price target.

        If all targets are exhausted (close_idx >= len(price_close)), transitions state
        to POSITION_STATE_WAIT_SAFETY_SELL (long) or POSITION_STATE_WAIT_SAFETY_BUY
        (short) and returns the last target to signal a forced close.

        Returns:
            Next exit price target.
        """
        if self.close_idx < len(self.price_close):
            return self.price_close[self.close_idx]
        # Exhausted — force-close remainder
        if self.position_type == POSITION_TYPE_LONG:
            self.state = POSITION_STATE_WAIT_SAFETY_SELL
        else:
            self.state = POSITION_STATE_WAIT_SAFETY_BUY
        return self.price_close[-1]

    def log_revenue(self, revenue_pct: float, revenue_abs: float) -> None:
        """Print P&L results.

        Args:
            revenue_pct: Percentage return (e.g. 0.048 for 4.8%).
            revenue_abs: Absolute return in USD.
        """
        print(
            f"[BasePosition thread={self.thread_num}] "
            f"type={self.position_type} "
            f"revenue_pct={revenue_pct:.4f} ({revenue_pct*100:.2f}%) "
            f"revenue_abs={revenue_abs:.2f}"
        )

    def finalize(self) -> tuple:
        """Compute P&L from fill history, log it, reset all state, return results.

        P&L formulas (round-trip fees deducted):
          Long:  revenue_pct = (avg_close / avg_open - 1) - 2*fee
          Short: revenue_pct = (1 - avg_close / avg_open) - 2*fee

        revenue_abs = revenue_pct * full_position

        After computation all state is reset to idle:
          price_open, price_close, executed_*, coinUse.reset(), coinGet.reset(),
          state = POSITION_STATE_WAIT, action = STRATEGY_ACTION_NOTHING, close_idx = 0.

        Returns:
            Tuple of (revenue_pct, revenue_abs).
        """
        avg_open = self.avg_price_open()
        avg_close = self.avg_price_close()

        if avg_open != 0.0:
            if self.position_type == POSITION_TYPE_LONG:
                revenue_pct = (avg_close / avg_open - 1) - 2 * self.fee
            else:  # SHORT
                revenue_pct = (1 - avg_close / avg_open) - 2 * self.fee
        else:
            revenue_pct = 0.0

        revenue_abs = revenue_pct * self.full_position

        self.log_revenue(revenue_pct, revenue_abs)

        # Reset all state
        self.price_open = []
        self.price_close = []
        self.executed_open = []
        self.executed_open_amount = []
        self.executed_close = []
        self.executed_close_amount = []
        self.coinUse.reset()
        self.coinGet.reset()
        self.state = POSITION_STATE_WAIT
        self.action = STRATEGY_ACTION_NOTHING
        self.close_idx = 0
        self.open_time = 0.0
        self.price_stop_loss = 0.0

        return (revenue_pct, revenue_abs)

    def to_dict(self) -> dict:
        """Serialize full position state to a JSON-compatible dict.

        Returns:
            Dictionary containing all position state fields including Coin objects.
        """
        return {
            "state": self.state,
            "action": self.action,
            "position_type": self.position_type,
            "fee": self.fee,
            "full_position": self.full_position,
            "risk_per_trade": self.risk_per_trade,
            "thread_num": self.thread_num,
            "price_open": list(self.price_open),
            "price_close": list(self.price_close),
            "price_stop_loss": self.price_stop_loss,
            "executed_open": list(self.executed_open),
            "executed_open_amount": list(self.executed_open_amount),
            "executed_close": list(self.executed_close),
            "executed_close_amount": list(self.executed_close_amount),
            "open_time": self.open_time,
            "safety_close_time": self.safety_close_time,
            "close_idx": self.close_idx,
            "coinUse": self.coinUse.to_dict(),
            "coinGet": self.coinGet.to_dict(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore all position state from a dict (in-place update).

        Args:
            data: Dictionary previously produced by to_dict().
        """
        self.state = data["state"]
        self.action = data["action"]
        self.position_type = data["position_type"]
        self.fee = data["fee"]
        self.full_position = data["full_position"]
        self.risk_per_trade = data["risk_per_trade"]
        self.thread_num = data["thread_num"]
        self.price_open = list(data["price_open"])
        self.price_close = list(data["price_close"])
        self.price_stop_loss = data["price_stop_loss"]
        self.executed_open = list(data["executed_open"])
        self.executed_open_amount = list(data["executed_open_amount"])
        self.executed_close = list(data["executed_close"])
        self.executed_close_amount = list(data["executed_close_amount"])
        self.open_time = data["open_time"]
        self.safety_close_time = data["safety_close_time"]
        self.close_idx = data["close_idx"]
        self.coinUse.from_dict(data["coinUse"])
        self.coinGet.from_dict(data["coinGet"])
