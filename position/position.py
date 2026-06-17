"""Position — public-facing facade wrapping LongPosition / ShortPosition."""

from constants import (
    POSITION_STATE_WAIT,
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
)
from constants import POSITION_TYPE_UNKNOWN
from position.base_position import BasePosition
from position.long_position import LongPosition
from position.short_position import ShortPosition


class Position:
    """Facade over BasePosition implementations.

    Creates LongPosition or ShortPosition on demand based on the strategy action
    received in open(). All lifecycle calls are delegated to posImpl; when no
    position is open (posImpl is None) every method returns a sensible default.

    Args:
        thread_num: Thread identifier forwarded to the concrete implementation.
        fee: Trading fee fraction forwarded to the concrete implementation.
    """

    def __init__(self, thread_num: int = 0, fee: float = 0.0):
        self.posImpl: BasePosition | None = None
        self.fee: float = fee
        self.thread_num: int = thread_num
        self.full_position: float = 10000.0
        # Facade-level change buffer (Phase 14, Task 03): survives posImpl being
        # discarded at finalize so the robot can drain the close/settle events.
        self._changes: list = []

    @property
    def position_type(self) -> str:
        """Current position type, or UNKNOWN when no position is open."""
        if self.posImpl is None:
            return POSITION_TYPE_UNKNOWN
        return self.posImpl.position_type

    @property
    def open_time(self) -> float:
        """Open timestamp of the active position, or 0.0 when flat."""
        if self.posImpl is None:
            return 0.0
        return self.posImpl.open_time

    @open_time.setter
    def open_time(self, value: float) -> None:
        """Override the active position's open timestamp (sim clock)."""
        if self.posImpl is not None:
            self.posImpl.open_time = value

    def _pull_changes(self) -> None:
        """Move any pending changes off posImpl into the facade buffer."""
        if self.posImpl is not None:
            self._changes.extend(self.posImpl.drain_changes())

    def drain_changes(self) -> list:
        """Return accumulated lifecycle changes and clear the buffer."""
        self._pull_changes()
        changes = self._changes
        self._changes = []
        return changes

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open(
        self,
        strategy_action: int,
        price_open: list,
        price_close: list,
        price_stop_loss: float,
        time_period: int,
        action_msg,
    ) -> bool:
        """Create and open a long or short position.

        Returns False if a position is already open or the action is unknown.
        """
        if self.posImpl is not None:
            return False

        if strategy_action == STRATEGY_ACTION_OPEN_LONG:
            impl = LongPosition(self.fee, self.thread_num, self.full_position)
        elif strategy_action == STRATEGY_ACTION_OPEN_SHORT:
            impl = ShortPosition(self.fee, self.thread_num, self.full_position)
        else:
            return False

        opened = impl.open(
            strategy_action, price_open, price_close, price_stop_loss,
            time_period, action_msg,
        )
        if not opened:
            # Inner open rejected (e.g. risk <= 0). Keep the facade flat so
            # is_opened() does not lie, but salvage any recorded change.
            self._changes.extend(impl.drain_changes())
            return False
        self.posImpl = impl
        return True

    def close(
        self,
        strategy_action: int,
        price_close: list,
        price_stop_loss: float,
        time_period: int,
        action_msg,
    ) -> None:
        """Delegate close to the active position; no-op if none."""
        if self.posImpl is None:
            return
        self.posImpl.close(strategy_action, price_close, price_stop_loss, time_period, action_msg)

    def set_stop_loss(self, price: float, action_msg, force: bool = False) -> bool:
        """Delegate stop-loss update; returns False if no position is open."""
        if self.posImpl is None:
            return False
        return self.posImpl.set_stop_loss(price, action_msg, force)

    def record_entry_fill(self, coin_amount: float, price: float) -> None:
        """Delegate entry fill recording; no-op if no position is open."""
        if self.posImpl is None:
            return
        self.posImpl.record_entry_fill(coin_amount, price)

    def record_exit_fill(self, coin_amount: float, price: float) -> None:
        """Delegate exit fill recording; no-op if no position is open."""
        if self.posImpl is None:
            return
        self.posImpl.record_exit_fill(coin_amount, price)

    def finalize(self) -> tuple:
        """Finalize the active position, clear posImpl, and return P&L.

        Returns (0.0, 0.0) if no position is open.
        """
        if self.posImpl is None:
            return (0.0, 0.0)
        result = self.posImpl.finalize()
        self._pull_changes()  # capture close/settle before discarding posImpl
        self.posImpl = None
        return result

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def get_action(self) -> str:
        """Return current action, or STRATEGY_ACTION_NOTHING if no position."""
        if self.posImpl is None:
            return STRATEGY_ACTION_NOTHING
        return self.posImpl.get_action()

    def get_state(self) -> str:
        """Return current state, or POSITION_STATE_WAIT if no position."""
        if self.posImpl is None:
            return POSITION_STATE_WAIT
        return self.posImpl.state

    def get_target(self) -> float:
        """Return next exit target, or 0.0 if no position."""
        if self.posImpl is None:
            return 0.0
        return self.posImpl.get_target()

    def is_opened(self) -> bool:
        """Return True when a position is currently open."""
        return self.posImpl is not None

    def is_stop_loss_triggered(self, cur_price: float) -> bool:
        """Return True if stop-loss has been crossed; False if no position."""
        if self.posImpl is None:
            return False
        return self.posImpl.is_stop_loss_triggered(cur_price)

    def is_target_reached(self, cur_price: float) -> bool:
        """Return True if the take-profit target has been reached; False if flat."""
        if self.posImpl is None:
            return False
        return self.posImpl.is_target_reached(cur_price)

    def current_target(self) -> float:
        """Return the active take-profit target, or 0.0 when flat."""
        if self.posImpl is None:
            return 0.0
        return self.posImpl.current_target()

    def check_stop_open(self, cur_price: float) -> bool:
        """Return True if entry order is stale; False if no position."""
        if self.posImpl is None:
            return False
        return self.posImpl.check_stop_open(cur_price)

    def close_by_time(self, cur_time: float) -> bool:
        """Return True if position has timed out; False if no position."""
        if self.posImpl is None:
            return False
        return self.posImpl.close_by_time(cur_time)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize current position state; returns {} if no position."""
        if self.posImpl is None:
            return {}
        return self.posImpl.to_dict()

    def from_dict(self, data: dict) -> None:
        """Reconstruct posImpl from a previously serialized dict.

        Reads data["position_type"] to determine whether to create a
        LongPosition or ShortPosition, then restores all state in-place.
        After restoring, syncs the facade's own fields from posImpl to ensure
        consistency for subsequent finalize() / open() calls.
        """
        position_type = data.get("position_type")
        if position_type == POSITION_TYPE_LONG:
            self.posImpl = LongPosition(self.fee, self.thread_num, self.full_position)
        elif position_type == POSITION_TYPE_SHORT:
            self.posImpl = ShortPosition(self.fee, self.thread_num, self.full_position)
        else:
            return
        self.posImpl.from_dict(data)
        # Sync facade fields from the restored posImpl
        self.fee = self.posImpl.fee
        self.full_position = self.posImpl.full_position
        self.thread_num = self.posImpl.thread_num
