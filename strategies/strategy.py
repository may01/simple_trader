"""Strategy — abstract base class for all trading strategies (Phase 07, Task 01)."""

from abc import ABC, abstractmethod

from constants import (
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
    STRATEGY_ACTION_MOVE_STOP_LOSS_LONG,
    STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT,
    POSITION_STATE_WAIT,
    LEVEL_TYPE_AUTO_RESISTANCE,
    LEVEL_TYPE_SHORT_RESISTANCE,
)
from signals_lib.signal_manager import SignalManager


# Action priority buckets (lower index = higher priority within group)
_OPEN_ACTIONS = {STRATEGY_ACTION_OPEN_LONG, STRATEGY_ACTION_OPEN_SHORT}
_CLOSE_ACTIONS = {
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    STRATEGY_ACTION_CLOSE_LONG_PART,
    STRATEGY_ACTION_CLOSE_SHORT_PART,
}
_MOVE_SL_ACTIONS = {
    STRATEGY_ACTION_MOVE_STOP_LOSS_LONG,
    STRATEGY_ACTION_MOVE_STOP_LOSS_SHORT,
}


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement:
        - ``register_signals()`` — build and register signal chains into
          ``self.signals``
        - ``check_conditions()`` — strategy-level gate that returns True
          when the strategy should run this tick

    The ``check()`` template method orchestrates the full per-tick evaluation:
    build levels → evaluate signals → select action → compute prices → return
    a 5-tuple consumed by StrategyManager.

    Args:
        fee: Trading fee fraction.  No default — must be supplied by the caller.
    """

    def __init__(self, fee: float) -> None:
        self.fee: float = fee
        self.signals: SignalManager = SignalManager()
        self.register_signals()

    # ------------------------------------------------------------------
    # Abstract methods — must be overridden by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def register_signals(self) -> None:
        """Build and register signal chains into ``self.signals``."""

    @abstractmethod
    def check_conditions(self, data_point, position_state: int, action_msg) -> bool:
        """Return True if this strategy should run for the current tick.

        Args:
            data_point: Current market data point.
            position_state: Current position state constant.
            action_msg: Action accumulator (may be None in tests).

        Returns:
            True to proceed with signal evaluation; False to skip entirely.
        """

    # ------------------------------------------------------------------
    # Template method — do not override
    # ------------------------------------------------------------------

    def check(
        self,
        data_point,
        position_state: int,
        cur_time: float,
        action_msg,
    ) -> tuple:
        """Per-tick evaluation template.

        Calls in order:
            1. ``check_conditions()`` — skip entirely if False
            2. ``get_level_values()`` — build levels dict for signals
            3. ``signals.check()`` — evaluate all chains
            4. ``select_final_action()`` — pick single winning action
            5. ``get_open_prices()`` / ``get_close_prices()`` /
               ``get_stop_loss_price()`` — compute price targets

        Args:
            data_point: Current market data point.
            position_state: Current position state constant.
            cur_time: Unix timestamp of current tick.
            action_msg: Action accumulator (may be None in tests).

        Returns:
            5-tuple: ``(action, open_prices, close_prices, stop_price, tf)``
        """
        if not self.check_conditions(data_point, position_state, action_msg):
            open_prices = self.get_open_prices(data_point, 0)
            close_prices = self.get_close_prices(data_point, 0, STRATEGY_ACTION_NOTHING)
            stop_price = self.get_stop_loss_price(data_point, 0, STRATEGY_ACTION_NOTHING)
            return (STRATEGY_ACTION_NOTHING, open_prices, close_prices, stop_price, 0)

        levels = self.get_level_values(data_point)
        actions_list = self.signals.check(data_point, levels, cur_time, action_msg)
        action, tf = self.select_final_action(actions_list, position_state)

        open_prices = self.get_open_prices(data_point, tf)
        close_prices = self.get_close_prices(data_point, tf, action)
        stop_price = self.get_stop_loss_price(data_point, tf, action)

        return (action, open_prices, close_prices, stop_price, tf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_level_values(self, data_point) -> dict:
        """Build levels dict from support/resistance level objects.

        Populates only LEVEL_TYPE_AUTO_RESISTANCE and
        LEVEL_TYPE_SHORT_RESISTANCE for now (long support/resistance entries
        are left commented out per implementation spec).

        Returns empty lists for all keys when the data environment (env vars
        or levels.txt file) is not available — this occurs in unit-test
        contexts where no Docker volume is mounted.

        Args:
            data_point: Current market data point.

        Returns:
            dict mapping level type constants to lists of price values.
        """
        from levels import Levels

        _empty: dict = {
            LEVEL_TYPE_AUTO_RESISTANCE: [],
            LEVEL_TYPE_SHORT_RESISTANCE: [],
        }

        try:
            # Load levels fresh every tick — not cached between ticks.
            levels_obj = Levels()
        except (KeyError, FileNotFoundError, OSError):
            # Environment variables not set (e.g. unit tests) or file missing.
            return _empty

        cur_time: float = data_point.timestamp.timestamp()

        levels: dict = {
            LEVEL_TYPE_AUTO_RESISTANCE: [
                Levels.get_level_value(lvl, cur_time)
                for lvl in levels_obj.get_active_levels(LEVEL_TYPE_AUTO_RESISTANCE, cur_time)
            ],
            LEVEL_TYPE_SHORT_RESISTANCE: [
                Levels.get_level_value(lvl, cur_time)
                for lvl in levels_obj.get_active_levels(LEVEL_TYPE_SHORT_RESISTANCE, cur_time)
            ],
            # LEVEL_TYPE_LONG_SUPPORT and LEVEL_TYPE_LONG_RESISTANCE
            # are commented out in current implementation.
        }
        return levels

    def select_final_action(
        self, actions_list: list, position_state: int
    ) -> tuple:
        """Pick a single action from the list returned by signals.check().

        Priority rules:
        - Empty list → (STRATEGY_ACTION_NOTHING, 0)
        - Single action → use it directly
        - Multiple, position_state == POSITION_STATE_WAIT → prefer OPEN_LONG /
          OPEN_SHORT; fall back to first action
        - Multiple, position open → prefer CLOSE_* first, then MOVE_STOP_LOSS_*;
          fall back to first action

        Args:
            actions_list: List of action tuples from SignalManager.check();
                each tuple is ``[action_str, tf, price, data_dict]``.
            position_state: Current position state constant.

        Returns:
            ``(final_action, tf)`` where tf is the timeframe of the winning
            action, or ``(STRATEGY_ACTION_NOTHING, 0)`` when the list is empty.
        """
        if not actions_list:
            return (STRATEGY_ACTION_NOTHING, 0)

        if len(actions_list) == 1:
            entry = actions_list[0]
            return (entry[0], entry[1])

        # Multiple actions — apply priority rules.
        if position_state == POSITION_STATE_WAIT:
            # Prefer OPEN actions.
            for entry in actions_list:
                if entry[0] in _OPEN_ACTIONS:
                    return (entry[0], entry[1])
        else:
            # Position is open — prefer CLOSE over MOVE_STOP_LOSS.
            for entry in actions_list:
                if entry[0] in _CLOSE_ACTIONS:
                    return (entry[0], entry[1])
            for entry in actions_list:
                if entry[0] in _MOVE_SL_ACTIONS:
                    return (entry[0], entry[1])

        # Fallback: first action in the list.
        entry = actions_list[0]
        return (entry[0], entry[1])

    # ------------------------------------------------------------------
    # Overridable price methods — defaults provided
    # ------------------------------------------------------------------

    def get_open_prices(self, data_point, tf: int) -> list:
        """Return entry price targets.

        Default: single entry at the current 1-min close.

        Args:
            data_point: Current market data point.
            tf: Timeframe of the fired chain (unused in default implementation).

        Returns:
            List of entry prices.
        """
        return [data_point.get("close", 1, 0)]

    def get_close_prices(self, data_point, tf: int, action: int) -> list:
        """Return exit price targets.

        Default: +0.8% for OPEN_LONG, -0.8% for OPEN_SHORT (and NOTHING).

        Args:
            data_point: Current market data point.
            tf: Timeframe of the fired chain.
            action: The selected strategy action.

        Returns:
            List of exit prices.
        """
        open_price = self.get_open_prices(data_point, tf)[0]
        if action == STRATEGY_ACTION_OPEN_SHORT:
            return [open_price * (1 - 0.008)]
        return [open_price * (1 + 0.008)]

    def get_stop_loss_price(self, data_point, tf: int, action: int) -> float:
        """Return stop-loss price.

        Default:
            - OPEN_LONG: sar(tf) - 0.3 × atr_14(tf)
            - OPEN_SHORT: sar(tf) + 0.3 × atr_14(tf)
            - other: sar(tf) - 0.3 × atr_14(tf)

        Falls back to tf=1 when tf is 0.

        Args:
            data_point: Current market data point.
            tf: Timeframe of the fired chain.
            action: The selected strategy action.

        Returns:
            Stop-loss price as float.
        """
        effective_tf = tf if tf > 0 else 1
        sar = data_point.get("sar", effective_tf, 0)
        atr = data_point.get("atr_14", effective_tf, 0)
        if action == STRATEGY_ACTION_OPEN_SHORT:
            return float(sar + 0.3 * atr)
        return float(sar - 0.3 * atr)
