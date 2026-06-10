"""StrategyManager — aggregates strategy actions with conflict resolution (Phase 07, Task 03).

Orchestrates multiple registered strategies, collects their per-tick outputs,
applies priority-based conflict resolution, and returns a single resolved action
to Robot.
"""

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
    STRATEGY_ACTION_DO_STOP_LOSS,
)


# Action priority buckets
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


class StrategyManager:
    """Aggregates multiple strategies with priority-based conflict resolution.

    Robot calls ``check()`` once per tick. Each registered strategy's ``check()``
    fires; results are collected and conflict-resolved. Priority order:
    DO_STOP_LOSS > CLOSE > MOVE_STOP_LOSS > OPEN.

    Multiple competing OPEN signals from different strategies → NOTHING.

    Args:
        fee: Trading fee fraction.
    """

    def __init__(self, fee: float) -> None:
        """Initialize StrategyManager.

        Args:
            fee: Trading fee fraction.
        """
        self.fee: float = fee
        self.strategies: list = []

    def register(self, strategy) -> None:
        """Register a strategy instance.

        Args:
            strategy: Strategy instance to register and evaluate on each tick.
        """
        self.strategies.append(strategy)

    def check(
        self,
        data_point,
        position_state: int,
        cur_time: float,
        action_msg,
    ) -> tuple:
        """Aggregate and resolve actions from all registered strategies.

        Calls ``strategy.check()`` for each registered strategy in order,
        collects non-NOTHING results, applies conflict resolution, and returns
        the winning action tuple.

        Args:
            data_point: Current market data point.
            position_state: Current position state constant.
            cur_time: Unix timestamp of current tick.
            action_msg: Action accumulator (may be None in tests).

        Returns:
            5-tuple: ``(action, open_prices, close_prices, stop_price, tf)``
            where action is the resolved winning action, or NOTHING if no
            consensus or empty strategies list.
        """
        if not self.strategies:
            return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)

        # Collect all non-NOTHING results from strategies
        results = []
        for strategy in self.strategies:
            action, open_prices, close_prices, stop_price, tf = strategy.check(
                data_point, position_state, cur_time, action_msg
            )
            if action != STRATEGY_ACTION_NOTHING:
                results.append((action, open_prices, close_prices, stop_price, tf))

        # Resolve conflicts
        return self._resolve(results)

    def _resolve(self, results: list) -> tuple:
        """Apply conflict resolution to collected strategy results.

        Priority rules (highest to lowest):
        1. DO_STOP_LOSS (return immediately if found)
        2. CLOSE_* (return first found)
        3. MOVE_STOP_LOSS_* (return first found)
        4. OPEN_* (return if single; return NOTHING if multiple)

        Args:
            results: List of non-NOTHING result tuples from strategies.

        Returns:
            5-tuple: ``(action, open_prices, close_prices, stop_price, tf)``
        """
        if not results:
            return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)

        # Priority 1: DO_STOP_LOSS
        for result in results:
            action, open_prices, close_prices, stop_price, tf = result
            if action == STRATEGY_ACTION_DO_STOP_LOSS:
                return result

        # Priority 2: CLOSE actions
        for result in results:
            action, open_prices, close_prices, stop_price, tf = result
            if action in _CLOSE_ACTIONS:
                return result

        # Priority 3: MOVE_STOP_LOSS actions
        for result in results:
            action, open_prices, close_prices, stop_price, tf = result
            if action in _MOVE_SL_ACTIONS:
                return result

        # Priority 4: OPEN actions (only if single; multiple = conflict)
        open_results = [r for r in results if r[0] in _OPEN_ACTIONS]
        if len(open_results) == 1:
            return open_results[0]
        elif len(open_results) > 1:
            # Multiple OPEN signals from different strategies = conflict → NOTHING
            return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)

        # No action matched (should not reach here if results non-empty)
        return (STRATEGY_ACTION_NOTHING, [], [], 0.0, 0)
