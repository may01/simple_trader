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
    POSITION_TYPE_LONG,
)
from logs import log_error
from position.position import Position
from backtesting.action import (
    Action,
    ActionLog,
    EVENT_SIGNAL_FIRED,
)

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
        # Action capture (Phase 14, Task 04)
        self.sim_id: int = 0
        self.action_log: ActionLog = ActionLog(0)
        self.tick_index: int = 0

    def set_sim_context(self, sim_id: int) -> None:
        """Bind this robot's action log to a simulation id; reset tick counter.

        Args:
            sim_id: Owning simulation id (shared across all workers of a run).
        """
        self.sim_id = sim_id
        self.action_log = ActionLog(sim_id)
        self.tick_index = 0

    def get_action_log(self) -> ActionLog:
        """Return the accumulated ActionLog for this robot."""
        return self.action_log

    # ------------------------------------------------------------------
    # Public tick entry point
    # ------------------------------------------------------------------

    def step(self, data_point) -> None:
        """Main tick: call _do() and catch any exception without crashing.

        Args:
            data_point: Current market data point (NOT stored as attribute).
        """
        self.tick_index += 1
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
            # Signal-driven close executes at the current market price, not the
            # strategy's +/-0.8% target (which is an OPEN-time take-profit).
            mkt = data_point.cur_price('close')
            self.sell(data_point, strategy_action, [mkt], stop_price, tf)
        else:
            self.wait(data_point)

        # Action capture (Phase 14, Task 04): record signal firings (pre-resolution)
        # and the position changes they produced this tick.
        self._capture_actions(data_point, open_prices, close_prices, stop_price)

    def _capture_actions(self, data_point, open_prices, close_prices, stop_price) -> None:
        """Assemble Action records from pre-resolution firings + position changes.

        Args:
            data_point: Current market data point (not stored).
            open_prices: Resolved entry targets the strategy provided this tick.
            close_prices: Resolved exit targets the strategy provided this tick.
            stop_price: Resolved stop-loss price the strategy provided this tick.
        """
        # WideDataPoint.timestamp is a pd.Timestamp (.timestamp() → epoch s);
        # test stubs may pass a plain float. Accept both.
        ts_attr = data_point.timestamp
        ts = ts_attr.timestamp() if hasattr(ts_attr, "timestamp") else float(ts_attr)

        # SIGNAL_FIRED — one per strategy firing before conflict resolution.
        fired = getattr(self.strategy_manager, "last_fired", []) or []
        target = open_prices[0] if open_prices else (close_prices[0] if close_prices else 0.0)
        for action_type in fired:
            self.action_log.record(Action(
                sim_id=self.sim_id, timestamp=ts, tick_index=self.tick_index,
                event=EVENT_SIGNAL_FIRED, action_type=action_type,
                position_type=self.position.position_type, was_stop_loss=False,
                target_price=float(target), executed_price=0.0,
                stop_loss_price=float(stop_price), revenue_pct=0.0, revenue_abs=0.0,
            ))

        # Position lifecycle changes — kind maps 1:1 to event.
        for ch in self.position.drain_changes():
            executed = ch["executed_price"] or ch["target_price"]
            self.action_log.record(Action(
                sim_id=self.sim_id, timestamp=ts, tick_index=self.tick_index,
                event=ch["kind"], action_type=ch["kind"],
                position_type=ch["position_type"], was_stop_loss=ch["was_stop_loss"],
                target_price=ch["target_price"], executed_price=float(executed),
                stop_loss_price=ch["stop_loss_price"],
                revenue_pct=ch["revenue_pct"], revenue_abs=ch["revenue_abs"],
            ))

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
            # Stamp open_time with the simulation clock, not wall-clock time —
            # close_by_time() compares against the data timestamp.
            ts_attr = data_point.timestamp
            self.position.open_time = (
                ts_attr.timestamp() if hasattr(ts_attr, "timestamp") else float(ts_attr)
            )

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
        # No open position → nothing to close. Guard against phantom finalize
        # (which would append a (0,0) trade and inflate the trade count).
        if not self.position.is_opened():
            return
        self.position.close(strategy_action, close_prices, stop_price, tf, None)
        self.position.record_exit_fill(self._current_coin_amount, close_prices[0])
        self._finalize()

    def wait(self, data_point) -> None:
        """Check passive exit conditions (stop-loss, timeout) and act if triggered.

        Args:
            data_point: Current market data point (not stored).
        """
        cur_price = data_point.cur_price('close')
        ts_attr = data_point.timestamp
        cur_time = ts_attr.timestamp() if hasattr(ts_attr, "timestamp") else float(ts_attr)

        if self.position.is_stop_loss_triggered(cur_price):
            self.sell(data_point, STRATEGY_ACTION_DO_STOP_LOSS, [cur_price], 0.0, 0)
            return

        # Take-profit: close when price reaches the target, filling at the target.
        if self.position.is_target_reached(cur_price):
            target = self.position.current_target()
            close_action = (
                STRATEGY_ACTION_CLOSE_LONG
                if self.position.position_type == POSITION_TYPE_LONG
                else STRATEGY_ACTION_CLOSE_SHORT
            )
            self.sell(data_point, close_action, [target], 0.0, 0)
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
