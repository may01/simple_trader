"""robots/robot.py — Live trading robot with polling loop and crash recovery (Phase 10, Task 02).

Drives live trading: polls live_data for the current DataPoint each second,
asks StrategyManager what to do, and dispatches to order management methods
(implemented in Task 03).
"""

import logging
import math
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
from backtesting.action import Action
from config_loader import load_shared_indicators_config
from position.position import Position
from robots.live_action_log import LIVE_SIM_ID, LiveActionLog
from robots.live_order_tracker import LiveOrderTracker
from stocks.base_stock import StockInterface
from strategies.strategy_manager import StrategyManager

try:
    from mq.indicator_publisher import IndicatorPublisher
except ImportError:  # pragma: no cover - exercised via sys.modules patching
    # Telemetry must never be a startup dependency of trading. The deployed
    # `live` image does not necessarily carry pyzmq, and without this guard
    # a missing optional dependency would abort `python3 trader.py` with an
    # ImportError before any trading logic ran. With it, a Robot simply gets
    # indicator_publisher=None and every publish step is a no-op.
    IndicatorPublisher = None

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
        indicator_publisher: Optional broadcaster for the allowlisted
            indicator readings (level-broadcast-plan Task 9). None (default)
            makes the publish step in do() a complete no-op, so pre-existing
            callers/tests are unaffected.
        indicator_publish_interval_sec: Minimum seconds between indicator
            broadcasts, independent of the 1s tick. 0 publishes every tick.
            Default 30 -- a 10x margin under the publisher's 300s TTL.
    """

    def __init__(
        self,
        strategy_manager: StrategyManager,
        live_data,
        stock: StockInterface,
        fee: float,
        persist_path: str,
        action_log_path: str | None = None,
        indicator_publisher: "IndicatorPublisher | None" = None,
        indicator_publish_interval_sec: float = 30.0,
    ) -> None:
        self.strategy_manager: StrategyManager = strategy_manager
        self.live_data = live_data
        self.stock: StockInterface = stock
        self.fee: float = fee
        self.running: bool = False

        # Optional indicator broadcast to trade_executor (level-broadcast-plan
        # Task 9). None -> inert; do() skips the publish block entirely, and
        # construction itself skips the allowlist load, so a Robot built
        # without indicator_publisher takes no dependency on
        # configs/shared_indicators_config.yaml existing at a CWD-relative
        # path (deviation from the brief's Step 3, per reviewer ruling).
        self.indicator_publisher = indicator_publisher
        self._shared_indicators = (
            load_shared_indicators_config() if indicator_publisher is not None else []
        )
        # Publish cadence is decoupled from the tick (do() runs once a
        # second). Each reading carries a TTL of ttl_seconds (300s by
        # default), so republishing every second rewrote the same row ~300
        # times before it could ever expire -- ~780k rows/day into an
        # append-only table with no retention. 30s keeps a 10x margin under
        # the TTL while cutting the write volume by 30x. Monotonic clock:
        # an NTP step must not freeze or spam the heartbeat.
        self._indicator_publish_interval_sec: float = indicator_publish_interval_sec
        self._last_indicator_publish_at: float | None = None
        # (kind, name, tf) keys already logged about, so a permanently
        # broken allowlist entry produces one log line rather than one per
        # tick per entry (9 tracebacks/second for a single typo).
        self._warned_indicator_keys: set = set()

        # Created internally
        self.position: Position = Position(thread_num=0, fee=fee)
        self.tracker: LiveOrderTracker = LiveOrderTracker(
            position=self.position,
            stock=stock,
            persist_path=persist_path,
        )
        # Optional live Action persistence (Phase 15 Task 03). None → inert.
        self._action_log: LiveActionLog | None = (
            LiveActionLog(action_log_path) if action_log_path else None
        )
        self._tick_index: int = 0

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
        """Execute one tick: refresh candles, query strategy, dispatch action.

        LiveData contract (phase-03 task-05): build_candles() refreshes from
        the exchange, get_data_point() returns the resulting LiveDataPoint.
        """
        self.live_data.build_candles()
        data_point = self.live_data.get_data_point()

        self._publish_shared_indicators(data_point)

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

        self._record_live_actions(data_point)
        self._tick_index += 1

    def _warn_once(self, key, message, *args, exc_info: bool = False) -> None:
        """Log `message` the first time `key` is seen, then never again.

        _publish_shared_indicators runs inside the per-tick loop, so an
        unconditional log there is a tick-rate log stream: one typo'd
        allowlist entry produced nine tracebacks a second, forever. These
        conditions are static (a name that does not exist never starts
        existing mid-run), so the first occurrence carries all the
        information the operator needs.
        """
        if key in self._warned_indicator_keys:
            return
        self._warned_indicator_keys.add(key)
        if exc_info:
            logger.exception(message, *args)
        else:
            logger.warning(message, *args)

    def _publish_shared_indicators(self, data_point) -> None:
        """Broadcast every allowlisted (name, tf) reading as a heartbeat.

        No-op when no indicator_publisher was configured. This is a
        heartbeat, not a change notification -- the executor ages readings
        out via each message's own TTL -- but it is throttled to
        `indicator_publish_interval_sec` rather than fired every tick: the
        TTL is 300s and do() runs every second, so an unthrottled heartbeat
        rewrote each reading ~300 times before it could expire.

        IndicatorPublisher.publish() itself never blocks and never raises
        (Task 8), but the read that feeds it, data_point.get(cfg.name, tf),
        can raise (e.g. KeyError when tf isn't in the live ohlc mapping, or
        the column doesn't exist yet because the indicator hasn't warmed
        up) -- that must never abort do() before the trading logic below it
        runs, so each read is individually guarded. A successfully-read but
        non-finite value (NaN/inf, e.g. still warming up) is skipped rather
        than published, since the executor's wire format cannot decode one
        anyway. Both of those are logged once per (name, tf).
        """
        if self.indicator_publisher is None:
            return
        now = time.monotonic()
        if (
            self._last_indicator_publish_at is not None
            and now - self._last_indicator_publish_at < self._indicator_publish_interval_sec
        ):
            return
        self._last_indicator_publish_at = now
        pair = self.stock.get_pair_name()
        for cfg in self._shared_indicators:
            for tf in cfg.timeframes:
                try:
                    value = data_point.get(cfg.name, tf)
                except Exception:
                    self._warn_once(
                        ("read_failed", cfg.name, tf),
                        "indicator publish: data_point.get(%s, %s) failed; "
                        "skipping this reading from now on (logged once per name/tf)",
                        cfg.name, tf,
                        exc_info=True,
                    )
                    continue
                if not math.isfinite(value):
                    # NaN/inf is the warm-up-not-ready case (data.py returns
                    # NaN rather than raising when there isn't enough
                    # history yet). The executor's wire format can't decode
                    # a non-finite number at all -- send_json would emit the
                    # bare token NaN, which isn't valid JSON, so the message
                    # is guaranteed to be dropped undecoded on the other
                    # side. Skip it instead of sending something that can
                    # never be delivered -- but say so once: an indicator
                    # that never warms up would otherwise be invisible
                    # forever, indistinguishable from one nobody configured.
                    self._warn_once(
                        ("non_finite", cfg.name, tf),
                        "indicator publish: %s on tf=%s is non-finite (%s); not publishing "
                        "until it warms up (logged once per name/tf)",
                        cfg.name, tf, value,
                    )
                    continue
                self.indicator_publisher.publish(pair=pair, name=f"{tf}_{cfg.name}", value=value)

    def _record_live_actions(self, data_point) -> None:
        """Persist any position lifecycle changes from this tick as Action records.

        No-op when no action log is configured. Drains the position's change
        buffer (OPEN / CLOSE / STOP_LOSS events produced by open / close /
        finalize) and appends one Action per change to the live store.
        """
        if self._action_log is None:
            return
        ts_attr = data_point.timestamp
        ts = ts_attr.timestamp() if hasattr(ts_attr, "timestamp") else float(ts_attr)
        for ch in self.position.drain_changes():
            executed = ch["executed_price"] or ch["target_price"]
            self._action_log.record(Action(
                sim_id=LIVE_SIM_ID,
                timestamp=ts,
                tick_index=self._tick_index,
                event=ch["kind"],
                action_type=ch["kind"],
                position_type=ch["position_type"],
                was_stop_loss=ch["was_stop_loss"],
                target_price=ch["target_price"],
                executed_price=float(executed),
                stop_loss_price=ch["stop_loss_price"],
                revenue_pct=ch["revenue_pct"],
                revenue_abs=ch["revenue_abs"],
            ))

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
