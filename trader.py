"""trader.py — live trading entry point (Docker path E).

Paper mode is the code DEFAULT (STOCK_TYPE=mock_binance: a frozen candle
fixture, so indicators never change). STOCK_TYPE=binance_candles reads real
Binance market data with no API keys and refuses every order/account call --
what configs/live.env uses. Real trading requires STOCK_TYPE=binance set
explicitly in the env file.

Live strategy/size knobs (Phase 15):
- STRATEGY_SET=ema registers the EMA test strategies via EmaStrategyFactory;
  unset/other keeps the inert default (empty StrategyManager).
- LIVE_POSITION_USDT sets the per-trade notional (Position.full_position),
  hard-clamped to <= 50 as the money-safety rail.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Hard cap on live per-trade notional. Safety rail, not a suggestion.
LIVE_POSITION_USDT_CAP = 50.0
DEFAULT_LIVE_POSITION_USDT = 40.0


def build_strategy_manager(stock, strategy_set: str):
    """Build the live StrategyManager for the given STRATEGY_SET value.

    ``"ema"`` returns the EMA test strategies via EmaStrategyFactory (2
    strategies). Any other value (including "") returns an inert
    StrategyManager with nothing registered, preserving the default
    behaviour from before Phase 15.

    Args:
        stock: Initialised stock; only ``stock.fee`` is used.
        strategy_set: Value of the STRATEGY_SET env var.
    """
    if strategy_set == "ema":
        from strategies.test_factory import EmaStrategyFactory

        return EmaStrategyFactory(stock.fee)()

    from strategies.strategy_manager import StrategyManager

    return StrategyManager(stock.fee)


def resolve_live_usdt() -> float:
    """Resolve LIVE_POSITION_USDT, defaulting to 40 and clamping to <= 50.

    The clamp is the phase's money-safety guard: an over-cap request is
    logged and reduced to the cap rather than honoured.
    """
    live_usdt = float(os.environ.get("LIVE_POSITION_USDT", str(DEFAULT_LIVE_POSITION_USDT)))
    if live_usdt > LIVE_POSITION_USDT_CAP:
        logger.warning(
            "LIVE_POSITION_USDT=%s exceeds cap %s; clamping to cap.",
            live_usdt,
            LIVE_POSITION_USDT_CAP,
        )
        live_usdt = LIVE_POSITION_USDT_CAP
    return live_usdt


def select_stock(inner, mode: str | None):
    """Return the stock `Robot` should trade through.

    trade_executor owns the live position (position-management design, D2),
    so main/ places no orders by default. The refusal happens at the
    StockInterface boundary in `DisarmedStock`, not as conditionals at each
    of robot.py's five order call sites — the same shape as the executor's
    own NoTradeAccount, and for the same reason: it must not depend on
    API-key permissions or on anything outside this process.

    Args:
        inner: The real, initialised StockInterface.
        mode: `MAIN_ORDER_PLACEMENT`. Unset, "disabled", or anything
            unrecognised disarms. Only the exact string "enabled" arms.

    Returns:
        Either `inner` or a `DisarmedStock` wrapping it.
    """
    from stocks.disarmed_stock import DisarmedStock

    normalised = (mode or "disabled").strip().lower()
    if normalised == "enabled":
        logger.warning(
            "MAIN_ORDER_PLACEMENT=enabled -- main/ WILL place real orders. "
            "Running this against a trade_executor with EXECUTION_MODE=live "
            "double-trades the same account: both processes will open and "
            "close positions independently, and neither knows about the other."
        )
        return inner
    if normalised != "disabled":
        # A typo must not arm anything. Disarm and say why, loudly, rather
        # than falling through to the safe branch in silence.
        logger.warning(
            "MAIN_ORDER_PLACEMENT=%r is not recognised (expected 'enabled' or "
            "'disabled'); order placement stays DISABLED.",
            mode,
        )
    logger.info(
        "main/ order placement is DISABLED: no order, loan or cancellation "
        "will reach the exchange from this process. trade_executor places "
        "orders instead. Set MAIN_ORDER_PLACEMENT=enabled to re-arm the "
        "legacy path (see stocks/disarmed_stock.py)."
    )
    return DisarmedStock(inner)


def main() -> None:
    from stocks_holder import do_stock_init, stock_holder

    stock_type = os.environ.get("STOCK_TYPE", "mock_binance")
    do_stock_init(stock_type)
    stock = select_stock(stock_holder.item, os.environ.get("MAIN_ORDER_PLACEMENT"))
    # `stock_holder.item` stays the raw stock: anything reaching for the
    # holder directly (LiveData's candle reads) is a read path and must not
    # be routed through the disarm wrapper's delegation for no reason.

    from data import LiveData
    from helpers import shared_folder
    from mq.indicator_publisher import IndicatorPublisher
    from mq.position_consumer import PositionConsumer
    from robots.robot import Robot

    strategy_set = os.environ.get("STRATEGY_SET", "")
    live_data = LiveData()
    strategy_manager = build_strategy_manager(stock, strategy_set)

    # trade_executor lives in a separate repo/deployment. Both compose
    # projects attach to the external `trader_mq` docker network, so the
    # executor is addressed by its compose service name. Not via the host:
    # trade_executor publishes its MQ port on 127.0.0.1 only, which this
    # container cannot reach through host-gateway -- and since the publisher
    # never raises, that failure would be silent.
    mq_executor_addr = os.environ.get("MQ_EXECUTOR_ADDR", "tcp://executor:5555")
    # Publish cadence, decoupled from the 1s tick: the readings carry a 300s
    # TTL, so republishing every second wrote each one ~300 times over before
    # it could expire. Same env-driven shape as MQ_EXECUTOR_ADDR above.
    try:
        mq_publish_interval = float(os.environ.get("MQ_INDICATOR_PUBLISH_INTERVAL_SEC", "30"))
    except ValueError:
        mq_publish_interval = 30.0
    # Constructing the publisher is the one step that genuinely needs pyzmq
    # (the module itself imports fine without it), and the deployed `live`
    # image does not carry pyzmq. Unguarded, that ImportError would land
    # here -- past the imports, before Robot is built and before
    # run_instantly() -- i.e. a missing telemetry dependency would stop
    # trading, the exact failure the guard in robots/robot.py exists to
    # prevent. Degrade to no telemetry instead: Robot treats
    # indicator_publisher=None as a complete no-op.
    try:
        indicator_publisher = IndicatorPublisher(connect_addr=mq_executor_addr, ttl_seconds=300.0)
    except ImportError:
        indicator_publisher = None
        logger.warning(
            "indicator broadcast DISABLED: pyzmq is not installed in this image, so "
            "IndicatorPublisher could not be constructed. Trading continues normally; "
            "no indicator readings will be published to trade_executor at %s. "
            "Install pyzmq to re-enable telemetry.",
            mq_executor_addr,
        )

    # Position feed from trade_executor (position-management design §6.4).
    # Same degrade-don't-crash rule as the publisher above and for the same
    # reason: this is telemetry, and a missing pyzmq must not stop trading.
    #
    # The executor binds two sockets -- one it PULLs main/'s messages from,
    # one it PUSHes position state onto. `MQ_EXECUTOR_OUTBOUND_ADDR` is the
    # second; `mq_executor_addr` above is the first, and the snapshot
    # request goes back out on it.
    mq_executor_outbound_addr = os.environ.get(
        "MQ_EXECUTOR_OUTBOUND_ADDR", "tcp://executor:5556"
    )
    try:
        position_consumer = PositionConsumer(connect_addr=mq_executor_outbound_addr)
        position_consumer.connect_request_socket(mq_executor_addr)
        # A freshly started main/ has seen no changes yet, and the outbound
        # topic carries only changes -- without this it could wait hours to
        # learn about a position that is open right now.
        position_consumer.request_snapshot()
    except ImportError:
        position_consumer = None
        logger.warning(
            "position feed DISABLED: pyzmq is not installed in this image, so "
            "PositionConsumer could not be constructed. Trading continues normally; "
            "main/ will not see what trade_executor holds at %s, and its own "
            "Position will reflect intent only.",
            mq_executor_outbound_addr,
        )

    os.makedirs(shared_folder(), exist_ok=True)
    robot = Robot(
        strategy_manager,
        live_data,
        stock,
        stock.fee,
        persist_path=shared_folder() + "live_tracker.json",
        action_log_path=shared_folder() + "live_actions.jsonl",
        indicator_publisher=indicator_publisher,
        indicator_publish_interval_sec=mq_publish_interval,
        position_consumer=position_consumer,
    )
    robot.position.full_position = resolve_live_usdt()
    robot.run_instantly()


if __name__ == "__main__":
    main()
