"""trader.py — live trading entry point (Docker path E).

Paper mode is the DEFAULT (STOCK_TYPE=mock_binance). Real trading
requires STOCK_TYPE=binance set explicitly in the env file.

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


def main() -> None:
    from stocks_holder import do_stock_init, stock_holder

    stock_type = os.environ.get("STOCK_TYPE", "mock_binance")
    do_stock_init(stock_type)
    stock = stock_holder.item

    from data import LiveData
    from helpers import shared_folder
    from robots.robot import Robot

    strategy_set = os.environ.get("STRATEGY_SET", "")
    live_data = LiveData()
    strategy_manager = build_strategy_manager(stock, strategy_set)

    os.makedirs(shared_folder(), exist_ok=True)
    robot = Robot(
        strategy_manager,
        live_data,
        stock,
        stock.fee,
        persist_path=shared_folder() + "live_tracker.json",
    )
    robot.position.full_position = resolve_live_usdt()
    robot.run_instantly()


if __name__ == "__main__":
    main()
