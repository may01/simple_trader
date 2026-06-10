"""trader.py — live trading entry point (Docker path E).

Paper mode is the DEFAULT (STOCK_TYPE=mock_binance). Real trading
requires STOCK_TYPE=binance set explicitly in the env file.
"""

import os


def main() -> None:
    from stocks_holder import do_stock_init, stock_holder

    stock_type = os.environ.get("STOCK_TYPE", "mock_binance")
    do_stock_init(stock_type)
    stock = stock_holder.item

    from data import LiveData
    from helpers import shared_folder
    from robots.robot import Robot
    from strategies.strategy_manager import StrategyManager

    live_data = LiveData()
    strategy_manager = StrategyManager(stock.fee)

    os.makedirs(shared_folder(), exist_ok=True)
    robot = Robot(
        strategy_manager,
        live_data,
        stock,
        stock.fee,
        persist_path=shared_folder() + "live_tracker.json",
    )
    robot.run_instantly()


if __name__ == "__main__":
    main()
