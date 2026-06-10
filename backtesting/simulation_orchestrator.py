"""SimulationOrchestrator — parallel backtesting runner (Phase 08, Task 02).

Splits a SimulationData into segments and runs each segment in a separate
process via ProcessPoolExecutor.  A single-threaded variant is also provided
for debugging and small datasets.
"""

import os
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

from robots.train_robot import TrainRobot

logger = logging.getLogger(__name__)

# Empty result returned when a worker crashes
_EMPTY_RESULT = {
    'revenue_history': [],
    'trade_count': 0,
    'total_trades': 0,
    'total_revenue_abs': 0.0,
    'avg_revenue_pct': 0.0,
}


def _worker(segment_data, strategy_factory: Callable, fee: float) -> dict:
    """Worker function executed in a child process.

    Must be a module-level function (not a method) so that multiprocessing
    can pickle it.

    Args:
        segment_data: A SimulationData slice to iterate over.
        strategy_factory: Zero-arg callable returning a configured StrategyManager.
        fee: Trading fee fraction passed to TrainRobot.

    Returns:
        Result dict from robot.get_results().
    """
    strategy_manager = strategy_factory()
    robot = TrainRobot(strategy_manager, fee)

    while not segment_data.is_end():
        dp = segment_data.get()
        robot.step(dp)
        segment_data.next()

    return robot.get_results()


class SimulationOrchestrator:
    """Orchestrates parallel backtesting over a SimulationData dataset.

    Args:
        strategy_factory: Zero-arg callable; called once per worker to produce
            an independent StrategyManager.
        fee: Trading fee fraction forwarded to each TrainRobot.
    """

    def __init__(self, strategy_factory: Callable, fee: float) -> None:
        self.strategy_factory = strategy_factory
        self.fee = fee
        self.num_workers: int = int(os.environ.get('NUM_WORKERS', '4'))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, simulation_data) -> list:
        """Run the simulation in parallel using one process per worker.

        Args:
            simulation_data: SimulationData instance to split and process.

        Returns:
            List of result dicts, one per worker segment.  Crashed workers
            contribute an empty result dict.
        """
        segments = simulation_data.split(self.num_workers)

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(_worker, seg, self.strategy_factory, self.fee)
                for seg in segments
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Simulation worker crashed: %s", exc)
                    results.append({**_EMPTY_RESULT, 'revenue_history': []})

        return results

    def run_single(self, simulation_data) -> dict:
        """Run the simulation single-threaded (no parallelism).

        Useful for debugging or when the overhead of spawning processes is
        not justified.

        Args:
            simulation_data: SimulationData instance to iterate over.

        Returns:
            Result dict from the single TrainRobot.
        """
        strategy_manager = self.strategy_factory()
        robot = TrainRobot(strategy_manager, self.fee)

        while not simulation_data.is_end():
            dp = simulation_data.get()
            robot.step(dp)
            simulation_data.next()

        return robot.get_results()
