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
    'action_log_jsonl': '',
}


def _worker(segment_data, strategy_factory: Callable, fee: float, sim_id: int) -> dict:
    """Worker function executed in a child process.

    Must be a module-level function (not a method) so that multiprocessing
    can pickle it.

    Args:
        segment_data: A SimulationData slice to iterate over.
        strategy_factory: Zero-arg callable returning a configured StrategyManager.
        fee: Trading fee fraction passed to TrainRobot.
        sim_id: Shared simulation id — all workers stamp the same id.

    Returns:
        Result dict from robot.get_results() plus 'action_log_jsonl'.
    """
    strategy_manager = strategy_factory()
    robot = TrainRobot(strategy_manager, fee)
    robot.set_sim_context(sim_id)

    while not segment_data.is_end():
        dp = segment_data.get()
        robot.step(dp)
        segment_data.next()

    results = robot.get_results()
    results['action_log_jsonl'] = robot.get_action_log().to_jsonl()
    return results


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

    def run(self, simulation_data) -> dict:
        """Run the simulation in parallel using one process per worker.

        Allocates a simulation id once, threads it to every worker so all
        workers write into the same sim_<id> folder, merges the per-worker
        action logs, and persists actions.jsonl.

        Args:
            simulation_data: SimulationData instance to split and process.

        Returns:
            Dict with keys: sim_id, results (per-worker list), action_count,
            actions_path. Crashed workers contribute an empty result dict.
        """
        sim_id = self._allocate_sim_id()
        segments = simulation_data.split(self.num_workers)

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(_worker, seg, self.strategy_factory, self.fee, sim_id)
                for seg in segments
            ]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.error("Simulation worker crashed: %s", exc)
                    results.append({**_EMPTY_RESULT})

        return self._finish(sim_id, results)

    def run_single(self, simulation_data) -> dict:
        """Run the simulation single-threaded (no parallelism).

        Args:
            simulation_data: SimulationData instance to iterate over.

        Returns:
            Same dict shape as run().
        """
        sim_id = self._allocate_sim_id()
        strategy_manager = self.strategy_factory()
        robot = TrainRobot(strategy_manager, self.fee)
        robot.set_sim_context(sim_id)

        while not simulation_data.is_end():
            dp = simulation_data.get()
            robot.step(dp)
            simulation_data.next()

        results = robot.get_results()
        results['action_log_jsonl'] = robot.get_action_log().to_jsonl()
        return self._finish(sim_id, [results])

    # ------------------------------------------------------------------
    # Internal — id allocation + persistence
    # ------------------------------------------------------------------

    def _allocate_sim_id(self) -> int:
        """Allocate the next simulation id, stash it on the instance."""
        from helpers import next_simulation_id  # lazy: avoids env read at import
        self.sim_id = next_simulation_id()
        return self.sim_id

    def _finish(self, sim_id: int, results: list) -> dict:
        """Persist merged action logs and return the run summary dict."""
        actions_path, action_count = self._persist_actions(sim_id, results)
        return {
            'sim_id': sim_id,
            'results': results,
            'action_count': action_count,
            'actions_path': actions_path,
        }

    def _persist_actions(self, sim_id: int, results: list) -> tuple:
        """Concatenate worker action logs (in segment order) into actions.jsonl.

        Args:
            sim_id: The run's simulation id.
            results: Per-worker result dicts carrying 'action_log_jsonl'.

        Returns:
            (actions_path, action_count).
        """
        from helpers import simulation_folder  # lazy
        from backtesting.action import ActionLog  # lazy

        folder = simulation_folder(sim_id)
        os.makedirs(folder, exist_ok=True)

        merged = ActionLog(sim_id)
        for res in results:
            jsonl = res.get('action_log_jsonl', '')
            if jsonl:
                merged.extend(ActionLog.from_jsonl(sim_id, jsonl))

        actions_path = folder + 'actions.jsonl'
        with open(actions_path, 'w') as f:
            f.write(merged.to_jsonl())

        return actions_path, len(merged)
