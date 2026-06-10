"""Tests for SimulationOrchestrator — Phase 08, Task 02.

TDD: tests written before implementation.
All dependencies (SimulationData, TrainRobot, strategy_factory) are mocked.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMPTY_RESULT = {
    'revenue_history': [],
    'trade_count': 0,
    'total_trades': 0,
    'total_revenue_abs': 0.0,
    'avg_revenue_pct': 0.0,
}

SAMPLE_RESULT = {
    'revenue_history': [(0.02, 200.0)],
    'trade_count': 1,
    'total_trades': 1,
    'total_revenue_abs': 200.0,
    'avg_revenue_pct': 0.02,
}


def make_strategy_manager():
    """Return a mock StrategyManager."""
    return MagicMock()


def _make_strategy_factory(sm=None):
    """Return a zero-arg callable that produces a mock StrategyManager."""
    if sm is None:
        sm = make_strategy_manager()
    def factory():
        return sm
    return factory


def make_simulation_data(num_points=3):
    """Return a mock SimulationData that iterates num_points times."""
    sd = MagicMock()
    # is_end returns False for the first num_points calls, then True
    is_end_side = [False] * num_points + [True]
    sd.is_end.side_effect = is_end_side
    sd.get.return_value = MagicMock()  # data point
    sd.next.return_value = None
    return sd


def make_simulation_data_with_result(num_points=3, result=None):
    """Return (mock SimulationData, mock TrainRobot that will be created)."""
    if result is None:
        result = SAMPLE_RESULT
    sd = make_simulation_data(num_points)
    return sd


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

from backtesting.simulation_orchestrator import SimulationOrchestrator, _worker


# ---------------------------------------------------------------------------
# Construction / init
# ---------------------------------------------------------------------------

class TestSimulationOrchestratorInit:

    def test_stores_strategy_factory(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        assert orch.strategy_factory is factory

    def test_stores_fee(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.005)
        assert orch.fee == 0.005

    def test_default_num_workers_is_4(self, monkeypatch):
        monkeypatch.delenv('NUM_WORKERS', raising=False)
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        assert orch.num_workers == 4

    def test_num_workers_reads_env_var(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '8')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        assert orch.num_workers == 8

    def test_num_workers_env_var_1(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '1')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        assert orch.num_workers == 1

    def test_num_workers_is_int(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '3')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        assert isinstance(orch.num_workers, int)


# ---------------------------------------------------------------------------
# _worker module-level function
# ---------------------------------------------------------------------------

class TestWorkerFunction:

    def test_worker_calls_strategy_factory(self):
        sm = make_strategy_manager()
        factory = _make_strategy_factory(sm)
        sd = make_simulation_data(0)
        robot_result = SAMPLE_RESULT.copy()

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = robot_result
            MockRobot.return_value = mock_robot_instance

            result = _worker(sd, factory, 0.001)

        MockRobot.assert_called_once_with(sm, 0.001)

    def test_worker_creates_train_robot_with_fee(self):
        sm = make_strategy_manager()
        factory = _make_strategy_factory(sm)
        sd = make_simulation_data(0)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            _worker(sd, factory, 0.007)

        MockRobot.assert_called_once_with(sm, 0.007)

    def test_worker_calls_step_for_each_data_point(self):
        factory = _make_strategy_factory()
        num_points = 5
        sd = make_simulation_data(num_points)
        dp = MagicMock()
        sd.get.return_value = dp

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            _worker(sd, factory, 0.001)

        assert mock_robot_instance.step.call_count == num_points

    def test_worker_calls_next_after_each_get(self):
        factory = _make_strategy_factory()
        num_points = 3
        sd = make_simulation_data(num_points)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            _worker(sd, factory, 0.001)

        assert sd.next.call_count == num_points

    def test_worker_returns_get_results(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(2)
        expected = SAMPLE_RESULT.copy()

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = expected
            MockRobot.return_value = mock_robot_instance

            result = _worker(sd, factory, 0.001)

        assert result == expected

    def test_worker_zero_data_points_returns_results(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(0)
        empty = {'total_trades': 0, 'revenue_history': [], 'total_revenue_abs': 0.0, 'avg_revenue_pct': 0.0}

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = empty
            MockRobot.return_value = mock_robot_instance

            result = _worker(sd, factory, 0.001)

        assert mock_robot_instance.step.call_count == 0
        assert result == empty


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------

class TestRunSingle:

    def test_run_single_creates_one_robot(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        sd = make_simulation_data(3)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            orch.run_single(sd)

        assert MockRobot.call_count == 1

    def test_run_single_calls_step_n_times(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        num_points = 7
        sd = make_simulation_data(num_points)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            orch.run_single(sd)

        assert mock_robot_instance.step.call_count == num_points

    def test_run_single_returns_dict(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        sd = make_simulation_data(2)
        expected = SAMPLE_RESULT.copy()

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = expected
            MockRobot.return_value = mock_robot_instance

            result = orch.run_single(sd)

        assert result == expected

    def test_run_single_uses_factory_result_as_strategy_manager(self):
        sm = make_strategy_manager()
        factory = _make_strategy_factory(sm)
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.002)
        sd = make_simulation_data(1)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            orch.run_single(sd)

        MockRobot.assert_called_once_with(sm, 0.002)

    def test_run_single_does_not_split_data(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        sd = make_simulation_data(2)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            orch.run_single(sd)

        sd.split.assert_not_called()

    def test_run_single_calls_next_for_each_point(self):
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)
        num_points = 4
        sd = make_simulation_data(num_points)

        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            mock_robot_instance = MagicMock()
            mock_robot_instance.get_results.return_value = SAMPLE_RESULT
            MockRobot.return_value = mock_robot_instance

            orch.run_single(sd)

        assert sd.next.call_count == num_points


# ---------------------------------------------------------------------------
# run (parallel)
# ---------------------------------------------------------------------------

class TestRun:

    def test_run_calls_split_with_num_workers(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '3')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)

        # Build 3 segment mocks
        segments = [make_simulation_data(2) for _ in range(3)]

        sd = MagicMock()
        sd.split.return_value = segments

        with patch('backtesting.simulation_orchestrator._worker',
                   return_value=SAMPLE_RESULT) as mock_worker:
            with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
                # Set up the executor mock to call _worker synchronously
                mock_executor = MagicMock()
                MockPool.return_value.__enter__.return_value = mock_executor

                futures = [MagicMock() for _ in segments]
                for f, seg in zip(futures, segments):
                    f.result.return_value = SAMPLE_RESULT
                mock_executor.submit.side_effect = futures

                orch.run(sd)

        sd.split.assert_called_once_with(3)

    def test_run_returns_list_of_dicts(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '2')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)

        segments = [make_simulation_data(2), make_simulation_data(2)]
        sd = MagicMock()
        sd.split.return_value = segments

        with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
            mock_executor = MagicMock()
            MockPool.return_value.__enter__.return_value = mock_executor

            futures = [MagicMock(), MagicMock()]
            futures[0].result.return_value = SAMPLE_RESULT
            futures[1].result.return_value = SAMPLE_RESULT
            mock_executor.submit.side_effect = futures

            results = orch.run(sd)

        assert isinstance(results, list)
        assert len(results) == 2

    def test_run_submits_one_future_per_segment(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '3')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)

        segments = [make_simulation_data(1) for _ in range(3)]
        sd = MagicMock()
        sd.split.return_value = segments

        with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
            mock_executor = MagicMock()
            MockPool.return_value.__enter__.return_value = mock_executor

            futures = [MagicMock() for _ in segments]
            for f in futures:
                f.result.return_value = SAMPLE_RESULT
            mock_executor.submit.side_effect = futures

            orch.run(sd)

        assert mock_executor.submit.call_count == 3

    def test_run_worker_crash_returns_empty_dict(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '2')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)

        segments = [make_simulation_data(2), make_simulation_data(2)]
        sd = MagicMock()
        sd.split.return_value = segments

        with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
            mock_executor = MagicMock()
            MockPool.return_value.__enter__.return_value = mock_executor

            future_ok = MagicMock()
            future_ok.result.return_value = SAMPLE_RESULT

            future_fail = MagicMock()
            future_fail.result.side_effect = RuntimeError("worker crashed")

            mock_executor.submit.side_effect = [future_ok, future_fail]

            results = orch.run(sd)

        assert len(results) == 2
        # The crashed worker returns empty dict
        assert results[1] == EMPTY_RESULT

    def test_run_collects_results_from_all_workers(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '2')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)

        result_a = {'total_trades': 1, 'revenue_history': [(0.01, 100.0)],
                    'total_revenue_abs': 100.0, 'avg_revenue_pct': 0.01}
        result_b = {'total_trades': 2, 'revenue_history': [(0.02, 200.0), (0.03, 300.0)],
                    'total_revenue_abs': 500.0, 'avg_revenue_pct': 0.025}

        segments = [make_simulation_data(1), make_simulation_data(1)]
        sd = MagicMock()
        sd.split.return_value = segments

        with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
            mock_executor = MagicMock()
            MockPool.return_value.__enter__.return_value = mock_executor

            future_a = MagicMock()
            future_a.result.return_value = result_a
            future_b = MagicMock()
            future_b.result.return_value = result_b
            mock_executor.submit.side_effect = [future_a, future_b]

            results = orch.run(sd)

        assert results[0] == result_a
        assert results[1] == result_b

    def test_run_uses_num_workers_for_pool(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '5')
        factory = _make_strategy_factory()
        orch = SimulationOrchestrator(strategy_factory=factory, fee=0.001)

        segments = [make_simulation_data(1) for _ in range(5)]
        sd = MagicMock()
        sd.split.return_value = segments

        with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
            mock_executor = MagicMock()
            MockPool.return_value.__enter__.return_value = mock_executor

            futures = [MagicMock() for _ in segments]
            for f in futures:
                f.result.return_value = SAMPLE_RESULT
            mock_executor.submit.side_effect = futures

            orch.run(sd)

        MockPool.assert_called_once_with(max_workers=5)
