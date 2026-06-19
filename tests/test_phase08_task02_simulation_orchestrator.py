"""Tests for SimulationOrchestrator — Phase 08, Task 02.

TDD: tests written before implementation.
All dependencies (SimulationData, TrainRobot, strategy_factory) are mocked.

Phase 14 (Task 05) changed the contract:
  - ``_worker(sd, factory, fee, sim_id)`` — gains sim_id, returns dict + 'action_log_jsonl'
  - ``run`` / ``run_single`` allocate a simulation id, persist actions, and
    return a dict ``{sim_id, results, action_count, actions_path}``.
Tests stub id-allocation + persistence so they stay env-free.
"""

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMPTY_RESULT = {
    'revenue_history': [],
    'trade_count': 0,
    'total_trades': 0,
    'total_revenue_abs': 0.0,
    'avg_revenue_pct': 0.0,
    'action_log_jsonl': '',
}

SAMPLE_RESULT = {
    'revenue_history': [(0.02, 200.0)],
    'trade_count': 1,
    'total_trades': 1,
    'total_revenue_abs': 200.0,
    'avg_revenue_pct': 0.02,
}


def make_strategy_manager():
    return MagicMock()


def _make_strategy_factory(sm=None):
    if sm is None:
        sm = make_strategy_manager()
    def factory():
        return sm
    return factory


def make_simulation_data(num_points=3):
    sd = MagicMock()
    sd.is_end.side_effect = [False] * num_points + [True]
    sd.get.return_value = MagicMock()
    sd.next.return_value = None
    return sd


def _mock_robot(MockRobot, result=None):
    inst = MagicMock()
    inst.get_results.return_value = (result or SAMPLE_RESULT).copy()
    inst.get_action_log.return_value.to_jsonl.return_value = ""
    MockRobot.return_value = inst
    return inst


def _stub_persistence(orch):
    """Replace id-allocation + persistence so run() needs no env/filesystem."""
    orch._allocate_sim_id = lambda: setattr(orch, 'sim_id', 1) or 1
    orch._persist_actions = lambda sim_id, results: ('/tmp/actions.jsonl', 0)


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
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.005)
        assert orch.fee == 0.005

    def test_default_num_workers_is_4(self, monkeypatch):
        monkeypatch.delenv('NUM_WORKERS', raising=False)
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        assert orch.num_workers == 4

    def test_num_workers_reads_env_var(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '8')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        assert orch.num_workers == 8

    def test_num_workers_is_int(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '3')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        assert isinstance(orch.num_workers, int)


# ---------------------------------------------------------------------------
# _worker module-level function
# ---------------------------------------------------------------------------

class TestWorkerFunction:

    def test_worker_calls_strategy_factory(self):
        sm = make_strategy_manager()
        factory = _make_strategy_factory(sm)
        sd = make_simulation_data(0)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot)
            _worker(sd, factory, 0.001, 1)
        MockRobot.assert_called_once_with(sm, 0.001)

    def test_worker_sets_sim_context(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(0)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            inst = _mock_robot(MockRobot)
            _worker(sd, factory, 0.001, 42)
        inst.set_sim_context.assert_called_once_with(42)

    def test_worker_calls_step_for_each_data_point(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(5)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            inst = _mock_robot(MockRobot)
            _worker(sd, factory, 0.001, 1)
        assert inst.step.call_count == 5

    def test_worker_calls_next_after_each_get(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(3)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot)
            _worker(sd, factory, 0.001, 1)
        assert sd.next.call_count == 3

    def test_worker_returns_results_with_action_jsonl(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(2)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot, SAMPLE_RESULT)
            result = _worker(sd, factory, 0.001, 1)
        assert result['total_trades'] == 1
        assert 'action_log_jsonl' in result

    def test_worker_zero_data_points(self):
        factory = _make_strategy_factory()
        sd = make_simulation_data(0)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            inst = _mock_robot(MockRobot)
            _worker(sd, factory, 0.001, 1)
        assert inst.step.call_count == 0


# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------

class TestRunSingle:

    def test_run_single_creates_one_robot(self):
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = make_simulation_data(3)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot)
            orch.run_single(sd)
        assert MockRobot.call_count == 1

    def test_run_single_calls_step_n_times(self):
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = make_simulation_data(7)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            inst = _mock_robot(MockRobot)
            orch.run_single(sd)
        assert inst.step.call_count == 7

    def test_run_single_returns_dict_with_results(self):
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = make_simulation_data(2)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot, SAMPLE_RESULT)
            result = orch.run_single(sd)
        assert result['sim_id'] == 1
        assert result['results'][0]['total_trades'] == 1

    def test_run_single_uses_factory_result(self):
        sm = make_strategy_manager()
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(sm), fee=0.002)
        _stub_persistence(orch)
        sd = make_simulation_data(1)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot)
            orch.run_single(sd)
        MockRobot.assert_called_once_with(sm, 0.002)

    def test_run_single_does_not_split_data(self):
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = make_simulation_data(2)
        with patch('backtesting.simulation_orchestrator.TrainRobot') as MockRobot:
            _mock_robot(MockRobot)
            orch.run_single(sd)
        sd.split.assert_not_called()


# ---------------------------------------------------------------------------
# run (parallel)
# ---------------------------------------------------------------------------

class TestRun:

    def _run_with_pool(self, orch, sd, segment_results):
        with patch('backtesting.simulation_orchestrator.ProcessPoolExecutor') as MockPool:
            mock_executor = MagicMock()
            MockPool.return_value.__enter__.return_value = mock_executor
            futures = []
            for r in segment_results:
                f = MagicMock()
                if isinstance(r, Exception):
                    f.result.side_effect = r
                else:
                    f.result.return_value = r
                futures.append(f)
            mock_executor.submit.side_effect = futures
            return orch.run(sd), MockPool, mock_executor

    def test_run_calls_split_with_num_workers(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '3')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = MagicMock()
        sd.split.return_value = [make_simulation_data(2) for _ in range(3)]
        self._run_with_pool(orch, sd, [SAMPLE_RESULT] * 3)
        sd.split.assert_called_once_with(3)

    def test_run_returns_dict_with_results_list(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '2')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = MagicMock()
        sd.split.return_value = [make_simulation_data(2)] * 2
        run_result, _, _ = self._run_with_pool(orch, sd, [SAMPLE_RESULT, SAMPLE_RESULT])
        assert isinstance(run_result['results'], list)
        assert len(run_result['results']) == 2
        assert run_result['sim_id'] == 1

    def test_run_submits_one_future_per_segment(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '3')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = MagicMock()
        sd.split.return_value = [make_simulation_data(1) for _ in range(3)]
        _, _, executor = self._run_with_pool(orch, sd, [SAMPLE_RESULT] * 3)
        assert executor.submit.call_count == 3

    def test_run_worker_crash_returns_empty_dict(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '2')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = MagicMock()
        sd.split.return_value = [make_simulation_data(2)] * 2
        run_result, _, _ = self._run_with_pool(
            orch, sd, [SAMPLE_RESULT, RuntimeError("crash")]
        )
        assert len(run_result['results']) == 2
        assert run_result['results'][1] == EMPTY_RESULT

    def test_run_uses_num_workers_for_pool(self, monkeypatch):
        monkeypatch.setenv('NUM_WORKERS', '5')
        orch = SimulationOrchestrator(strategy_factory=_make_strategy_factory(), fee=0.001)
        _stub_persistence(orch)
        sd = MagicMock()
        sd.split.return_value = [make_simulation_data(1) for _ in range(5)]
        _, MockPool, _ = self._run_with_pool(orch, sd, [SAMPLE_RESULT] * 5)
        MockPool.assert_called_once_with(max_workers=5)
