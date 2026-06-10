"""Tests for training/trainer.py — Trainer orchestrator (Phase 09, Task 03).

Tests use --noconftest so all fixtures must be self-contained.
All external dependencies are mocked.
"""

import json
import os
import pickle
import pytest
from unittest.mock import MagicMock, patch, call, mock_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_env(monkeypatch, extra: dict = None):
    """Set minimum required env vars for Trainer construction."""
    env = {
        "RUN_TYPE": "grab_data",
        "ROOT_FOLDER": "short",
        "DATA_ROOT": "test_data",
        "DATA_SET_NAME": "test_set",
        "PAIR": "link_usdt",
        "DATA_START": "1700000000000",
        "DATA_END": "1700100000000",
    }
    if extra:
        env.update(extra)
    for key, val in env.items():
        monkeypatch.setenv(key, val)


# ---------------------------------------------------------------------------
# 1. Trainer reads RUN_TYPE from env at construction
# ---------------------------------------------------------------------------

class TestTrainerConstruction:
    def test_reads_run_type_from_env(self, monkeypatch):
        """RUN_TYPE env var is captured into self.run_type at construction."""
        _set_env(monkeypatch, {"RUN_TYPE": "simulate"})
        from training.trainer import Trainer
        t = Trainer(config_path="config/")
        assert t.run_type == "simulate"

    def test_run_type_grab_data(self, monkeypatch):
        """run_type='grab_data' is stored correctly."""
        _set_env(monkeypatch, {"RUN_TYPE": "grab_data"})
        from training.trainer import Trainer
        t = Trainer()
        assert t.run_type == "grab_data"

    def test_default_config_path(self, monkeypatch):
        """config_path defaults to 'config/'."""
        _set_env(monkeypatch)
        from training.trainer import Trainer
        t = Trainer()
        assert t.config_path == "config/"

    def test_custom_config_path(self, monkeypatch):
        """config_path can be overridden."""
        _set_env(monkeypatch)
        from training.trainer import Trainer
        t = Trainer(config_path="/custom/config/")
        assert t.config_path == "/custom/config/"

    def test_metadata_initialized_as_dict(self, monkeypatch):
        """metadata is initialized as an empty dict."""
        _set_env(monkeypatch)
        from training.trainer import Trainer
        t = Trainer()
        assert isinstance(t.metadata, dict)

    def test_raises_key_error_when_run_type_missing(self, monkeypatch):
        """Raises KeyError if RUN_TYPE env var is not set."""
        _set_env(monkeypatch)
        monkeypatch.delenv("RUN_TYPE", raising=False)
        # Reimport to pick up env state
        import importlib
        import training.trainer as mod
        importlib.reload(mod)
        with pytest.raises(KeyError):
            mod.Trainer()


# ---------------------------------------------------------------------------
# 2. run() dispatches to correct method for each run_type
# ---------------------------------------------------------------------------

class TestRunDispatch:
    """run() calls the correct _run_* method for each RUN_TYPE value."""

    def _make_trainer(self, monkeypatch, run_type: str):
        _set_env(monkeypatch, {"RUN_TYPE": run_type})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)
        return mod.Trainer()

    def test_dispatches_grab_data(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "grab_data")
        t._run_grab_data = MagicMock()
        t.run()
        t._run_grab_data.assert_called_once()

    def test_dispatches_prepare_data(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "prepare_data")
        t._run_prepare_data = MagicMock()
        t.run()
        t._run_prepare_data.assert_called_once()

    def test_dispatches_simulate(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "simulate")
        t._run_simulate = MagicMock()
        t.run()
        t._run_simulate.assert_called_once()

    def test_dispatches_train_nn(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "train_nn")
        t._run_train_nn = MagicMock()
        t.run()
        t._run_train_nn.assert_called_once()

    def test_dispatches_simulate_nn(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "simulate_nn")
        t._run_simulate_nn = MagicMock()
        t.run()
        t._run_simulate_nn.assert_called_once()


# ---------------------------------------------------------------------------
# 3. run() raises ValueError for unknown run_type
# ---------------------------------------------------------------------------

class TestUnknownRunType:
    def test_raises_value_error_for_unknown_type(self, monkeypatch):
        """run() raises ValueError when run_type is unrecognized."""
        _set_env(monkeypatch, {"RUN_TYPE": "unknown_type"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)
        t = mod.Trainer()
        with pytest.raises(ValueError, match="unknown_type"):
            t.run()

    def test_raises_value_error_for_empty_string(self, monkeypatch):
        """run() raises ValueError for empty run_type."""
        _set_env(monkeypatch, {"RUN_TYPE": ""})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)
        t = mod.Trainer()
        with pytest.raises(ValueError):
            t.run()


# ---------------------------------------------------------------------------
# 4. _run_grab_data() constructs Graber with correct params
# ---------------------------------------------------------------------------

class TestRunGrabData:
    """Tests for _run_grab_data().

    Since imports in _run_grab_data are lazy (inside method body), we must
    patch at the source module level, not at training.trainer namespace.
    """

    def test_graber_constructed_with_stock_and_path(self, monkeypatch):
        """_run_grab_data instantiates Graber(stock, output_path=graber_data_path())."""
        _set_env(monkeypatch, {"RUN_TYPE": "grab_data"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_stock = MagicMock()
        mock_graber_instance = MagicMock()
        mock_graber_cls = MagicMock(return_value=mock_graber_instance)

        # Patch at source since imports are lazy inside the method
        with patch("stocks_holder.do_stock_init"), \
             patch("stocks_holder.stock_holder") as mock_holder, \
             patch("training.graber.Graber", mock_graber_cls), \
             patch("helpers.graber_data_path", return_value="/data/graber_data.pkl"):
            mock_holder.item = mock_stock
            t = mod.Trainer()
            t._run_grab_data()

        mock_graber_cls.assert_called_once_with(mock_stock, output_path="/data/graber_data.pkl")

    def test_ensure_data_called_with_symbol_and_times(self, monkeypatch):
        """ensure_data called with converted symbol and DATA_START/DATA_END as ints."""
        _set_env(monkeypatch, {
            "RUN_TYPE": "grab_data",
            "PAIR": "link_usdt",
            "DATA_START": "1700000000000",
            "DATA_END": "1700100000000",
        })
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_graber_instance = MagicMock()
        mock_graber_cls = MagicMock(return_value=mock_graber_instance)

        with patch("stocks_holder.do_stock_init"), \
             patch("stocks_holder.stock_holder"), \
             patch("training.graber.Graber", mock_graber_cls), \
             patch("helpers.graber_data_path", return_value="/data/graber_data.pkl"):
            t = mod.Trainer()
            t._run_grab_data()

        # PAIR "link_usdt" → "LINKUSDT"
        mock_graber_instance.ensure_data.assert_called_once_with(
            "LINKUSDT", 1700000000000, 1700100000000
        )

    def test_pair_conversion_btc_usdt(self, monkeypatch):
        """PAIR 'btc_usdt' converts to symbol 'BTCUSDT'."""
        _set_env(monkeypatch, {
            "RUN_TYPE": "grab_data",
            "PAIR": "btc_usdt",
        })
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_graber_instance = MagicMock()
        mock_graber_cls = MagicMock(return_value=mock_graber_instance)

        with patch("stocks_holder.do_stock_init"), \
             patch("stocks_holder.stock_holder"), \
             patch("training.graber.Graber", mock_graber_cls), \
             patch("helpers.graber_data_path", return_value="/data/graber_data.pkl"):
            t = mod.Trainer()
            t._run_grab_data()

        call_args = mock_graber_instance.ensure_data.call_args
        assert call_args[0][0] == "BTCUSDT"


# ---------------------------------------------------------------------------
# 5. "full" mode calls all 6 steps in correct order
# ---------------------------------------------------------------------------

class TestFullMode:
    def test_full_calls_all_steps_in_order(self, monkeypatch):
        """'full' run_type calls all 6 pipeline steps in order."""
        _set_env(monkeypatch, {"RUN_TYPE": "full"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        t = mod.Trainer()
        call_order = []

        t._run_grab_data = MagicMock(side_effect=lambda: call_order.append("grab_data"))
        t._run_prepare_data = MagicMock(side_effect=lambda: call_order.append("prepare_data"))
        t._run_simulate = MagicMock(side_effect=lambda: call_order.append("simulate"))
        t._run_train_nn = MagicMock(side_effect=lambda: call_order.append("train_nn"))
        t._run_simulate_nn = MagicMock(side_effect=lambda: call_order.append("simulate_nn"))

        t.run()

        # "full" = grab_data → prepare_data → simulate → train_nn → simulate_nn → prepare_data
        assert call_order == [
            "grab_data",
            "prepare_data",
            "simulate",
            "train_nn",
            "simulate_nn",
            "prepare_data",
        ]

    def test_full_prepare_data_called_twice(self, monkeypatch):
        """In 'full' mode, _run_prepare_data is called exactly twice."""
        _set_env(monkeypatch, {"RUN_TYPE": "full"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        t = mod.Trainer()
        t._run_grab_data = MagicMock()
        t._run_prepare_data = MagicMock()
        t._run_simulate = MagicMock()
        t._run_train_nn = MagicMock()
        t._run_simulate_nn = MagicMock()

        t.run()

        assert t._run_prepare_data.call_count == 2


# ---------------------------------------------------------------------------
# 6. _save_metadata() writes JSON to the given path
# ---------------------------------------------------------------------------

class TestSaveMetadata:
    def test_saves_metadata_as_json(self, monkeypatch, tmp_path):
        """_save_metadata writes self.metadata as JSON to the given path."""
        _set_env(monkeypatch)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        t = mod.Trainer()
        t.metadata = {"stage": "simulate", "trades": 42, "win_rate": 0.55}

        out_path = str(tmp_path / "metadata.json")
        t._save_metadata(out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == {"stage": "simulate", "trades": 42, "win_rate": 0.55}

    def test_saves_empty_metadata(self, monkeypatch, tmp_path):
        """_save_metadata works with an empty metadata dict."""
        _set_env(monkeypatch)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        t = mod.Trainer()
        out_path = str(tmp_path / "metadata.json")
        t._save_metadata(out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded == {}


# ---------------------------------------------------------------------------
# 7. Mock all external dependencies — _run_prepare_data, _run_simulate, etc.
# ---------------------------------------------------------------------------

class TestRunPrepareData:
    """Tests for _run_prepare_data().

    Patches at source since imports are lazy inside the method body.
    """

    def test_data_preparer_instantiated_with_correct_args(self, monkeypatch):
        """_run_prepare_data instantiates DataPreparer with correct arguments."""
        _set_env(monkeypatch, {"RUN_TYPE": "prepare_data"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_dp_instance = MagicMock()
        mock_dp_cls = MagicMock(return_value=mock_dp_instance)

        # Patch DataPreparer in its source module + helpers at source
        with patch("training.data_preparer.DataPreparer", mock_dp_cls), \
             patch("helpers.graber_data_path", return_value="/data/graber.pkl"), \
             patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
             patch("helpers.data_attributes_path", return_value="/data/attrs.pkl"), \
             patch("helpers.nn_folder", return_value="/data/nn/"):
            t = mod.Trainer(config_path="/cfg/")
            t._run_prepare_data()

        mock_dp_cls.assert_called_once_with(
            "/cfg/",
            "/data/wide.pkl",
            "/data/attrs.pkl",
            nn_output_path="/data/nn/df_with_nn.pkl",
        )

    def test_prepare_called_with_graber_data_path(self, monkeypatch):
        """_run_prepare_data calls prepare(graber_data_path())."""
        _set_env(monkeypatch, {"RUN_TYPE": "prepare_data"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_dp_instance = MagicMock()
        mock_dp_cls = MagicMock(return_value=mock_dp_instance)

        with patch("training.data_preparer.DataPreparer", mock_dp_cls), \
             patch("helpers.graber_data_path", return_value="/data/graber.pkl"), \
             patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
             patch("helpers.data_attributes_path", return_value="/data/attrs.pkl"), \
             patch("helpers.nn_folder", return_value="/data/nn/"):
            t = mod.Trainer()
            t._run_prepare_data()

        mock_dp_instance.prepare.assert_called_once_with("/data/graber.pkl")


class TestRunSimulate:
    """Tests for _run_simulate().

    Uses sys.modules injection so the lazy import inside _run_simulate finds
    our mock modules without needing talib or real backtesting code.
    """

    def test_simulation_orchestrator_called(self, monkeypatch, tmp_path):
        """_run_simulate instantiates SimulationOrchestrator and calls run()."""
        import sys
        _set_env(monkeypatch, {"RUN_TYPE": "simulate"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_sim_data = MagicMock()
        mock_sim_data_cls = MagicMock(return_value=mock_sim_data)
        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = []
        mock_orch_cls = MagicMock(return_value=mock_orch_instance)
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"total_trades": 0}
        mock_analyzer_cls = MagicMock(return_value=mock_analyzer_instance)

        # Inject mock modules so lazy imports find them
        mock_data_module = MagicMock()
        mock_data_module.SimulationData = mock_sim_data_cls
        mock_backtesting_sim = MagicMock()
        mock_backtesting_sim.SimulationOrchestrator = mock_orch_cls
        mock_backtesting_perf = MagicMock()
        mock_backtesting_perf.PerformanceAnalyzer = mock_analyzer_cls
        mock_robots = MagicMock()

        _keys_to_mock = ["data", "backtesting.simulation_orchestrator",
                         "backtesting.performance_analyzer", "robots.train_robot"]
        _saved = {k: sys.modules.get(k) for k in _keys_to_mock}
        sys.modules["data"] = mock_data_module
        sys.modules["backtesting.simulation_orchestrator"] = mock_backtesting_sim
        sys.modules["backtesting.performance_analyzer"] = mock_backtesting_perf
        sys.modules["robots.train_robot"] = mock_robots

        # Also patch pandas.read_pickle to avoid actual file access
        import pandas as real_pd
        mock_pd = MagicMock(wraps=real_pd)
        mock_pd.read_pickle.return_value = MagicMock()
        sys.modules["pandas"] = mock_pd

        try:
            with patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
                 patch("helpers.shared_folder", return_value=str(tmp_path) + "/"):
                t = mod.Trainer()
                t._run_simulate()
        finally:
            for key, original in _saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original
            sys.modules["pandas"] = real_pd

        mock_orch_instance.run.assert_called_once_with(mock_sim_data)
        mock_analyzer_instance.analyze.assert_called_once()


class TestRunSimulateNN:
    """Tests for _run_simulate_nn().

    Uses sys.modules injection since NNOrchestrator, DataAttributes, and
    pandas are all imported lazily inside the method body.
    """

    def test_simulate_nn_saves_result(self, monkeypatch, tmp_path):
        """_run_simulate_nn calls NNOrchestrator.run_inference and saves result."""
        import sys
        import pandas as real_pd
        _set_env(monkeypatch, {"RUN_TYPE": "simulate_nn"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        # Build a real small DataFrame as inference output so to_pickle works
        nn_result_df = real_pd.DataFrame({"nn_col": [1, 2, 3]})

        mock_nn_orch = MagicMock()
        mock_nn_orch.run_inference.return_value = nn_result_df
        mock_nn_orch_cls = MagicMock(return_value=mock_nn_orch)

        mock_da_instance = MagicMock()
        mock_da_cls = MagicMock()
        mock_da_cls.load.return_value = mock_da_instance

        # Mock nn_orchestrator module
        mock_nn_module = MagicMock()
        mock_nn_module.NNOrchestrator = mock_nn_orch_cls

        # Mock indicators module (contains DataAttributes)
        mock_indicators_module = MagicMock()
        mock_indicators_module.DataAttributes = mock_da_cls

        # Mock config_loader for CANDLES
        mock_config_module = MagicMock()
        mock_config_module.CANDLES = [1, 5, 15, 60]

        _nn_keys = ["training.nn_orchestrator", "indicators", "config_loader"]
        _nn_saved = {k: sys.modules.get(k) for k in _nn_keys}
        sys.modules["training.nn_orchestrator"] = mock_nn_module
        sys.modules["indicators"] = mock_indicators_module
        sys.modules["config_loader"] = mock_config_module

        try:
            with patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
                 patch("helpers.data_attributes_path", return_value="/data/attrs.pkl"), \
                 patch("helpers.nn_folder", return_value=str(tmp_path) + "/"), \
                 patch("pandas.read_pickle", return_value=MagicMock()):
                t = mod.Trainer()
                t._run_simulate_nn()
        finally:
            for key, original in _nn_saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

        mock_nn_orch.run_inference.assert_called_once()
        # Result should be saved as a pkl file
        out_path = str(tmp_path) + "/df_with_nn.pkl"
        assert os.path.exists(out_path)
