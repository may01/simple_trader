"""Tests for training/trainer.py — Trainer orchestrator (Phase 09, Task 03).

Tests use --noconftest so all fixtures must be self-contained.
All external dependencies are mocked.
"""

import json
import os
import pickle
import pytest
from contextlib import contextmanager
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
        """config_path defaults to 'configs/' — the repo's config directory."""
        _set_env(monkeypatch)
        from training.trainer import Trainer
        t = Trainer()
        assert t.config_path == "configs/"

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

    def test_dispatches_nn_train(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "nn_train")
        t._run_train_nn = MagicMock()
        t.run()
        t._run_train_nn.assert_called_once()

    def test_dispatches_infer_nn(self, monkeypatch):
        t = self._make_trainer(monkeypatch, "infer_nn")
        t._run_infer_nn = MagicMock()
        t.run()
        t._run_infer_nn.assert_called_once()

    def test_dispatches_simulate_nn_alias_to_infer(self, monkeypatch):
        """Legacy RUN_TYPE 'simulate_nn' is an alias for infer_nn."""
        t = self._make_trainer(monkeypatch, "simulate_nn")
        t._run_infer_nn = MagicMock()
        t.run()
        t._run_infer_nn.assert_called_once()

    def test_group_nn_run_type_removed(self, monkeypatch):
        """The legacy 'group_nn' RUN_TYPE no longer dispatches → ValueError."""
        t = self._make_trainer(monkeypatch, "group_nn")
        with pytest.raises(ValueError, match="group_nn"):
            t.run()


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
        """ensure_data called with symbol, warmup-extended DATA_START, and DATA_END.

        Grab start is DATA_START minus the indicator warmup margin
        (105 rows × max(CANDLES)=1440 min = 105 days = 9_072_000_000 ms).
        """
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

        # PAIR "link_usdt" → "LINKUSDT"; start extended back by 105 days warmup
        mock_graber_instance.ensure_data.assert_called_once_with(
            "LINKUSDT", 1690928000000, 1700100000000
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
        t._run_infer_nn = MagicMock(side_effect=lambda: call_order.append("infer_nn"))

        t.run()

        # "full" = grab_data → prepare_data → simulate → train_nn → infer_nn → prepare_data
        assert call_order == [
            "grab_data",
            "prepare_data",
            "simulate",
            "train_nn",
            "infer_nn",
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
        t._run_infer_nn = MagicMock()

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
            "/cfg/indicators_config.yaml",
            "/data/wide.pkl",
            "/data/attrs.pkl",
        )

    def test_prepare_chunked_called_with_graber_data_path(self, monkeypatch):
        """_run_prepare_data calls prepare_chunked(graber_data_path(),
        data_start_ms=DATA_START, data_end_ms=DATA_END)."""
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

        # Chunked entry point: below CHUNK_MIN_ROWS it is the unchanged single
        # pass; indicators are computed from DATA_START on, warmup rows trimmed.
        mock_dp_instance.prepare_chunked.assert_called_once_with(
            "/data/graber.pkl",
            data_start_ms=1700000000000,
            data_end_ms=1700100000000,
        )


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
        _actions_file = tmp_path / "actions.jsonl"
        _actions_file.write_text("")
        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = {
            "results": [], "sim_id": 1,
            "actions_path": str(_actions_file), "action_count": 0,
        }
        mock_orch_cls = MagicMock(return_value=mock_orch_instance)
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"total_trades": 0}
        mock_analyzer_cls = MagicMock(return_value=mock_analyzer_instance)
        mock_report_instance = MagicMock()
        mock_report_instance.write.return_value = str(tmp_path / "report.json")
        mock_backtesting_report = MagicMock()
        mock_backtesting_report.SimulationReport = MagicMock(return_value=mock_report_instance)
        mock_backtesting_report.SimulationReport.action_counts_from_jsonl = MagicMock(return_value={})

        # Inject mock modules so lazy imports find them
        mock_data_module = MagicMock()
        mock_data_module.SimulationData = mock_sim_data_cls
        mock_backtesting_sim = MagicMock()
        mock_backtesting_sim.SimulationOrchestrator = mock_orch_cls
        mock_backtesting_perf = MagicMock()
        mock_backtesting_perf.PerformanceAnalyzer = mock_analyzer_cls
        mock_robots = MagicMock()

        _keys_to_mock = ["data", "backtesting.simulation_orchestrator",
                         "backtesting.performance_analyzer",
                         "backtesting.simulation_report", "robots.train_robot"]
        _saved = {k: sys.modules.get(k) for k in _keys_to_mock}
        sys.modules["data"] = mock_data_module
        sys.modules["backtesting.simulation_orchestrator"] = mock_backtesting_sim
        sys.modules["backtesting.performance_analyzer"] = mock_backtesting_perf
        sys.modules["backtesting.simulation_report"] = mock_backtesting_report
        sys.modules["robots.train_robot"] = mock_robots

        # Also patch pandas.read_pickle to avoid actual file access
        import pandas as real_pd
        mock_pd = MagicMock(wraps=real_pd)
        mock_pd.read_pickle.return_value = MagicMock()
        sys.modules["pandas"] = mock_pd

        try:
            with patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
                 patch("helpers.shared_folder", return_value=str(tmp_path) + "/"), \
                 patch("helpers.simulation_folder", return_value=str(tmp_path) + "/"):
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

    def test_simulation_window_not_warmup_extended(self, monkeypatch, tmp_path):
        """_run_simulate keeps begin_ts = DATA_START // 1000 — warmup margin
        applies only to the grab range, never to the simulation window."""
        import sys
        _set_env(monkeypatch, {"RUN_TYPE": "simulate"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        mock_sim_data = MagicMock()
        mock_sim_data_cls = MagicMock(return_value=mock_sim_data)
        _actions_file = tmp_path / "actions.jsonl"
        _actions_file.write_text("")
        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = {
            "results": [], "sim_id": 1,
            "actions_path": str(_actions_file), "action_count": 0,
        }
        mock_orch_cls = MagicMock(return_value=mock_orch_instance)
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"total_trades": 0}
        mock_analyzer_cls = MagicMock(return_value=mock_analyzer_instance)
        mock_report_instance = MagicMock()
        mock_report_instance.write.return_value = str(tmp_path / "report.json")
        mock_backtesting_report = MagicMock()
        mock_backtesting_report.SimulationReport = MagicMock(return_value=mock_report_instance)
        mock_backtesting_report.SimulationReport.action_counts_from_jsonl = MagicMock(return_value={})

        mock_data_module = MagicMock()
        mock_data_module.SimulationData = mock_sim_data_cls
        mock_backtesting_sim = MagicMock()
        mock_backtesting_sim.SimulationOrchestrator = mock_orch_cls
        mock_backtesting_perf = MagicMock()
        mock_backtesting_perf.PerformanceAnalyzer = mock_analyzer_cls
        mock_robots = MagicMock()

        _keys_to_mock = ["data", "backtesting.simulation_orchestrator",
                         "backtesting.performance_analyzer",
                         "backtesting.simulation_report", "robots.train_robot"]
        _saved = {k: sys.modules.get(k) for k in _keys_to_mock}
        sys.modules["data"] = mock_data_module
        sys.modules["backtesting.simulation_orchestrator"] = mock_backtesting_sim
        sys.modules["backtesting.performance_analyzer"] = mock_backtesting_perf
        sys.modules["backtesting.simulation_report"] = mock_backtesting_report
        sys.modules["robots.train_robot"] = mock_robots

        try:
            with patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
                 patch("helpers.shared_folder", return_value=str(tmp_path) + "/"), \
                 patch("helpers.simulation_folder", return_value=str(tmp_path) + "/"):
                t = mod.Trainer()
                t._run_simulate()
        finally:
            for key, original in _saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

        # DATA_START=1700000000000, DATA_END=1700100000000 (from _set_env)
        mock_sim_data_cls.assert_called_once_with(
            "link_usdt", 1700000000, 1700100000, 1
        )


class TestAvailableThreads:
    """Trainer.available_threads() reads NUM_WORKERS (fallback AVAIABLE_THREADS,
    default 4) — the orchestrator's from_trainer factory consumes it."""

    def _make(self, monkeypatch, env):
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", **env})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)
        return mod.Trainer()

    def test_defaults_to_4(self, monkeypatch):
        monkeypatch.delenv("NUM_WORKERS", raising=False)
        monkeypatch.delenv("AVAIABLE_THREADS", raising=False)
        t = self._make(monkeypatch, {})
        assert t.available_threads() == 4

    def test_reads_num_workers(self, monkeypatch):
        t = self._make(monkeypatch, {"NUM_WORKERS": "9"})
        assert t.available_threads() == 9

    def test_falls_back_to_available_threads(self, monkeypatch):
        monkeypatch.delenv("NUM_WORKERS", raising=False)
        t = self._make(monkeypatch, {"AVAIABLE_THREADS": "7"})
        assert t.available_threads() == 7

    def test_pair_property(self, monkeypatch):
        t = self._make(monkeypatch, {"PAIR": "btc_usdt"})
        assert t.pair == "btc_usdt"


class TestRunInferNN:
    """Tests for _run_infer_nn() (replaces _run_simulate_nn).

    Delegates dataset resolution + atomic write to
    NNOrchestrator.run_inference_dataset; Trainer no longer touches pandas
    or DataAttributes on this path.
    """

    @contextmanager
    def _injected(self, tmp_path, run_inference_dataset_return):
        import sys

        mock_orch_instance = MagicMock()
        mock_orch_instance.run_inference_dataset.return_value = (
            run_inference_dataset_return
        )
        mock_orch_cls = MagicMock()
        mock_orch_cls.from_trainer.return_value = mock_orch_instance

        mods = {"nn.nn_orchestrator": MagicMock(NNOrchestrator=mock_orch_cls)}
        saved = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            # wide_df_path() dir is the default dataset folder.
            with patch(
                "helpers.wide_df_path",
                return_value=str(tmp_path) + "/df_with_indicators.pkl",
            ):
                yield (mock_orch_cls, mock_orch_instance)
        finally:
            for key, original in saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

    def test_infer_nn_builds_orch_from_trainer(self, monkeypatch, tmp_path):
        """_run_infer_nn constructs the orchestrator via from_trainer(pair, self)."""
        import pandas as real_pd
        _set_env(monkeypatch, {"RUN_TYPE": "infer_nn"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        result = real_pd.DataFrame({"nn_res_x": [1, 2, 3]})
        with self._injected(tmp_path, result) as (orch_cls, orch_instance):
            t = mod.Trainer()
            t._run_infer_nn()

        orch_cls.from_trainer.assert_called_once_with(t.pair, t)
        orch_instance.run_inference_dataset.assert_called_once()

    def test_infer_nn_default_dataset_is_wide_df_dir(self, monkeypatch, tmp_path):
        """Default dataset = the dir CONTAINING df_with_indicators.pkl."""
        import pandas as real_pd
        _set_env(monkeypatch, {"RUN_TYPE": "infer_nn"})
        monkeypatch.delenv("NN_INFER_DATASET", raising=False)
        monkeypatch.delenv("NN_INFER_CHECKPOINT", raising=False)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, real_pd.DataFrame({"nn_res_x": [1]})) as (
            _,
            orch_instance,
        ):
            mod.Trainer()._run_infer_nn()

        kwargs = orch_instance.run_inference_dataset.call_args.kwargs
        assert kwargs["dataset_dir"] == str(tmp_path)
        assert kwargs["checkpoint_id"] == "best"

    def test_infer_nn_env_overrides(self, monkeypatch, tmp_path):
        """NN_INFER_DATASET / NN_INFER_CHECKPOINT override the defaults."""
        import pandas as real_pd
        _set_env(
            monkeypatch,
            {
                "RUN_TYPE": "infer_nn",
                "NN_INFER_DATASET": "/custom/ds",
                "NN_INFER_CHECKPOINT": "epoch_5",
            },
        )
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, real_pd.DataFrame({"nn_res_x": [1]})) as (
            _,
            orch_instance,
        ):
            mod.Trainer()._run_infer_nn()

        kwargs = orch_instance.run_inference_dataset.call_args.kwargs
        assert kwargs["dataset_dir"] == "/custom/ds"
        assert kwargs["checkpoint_id"] == "epoch_5"

    def test_infer_nn_absence_safe_metadata(self, monkeypatch, tmp_path):
        """result None → metadata records the absence without crashing."""
        _set_env(monkeypatch, {"RUN_TYPE": "infer_nn"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, None):
            t = mod.Trainer()
            t._run_infer_nn()  # must not raise

        assert "infer_nn" in t.metadata

    def test_infer_nn_empty_env_falls_back_to_wide_df_dir(self, monkeypatch, tmp_path):
        """Regression: NN_INFER_DATASET="" (empty string) must be treated as unset.

        Compose passes NN_INFER_DATASET= (empty string) by default.
        ``os.getenv("NN_INFER_DATASET", default)`` would return "" instead of the
        default, so the fix uses ``os.getenv(...) or <default>`` to treat "" as
        falsy and fall through to wide_df_path()-based default.

        This test FAILS without the fix (dataset_dir would be "") and passes with it.
        """
        import pandas as real_pd
        _set_env(monkeypatch, {"RUN_TYPE": "infer_nn", "NN_INFER_DATASET": ""})
        monkeypatch.delenv("NN_INFER_CHECKPOINT", raising=False)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, real_pd.DataFrame({"nn_res_x": [1]})) as (
            _,
            orch_instance,
        ):
            mod.Trainer()._run_infer_nn()

        kwargs = orch_instance.run_inference_dataset.call_args.kwargs
        # Must use the parent dir of wide_df_path(), NOT the empty string "".
        assert kwargs["dataset_dir"] == str(tmp_path), (
            f"Expected dataset_dir={str(tmp_path)!r} (wide_df dir), "
            f"got {kwargs['dataset_dir']!r}; empty NN_INFER_DATASET was not treated as unset"
        )
        assert kwargs["dataset_dir"] != "", "dataset_dir must not be empty string"

    def test_infer_nn_nonempty_env_is_honoured(self, monkeypatch, tmp_path):
        """A non-empty NN_INFER_DATASET=/some/path is used as-is (not overridden).

        Complements the empty-string regression test: ensures the fix does not
        accidentally swallow legitimate non-empty values.
        """
        import pandas as real_pd
        _set_env(
            monkeypatch,
            {"RUN_TYPE": "infer_nn", "NN_INFER_DATASET": "/explicit/dataset/dir"},
        )
        monkeypatch.delenv("NN_INFER_CHECKPOINT", raising=False)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, real_pd.DataFrame({"nn_res_x": [1]})) as (
            _,
            orch_instance,
        ):
            mod.Trainer()._run_infer_nn()

        kwargs = orch_instance.run_inference_dataset.call_args.kwargs
        assert kwargs["dataset_dir"] == "/explicit/dataset/dir", (
            f"Non-empty NN_INFER_DATASET was not honoured; got {kwargs['dataset_dir']!r}"
        )


class TestRunSimulateRealSignatures:
    """_run_simulate must construct collaborators with their REAL signatures:
    SimulationData(pair, begin_ts, end_ts, step_min) — seconds, loads its own
    pickle; StrategyManager(fee) from strategies.strategy_manager."""

    @contextmanager
    def _injected(self, tmp_path):
        import sys

        _actions_file = tmp_path / "actions.jsonl"
        _actions_file.write_text("")
        mock_sim_data_cls = MagicMock(return_value=MagicMock())
        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = {
            "results": [], "sim_id": 1,
            "actions_path": str(_actions_file), "action_count": 0,
        }
        mock_orch_cls = MagicMock(return_value=mock_orch_instance)
        mock_analyzer_cls = MagicMock(
            return_value=MagicMock(analyze=MagicMock(return_value={"total_trades": 0}))
        )
        mock_strategy_manager_cls = MagicMock(return_value=MagicMock())
        mock_report_cls = MagicMock(
            return_value=MagicMock(write=MagicMock(return_value=str(tmp_path / "report.json")))
        )
        mock_report_cls.action_counts_from_jsonl = MagicMock(return_value={})

        mods = {
            "data": MagicMock(SimulationData=mock_sim_data_cls),
            "backtesting.simulation_orchestrator": MagicMock(
                SimulationOrchestrator=mock_orch_cls
            ),
            "backtesting.performance_analyzer": MagicMock(
                PerformanceAnalyzer=mock_analyzer_cls
            ),
            "backtesting.simulation_report": MagicMock(
                SimulationReport=mock_report_cls
            ),
            "strategies.strategy_manager": MagicMock(
                StrategyManager=mock_strategy_manager_cls
            ),
        }
        saved = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            with patch("helpers.shared_folder", return_value=str(tmp_path) + "/"), \
                 patch("helpers.simulation_folder", return_value=str(tmp_path) + "/"):
                yield (mock_sim_data_cls, mock_orch_cls, mock_strategy_manager_cls)
        finally:
            for key, original in saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

    def test_simulation_data_gets_pair_and_window_seconds(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, {"RUN_TYPE": "simulate"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as (sim_data_cls, _, __):
            mod.Trainer()._run_simulate()

        # env DATA_START/DATA_END are epoch ms; SimulationData wants seconds
        sim_data_cls.assert_called_once_with("link_usdt", 1700000000, 1700100000, 1)

    def test_strategy_factory_builds_strategy_manager_with_fee(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, {"RUN_TYPE": "simulate", "EXCHANGE_FEE": "0.002"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as (_, orch_cls, strategy_manager_cls):
            mod.Trainer()._run_simulate()

            factory = orch_cls.call_args.kwargs["strategy_factory"]
            factory()

        strategy_manager_cls.assert_called_once_with(0.002)
        assert orch_cls.call_args.kwargs["fee"] == 0.002


class TestRunTrainNNSingleMode:
    """NN_TRAIN_MODE=single preserves the OLD single-shot behaviour.

    _run_train_nn builds the orchestrator via NNOrchestrator.from_trainer and
    calls orch.train(...) directly with a group-keyed epoch callback that pickles
    a {"phase": "nn_train", "group": ..., "epoch": ..., "metrics": ...} state.
    The TrainingLoop is NOT used on this path.
    """

    @contextmanager
    def _injected(self, tmp_path):
        import sys

        mock_orch_instance = MagicMock()
        mock_orch_instance.train.return_value = {"all": {"loss": 0.1}}
        mock_orch_cls = MagicMock()
        mock_orch_cls.from_trainer.return_value = mock_orch_instance

        # If the loop is (incorrectly) reached, this MagicMock lets us assert it
        # was NOT constructed/run on the single-shot path.
        mock_loop_cls = MagicMock()

        mock_indicators_mod = MagicMock()
        mock_indicators_mod.DataAttributes.load.return_value = MagicMock()

        mods = {
            "nn.nn_orchestrator": MagicMock(NNOrchestrator=mock_orch_cls),
            "nn.training_loop": MagicMock(TrainingLoop=mock_loop_cls),
            "indicators": mock_indicators_mod,
        }
        saved = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            with patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
                 patch("helpers.data_attributes_path", return_value="/data/attrs.pkl"), \
                 patch("helpers.shared_folder", return_value=str(tmp_path) + "/"), \
                 patch("pandas.read_pickle", return_value=MagicMock()):
                yield (mock_orch_cls, mock_orch_instance, mock_loop_cls)
        finally:
            for key, original in saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

    def test_single_mode_constructs_orchestrator_from_trainer(
        self, monkeypatch, tmp_path
    ):
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "single"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as (orch_cls, orch_instance, loop_cls):
            t = mod.Trainer(config_path="configs/")
            t._run_train_nn()

        orch_cls.from_trainer.assert_called_once_with(t.pair, t)
        orch_instance.train.assert_called_once()
        # The TrainingLoop is NOT used on the single-shot path.
        loop_cls.assert_not_called()
        # No legacy NNOrchestrator(checkpoint_dir=..., feature_cols=...) call.
        orch_cls.assert_not_called()

    def test_single_mode_metadata_is_group_keyed(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "single"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path):
            t = mod.Trainer(config_path="configs/")
            t._run_train_nn()

        assert t.metadata["train_nn"] == {"all": {"loss": 0.1}}

    def test_single_mode_epoch_callback_is_group_keyed(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "single"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as (_, orch_instance, __):
            mod.Trainer(config_path="configs/")._run_train_nn()
            cb = orch_instance.train.call_args.kwargs["epoch_callback"]
            cb("all", 3, {"loss": 0.5})  # contract: (group_key, epoch, metrics)

        state_path = str(tmp_path) + "/training_state.pkl"
        assert os.path.exists(state_path)
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        assert state["phase"] == "nn_train"
        assert state["group"] == "all"
        assert state["epoch"] == 3
        assert state["metrics"] == {"loss": 0.5}
        assert "tf" not in state


class TestRunTrainNNSearchMode:
    """NN_TRAIN_MODE=search (the DEFAULT) runs the agentic TrainingLoop.

    _run_train_nn builds the orchestrator via from_trainer, resolves a
    search_config (configs/nn_search.yaml), constructs an ExperimentTracker and a
    TrainingLoop, and calls loop.run(df, data_attributes). study_name defaults to
    "{pair}_{spec_hash[:8]}" and NN_STUDY overrides it; NN_STRATEGIST toggles the
    strategist arg between None and an NNStrategist.
    """

    @contextmanager
    def _injected(self, tmp_path, spec_hash="abcdef0123456789", best=None):
        import sys

        if best is None:
            best = {"all": {"holdout": {"holdout_score": 0.7}}}

        # Orchestrator with a base_spec exposing a spec_hash.
        mock_base_spec = MagicMock()
        mock_base_spec.spec_hash = spec_hash
        mock_orch_instance = MagicMock()
        mock_orch_instance.base_spec = mock_base_spec
        mock_orch_cls = MagicMock()
        mock_orch_cls.from_trainer.return_value = mock_orch_instance

        # ExperimentTracker — best() returns a summary-able incumbent map.
        mock_tracker_instance = MagicMock()
        mock_tracker_instance.best.return_value = best
        mock_tracker_cls = MagicMock(return_value=mock_tracker_instance)

        # TrainingLoop — run() returns a RunResult-like object.
        mock_result = MagicMock()
        mock_result.best = best
        mock_result.study_name = "study-x"
        mock_result.rounds_run = 2
        mock_result.trials_run = 6
        mock_loop_instance = MagicMock()
        mock_loop_instance.run.return_value = mock_result
        mock_loop_cls = MagicMock(return_value=mock_loop_instance)

        # NNStrategist — constructor captured so we can assert it is/ isn't used.
        mock_strategist_instance = MagicMock()
        mock_strategist_cls = MagicMock(return_value=mock_strategist_instance)

        # device.nn_artefact_root(pair) → a tracking root.
        mock_device_mod = MagicMock()
        mock_device_mod.nn_artefact_root.return_value = str(tmp_path / "artefacts")

        mock_indicators_mod = MagicMock()
        mock_indicators_mod.DataAttributes.load.return_value = MagicMock()

        mods = {
            "nn.nn_orchestrator": MagicMock(NNOrchestrator=mock_orch_cls),
            "nn.experiment_tracker": MagicMock(
                ExperimentTracker=mock_tracker_cls
            ),
            "nn.training_loop": MagicMock(TrainingLoop=mock_loop_cls),
            "nn.nn_strategist": MagicMock(NNStrategist=mock_strategist_cls),
            "nn.device": mock_device_mod,
            "indicators": mock_indicators_mod,
        }
        saved = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            with patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
                 patch("helpers.data_attributes_path", return_value="/data/attrs.pkl"), \
                 patch("helpers.shared_folder", return_value=str(tmp_path) + "/"), \
                 patch("pandas.read_pickle", return_value=MagicMock()):
                yield {
                    "orch_cls": mock_orch_cls,
                    "orch": mock_orch_instance,
                    "tracker_cls": mock_tracker_cls,
                    "tracker": mock_tracker_instance,
                    "loop_cls": mock_loop_cls,
                    "loop": mock_loop_instance,
                    "strategist_cls": mock_strategist_cls,
                    "result": mock_result,
                }
        finally:
            for key, original in saved.items():
                if original is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original

    def test_search_mode_is_default(self, monkeypatch, tmp_path):
        """No NN_TRAIN_MODE env → the loop path runs (search is the default)."""
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train"})
        monkeypatch.delenv("NN_TRAIN_MODE", raising=False)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as m:
            mod.Trainer(config_path="configs/")._run_train_nn()

        m["loop_cls"].assert_called_once()
        m["loop"].run.assert_called_once()
        # Single-shot orch.train is NOT called directly by the search path.
        m["orch"].train.assert_not_called()

    def test_search_mode_constructs_tracker_and_loop(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "search"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as m:
            t = mod.Trainer(config_path="configs/")
            t._run_train_nn()

        m["orch_cls"].from_trainer.assert_called_once_with(t.pair, t)
        m["tracker_cls"].assert_called_once()
        m["loop_cls"].assert_called_once()
        # loop.run is called with (df, data_attributes).
        assert m["loop"].run.call_count == 1
        # TrainingLoop constructed with orchestrator/tracker/strategist/search_config.
        loop_kwargs = m["loop_cls"].call_args.kwargs
        assert loop_kwargs["orchestrator"] is m["orch"]
        assert loop_kwargs["tracker"] is m["tracker"]
        assert isinstance(loop_kwargs["search_config"], dict)

    def test_default_study_name_derivation(self, monkeypatch, tmp_path):
        """study_name defaults to f"{pair}_{spec_hash[:8]}"."""
        _set_env(
            monkeypatch,
            {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "search", "PAIR": "link_usdt"},
        )
        monkeypatch.delenv("NN_STUDY", raising=False)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, spec_hash="deadbeefcafef00d") as m:
            mod.Trainer(config_path="configs/")._run_train_nn()

        # Tracker constructed with the default study name.
        args, kwargs = m["tracker_cls"].call_args
        # ExperimentTracker(tracking_dir, study_name, metric=..., margin=..., mode=...)
        study_name = kwargs.get("study_name", args[1] if len(args) > 1 else None)
        assert study_name == "link_usdt_deadbeef"

    def test_nn_study_env_overrides_study_name(self, monkeypatch, tmp_path):
        """NN_STUDY env overrides the derived default."""
        _set_env(
            monkeypatch,
            {
                "RUN_TYPE": "nn_train",
                "NN_TRAIN_MODE": "search",
                "NN_STUDY": "my_custom_study",
            },
        )
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as m:
            mod.Trainer(config_path="configs/")._run_train_nn()

        args, kwargs = m["tracker_cls"].call_args
        study_name = kwargs.get("study_name", args[1] if len(args) > 1 else None)
        assert study_name == "my_custom_study"

    def test_invalid_study_name_with_separator_is_rejected(
        self, monkeypatch, tmp_path
    ):
        """A study_name containing a path separator is rejected early."""
        _set_env(
            monkeypatch,
            {
                "RUN_TYPE": "nn_train",
                "NN_TRAIN_MODE": "search",
                "NN_STUDY": "bad/name",
            },
        )
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path):
            with pytest.raises(ValueError):
                mod.Trainer(config_path="configs/")._run_train_nn()

    def test_strategist_off_by_default(self, monkeypatch, tmp_path):
        """No NN_STRATEGIST → strategist arg is None (pure-Optuna)."""
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "search"})
        monkeypatch.delenv("NN_STRATEGIST", raising=False)
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as m:
            mod.Trainer(config_path="configs/")._run_train_nn()

        loop_kwargs = m["loop_cls"].call_args.kwargs
        assert loop_kwargs["strategist"] is None
        m["strategist_cls"].assert_not_called()

    def test_strategist_on_when_env_set(self, monkeypatch, tmp_path):
        """NN_STRATEGIST=1 → an NNStrategist is constructed and passed in."""
        _set_env(
            monkeypatch,
            {
                "RUN_TYPE": "nn_train",
                "NN_TRAIN_MODE": "search",
                "NN_STRATEGIST": "1",
            },
        )
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as m:
            mod.Trainer(config_path="configs/")._run_train_nn()

        m["strategist_cls"].assert_called_once()
        loop_kwargs = m["loop_cls"].call_args.kwargs
        assert loop_kwargs["strategist"] is m["strategist_cls"].return_value

    def test_search_mode_writes_training_state_summary(self, monkeypatch, tmp_path):
        """A final training_state.pkl summary is written for the search path."""
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "search"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path):
            mod.Trainer(config_path="configs/")._run_train_nn()

        state_path = str(tmp_path) + "/training_state.pkl"
        assert os.path.exists(state_path)
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        assert state["phase"] == "nn_train"
        assert "study" in state
        assert "best" in state

    def test_search_mode_sets_light_metadata(self, monkeypatch, tmp_path):
        """metadata['train_nn'] is a light summary (study_name + best)."""
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "search"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path) as m:
            t = mod.Trainer(config_path="configs/")
            t._run_train_nn()

        meta = t.metadata["train_nn"]
        assert isinstance(meta, dict)
        assert "study_name" in meta

    def test_search_mode_degenerate_result_does_not_crash(
        self, monkeypatch, tmp_path
    ):
        """An empty tracker.best()/RunResult must not crash the handler."""
        _set_env(monkeypatch, {"RUN_TYPE": "nn_train", "NN_TRAIN_MODE": "search"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        with self._injected(tmp_path, best={}) as m:
            m["result"].best = {}
            m["tracker"].best.return_value = {}
            t = mod.Trainer(config_path="configs/")
            t._run_train_nn()  # must not raise

        assert "train_nn" in t.metadata


class TestStrategyFactoryPicklable:
    """SimulationOrchestrator runs workers via ProcessPoolExecutor — the
    strategy_factory crosses the process boundary and must be picklable.
    A closure inside _run_simulate is not."""

    def test_strategy_factory_pickles(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, {"RUN_TYPE": "simulate"})
        import importlib
        import training.trainer as mod
        importlib.reload(mod)

        helper = TestRunSimulateRealSignatures()
        with helper._injected(tmp_path) as (_, orch_cls, __):
            mod.Trainer()._run_simulate()
            factory = orch_cls.call_args.kwargs["strategy_factory"]

        pickle.dumps(factory)  # must not raise
