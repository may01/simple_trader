"""training/trainer.py — Trainer: top-level orchestrator for the training pipeline.

Dispatches to the correct pipeline path based on the RUN_TYPE env var.
No trading logic here — pure orchestration and dispatch.

All domain imports are deferred to method bodies so that:
1. ``import talib`` is never triggered at module import time (avoids test-time issues).
2. NNOrchestrator can be absent without breaking the rest of the pipeline.
"""

from __future__ import annotations

import json
import os
import pickle


def _nn_output_path() -> str:
    """Return the path for df_with_nn.pkl inside nn_folder()."""
    from helpers import nn_folder  # lazy — reads env at call time
    return nn_folder() + "df_with_nn.pkl"


class Trainer:
    """Top-level pipeline orchestrator.

    Args:
        config_path: Path to the config directory.  Defaults to ``"config/"``.
                     Override in tests to avoid touching real config files.

    Attributes:
        config_path: Path to config directory.
        run_type:    Value of the ``RUN_TYPE`` environment variable read at
                     construction time.
        metadata:    Dict accumulated across pipeline stages for logging.
    """

    def __init__(self, config_path: str = "config/") -> None:
        self.config_path = config_path
        self.run_type: str = os.environ["RUN_TYPE"]
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Dispatch to the correct pipeline method based on self.run_type.

        Raises:
            ValueError: If self.run_type is not one of the known values.
        """
        dispatch = {
            "grab_data": self._run_grab_data,
            "prepare_data": self._run_prepare_data,
            "simulate": self._run_simulate,
            "train_nn": self._run_train_nn,
            "simulate_nn": self._run_simulate_nn,
        }

        if self.run_type == "full":
            self._run_grab_data()
            self._run_prepare_data()
            self._run_simulate()
            self._run_train_nn()
            self._run_simulate_nn()
            self._run_prepare_data()
            return

        if self.run_type not in dispatch:
            raise ValueError(
                f"Unknown RUN_TYPE: {self.run_type!r}. "
                f"Expected one of: {sorted(dispatch) + ['full']}"
            )

        dispatch[self.run_type]()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_grab_data(self) -> None:
        """Ensure raw OHLCV data is present and up-to-date.

        Reads PAIR, DATA_START, DATA_END from environment.
        Converts PAIR "link_usdt" → symbol "LINKUSDT".
        """
        from helpers import graber_data_path  # lazy
        from stocks_holder import do_stock_init, stock_holder  # lazy
        from training.graber import Graber  # lazy

        stock_type = os.environ.get("STOCK_TYPE", "binance")
        do_stock_init(stock_type)
        stock = stock_holder.item

        graber = Graber(stock, output_path=graber_data_path())

        pair = os.environ["PAIR"]
        symbol = pair.replace("_", "").upper()
        start_ms = int(os.environ["DATA_START"])
        end_ms = int(os.environ["DATA_END"])

        graber.ensure_data(symbol, start_ms, end_ms)

        self.metadata["grab_data"] = {
            "symbol": symbol,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

    def _run_prepare_data(self) -> None:
        """Run the data-preparation pipeline (raw OHLCV → indicators → attributes)."""
        from helpers import (  # lazy
            data_attributes_path,
            graber_data_path,
            wide_df_path,
        )
        from training.data_preparer import DataPreparer  # lazy

        preparer = DataPreparer(
            self.config_path,
            wide_df_path(),
            data_attributes_path(),
            nn_output_path=_nn_output_path(),
        )
        preparer.prepare(graber_data_path())

        self.metadata["prepare_data"] = {"status": "complete"}

    def _run_simulate(self) -> None:
        """Run the backtesting simulation and analyze results.

        SimulationData loads df_with_indicators.pkl itself (by pair); the
        simulation window comes from DATA_START/DATA_END (epoch ms in env,
        seconds for SimulationData). Writes training_state.pkl to
        shared_folder().
        """
        from backtesting.performance_analyzer import PerformanceAnalyzer  # lazy
        from backtesting.simulation_orchestrator import SimulationOrchestrator  # lazy
        from data import SimulationData  # lazy
        from helpers import shared_folder  # lazy

        pair = os.environ["PAIR"]
        begin_ts = int(os.environ["DATA_START"]) // 1000
        end_ts = int(os.environ["DATA_END"]) // 1000
        step_min = int(os.environ.get("STEP_MIN", "1"))

        simulation_data = SimulationData(pair, begin_ts, end_ts, step_min)

        fee = float(os.environ.get("FEE", os.environ.get("EXCHANGE_FEE", "0.001")))

        def _default_strategy_factory():
            from strategies.strategy_manager import StrategyManager  # lazy
            return StrategyManager(fee)

        orch = SimulationOrchestrator(
            strategy_factory=_default_strategy_factory,
            fee=fee,
        )
        results = orch.run(simulation_data)

        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.analyze()

        # Write training state checkpoint
        shared = shared_folder()
        os.makedirs(shared, exist_ok=True)
        state_path = shared + "training_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump({"phase": "simulate", "metrics": metrics}, f)

        self.metadata["simulate"] = metrics

    def _run_train_nn(self) -> None:
        """Train the neural network using NNOrchestrator.

        NNOrchestrator is imported lazily so a missing module does not break
        other run types.
        """
        try:
            from training.nn_orchestrator import NNOrchestrator  # noqa: F401 lazy
        except ModuleNotFoundError:
            raise RuntimeError(
                "NNOrchestrator is not yet implemented (Phase 11). "
                "Cannot run run_type='train_nn'."
            )

        import pandas as pd  # lazy
        from config_loader import CANDLES as tfs  # lazy
        from helpers import data_attributes_path, shared_folder, wide_df_path  # lazy
        from indicators import DataAttributes  # lazy
        from training.nn_orchestrator import NNOrchestrator  # now safe to import

        df = pd.read_pickle(wide_df_path())
        data_attributes = DataAttributes.load(data_attributes_path())

        orch = NNOrchestrator(self.config_path)

        shared = shared_folder()
        os.makedirs(shared, exist_ok=True)
        state_path = shared + "training_state.pkl"

        def epoch_callback(epoch: int, metrics: dict) -> None:
            with open(state_path, "wb") as f:
                pickle.dump({"phase": "nn_train", "epoch": epoch, **metrics}, f)

        train_metrics = orch.train(df, data_attributes, tfs, epoch_callback=epoch_callback)

        self.metadata["train_nn"] = train_metrics or {}

    def _run_simulate_nn(self) -> None:
        """Run NN inference and save the result to nn_output_path atomically.

        NNOrchestrator is imported lazily so a missing module does not break
        other run types.
        """
        try:
            from training.nn_orchestrator import NNOrchestrator  # noqa: F401 lazy
        except ModuleNotFoundError:
            raise RuntimeError(
                "NNOrchestrator is not yet implemented (Phase 11). "
                "Cannot run run_type='simulate_nn'."
            )

        import pandas as pd  # lazy
        from config_loader import CANDLES as tfs  # lazy
        from helpers import data_attributes_path, wide_df_path  # lazy
        from indicators import DataAttributes  # lazy
        from training.nn_orchestrator import NNOrchestrator  # now safe to import

        df = pd.read_pickle(wide_df_path())
        data_attributes = DataAttributes.load(data_attributes_path())

        orch = NNOrchestrator(self.config_path)
        nn_df = orch.run_inference(df, data_attributes, tfs)

        out_path = _nn_output_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path + ".tmp"
        nn_df.to_pickle(tmp_path)
        os.rename(tmp_path, out_path)

        self.metadata["simulate_nn"] = {
            "rows": len(nn_df),
            "columns": list(nn_df.columns),
        }

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _save_metadata(self, path: str) -> None:
        """Write self.metadata as JSON to *path*.

        Args:
            path: Filesystem path for the output JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=2)
