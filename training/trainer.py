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


class _DefaultStrategyFactory:
    """Picklable StrategyManager factory.

    SimulationOrchestrator sends the factory to worker processes via
    ProcessPoolExecutor — a closure can't cross that boundary.
    """

    def __init__(self, fee: float) -> None:
        self.fee = fee

    def __call__(self):
        from strategies.strategy_manager import StrategyManager  # lazy
        return StrategyManager(self.fee)


class Trainer:
    """Top-level pipeline orchestrator.

    Args:
        config_path: Path to the config directory.  Defaults to ``"configs/"``.
                     Override in tests to avoid touching real config files.

    Attributes:
        config_path: Path to config directory.
        run_type:    Value of the ``RUN_TYPE`` environment variable read at
                     construction time.
        metadata:    Dict accumulated across pipeline stages for logging.
    """

    def __init__(self, config_path: str = "configs/") -> None:
        self.config_path = config_path
        self.run_type: str = os.environ["RUN_TYPE"]
        self.metadata: dict = {}

    @property
    def pair(self) -> str:
        """Trading pair (e.g. ``"link_usdt"``) from the PAIR env var."""
        return os.environ["PAIR"]

    def available_threads(self) -> int:
        """Worker count for the NN orchestrator.

        Reads NUM_WORKERS (fallback the misspelled-but-historical
        AVAIABLE_THREADS, default 4). Consumed by
        ``NNOrchestrator.from_trainer(pair, self)``.
        """
        return int(
            os.environ.get("NUM_WORKERS", os.environ.get("AVAIABLE_THREADS", "4"))
        )

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
            "nn_train": self._run_train_nn,
            "infer_nn": self._run_infer_nn,
            "simulate_nn": self._run_infer_nn,  # legacy alias → infer_nn
        }

        if self.run_type == "full":
            self._run_grab_data()
            self._run_prepare_data()
            self._run_simulate()
            self._run_train_nn()
            self._run_infer_nn()
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
        from config_loader import warmup_start_ms  # lazy
        from helpers import graber_data_path  # lazy
        from stocks_holder import do_stock_init, stock_holder  # lazy
        from training.graber import Graber  # lazy

        stock_type = os.environ.get("STOCK_TYPE", "binance")
        do_stock_init(stock_type)
        stock = stock_holder.item

        graber = Graber(stock, output_path=graber_data_path())

        pair = os.environ["PAIR"]
        symbol = pair.replace("_", "").upper()
        # Grab starts earlier than DATA_START so indicators have full warmup
        # history at the simulation start date (sim window stays DATA_START).
        start_ms = warmup_start_ms(int(os.environ["DATA_START"]))
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

        # DataPreparer wants the path to indicators_config.yaml (a file),
        # not the config directory.
        preparer = DataPreparer(
            self.config_path + "indicators_config.yaml",
            wide_df_path(),
            data_attributes_path(),
            nn_output_path=_nn_output_path(),
        )
        # Indicators are computed from DATA_START on; the warmup head grabbed
        # by _run_grab_data is lookback input only and is trimmed before save.
        preparer.prepare(
            graber_data_path(), data_start_ms=int(os.environ["DATA_START"])
        )

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
        from backtesting.simulation_report import SimulationReport  # lazy
        from data import SimulationData  # lazy
        from helpers import shared_folder, simulation_folder  # lazy

        pair = os.environ["PAIR"]
        begin_ts = int(os.environ["DATA_START"]) // 1000
        end_ts = int(os.environ["DATA_END"]) // 1000
        step_min = int(os.environ.get("STEP_MIN", "1"))

        simulation_data = SimulationData(pair, begin_ts, end_ts, step_min)

        fee = float(os.environ.get("FEE", os.environ.get("EXCHANGE_FEE", "0.001")))

        # Select strategy set: STRATEGY_SET=test registers the Phase 14 test
        # strategies; otherwise the default (empty) factory.
        strategy_set = os.environ.get("STRATEGY_SET", "default")
        if strategy_set == "test":
            from strategies.test_factory import TestStrategyFactory  # lazy
            factory = TestStrategyFactory(fee)
        elif strategy_set == "ema":
            from strategies.test_factory import EmaStrategyFactory  # lazy
            factory = EmaStrategyFactory(fee)
        else:
            factory = _DefaultStrategyFactory(fee)

        num_workers = int(os.environ.get("NUM_WORKERS", "4"))
        orch = SimulationOrchestrator(strategy_factory=factory, fee=fee)
        run_result = orch.run(simulation_data)

        results = run_result["results"]
        sim_id = run_result["sim_id"]

        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.analyze()

        # Build and write the per-simulation report beside actions.jsonl.
        with open(run_result["actions_path"]) as f:
            actions_jsonl = f.read()
        action_counts = SimulationReport.action_counts_from_jsonl(actions_jsonl)
        context = {
            "strategy_set": strategy_set,
            "pair": pair,
            "begin_ts": begin_ts,
            "end_ts": end_ts,
            "step_min": step_min,
            "num_workers": num_workers,
            "fee": fee,
        }
        report_path = SimulationReport(
            sim_id, metrics, action_counts, context
        ).write(simulation_folder(sim_id))

        # Write training state checkpoint
        shared = shared_folder()
        os.makedirs(shared, exist_ok=True)
        state_path = shared + "training_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump({"phase": "simulate", "metrics": metrics}, f)

        self.metadata["simulate"] = {
            **metrics,
            "sim_id": sim_id,
            "report_path": report_path,
            "action_count": run_result["action_count"],
        }

    def _run_train_nn(self) -> None:
        """Train the NN via NNOrchestrator.from_trainer (nn/, Phase 11).

        The orchestrator owns all architecture/grouping/timeframe/target
        knowledge (configs/nn_spec.yaml). Trainer only loads the dataset df +
        DataAttributes and pickles group-keyed epoch progress.
        """
        import pandas as pd  # lazy
        from helpers import data_attributes_path, shared_folder, wide_df_path  # lazy
        from indicators import DataAttributes  # lazy
        from nn.nn_orchestrator import NNOrchestrator  # lazy

        df = pd.read_pickle(wide_df_path())
        data_attributes = DataAttributes.load(data_attributes_path())

        orch = NNOrchestrator.from_trainer(self.pair, self)

        shared = shared_folder()
        os.makedirs(shared, exist_ok=True)
        state_path = shared + "training_state.pkl"

        # Orchestrator contract (v3.0): epoch_callback(group_key, epoch, metrics).
        def epoch_callback(group_key: str, epoch: int, metrics: dict) -> None:
            with open(state_path, "wb") as f:
                pickle.dump(
                    {
                        "phase": "nn_train",
                        "group": group_key,
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    f,
                )

        train_metrics = orch.train(df, data_attributes, epoch_callback=epoch_callback)

        self.metadata["train_nn"] = train_metrics or {}

    def _run_infer_nn(self) -> None:
        """Run NN inference over a dataset folder (Phase 11).

        The orchestrator loads df_with_indicators.pkl from the target dataset,
        scores it, and atomically writes {dataset}/df_with_nn.pkl (absence-safe;
        never mutates df_with_indicators.pkl). Trainer only resolves the dataset
        dir + checkpoint id and records a light summary.
        """
        from helpers import wide_df_path  # lazy
        from nn.nn_orchestrator import NNOrchestrator  # lazy

        # Default dataset = the dir CONTAINING df_with_indicators.pkl.
        dataset = os.getenv("NN_INFER_DATASET", os.path.dirname(wide_df_path()))
        checkpoint_id = os.getenv("NN_INFER_CHECKPOINT", "best")

        orch = NNOrchestrator.from_trainer(self.pair, self)
        result = orch.run_inference_dataset(
            dataset_dir=dataset, checkpoint_id=checkpoint_id
        )

        if result is None:
            self.metadata["infer_nn"] = {
                "dataset": dataset,
                "checkpoint_id": checkpoint_id,
                "written": False,
            }
        else:
            self.metadata["infer_nn"] = {
                "dataset": dataset,
                "checkpoint_id": checkpoint_id,
                "written": True,
                "rows": len(result),
                "columns": list(result.columns),
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
