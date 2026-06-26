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
        )
        # Indicators are computed from DATA_START on; the warmup head grabbed
        # by _run_grab_data is lookback input only and is trimmed before save.
        # prepare_chunked splits big windows into resumable, progress-logged
        # time portions; below CHUNK_MIN_ROWS it is the unchanged single pass.
        preparer.prepare_chunked(
            graber_data_path(),
            data_start_ms=int(os.environ["DATA_START"]),
            data_end_ms=int(os.environ["DATA_END"]),
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
        """Train the NN (nn/, Phase 11).

        Two modes, selected by ``NN_TRAIN_MODE`` (default ``"search"``):

        - ``"search"`` (default): NN training is a *search for a better model*.
          The agentic ``TrainingLoop`` (Optuna sampling/pruning + an OPTIONAL
          ``NNStrategist`` LLM) trains many candidate specs and promotes the best
          on a time-ordered holdout via the ``ExperimentTracker``. This is the
          "investigation" capability.
        - ``"single"``: the PROVEN single-shot path — one ``orch.train`` over the
          base spec with a group-keyed epoch callback. Kept reachable so the
          minimal training behaviour is always available.

        The orchestrator owns all architecture/grouping/timeframe/target
        knowledge (configs/nn_spec.yaml); the search_config (configs/nn_search.yaml)
        owns the search caps + numeric bounds. All NN imports stay lazy so the
        base image (no optuna) can import this module and run non-NN RUN_TYPEs.
        """
        import pandas as pd  # lazy
        from helpers import data_attributes_path, shared_folder, wide_df_path  # lazy
        from indicators import DataAttributes  # lazy
        from nn.nn_orchestrator import NNOrchestrator  # lazy

        df = pd.read_pickle(wide_df_path())
        data_attributes = DataAttributes.load(data_attributes_path())

        orch = NNOrchestrator.from_trainer(self.pair, self)

        mode = os.getenv("NN_TRAIN_MODE", "search").strip().lower()
        if mode == "single":
            self._run_train_nn_single(orch, df, data_attributes)
        else:
            self._run_train_nn_search(orch, df, data_attributes)

    def _run_train_nn_single(self, orch, df, data_attributes) -> None:
        """Single-shot NN training (NN_TRAIN_MODE=single): one orch.train pass.

        Preserves the proven behaviour: train the base spec once with a
        group-keyed epoch callback that pickles
        ``{"phase": "nn_train", "group", "epoch", "metrics"}`` progress.
        """
        from helpers import shared_folder  # lazy

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

    def _run_train_nn_search(self, orch, df, data_attributes) -> None:
        """Agentic NN search (NN_TRAIN_MODE=search, the default).

        Wires the hybrid ``TrainingLoop`` (Optuna + optional ``NNStrategist``):
        resolve the search_config (configs/nn_search.yaml), construct an
        ExperimentTracker over ``{nn_artefact_root(pair)}/tracking``, build the
        loop, and ``loop.run(df, data_attributes)``. The loop OWNS orchestration;
        it never derives/mutates ``study_name`` (resolved here).

        Everything is guarded so a degenerate result never crashes nn_train.
        """
        from helpers import shared_folder  # lazy
        from nn.device import nn_artefact_root  # lazy
        from nn.experiment_tracker import ExperimentTracker  # lazy
        from nn.training_loop import TrainingLoop  # lazy (optuna stays inside run)

        search_config = self._load_nn_search_config()

        # study_name: NN_STUDY override, else "{pair}_{spec_hash[:8]}". The tracker
        # OWNS the name; we validate it as a filesystem-safe slug before disk I/O.
        study_name = os.getenv("NN_STUDY") or (
            f"{self.pair}_{orch.base_spec.spec_hash[:8]}"
        )
        if any(sep in study_name for sep in ("/", "\\", os.sep)) or study_name in (
            "",
            ".",
            "..",
        ):
            raise ValueError(
                f"NN_STUDY must be a filesystem-safe slug (no path separators); "
                f"got {study_name!r}"
            )

        tracking_dir = f"{nn_artefact_root(self.pair)}/tracking"
        margin = float(search_config.get("margin", 0.0) or 0.0)
        tracker = ExperimentTracker(
            tracking_dir,
            study_name,
            metric="holdout_score",
            margin=margin,
            mode="max",
        )

        # Strategist OFF by default (pure-Optuna); enable via NN_STRATEGIST.
        strategist = None
        if os.getenv("NN_STRATEGIST", "").strip().lower() in ("1", "true", "on"):
            from nn.nn_strategist import NNStrategist  # lazy (anthropic optional)

            strategist = NNStrategist(
                search_config=search_config.get("search_space", {})
            )

        loop = TrainingLoop(
            orchestrator=orch,
            tracker=tracker,
            strategist=strategist,
            search_config=search_config,
        )
        result = loop.run(df, data_attributes)

        # Final summary state (guarded — a degenerate result must not crash).
        try:
            best = tracker.best() or {}
        except Exception:  # noqa: BLE001 — never let summary I/O abort the run
            best = {}

        shared = shared_folder()
        os.makedirs(shared, exist_ok=True)
        state_path = shared + "training_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(
                {"phase": "nn_train", "study": study_name, "best": best},
                f,
            )

        self.metadata["train_nn"] = {
            "mode": "search",
            "study_name": study_name,
            "best": best,
            "n_trials": int(getattr(result, "trials_run", 0) or 0),
            "rounds_run": int(getattr(result, "rounds_run", 0) or 0),
        }

    def _load_nn_search_config(self) -> dict:
        """Load configs/nn_search.yaml and apply trivial env cap overrides.

        Env caps (when set) override the file: NN_MAX_ROUNDS,
        NN_TRIALS_PER_ROUND, NN_MAX_WALL_CLOCK_S, NN_MAX_COMPUTE, NN_SEED.
        """
        import yaml  # lazy

        path = os.path.join(self.config_path, "nn_search.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}

        def _override(key: str, env: str, cast):
            raw = os.getenv(env)
            if raw not in (None, ""):
                try:
                    cfg[key] = cast(raw)
                except (TypeError, ValueError):
                    pass

        _override("max_rounds", "NN_MAX_ROUNDS", int)
        _override("trials_per_round", "NN_TRIALS_PER_ROUND", int)
        _override("max_wall_clock_s", "NN_MAX_WALL_CLOCK_S", float)
        _override("max_compute", "NN_MAX_COMPUTE", int)
        _override("seed", "NN_SEED", int)
        return cfg

    def _run_infer_nn(self) -> None:
        """Run NN inference over a dataset folder (Phase 11).

        The orchestrator loads df_with_indicators.pkl from the target dataset,
        scores it, and atomically writes {dataset}/df_with_nn.pkl (absence-safe;
        never mutates df_with_indicators.pkl). Trainer only resolves the dataset
        dir + checkpoint id and records a light summary.
        """
        from helpers import wide_df_path  # lazy
        from nn.nn_orchestrator import NNOrchestrator  # lazy

        # Default dataset = the dir CONTAINING df_with_indicators.pkl. Compose sets
        # NN_INFER_DATASET to an EMPTY string by default, so treat empty as unset
        # (`os.getenv(..., default)` would otherwise return "" and skip the default).
        dataset = os.getenv("NN_INFER_DATASET") or os.path.dirname(wide_df_path())
        checkpoint_id = os.getenv("NN_INFER_CHECKPOINT") or "best"

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
