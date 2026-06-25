"""Integration: Trainer._run_train_nn in SEARCH mode end-to-end (Phase-11).

Runs in the nn-train image only (optuna present). The base image SKIPS the whole
module via importorskip — the search path lazy-imports optuna inside the loop.

Wires the REAL stack the same way ``Trainer._run_train_nn`` does:
``NNOrchestrator.from_trainer`` → ``configs/nn_search.yaml`` → ``ExperimentTracker``
→ ``TrainingLoop.run(df, data_attributes)`` → final ``training_state.pkl`` + light
metadata. Kept FAST: a tiny prepared wide df, a tiny spec (1 narrow layer, 1-2
epochs), max_rounds=1 / trials_per_round=1.

Asserts the tracking dir is populated (index.sqlite + trials/) and a promoted
best is recorded — i.e. the loop really searched + promoted, not just ran.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

optuna = pytest.importorskip("optuna")  # skip whole file in the base image

from indicators import DataAttributes
from indicators.labels import _fmt


# ---------------------------------------------------------------------------
# Tiny prepared dataset (mirrors the orchestrator/loop unit fixtures)
# ---------------------------------------------------------------------------


def _label_suffix(n: int, m: float, x: float) -> str:
    return f"n{_fmt(n)}_m{_fmt(m)}_x{_fmt(x)}"


def make_wide_df(rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """Small synthetic wide frame: 15-min closed flags, features, dir labels."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=rows, freq="1min")
    df = pd.DataFrame(index=idx)
    pos = np.arange(rows)

    df["15_is_closed"] = (pos + 1) % 15 == 0
    df["15_logret"] = (pos.astype(float) * 0.001) - 0.01
    df["15_rsi_14"] = 50.0 + (pos.astype(float) % 30)
    df["15_close"] = 100.0 + np.cumsum(rng.normal(0, 0.1, rows))

    suf = _label_suffix(1, 1.0, 0.3)
    df[f"15_plong_{suf}"] = np.where(pos % 3 == 0, 1.0, 0.0)
    df[f"15_pshort_{suf}"] = np.where(pos % 3 == 1, 1.0, 0.0)
    return df


# Tiny, fast nn_spec.yaml — one narrow dense layer, short lookback, 2 epochs.
_TINY_NN_SPEC = """
name: t
timeframes: [15]
indicators:
  - logret
  - rsi_14
history_points: 4
grouping:
  mode: single
layers:
  - kind: dense
    units: 8
activation: relu
dropout: 0.1
targets:
  - name: dir15
    kind: direction
    horizons: [1]
    label_tf: 15
    label_m: 1.0
    label_x: 0.3
    strict: false
loss_fn: auto
optimizer: adam
learning_rate: 0.001
batch_size: 32
epochs: 2
validation_split: 0.2
val_strategy: time_holdout
seed: 0
"""

# Tiny, fast nn_search.yaml — 1 round, 1 trial, generous caps.
_TINY_NN_SEARCH = """
max_rounds: 1
trials_per_round: 1
sampler: tpe
pruner: median
max_wall_clock_s: 120
max_compute: null
margin: 0.0
seed: 0
search_space:
  lr: [1.0e-4, 1.0e-2]
  depth: [1, 2]
  units: [8, 16]
  dropout: [0.0, 0.2]
"""


@pytest.fixture
def search_env(tmp_path, monkeypatch):
    """Set up env + a temp config dir with tiny nn_spec/nn_search and a df."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "nn_spec.yaml").write_text(_TINY_NN_SPEC)
    (cfg_dir / "nn_search.yaml").write_text(_TINY_NN_SEARCH)

    artefacts = tmp_path / "artefacts"
    artefacts.mkdir()

    monkeypatch.setenv("RUN_TYPE", "nn_train")
    monkeypatch.setenv("NN_TRAIN_MODE", "search")
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("DATA_ROOT", "test_data")
    monkeypatch.setenv("NN_ARTEFACT_ROOT", str(artefacts))
    monkeypatch.delenv("NN_STRATEGIST", raising=False)
    monkeypatch.delenv("NN_STUDY", raising=False)
    monkeypatch.setenv("NUM_WORKERS", "1")

    # NNOrchestrator.from_trainer reads "configs/nn_spec.yaml" by literal path —
    # point cwd at tmp so it resolves to our tiny spec, and Trainer config_path
    # at the same dir so nn_search.yaml resolves there too.
    monkeypatch.chdir(tmp_path)

    df = make_wide_df()
    df.to_pickle(str(tmp_path / "df_with_indicators.pkl"))
    DataAttributes().save(str(tmp_path / "data_attributes.pkl"))

    shared = tmp_path / "shared"
    shared.mkdir()

    monkeypatch.setattr(
        "helpers.wide_df_path",
        lambda: str(tmp_path / "df_with_indicators.pkl"),
    )
    monkeypatch.setattr(
        "helpers.data_attributes_path",
        lambda: str(tmp_path / "data_attributes.pkl"),
    )
    monkeypatch.setattr("helpers.shared_folder", lambda: str(shared) + "/")

    return tmp_path, artefacts, shared


def test_run_train_nn_search_end_to_end(search_env):
    """_run_train_nn (search) populates the tracking dir + records a best."""
    tmp_path, artefacts, shared = search_env

    from training.trainer import Trainer

    t = Trainer(config_path="configs/")
    t._run_train_nn()  # must not raise

    # tracking_dir = {nn_artefact_root(pair)}/tracking;
    # nn_artefact_root = {NN_ARTEFACT_ROOT}/{DATA_ROOT}/{pair}/nn
    tracking_root = artefacts / "test_data" / "link_usdt" / "nn" / "tracking"
    assert tracking_root.is_dir(), f"tracking root missing: {tracking_root}"

    studies = [p for p in tracking_root.iterdir() if p.is_dir()]
    assert studies, "no study dir created under tracking/"
    study_dir = studies[0]
    # Default study name = {pair}_{spec_hash[:8]}.
    assert study_dir.name.startswith("link_usdt_")

    assert (study_dir / "index.sqlite").exists(), "index.sqlite not written"
    assert (study_dir / "trials").is_dir(), "trials/ dir not written"

    # At least one trial recorded.
    trial_files = list((study_dir / "trials").glob("*.json"))
    assert trial_files, "no trial JSON records written"

    # A promoted best was recorded (best.json + tracker metadata).
    best_json = study_dir / "best.json"
    assert best_json.exists(), "best.json not written (nothing promoted)"

    meta = t.metadata["train_nn"]
    assert meta["mode"] == "search"
    assert meta["study_name"] == study_dir.name
    assert meta["n_trials"] >= 1
    assert meta["best"], "metadata best is empty (no incumbent)"
    assert "all" in meta["best"], "group 'all' not promoted"


def test_run_train_nn_search_writes_training_state(search_env):
    """The final training_state.pkl summary is written for the search path."""
    tmp_path, _, shared = search_env

    from training.trainer import Trainer

    Trainer(config_path="configs/")._run_train_nn()

    state_path = str(shared) + "/training_state.pkl"
    assert os.path.exists(state_path)
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    assert state["phase"] == "nn_train"
    assert state["study"].startswith("link_usdt_")
    assert "best" in state
