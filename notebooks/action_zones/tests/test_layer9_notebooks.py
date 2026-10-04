"""Layer 9 tests — the 5 action_zones papermill notebooks (Task 9).

Memory-safe smoke gate (see task-9-brief.md's "Memory safety" section): runs
each of notebooks/action_zones/nb/{10,40,60,80,90}_*.ipynb via papermill
against a TINY SYNTHETIC wide df written to the volume at a dedicated
"azsmoke" dataset dir — NEVER the real ~6 GB df_with_indicators.pkl. Reuses
the SAME `synthetic_wide_df` fixture (Appendix A, see conftest.py) every
other layer's tests already depend on — no new random-data recipe — pickled
to disk twice (train + "OOS", see `smoke_datasets`' own docstring for why an
independently-seeded second copy isn't needed here), and self-cleaning (the
smoke dataset dir + every action_zones/ artifact a notebook wrote under it
is removed in the fixture's teardown, pass or fail).

Each notebook is invoked as a REAL `papermill` subprocess (not the
in-process API) with the smoke dataset selected via ambient env vars passed
to that subprocess's own `env=` — mirrors EXACTLY how the real Docker
service selects a dataset (`docker compose run --env-file ...`, see
nb/README.md), just without actually needing Docker's `--env-file` flag
(this test already runs *inside* the experiment container, so the plain env
dict achieves the same effect for the child papermill process).

Filename carries "layer9" so `pytest -k layer9` selects every test here
(matches every earlier layer's own module-naming convention).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

# --- paths -------------------------------------------------------------

# tests/ -> action_zones/ -> notebooks/ -> repo root (== /code in the
# experiment Docker service, where docker-compose.experiment.yml sets
# working_dir: /code and PYTHONPATH=/code comes from the base image).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NB_DIR = _REPO_ROOT / "notebooks" / "action_zones" / "nb"

# Smoke dataset identity — task-9-brief.md's own example, verbatim:
# ROOT_FOLDER=long, DATA_ROOT=train, PAIR=link_usdt -> the real mounted
# /trader_data_long volume (docker-compose.experiment.yml), never a path
# under the worktree.
_ROOT_FOLDER = "long"
_DATA_ROOT = "train"
_PAIR = "link_usdt"
_TRAIN_NAME = "azsmoke"
_OOS_NAME = "azsmoke_oos"
_VOLUME_ROOT = Path("/trader_data_long")

_PAPERMILL_TIMEOUT_S = 600  # generous per-notebook ceiling; each is a fast, tiny-df smoke run


def _dataset_dir(name: str) -> Path:
    return _VOLUME_ROOT / _DATA_ROOT / f"{name}_{_PAIR}"


def _base_env() -> dict:
    """Ambient env for the smoke TRAIN dataset — same 5 keys
    `nbutil.snapshot_env_file`/`azlib.validate._apply_env_file` care about
    (see nb/nbutil.py), passed to the papermill SUBPROCESS's own `env=`
    (mirrors `docker compose run --env-file ...` for the real service).
    """
    env = dict(os.environ)
    env.update(
        {
            "ROOT_FOLDER": _ROOT_FOLDER,
            "DATA_ROOT": _DATA_ROOT,
            "DATA_SET_NAME": _TRAIN_NAME,
            "PAIR": _PAIR,
            "EXCHANGE_FEE": "0.001",
        }
    )
    return env


@pytest.fixture
def smoke_datasets(tmp_path, synthetic_wide_df):
    """Write the tiny synthetic wide df (`synthetic_wide_df` — 750 rows,
    every `{tf}_is_closed`/indicator/label-input column the pipeline reads,
    see conftest.py's Appendix A docstring) as `df_with_indicators.pkl` to
    BOTH the smoke TRAIN and smoke OOS dataset dirs on the mounted volume
    (never the worktree).

    A single fixture instance pickled twice — not two independently-seeded
    frames — is deliberate: `run_oos` does not require its OOS frame to
    differ from train (it only needs a VALID, same-shape wide df at the
    env-pointed-to path with no refit happening — see azlib.validate's own
    "Train-only freeze / no leakage" docstring), and this smoke gate's job
    is purely structural (papermill exits 0, artifacts land where expected)
    — not a check on any particular metric VALUE, which is what an
    independently-seeded OOS copy would matter for.

    Cleans up BOTH dataset dirs (the input pickle + every action_zones/
    artifact a notebook wrote under them, e.g.
    action_zones/long/15/results/...) in a `finally`, so the volume is left
    clean whether the test passes or fails.
    """
    if not _VOLUME_ROOT.is_dir():
        pytest.skip(f"{_VOLUME_ROOT} not mounted -- run inside the experiment Docker service")

    train_dir = _dataset_dir(_TRAIN_NAME)
    oos_dir = _dataset_dir(_OOS_NAME)
    for d in (train_dir, oos_dir):
        d.mkdir(parents=True, exist_ok=True)

    synthetic_wide_df.to_pickle(train_dir / "df_with_indicators.pkl")
    synthetic_wide_df.to_pickle(oos_dir / "df_with_indicators.pkl")

    oos_env_path = tmp_path / "oos_smoke.env"
    oos_env_path.write_text(
        f"ROOT_FOLDER={_ROOT_FOLDER}\n"
        f"DATA_ROOT={_DATA_ROOT}\n"
        f"DATA_SET_NAME={_OOS_NAME}\n"
        f"PAIR={_PAIR}\n"
        "EXCHANGE_FEE=0.001\n"
    )

    try:
        yield {"train_dir": train_dir, "oos_dir": oos_dir, "oos_env": str(oos_env_path)}
    finally:
        for d in (train_dir, oos_dir):
            shutil.rmtree(d, ignore_errors=True)


def _run_papermill(notebook: str, out_path: Path, params: list, env: dict) -> subprocess.CompletedProcess:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["papermill", str(_NB_DIR / notebook), str(out_path), *params]
    return subprocess.run(
        cmd, cwd=str(_REPO_ROOT), env=env, capture_output=True, text=True, timeout=_PAPERMILL_TIMEOUT_S
    )


def _assert_ok(proc: subprocess.CompletedProcess, notebook: str) -> None:
    assert proc.returncode == 0, (
        f"papermill {notebook} failed (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _assert_json_readable(path: Path) -> dict:
    assert path.is_file(), f"missing artifact: {path}"
    with open(path) as f:
        return json.load(f)


_PARAMS_BASE = ["-p", "tf", "15", "-p", "direction", "long"]


# --- 10_action_space ---------------------------------------------------


def test_10_action_space_smoke(smoke_datasets, tmp_path):
    out = tmp_path / "out" / "10_smoke.ipynb"
    proc = _run_papermill("10_action_space.ipynb", out, _PARAMS_BASE, _base_env())
    _assert_ok(proc, "10_action_space.ipynb")

    space_dir = smoke_datasets["train_dir"] / "action_zones" / "long" / "15" / "space"
    assert (space_dir / "price_levels.png").is_file()
    assert (space_dir / "label_coeff_hist.png").is_file()
    report = _assert_json_readable(space_dir / "space_report.json")
    assert report["tf"] == 15
    assert report["direction"] == "long"


# --- 40_regression_classification ---------------------------------------


def test_40_regression_classification_smoke(smoke_datasets, tmp_path):
    out = tmp_path / "out" / "40_smoke.ipynb"
    proc = _run_papermill("40_regression_classification.ipynb", out, _PARAMS_BASE, _base_env())
    _assert_ok(proc, "40_regression_classification.ipynb")

    reg_clf_dir = smoke_datasets["train_dir"] / "action_zones" / "long" / "15" / "reg_clf"
    report = _assert_json_readable(reg_clf_dir / "reg_clf_report.json")
    assert report["n_records"] > 0
    # rsi's 1D "position" group always has ample non-NaN rows at 750-row/tf=15
    # (talib warm-up is tiny relative to the fixture) -- deterministic, not skipped.
    assert (reg_clf_dir / "rsi_position_linear_reg.json").is_file()
    assert (reg_clf_dir / "rsi_position_linear_reg.png").is_file()


# --- 60_rr_grid -----------------------------------------------------------


def test_60_rr_grid_smoke(smoke_datasets, tmp_path):
    out = tmp_path / "out" / "60_smoke.ipynb"
    params = [*_PARAMS_BASE, "-p", "fee", "0.001"]
    proc = _run_papermill("60_rr_grid.ipynb", out, params, _base_env())
    _assert_ok(proc, "60_rr_grid.ipynb")

    rr_dir = smoke_datasets["train_dir"] / "action_zones" / "long" / "15" / "rr"
    assert (rr_dir / "rr_grid.csv").is_file()
    assert (rr_dir / "reach_prob_sanity.png").is_file()
    levels = _assert_json_readable(rr_dir / "selected_levels.json")
    assert set(levels) == {"tgt_x", "sl_x", "rr", "exp_ret"}


# --- 80_zones ------------------------------------------------------------


def test_80_zones_smoke(smoke_datasets, tmp_path):
    out = tmp_path / "out" / "80_smoke.ipynb"
    proc = _run_papermill("80_zones.ipynb", out, _PARAMS_BASE, _base_env())
    _assert_ok(proc, "80_zones.ipynb")

    results_dir = smoke_datasets["train_dir"] / "action_zones" / "long" / "15" / "results"
    assert (results_dir / "results.json").is_file()
    assert (results_dir / "stats.json").is_file()
    assert (results_dir / "zoned_dataset_train.pkl").is_file()
    summary = _assert_json_readable(results_dir / "zones_summary.json")
    assert summary["n_rows"] > 0


# --- 90_validate -----------------------------------------------------------


def test_90_validate_smoke(smoke_datasets, tmp_path):
    out = tmp_path / "out" / "90_smoke.ipynb"
    params = [*_PARAMS_BASE, "-p", "oos_env", smoke_datasets["oos_env"]]
    proc = _run_papermill("90_validate.ipynb", out, params, _base_env())
    _assert_ok(proc, "90_validate.ipynb")

    validate_dir = smoke_datasets["train_dir"] / "action_zones" / "long" / "15" / "validate"
    assert (validate_dir / "zoned_dataset_oos.pkl").is_file()
    report = _assert_json_readable(validate_dir / "metrics_report.json")
    assert "strict_coverage" in report["metrics"]
    assert "reach_drift_max" in report["metrics"]
