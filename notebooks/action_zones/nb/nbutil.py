"""notebooks/action_zones/nb/nbutil.py — Task 9: thin orchestration helpers
shared by the action_zones notebooks (10/40/60/80/90).

No azlib business logic lives here — only (1) an artifact-directory
convention matching azlib.validate._results_dir's own env-derived volume
path shape (different `subdir` per notebook so each of Task 9's 5 notebooks
gets its own artifact namespace without clobbering another's), (2) a tiny
env-file-snapshot helper bridging this project's Docker convention (the
TRAIN dataset is selected via AMBIENT env vars — `docker compose run
--env-file ...` — not a notebook parameter, see nb/README.md) with
`azlib.validate.run_train`/`run_oos`'s own signature (both take an
env-FILE PATH, not env vars directly), and (3) a tiny JSON-report
formatting helper.

sys.path wiring for `azlib` itself is deliberately NOT here — this module
lives next to azlib, under notebooks/action_zones/nb/, so importing IT
already requires notebooks/action_zones/nb/ to be on sys.path (a
chicken-and-egg problem if the bootstrap lived inside the thing being
bootstrapped). Each notebook's own setup cell does that sys.path bootstrap
inline (a few lines, mirrors tests/conftest.py's own approach), then
imports this module for everything else.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Keys azlib.validate._apply_env_file / helpers.py actually read at call
# time (ROOT_FOLDER/DATA_ROOT/DATA_SET_NAME/PAIR -> helpers.dataset_folder();
# EXCHANGE_FEE -> azlib.validate._fee()). Kept as an explicit whitelist (not
# "dump all of os.environ") so a snapshot never accidentally leaks unrelated
# container env vars into a run_train/run_oos env-file argument.
_ENV_KEYS = ("ROOT_FOLDER", "DATA_ROOT", "DATA_SET_NAME", "PAIR", "EXCHANGE_FEE")


def snapshot_env_file(path: str) -> str:
    """Dump the currently-active dataset env vars (`_ENV_KEYS`) to `path` as
    a plain KEY=VALUE file `azlib.validate._apply_env_file` can parse.
    Returns `path` unchanged, for inline use.

    `run_train(train_env, tf, direction)`'s first argument is an env-FILE
    PATH (its own `_apply_env_file` mechanism) — but this project's Docker
    convention selects the TRAIN dataset via AMBIENT env vars (`docker
    compose run --env-file ...`), not a notebook parameter (see
    nb/README.md — matches the documented full-2y command, which passes no
    `-p train_env`). This snapshot bridges the two: re-applying it via
    `_apply_env_file` is a no-op against the already-active environment —
    it exists only so `run_train`/`run_oos` have a file path to call that
    on.
    """
    lines = [f"{k}={os.environ[k]}\n" for k in _ENV_KEYS if k in os.environ]
    Path(path).write_text("".join(lines))
    return path


def results_dir(tf: int, direction: str, subdir: str) -> str:
    """`{dataset_folder()}action_zones/{direction}/{tf}/{subdir}/`, created
    (`exist_ok=True`) if missing.

    Same env-derived volume convention `azlib.validate._results_dir` uses
    for its own `results/` subdir (Task 8) — extended here with a
    caller-chosen `subdir` so each of Task 9's 5 notebooks gets its own
    artifact namespace (`space`, `reg_clf`, `rr`, `results`, `validate`)
    under the SAME `.../action_zones/{direction}/{tf}/` root, never the
    worktree — `helpers.dataset_folder()` is entirely env-driven
    (ROOT_FOLDER/DATA_ROOT/DATA_SET_NAME/PAIR); ROOT_FOLDER=long always
    resolves under the mounted `/trader_data_long` volume, never a path
    under `/code`.
    """
    from helpers import dataset_folder  # /code on sys.path via the image's own PYTHONPATH

    base = os.path.join(dataset_folder(), "action_zones", direction, str(tf), subdir)
    os.makedirs(base, exist_ok=True)
    return base


def write_json_report(path: str, data: dict) -> str:
    """Write `data` to `path` as indented, human-reproducible JSON. Returns
    `path` unchanged, for inline use.

    Tiny shared formatting convention (indent=2, sorted keys, `default=str`
    as a safety net for any stray numpy scalar that isn't plain-JSON-native)
    so every notebook's own report file looks the same — not business
    logic, just consistent report formatting.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    return path
