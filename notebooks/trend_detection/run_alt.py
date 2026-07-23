"""notebooks/trend_detection/run_alt.py -- Task 8 driver.

Thin: wire sys.path, read the SLIM frame, run the truth-decomposition
diagnostic, print where its outputs landed. No logic of its own -- see
tdlib/alt_truth.py for the actual decomposition (run_alt/write_alt_report).

Env-driven, like run_loop.py/run_diag.py: reads ``config.slim_out_path()``
and writes under ``config.artifacts_dir()`` -- the SAME base directory the
loop run and diag run both used (``write_alt_report`` reads
``iter_01/{combo}/metrics.json`` and ``diag/{combo}/metrics.json`` straight
off that same base_dir -- see its own docstring). Meant to run AFTER both
``run_loop.py`` and ``run_diag.py``.

Usage (inside the experiment container, matching docker-compose.experiment.yml):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm experiment python notebooks/trend_detection/run_alt.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- sys.path wiring ---------------------------------------------------
# This file lives at notebooks/trend_detection/run_alt.py. Two directories
# need to be importable, mirroring run_loop.py's/run_diag.py's own wiring:
#   - _PACKAGE_DIR (notebooks/trend_detection/, this file's own directory):
#     so `from tdlib.alt_truth import run_alt` resolves `tdlib` as a bare
#     top-level package.
#   - _REPO_ROOT (three levels up: trend_detection/ -> notebooks/ -> repo
#     root): so tdlib's own `from helpers import dataset_folder` resolves.
#     Already satisfied inside Docker today (the base Dockerfile sets
#     ENV PYTHONPATH=/code), but wiring it explicitly here too means this
#     script also runs correctly outside that container, e.g. a bare
#     `python notebooks/trend_detection/run_alt.py` from a repo checkout
#     with no PYTHONPATH set.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402 - after sys.path wiring

from tdlib import config  # noqa: E402
from tdlib.alt_truth import run_alt, write_alt_report  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    slim = pd.read_pickle(config.slim_out_path())
    base = config.artifacts_dir()

    table = run_alt(slim, base)
    report_path = write_alt_report(table, base)

    n_combos = table["combo"].nunique() if not table.empty else 0
    print(f"run_alt: {len(table)} row(s) across {n_combos} combo(s)")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
