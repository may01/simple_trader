"""notebooks/trend_detection/run_diag.py -- Task 7 driver.

Thin: wire sys.path, read the SLIM frame, run the geometry-excluded
diagnostic, print where its outputs landed. No logic of its own -- see
tdlib/diag.py for the actual diagnostic (run_diag/write_diag_report).

Env-driven, like run_loop.py: reads ``config.slim_out_path()`` and writes
under ``config.artifacts_dir()`` -- the SAME base directory the loop run
itself used (this driver is meant to run AFTER run_loop.py, against the
loop's own frozen ``iter_01/`` artifacts, not a separately-located one).

Usage (inside the experiment container, matching docker-compose.experiment.yml):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm experiment python notebooks/trend_detection/run_diag.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- sys.path wiring ---------------------------------------------------
# This file lives at notebooks/trend_detection/run_diag.py. Two directories
# need to be importable, mirroring run_loop.py's own wiring:
#   - _PACKAGE_DIR (notebooks/trend_detection/, this file's own directory):
#     so `from tdlib.diag import run_diag` resolves `tdlib` as a bare
#     top-level package.
#   - _REPO_ROOT (three levels up: trend_detection/ -> notebooks/ -> repo
#     root): so tdlib's own `from helpers import dataset_folder` resolves.
#     Already satisfied inside Docker today (the base Dockerfile sets
#     ENV PYTHONPATH=/code), but wiring it explicitly here too means this
#     script also runs correctly outside that container, e.g. a bare
#     `python notebooks/trend_detection/run_diag.py` from a repo checkout
#     with no PYTHONPATH set.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402 - after sys.path wiring

from tdlib import config  # noqa: E402
from tdlib.diag import run_diag, write_diag_report  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    slim = pd.read_pickle(config.slim_out_path())
    base = config.artifacts_dir()

    table = run_diag(slim, base)
    loop_summary_path = f"{base}summary.md"
    report_path = write_diag_report(table, base, loop_summary_path)

    n_combos = table["combo"].nunique() if not table.empty else 0
    print(f"run_diag: {len(table)} row(s) across {n_combos} combo(s)")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
