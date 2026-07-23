"""notebooks/trend_detection/run_loop.py -- Task 5 driver.

Thin: wire sys.path, read the SLIM frame, run the autonomous improvement
loop, print where its outputs landed. No logic of its own -- see
tdlib/loop.py for the actual orchestration (run_iteration/improvement_loop/
write_iter_report/write_summary).

Usage (inside the experiment container, matching docker-compose.experiment.yml):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm experiment python notebooks/trend_detection/run_loop.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- sys.path wiring ---------------------------------------------------
# This file lives at notebooks/trend_detection/run_loop.py. Two directories
# need to be importable, mirroring run_extract.py's own wiring:
#   - _PACKAGE_DIR (notebooks/trend_detection/, this file's own directory):
#     so `from tdlib.loop import improvement_loop` resolves `tdlib` as a
#     bare top-level package.
#   - _REPO_ROOT (three levels up: trend_detection/ -> notebooks/ -> repo
#     root): so tdlib's own `from helpers import dataset_folder` resolves.
#     Already satisfied inside Docker today (the base Dockerfile sets
#     ENV PYTHONPATH=/code), but wiring it explicitly here too means this
#     script also runs correctly outside that container, e.g. a bare
#     `python notebooks/trend_detection/run_loop.py` from a repo checkout
#     with no PYTHONPATH set.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402 - after sys.path wiring

from tdlib import config  # noqa: E402
from tdlib.loop import improvement_loop, select_best  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    slim = pd.read_pickle(config.slim_out_path())
    results = improvement_loop(slim)

    base = config.artifacts_dir()
    summary_path = f"{base}summary.md"
    best_path = f"{base}best.json"
    print(f"improvement_loop: ran {len(results)} iteration(s)")
    print(f"summary: {summary_path}")
    print(f"best.json: {best_path}")

    # Reuse loop.py's OWN best-selection helper (not a re-derived max(...)
    # here) so this console line can never disagree with what best.json
    # actually names -- including the all-rejected fallback (select_best
    # falls back to the LAST iteration overall when nothing was ever kept,
    # same as write_summary's best.json does).
    best = select_best(results)
    if best is not None:
        print(
            f"best iteration: {best.cfg.iter_no:02d} ({best.cfg.transform}, "
            f"horizon={best.cfg.horizon}) mean_test_auc={best.mean_test_auc:.4f} kept={best.kept}"
        )
    else:
        print("best iteration: none (no iterations ran)")


if __name__ == "__main__":
    main()
