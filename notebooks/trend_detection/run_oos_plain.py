"""notebooks/trend_detection/run_oos_plain.py -- plain-truth OOS driver.

Thin: wire sys.path, read the OOS SLIM frame, score the PLAIN-truth 2y
loop's frozen artifacts (run_loop_plain.py's ``{2y base}/plain/best.json``
+ bundles) against the OOS dataset with truth_kind="plain", zero refit.
Identical to run_oos.py except: the default train base points at the plain
subdir, scoring marks truth via the plain race+fill pair, and the report
lands under ``{artifacts_dir()}plain/`` on the OOS dataset dir (never
clobbering the strict run's own oos_report.md).

Usage (inside the experiment container, oos2m env):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm --env-file configs/oos2m_dataset.env experiment \\
        python notebooks/trend_detection/run_oos_plain.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# --- sys.path wiring (mirrors run_oos.py's own) ------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402 - after sys.path wiring

from tdlib import config  # noqa: E402
from tdlib.loop import PLAIN_SUBDIR  # noqa: E402
from tdlib.oos import crosscheck_gates, run_oos, write_oos_report  # noqa: E402

_DEFAULT_TRAIN_BASE = f"/trader_data_long/train/2y_link_usdt/trend_detection/{PLAIN_SUBDIR}/"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    oos_slim = pd.read_pickle(config.slim_out_path())
    train_base = os.environ.get("TREND_TRAIN_BASE", _DEFAULT_TRAIN_BASE)

    gates = crosscheck_gates(oos_slim)

    table = run_oos(oos_slim, train_base, truth_kind="plain")

    out_dir = f"{config.artifacts_dir()}{PLAIN_SUBDIR}/"
    report_path = write_oos_report(table, gates, train_base, out_dir)
    table_path = f"{out_dir}oos_table.csv"

    n_combos = table["combo"].nunique() if len(table) else 0
    print(f"run_oos (plain truth): scored {len(table)} row(s) across {n_combos} combo(s) against train_base={train_base}")
    print(f"oos_report.md: {report_path}")
    print(f"oos_table.csv: {table_path}")


if __name__ == "__main__":
    main()
