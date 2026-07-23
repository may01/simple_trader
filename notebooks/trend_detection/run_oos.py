"""notebooks/trend_detection/run_oos.py -- Task 6 driver.

Thin: wire sys.path, read the OOS SLIM frame (env-driven, via
config.slim_out_path() -- run this with the oos2m dataset's own env vars
set, e.g. DATA_SET_NAME=oos2m), score the frozen best 2y iteration's
artifacts against it with ZERO refit, write the report, print both output
paths. No logic of its own -- see tdlib/oos.py for the actual frozen-
validation implementation.

The 2y TRAIN artifacts (best.json + iter_NN/.../bundle/) live in a
DIFFERENT dataset dir than this run's own config.artifacts_dir(): env-
driven config resolves ROOT_FOLDER/DATA_ROOT/DATA_SET_NAME/PAIR at CALL time
(see tdlib/config.py), which for an oos2m run point at the oos2m dataset
dir, never the 2y one. So the 2y train base is its own explicit env var,
TREND_TRAIN_BASE, independent of config's own env wiring -- defaulting to
the real 2y link_usdt trend_detection artifacts dir on the long-data
volume (task-6-brief.md's exact default).

Usage (inside the experiment container, matching docker-compose.experiment.yml,
with the oos2m dataset's env vars set):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm experiment python notebooks/trend_detection/run_oos.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# --- sys.path wiring ---------------------------------------------------
# Mirrors run_extract.py / run_loop.py's own wiring exactly (this file lives
# at the same notebooks/trend_detection/ depth).
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

from tdlib import config  # noqa: E402
from tdlib.oos import crosscheck_gates, run_oos, write_oos_report  # noqa: E402

_DEFAULT_TRAIN_BASE = "/trader_data_long/train/2y_link_usdt/trend_detection/"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    oos_slim = pd.read_pickle(config.slim_out_path())
    train_base = os.environ.get("TREND_TRAIN_BASE", _DEFAULT_TRAIN_BASE)

    # Computed once here (rather than pulled off run_oos's own return value,
    # which run_oos.py's contract pins to a bare pd.DataFrame) and handed
    # straight to write_oos_report -- run_oos itself recomputes the SAME
    # gate internally to enforce it (see tdlib/oos.py's module docstring
    # point 1 for why that cheap redundancy beats a side-channel).
    gates = crosscheck_gates(oos_slim)

    table = run_oos(oos_slim, train_base)

    out_dir = config.artifacts_dir()
    report_path = write_oos_report(table, gates, train_base, out_dir)
    table_path = f"{out_dir}oos_table.csv"

    n_combos = table["combo"].nunique() if len(table) else 0
    print(f"run_oos: scored {len(table)} row(s) across {n_combos} combo(s) against train_base={train_base}")
    print(f"oos_report.md: {report_path}")
    print(f"oos_table.csv: {table_path}")


if __name__ == "__main__":
    main()
