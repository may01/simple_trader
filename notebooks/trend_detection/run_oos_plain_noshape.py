"""notebooks/trend_detection/run_oos_plain_noshape.py -- plain-truth,
shape-excluded OOS driver.

Same as run_oos_plain.py but validating run_loop_plain_noshape.py's frozen
artifacts (``{2y base}/plain_noshape/best.json`` + bundles); the report
lands under ``{artifacts_dir()}plain_noshape/`` on the OOS dataset dir.

Scoring reuses ``run_oos(..., truth_kind="plain")`` unchanged: the OOS X
is rebuilt from the FULL plain feature set, which is a strict superset of
the noshape candidate set, so the frozen bundle's own feature_order (all
noshape columns) selects exactly the columns it was trained on -- same
selection-by-frozen-list mechanism every other transform reproduction in
tdlib.oos already relies on.

Usage (inside the experiment container, oos2m env):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm <-e VAR=... from configs/oos2m_dataset.env> experiment \\
        python notebooks/trend_detection/run_oos_plain_noshape.py
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
from tdlib.loop import PLAIN_NOSHAPE_SUBDIR  # noqa: E402
from tdlib.oos import crosscheck_gates, run_oos, write_oos_report  # noqa: E402

_DEFAULT_TRAIN_BASE = f"/trader_data_long/train/2y_link_usdt/trend_detection/{PLAIN_NOSHAPE_SUBDIR}/"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    oos_slim = pd.read_pickle(config.slim_out_path())
    train_base = os.environ.get("TREND_TRAIN_BASE", _DEFAULT_TRAIN_BASE)

    gates = crosscheck_gates(oos_slim)

    table = run_oos(oos_slim, train_base, truth_kind="plain")

    out_dir = f"{config.artifacts_dir()}{PLAIN_NOSHAPE_SUBDIR}/"
    report_path = write_oos_report(table, gates, train_base, out_dir)
    table_path = f"{out_dir}oos_table.csv"

    n_combos = table["combo"].nunique() if len(table) else 0
    print(
        f"run_oos (plain truth, noshape-trained artifacts): scored {len(table)} row(s) "
        f"across {n_combos} combo(s) against train_base={train_base}"
    )
    print(f"oos_report.md: {report_path}")
    print(f"oos_table.csv: {table_path}")


if __name__ == "__main__":
    main()
