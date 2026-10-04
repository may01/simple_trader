"""notebooks/trend_detection/run_loop_plain_noshape.py -- plain-truth,
shape-excluded loop driver.

Same as run_loop_plain.py but with feature_set="noshape"
(tdlib.diag.diag_feature_cols: candle-shape/geometry raw columns excluded;
engineered features stay) -- the direct measure of how much plain-truth
separation remains once the entry-fill mechanics' most direct fingerprint
(candle-k shape) is removed. Artifacts land under
``{artifacts_dir()}plain_noshape/``.

Usage (inside the experiment container):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm <-e VAR=... from configs/nn_train_dataset_2y.env> experiment \\
        python notebooks/trend_detection/run_loop_plain_noshape.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- sys.path wiring (mirrors run_loop.py's own) -----------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402 - after sys.path wiring

from tdlib import config  # noqa: E402
from tdlib.loop import improvement_loop, loop_base_dir, select_best  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    slim = pd.read_pickle(config.slim_out_path())
    results = improvement_loop(slim, truth_kind="plain", feature_set="noshape")

    base = loop_base_dir("plain", "noshape")
    print(f"improvement_loop (plain truth, noshape features): ran {len(results)} iteration(s)")
    print(f"summary: {base}summary.md")
    print(f"best.json: {base}best.json")

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
