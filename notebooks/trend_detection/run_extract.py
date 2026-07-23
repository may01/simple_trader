"""notebooks/trend_detection/run_extract.py -- Task 1 driver.

Thin: wire sys.path, call extract_slim(), print the result. No logic of its
own -- see tdlib/extract.py for the actual SLIM-extraction implementation.

Usage (inside the experiment container, matching docker-compose.experiment.yml):
    docker compose -f docker-compose.yml -f docker-compose.experiment.yml \\
        run --rm experiment python notebooks/trend_detection/run_extract.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- sys.path wiring ---------------------------------------------------
# This file lives at notebooks/trend_detection/run_extract.py. Two
# directories need to be importable, mirroring tests/conftest.py's own
# wiring (notebooks/trend_detection/tests/conftest.py):
#   - _PACKAGE_DIR (notebooks/trend_detection/, this file's own directory):
#     so `from tdlib.extract import extract_slim` resolves `tdlib` as a
#     bare top-level package.
#   - _REPO_ROOT (three levels up: trend_detection/ -> notebooks/ -> repo
#     root): so tdlib's own `from helpers import wide_df_path` resolves.
#     Already satisfied inside Docker today (the base Dockerfile sets
#     ENV PYTHONPATH=/code), but wiring it explicitly here too means this
#     script also runs correctly outside that container, e.g. a bare
#     `python notebooks/trend_detection/run_extract.py` from a repo
#     checkout with no PYTHONPATH set.
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent.parent
for _p in (_REPO_ROOT, _PACKAGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tdlib.extract import extract_slim  # noqa: E402 - after sys.path wiring


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows, cols = extract_slim()
    print(f"extract_slim: wrote {rows} rows x {cols} cols")


if __name__ == "__main__":
    main()
