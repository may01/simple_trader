"""tdlib/extract.py -- Task 1: SLIM extraction from the wide df.

``extract_slim()`` reads the pair's wide df (``helpers.wide_df_path()``,
env-driven, READ-ONLY -- the source file is never written to; the SLIM
output goes to a different file, ``config.slim_out_path()``, under its own
``trend_detection/`` subdirectory the source wide df never occupies),
reduces it to completed 15-minute rows (``df["15_is_closed"] == True``) and
the ``config.slim_columns()`` whitelist, and writes the result to
``config.slim_out_path()``. Logs shape + RSS before/after so a caller
running this against the real multi-GB wide df (see project memory:
data-prep perf bottleneck / NN OOM notes) can see memory behavior without
needing a profiler.
"""

from __future__ import annotations

import gc
import logging
import os

import pandas as pd

from helpers import wide_df_path
from tdlib.config import artifacts_dir, slim_columns, slim_out_path

logger = logging.getLogger(__name__)


def _rss_mb() -> float:
    """Current process resident set size, in MiB, read from
    /proc/self/status's VmRSS line (Linux-only -- this experiment always
    runs in Docker, see task-1 brief)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                # e.g. "VmRSS:\t   123456 kB\n"
                kb = int(line.split()[1])
                return kb / 1024.0
    return float("nan")  # pragma: no cover - VmRSS always present on Linux


def extract_slim() -> tuple:  # (rows, cols)
    """Read the wide df, reduce it to the SLIM frame, write it out, return
    ``(rows, cols)`` of the written frame.

    Steps (task-1 brief):
      1. Resolve the source path via ``helpers.wide_df_path()`` (env-driven;
         never hardcoded). Missing source -> ``FileNotFoundError`` whose
         message contains the resolved path.
      2. ``pd.read_pickle`` it, then filter to completed 15-minute rows:
         ``df["15_is_closed"] == True``.
      3. Reduce columns via ``config.slim_columns(df.columns.tolist())``.
      4. ``os.makedirs(artifacts_dir(), exist_ok=True)``, then write the
         result to ``config.slim_out_path()`` via ``to_pickle`` (overwrites
         any existing file there -- re-running is idempotent).
      5. Log shape + RSS, once before reading and once after writing.
      6. ``del df; gc.collect()`` -- drop the (potentially large) source
         frame reference before returning, so it doesn't linger for the
         rest of a long-running notebook/script process.
    """
    source_path = wide_df_path()
    logger.info("extract_slim: reading %s (RSS before: %.1f MiB)", source_path, _rss_mb())

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"extract_slim: source wide df not found at {source_path}")

    df = pd.read_pickle(source_path)

    slim = df[df["15_is_closed"] == True]  # noqa: E712 - explicit bool-column compare, not `is True`
    slim = slim[slim_columns(slim.columns.tolist())]

    os.makedirs(artifacts_dir(), exist_ok=True)
    out_path = slim_out_path()
    slim.to_pickle(out_path)

    rows, cols = slim.shape
    logger.info(
        "extract_slim: wrote %s rows x %s cols to %s (RSS after: %.1f MiB)",
        rows, cols, out_path, _rss_mb(),
    )

    del df
    gc.collect()

    return rows, cols
