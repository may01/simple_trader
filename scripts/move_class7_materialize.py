#!/usr/bin/env python3
"""One-shot: recompute {tf}_move_class as 7-class sym0 in baked wide frames.

Replaces the legacy 5-tier (diff_mean ± diff_std) values with the RSI
parameters-selection experiment winner (classes -3..3, frozen 2y-fit cuts —
single source: indicators.library.classification.MOVE_CLASS_CUTS). Also
drops the interim {tf}_move_class_sym7 columns where present. Avoids a full
data re-prep. Idempotent (7-class values re-digitize to themselves — the
recompute reads rsi_ma8_diff, not the old classes).

Run once per dataset via the dataset env (single df_with_indicators.pkl,
plus every df_with_indicators.part_*.pkl for chunked datasets):

    LONG_ENV=configs/oos2m_dataset.env docker compose run --rm view-full \
        python3 scripts/move_class7_materialize.py
    NN_TRAIN_ENV=configs/nn_train_dataset_2y.env docker compose run --rm nn-train \
        python3 scripts/move_class7_materialize.py

First run writes a one-time .premc7_bak hard-link backup per file.
"""
import glob
import os

import numpy as np
import pandas as pd

from helpers import wide_df_path
from indicators.library.classification import MOVE_CLASS_CUTS


def recompute_move_class(wide_df: pd.DataFrame) -> list[str]:
    """Recompute {tf}_move_class in place; drop {tf}_move_class_sym7. Returns
    the names of columns written/dropped (empty = frame already current)."""
    changed = []
    for tf, cuts in MOVE_CLASS_CUTS.items():
        src = f"{tf}_rsi_ma8_diff"
        col = f"{tf}_move_class"
        if src in wide_df.columns:
            x = wide_df[src].to_numpy(dtype=float)
            new = (np.digitize(np.nan_to_num(x, nan=0.0), cuts) - 3).astype(np.int8)
            old = wide_df[col].to_numpy() if col in wide_df.columns else None
            if old is None or old.dtype != new.dtype or not np.array_equal(old, new):
                wide_df[col] = new
                changed.append(col)
        sym7 = f"{tf}_move_class_sym7"
        if sym7 in wide_df.columns:
            wide_df.drop(columns=[sym7], inplace=True)
            changed.append(f"-{sym7}")
    return changed


def _patch_file(path: str) -> None:
    df = pd.read_pickle(path)
    changed = recompute_move_class(df)
    if not changed:
        print(f"[mc7] current -> {path} (no-op)", flush=True)
        return
    bak = path + ".premc7_bak"
    if not os.path.exists(bak):
        os.link(path, bak)
    tmp = path + ".tmp"
    df.to_pickle(tmp)
    os.replace(tmp, path)
    print(f"[mc7] {changed} -> {path}", flush=True)


def main() -> None:
    path = wide_df_path()
    dataset_dir = os.path.dirname(path)
    parts = sorted(glob.glob(os.path.join(dataset_dir, "df_with_indicators.part_*.pkl")))
    targets = parts if parts else [path]
    for p in targets:
        _patch_file(p)
    print(f"[mc7] done ({len(targets)} file(s))", flush=True)


if __name__ == "__main__":
    main()
