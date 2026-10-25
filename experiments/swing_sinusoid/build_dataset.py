"""Build per-tf closed-row series sidecars for one dataset.

    python -m experiments.swing_sinusoid.build_dataset --dataset 2y [--limit 2]
    python -m experiments.swing_sinusoid.build_dataset --dataset oos2m

Reads the wide frames one part at a time (read-only), extracts the closed rows
of each tf, concatenates, derives the swing series once over the continuous
series and writes `<dataset dir>/swing_sinusoid/series_tf{tf}.pkl` + meta.json.
On oos2m the recomputed rsi_ma8 is checked against the stored column
(reproduction gate for the EMA recipe).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os

import numpy as np
import pandas as pd

from experiments.rsi_params_selection.data_loading import add_labels_for_tfs

from . import config
from .provenance import stamp
from .series import derive, extract_raw


def part_paths(dataset: str, limit: int | None = None) -> list[str]:
    d = config.DATASETS[dataset]
    if d["parts"]:
        paths = sorted(glob.glob(os.path.join(d["dir"], "df_with_indicators.part_*.pkl")))
    else:
        paths = [os.path.join(d["dir"], "df_with_indicators.pkl")]
    if not paths:
        raise FileNotFoundError(f"no frames under {d['dir']}")
    return paths[:limit] if limit else paths


def build(dataset: str, limit: int | None = None, tfs=tuple(config.TFS)) -> dict:
    raws: dict[int, list] = {tf: [] for tf in tfs}
    closed_counts = {tf: 0 for tf in tfs}
    stored_rsi_ma8 = {tf: [] for tf in tfs}
    paths = part_paths(dataset, limit)
    for i, path in enumerate(paths):
        df = pd.read_pickle(path)
        has_labels = config.label_cols(tfs[0])["plain_n1_long"] in df.columns
        if not has_labels:
            add_labels_for_tfs(df, list(tfs), config.CONFIG_PATH)
        for tf in tfs:
            mask = df[f"{tf}_is_closed"].astype(bool).to_numpy()
            closed_counts[tf] += int(mask.sum())
            raws[tf].append(extract_raw(df, tf))
            col = f"{tf}_rsi_ma8"
            if col in df.columns:
                stored_rsi_ma8[tf].append(df[col].to_numpy()[mask])
        print(f"[{dataset}] part {i + 1}/{len(paths)} {os.path.basename(path)} rows={len(df)}", flush=True)
        del df
        gc.collect()

    out_dir = config.sidecar_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    meta = stamp(dataset, parts=[os.path.basename(p) for p in paths], tfs=list(tfs), rows={}, checks={})
    for tf in tfs:
        full = derive(pd.concat(raws[tf]))
        assert len(full) == closed_counts[tf], (tf, len(full), closed_counts[tf])
        assert full.index.is_monotonic_increasing and full.index.is_unique
        path = os.path.join(out_dir, f"series_tf{tf}.pkl")
        full.to_pickle(path)
        meta["rows"][str(tf)] = int(len(full))
        if stored_rsi_ma8[tf]:
            stored = np.concatenate(stored_rsi_ma8[tf])
            ours = full["rsi_ma8"].to_numpy()
            ok = ~np.isnan(stored) & ~np.isnan(ours)
            meta["checks"][f"rsi_ma8_max_abs_diff_tf{tf}"] = float(np.max(np.abs(stored[ok] - ours[ok])))
        print(f"[{dataset}] tf{tf}: {len(full)} rows → {path}", flush=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta["checks"]))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASETS), required=True)
    ap.add_argument("--limit", type=int, default=None, help="parts to read (smoke)")
    args = ap.parse_args()
    build(args.dataset, args.limit)


if __name__ == "__main__":
    main()
