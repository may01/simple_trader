"""Driver: fit cuts on 2y closed rows, score 2y + oos2m, dump one JSON.

Usage (from worktree root, host python):
    python3 -m experiments.rsi_params_selection.run_experiment --limit 2   # smoke
    python3 -m experiments.rsi_params_selection.run_experiment             # full
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import config
from .classify import apply_cuts, fit_cuts
from .data_loading import load_2y, load_oos
from .metrics import evaluate

DEFAULT_OUT = os.path.join(
    "external", "docs", "superpowers", "experiment", "results",
    "rsi_parameters_selection_results.json")

LABEL_KEYS = [f"{kind}_n{n}_{side}" for kind, n in config.LABEL_PAIRS
              for side in ("long", "short")]


def _score(frame: pd.DataFrame, window: int, cuts) -> dict:
    """Drop NaN-feature rows, classify, evaluate against the 8 label columns."""
    x = frame[f"diff{window}"].to_numpy(dtype=float)
    ok = ~np.isnan(x)
    classes = apply_cuts(x[ok], cuts)
    return evaluate(classes, frame.loc[ok, LABEL_KEYS], len(cuts) + 1)


def run_grid(train: dict[int, pd.DataFrame], oos: dict[int, pd.DataFrame]) -> dict:
    dropped = {"train": {}, "oos": {}}
    for name, dsets in (("train", train), ("oos", oos)):
        for tf, frame in dsets.items():
            dropped[name][str(tf)] = {
                str(w): int(frame[f"diff{w}"].isna().sum())
                for w in config.WINDOWS}

    cells = []
    for tf in config.TFS:
        for window in config.WINDOWS:
            fit_x = train[tf][f"diff{window}"].dropna().to_numpy(dtype=float)
            for technique in config.TECHNIQUES:
                for n_classes in config.CLASS_COUNTS:
                    cuts = fit_cuts(fit_x, technique, n_classes)
                    cells.append({
                        "window": window, "tf": tf,
                        "technique": technique, "n_classes": n_classes,
                        "cuts": [float(c) for c in cuts],
                        "train": _score(train[tf], window, cuts),
                        "oos": _score(oos[tf], window, cuts),
                    })
    return {
        "meta": {
            "spec": "2026-08-06-rsi-parameters-selection-design.md",
            "row_counts": {
                name: {str(tf): int(len(f)) for tf, f in dsets.items()}
                for name, dsets in (("train", train), ("oos", oos))},
            "dropped_nan_feature": dropped,
        },
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N 2y parts (smoke run)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    print("loading 2y parts…", flush=True)
    train = load_2y(limit=args.limit)
    print("loading oos2m…", flush=True)
    oos = load_oos()
    for tf in config.TFS:
        print(f"  tf {tf}: train {len(train[tf])} rows, oos {len(oos[tf])} rows")

    result = run_grid(train, oos)
    result["meta"]["parts_used"] = args.limit or 25

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out} ({len(result['cells'])} cells)")


if __name__ == "__main__":
    main()
