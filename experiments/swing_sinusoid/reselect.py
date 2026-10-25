"""Rebuild the frozen winners from the saved per-tf deviation CSVs with the current
selection rule (no recomputation). Usage:
    python -m experiments.swing_sinusoid.reselect --params <json>
Overwrites `winners` in the json and records the rule.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from . import config
from .sweep import MIN_COVERAGE, MIN_OK, select


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    args = ap.parse_args()
    params = json.load(open(args.params))
    ptag = f"p{params['prominence_frac']:g}".replace(".", "") + params.get("tag", "")
    sdir = config.sidecar_dir(params["dataset"])
    summ = pd.concat([pd.read_csv(os.path.join(sdir, f"deviation_tf{tf}_{ptag}.csv")) for tf in config.TFS],
                     ignore_index=True)
    params["winners"] = select(summ)
    params["min_ok"], params["min_coverage"] = MIN_OK, MIN_COVERAGE
    params["reselected"] = True
    with open(args.params, "w") as fh:
        json.dump(params, fh, indent=2, default=float)
    for k, w in params["winners"].items():
        print(k, w.get("method"), w.get("W"), round(w.get("hit_t", float("nan")), 3),
              "bl", (w.get("best_baseline") or {}).get("method"), round((w.get("best_baseline") or {}).get("hit_t", float("nan")), 3))


if __name__ == "__main__":
    main()
