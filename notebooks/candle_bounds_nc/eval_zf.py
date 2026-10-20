"""Evaluate the 6 zone-filtered NN models (zf{tf}_n2_{side}) on oos2m.

For each model: checkpoint-normalised inference over the full oos2m frame
(NNOrchestrator.run_inference — training stats bundled in the checkpoint),
then scoring restricted to that model's own zone rows
({tf}_cbnc_inzone_{side}): base rate, AUC, precision@k, lift.

Baseline: the unfiltered nn-features-only head (nnfo_ps{tf}_n2_{side}) from
the existing df_with_nn_heads.pkl, scored on the SAME rows — answers whether
zone-filtered *training* beats zone-filtered *inference* of the incumbent.

Run on the host (venv with torch, CPU) from the worktree root:
    python notebooks/candle_bounds_nc/eval_zf.py
Writes eval json to CBNC_STAGE_DIR (default: alongside this script).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
os.environ.setdefault("PAIR", "link_usdt")

from data import join_candle_bounds, join_candle_bounds_nc  # noqa: E402
from indicators import DataAttributes  # noqa: E402
from nn.nn_model_spec import NNModelSpec  # noqa: E402
from nn.nn_orchestrator import NNOrchestrator  # noqa: E402

VOL = "/media/om/Alexandria/simple_trader/simple_trader_vol_long/train"
OOS_DIR = f"{VOL}/oos2m_link_usdt"
NN_ROOT = f"{VOL}/link_usdt/nn"
SPEC_GLOB = "configs/nn_specs/zone_filtered/zf*_n2_*.yaml"
STAGE = os.environ.get("CBNC_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
KS = (0.05, 0.10, 0.20)


def precision_at(p: np.ndarray, y: np.ndarray, frac: float) -> tuple[float, int]:
    k = max(1, int(round(len(p) * frac)))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean()), k


def score(p: np.ndarray, y: np.ndarray) -> dict:
    base = float(y.mean())
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    pos, neg = y == 1, y == 0
    auc = float((ranks[pos].mean() - (pos.sum() - 1) / 2) / neg.sum()) if pos.any() and neg.any() else float("nan")
    out = {"n": int(len(p)), "base_rate": base, "auc": auc}
    for f in KS:
        prec, k = precision_at(p, y, f)
        out[f"p_at_{int(f*100)}"] = prec
        out[f"lift_at_{int(f*100)}"] = prec / base if base > 0 else float("nan")
    return out


def main() -> None:
    df = pd.read_pickle(f"{OOS_DIR}/df_with_indicators.pkl")
    df = join_candle_bounds(df, OOS_DIR)
    df = join_candle_bounds_nc(df, OOS_DIR)
    attrs_p = f"{OOS_DIR}/data_attributes.pkl"
    attrs = DataAttributes.load(attrs_p) if os.path.exists(attrs_p) else DataAttributes()
    heads_p = f"{OOS_DIR}/df_with_nn_heads.pkl"
    heads = pd.read_pickle(heads_p) if os.path.exists(heads_p) else pd.DataFrame(index=df.index)

    results: dict = {}
    for path in sorted(glob.glob(SPEC_GLOB)):
        spec = NNModelSpec.from_yaml(path)
        tf = spec.targets[0].label_tf
        side = spec.targets[0].side
        ck = f"{NN_ROOT}/checkpoints/{spec.spec_hash}"
        if not os.path.isfile(f"{ck}/all_best.pt"):
            print(f"SKIP {spec.name}: no checkpoint at {ck}", flush=True)
            continue
        orch = NNOrchestrator(checkpoint_dir=ck, dataset_dir=f"{NN_ROOT}/datasets", base_spec=spec)
        res = orch.run_inference(df, attrs)
        pcols = [c for c in res.columns if str(c).startswith("nn_res_")]
        pcol = pcols[0]

        # target label column (exact training target)
        from nn.nn_dataset import _profit_long_col, _profit_short_col
        t = spec.targets[0]
        ycol = _profit_long_col(t, 2) if side == "long" else _profit_short_col(t, 2)

        zone = df[f"{tf}_cbnc_inzone_{side}"].fillna(False).astype(bool)
        mask = zone & res[pcol].notna() & df[ycol].notna()
        p = res.loc[mask, pcol].to_numpy(dtype=float)
        y = df.loc[mask, ycol].to_numpy(dtype=float)
        entry = {"zf": score(p, y), "pred_col": pcol, "label_col": ycol}

        # incumbent baseline on the same rows
        bcol = next((c for c in heads.columns if f"ps{tf}_n2_{side}" in str(c)), None)
        if bcol is not None:
            bmask = mask & heads[bcol].notna()
            pb = heads.loc[bmask, bcol].to_numpy(dtype=float)
            yb = df.loc[bmask, ycol].to_numpy(dtype=float)
            entry["baseline_nnfo"] = score(pb, yb)
            entry["baseline_col"] = str(bcol)
        results[spec.name] = entry
        print(spec.name, json.dumps(entry["zf"]), "| baseline:",
              json.dumps(entry.get("baseline_nnfo", {})), flush=True)

    out = os.path.join(STAGE, "zf_eval_oos2m.json")
    json.dump(results, open(out, "w"), indent=1)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
