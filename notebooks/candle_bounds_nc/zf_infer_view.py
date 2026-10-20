"""Produce the viewer NN artifact for the zone-filtered models on oos2m.

Runs checkpoint-normalised inference for the 6 zf models over the full oos2m
frame (CPU — the 85k-row batch OOMs the 4GB GPU) and stages:

  df_with_nn_zf_heads.pkl — canonical 6 zf heads (nn_res_zf*)
  df_with_nn.pkl          — the viewer artifact, zf heads ONLY (this is the
                            established narrowing mechanism: the viewer draws
                            whatever nn_res_* columns df_with_nn.pkl carries;
                            the old 16/32-head views stay recoverable from
                            df_with_nn_heads.pkl / df_with_nn_psh8_heads.pkl)

Caller docker-cps the staged files into the volume dataset dir (root-owned)
after backing up the existing df_with_nn.pkl.

Run from the worktree root: CBNC_STAGE_DIR=<dir> python notebooks/candle_bounds_nc/zf_infer_view.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
os.environ.setdefault("PAIR", "link_usdt")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU inference

from indicators import DataAttributes  # noqa: E402
from nn.nn_model_spec import NNModelSpec  # noqa: E402
from nn.nn_orchestrator import NNOrchestrator  # noqa: E402

VOL = "/media/om/Alexandria/simple_trader/simple_trader_vol_long/train"
OOS_DIR = f"{VOL}/oos2m_link_usdt"
NN_ROOT = f"{VOL}/link_usdt/nn"
SPEC_GLOB = "configs/nn_specs/zone_filtered/zf*_n2_*.yaml"
STAGE = os.environ.get("CBNC_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    df = pd.read_pickle(f"{OOS_DIR}/df_with_indicators.pkl")
    attrs_p = f"{OOS_DIR}/data_attributes.pkl"
    attrs = DataAttributes.load(attrs_p) if os.path.exists(attrs_p) else DataAttributes()
    acc = pd.DataFrame(index=df.index)
    for path in sorted(glob.glob(SPEC_GLOB)):
        spec = NNModelSpec.from_yaml(path)
        ck = f"{NN_ROOT}/checkpoints/{spec.spec_hash}"
        if not os.path.isfile(f"{ck}/all_best.pt"):
            print(f"SKIP {spec.name}: no checkpoint", flush=True)
            continue
        orch = NNOrchestrator(checkpoint_dir=ck, dataset_dir=f"{NN_ROOT}/datasets", base_spec=spec)
        res = orch.run_inference(df, attrs)
        cols = [c for c in res.columns if str(c).startswith("nn_res_")]
        for c in cols:
            acc[c] = res[c]
        print(f"{spec.name}: {cols} valid={int(res[cols[0]].notna().sum())}", flush=True)

    for name in ("df_with_nn_zf_heads.pkl", "df_with_nn.pkl"):
        out = os.path.join(STAGE, name)
        acc.to_pickle(out)
    print(f"WROTE staged df_with_nn_zf_heads.pkl + df_with_nn.pkl | cols={list(acc.columns)}", flush=True)


if __name__ == "__main__":
    main()
