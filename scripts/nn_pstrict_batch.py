#!/usr/bin/env python3
"""Batch driver for the 16 profit_strict nn-features-only models.

Backbone = the promoted nn-features-only v2 winner (conv1d_seq -> lstm -> dense,
46 nn_features x 6 TFs); each model swaps in ONE ``label`` (profit_strict) head
(side long/short x TF 5/15/60/240 x horizon n1/n2). Specs live under
``configs/nn_specs/profit_strict/*.yaml``.

Runs INSIDE the nn-train container (torch + volume mounts present):

    python3 scripts/nn_pstrict_batch.py train
        Single-train every model on the env-selected dataset (2y via
        NN_TRAIN_ENV=configs/nn_train_dataset_2y.env). Resumable: skips any
        model whose {spec_hash}/all_best.pt already exists.

    python3 scripts/nn_pstrict_batch.py infer <oos_dataset_dir>
        Run inference for every trained model on the OOS dataset and ACCUMULATE
        all nn_res_* columns into a single {oos}/df_with_nn.pkl (the wired
        run_inference_dataset overwrites, so we merge here instead).

All artefacts land on the shared volume (checkpoints/datasets under the pair's
nn artefact root); nothing is written into /code.
"""
import glob
import os
import sys

import pandas as pd

from helpers import wide_df_path, data_attributes_path
from indicators import DataAttributes
from nn.nn_model_spec import NNModelSpec
from nn.nn_orchestrator import NNOrchestrator
from nn.device import nn_artefact_root

SPEC_GLOB = os.environ.get("NN_SPEC_GLOB", "/code/configs/nn_specs/profit_strict/*.yaml")
PAIR = os.environ.get("PAIR", "link_usdt")
HEADS_SIDECAR = "df_with_nn_heads.pkl"  # canonical 16-head predictions (viewer ignores)


def _specs():
    paths = sorted(glob.glob(SPEC_GLOB))
    if not paths:
        raise SystemExit(f"no specs matched {SPEC_GLOB}")
    return [(p, NNModelSpec.from_yaml(p)) for p in paths]


def _roots():
    root = nn_artefact_root(PAIR)
    return str(root), os.path.join(str(root), "checkpoints"), os.path.join(str(root), "datasets")


def train():
    root, ck_root, ds_dir = _roots()
    df = pd.read_pickle(wide_df_path())
    attrs = DataAttributes.load(data_attributes_path())
    specs = _specs()
    workers = int(os.environ.get("NUM_WORKERS", os.environ.get("AVAIABLE_THREADS", "4")))
    print(f"[train] {len(specs)} specs | df rows={len(df)} | workers={workers} | root={root}",
          flush=True)
    for i, (path, spec) in enumerate(specs, 1):
        ck = os.path.join(ck_root, spec.spec_hash)
        if os.path.isfile(os.path.join(ck, "all_best.pt")):
            print(f"[train] ({i}/{len(specs)}) SKIP {spec.name} ({spec.spec_hash[:8]}) "
                  f"— all_best.pt exists", flush=True)
            continue
        print(f"[train] ({i}/{len(specs)}) TRAIN {spec.name} ({spec.spec_hash[:8]})", flush=True)
        orch = NNOrchestrator(checkpoint_dir=ck, dataset_dir=ds_dir, base_spec=spec)
        orch.num_workers = workers
        metrics = orch.train(df, attrs, promote=True)
        print(f"[train] ({i}/{len(specs)}) DONE {spec.name} metrics={metrics}", flush=True)
    print("[train] all done", flush=True)


def infer(oos_dir):
    root, ck_root, ds_dir = _roots()
    df = pd.read_pickle(os.path.join(oos_dir, "df_with_indicators.pkl"))
    attrs_path = os.path.join(oos_dir, "data_attributes.pkl")
    attrs = DataAttributes.load(attrs_path) if os.path.exists(attrs_path) else DataAttributes()
    acc = pd.DataFrame(index=df.index)
    specs = _specs()
    print(f"[infer] {len(specs)} specs | oos rows={len(df)} | dir={oos_dir}", flush=True)
    for i, (path, spec) in enumerate(specs, 1):
        ck = os.path.join(ck_root, spec.spec_hash)
        if not os.path.isfile(os.path.join(ck, "all_best.pt")):
            print(f"[infer] ({i}/{len(specs)}) SKIP {spec.name} — no checkpoint", flush=True)
            continue
        orch = NNOrchestrator(checkpoint_dir=ck, dataset_dir=ds_dir, base_spec=spec)
        res = orch.run_inference(df, attrs)
        cols = [c for c in res.columns if str(c).startswith("nn_res_")]
        for c in cols:
            acc[c] = res[c]
        valid = int(res[cols[0]].notna().sum()) if cols else 0
        print(f"[infer] ({i}/{len(specs)}) {spec.name}: cols={cols} valid_rows={valid}",
              flush=True)
    # Canonical 16-head source (viewer ignores it; derive modes read it) + the
    # viewer artifact df_with_nn.pkl (the 16 heads, until a derive mode narrows it).
    for out in (os.path.join(oos_dir, HEADS_SIDECAR), os.path.join(oos_dir, "df_with_nn.pkl")):
        tmp = out + ".tmp"
        acc.to_pickle(tmp)
        os.replace(tmp, out)
    print(f"[infer] wrote df_with_nn.pkl + {HEADS_SIDECAR} | {acc.shape[1]} heads | "
          f"{len(acc)} rows", flush=True)
    print(f"[infer] columns: {list(acc.columns)}", flush=True)


def _load_heads(oos_dir):
    """Return (df, head_cols) of the canonical 16 per-head predictions.

    Prefers {oos}/df_with_nn_heads.pkl; falls back to df_with_nn.pkl if it still
    carries per-head cols. Errors if neither does (run `infer`).
    """
    for fn in (HEADS_SIDECAR, "df_with_nn.pkl"):
        p = os.path.join(oos_dir, fn)
        if not os.path.isfile(p):
            continue
        df = pd.read_pickle(p)
        heads = [c for c in df.columns
                 if str(c).startswith("nn_res_") and not str(c).startswith("nn_res_avg")
                 and (str(c).endswith("_long_prob") or str(c).endswith("_short_prob"))]
        if heads:
            return df, heads
    raise SystemExit(f"[signals] no per-head predictions under {oos_dir}; run `infer` first")


def _write_view(oos_dir, cols):
    """Write {oos}/df_with_nn.pkl exposing exactly *cols* (dict name->Series)."""
    out = pd.DataFrame(cols)
    path = os.path.join(oos_dir, "df_with_nn.pkl")
    tmp = path + ".tmp"
    out.to_pickle(tmp)
    os.replace(tmp, path)
    print(f"[signals] wrote {path} | cols={list(out.columns)} | rows={len(out)}", flush=True)
    for c in out.columns:
        v = out[c].dropna()
        print(f"[signals]   {c}: mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.3f} max={v.max():.3f} nan={out[c].isna().sum()}", flush=True)


def signals(oos_dir, what):
    """Derive viewer signals from the canonical heads and expose *what*.

    what ∈ {heads, avg, diff, zdiff, all}:
      avg   — mean of the 8 long / 8 short heads   (nn_res_avg_{long,short}_prob)
      diff  — first difference of each avg          (nn_res_diff_{long,short})
      zdiff — z-score of each diff (over the series)(nn_res_zdiff_{long,short})
      all   — avg + zdiff together
      heads — the raw 16 per-head predictions
    """
    df, heads = _load_heads(oos_dir)
    long_h = [c for c in heads if str(c).endswith("_long_prob")]
    short_h = [c for c in heads if str(c).endswith("_short_prob")]
    avg_l, avg_s = df[long_h].mean(axis=1), df[short_h].mean(axis=1)
    diff_l, diff_s = avg_l.diff(), avg_s.diff()
    zdiff_l = (diff_l - diff_l.mean()) / diff_l.std()
    zdiff_s = (diff_s - diff_s.mean()) / diff_s.std()
    table = {
        "avg":   {"nn_res_avg_long_prob": avg_l, "nn_res_avg_short_prob": avg_s},
        "diff":  {"nn_res_diff_long": diff_l, "nn_res_diff_short": diff_s},
        "zdiff": {"nn_res_zdiff_long": zdiff_l, "nn_res_zdiff_short": zdiff_s},
        "heads": {c: df[c] for c in heads},
    }
    if what == "all":
        cols = {**table["avg"], **table["zdiff"]}
    elif what in table:
        cols = table[what]
    else:
        raise SystemExit(f"[signals] unknown what={what!r}; use heads|avg|diff|zdiff|all")
    print(f"[signals] source heads: {len(long_h)} long / {len(short_h)} short | expose={what}",
          flush=True)
    _write_view(oos_dir, cols)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "train":
        train()
    elif mode == "infer":
        oos = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NN_INFER_DATASET", "")
        if not oos:
            raise SystemExit("infer needs <oos_dataset_dir> or NN_INFER_DATASET")
        infer(oos)
    elif mode in ("signals", "avg"):
        # `avg` kept as an alias for `signals <oos> avg`.
        oos = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NN_INFER_DATASET", "")
        if not oos:
            raise SystemExit(f"{mode} needs <oos_dataset_dir> or NN_INFER_DATASET")
        what = "avg" if mode == "avg" else (sys.argv[3] if len(sys.argv) > 3 else "zdiff")
        signals(oos, what)
    else:
        raise SystemExit(__doc__)
