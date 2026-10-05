"""Driver: zone-profitability shift search on 2y_az + oos2m validation.

Phases:
  1. 2y_az: refit closed-bound models (G1 gate vs stored candle_bounds_meta),
     infer closed bounds on 2y itself, compute profit labels (strict +
     non-strict, azlib defaults n=1 m=2 x=2 l=tf y=1), approx 7-class sym0
     move class, grid-search open/target shifts per (tf, class, side).
  2. oos2m: stored cb bounds + stored {tf}_move_class, same labels computed
     in-memory, apply the selected shifts, measure OOS winrate/reach-prob.
  3. Artifacts (ZP_OUT, default scratchpad): zone_profitability_params.json
     (reusable best shifts), zone_profitability_report.json (full grids +
     train/oos stats), df_with_zone_profitability.pkl markers for both frames.

Volume is root-owned: this writes to ZP_OUT on the host; copying markers next
to the datasets happens separately via docker. Nothing is committed.

Run: scratchpad zpenv python, from this directory.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebooks", "candle_bounds_nc"))

from cbnc import fit_side, infer_closed  # noqa: E402
from indicators.labels import (  # noqa: E402
    add_profit_labels, add_profit_strict_labels,
)
from zplib import (  # noqa: E402
    CLASSES, SIDES, TFS, approx_move_class, mark_zones, search_cells,
)

VOL = "/media/om/Alexandria/simple_trader/simple_trader_vol_long/train"
TRAIN_DIR = f"{VOL}/2y_az_link_usdt"
OOS_DIR = f"{VOL}/oos2m_link_usdt"
OUT = os.environ.get("ZP_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)

LABEL_N, LABEL_M, LABEL_X, LABEL_Y = 1, 2.0, 2.0, 1.0


def add_all_labels(df: pd.DataFrame) -> dict:
    labels = {}
    for tf in TFS:
        add_profit_strict_labels(df, tf, LABEL_N, LABEL_M, LABEL_X, tf, LABEL_Y)
        add_profit_labels(df, tf, LABEL_N, LABEL_M, LABEL_X)
        for side in SIDES:
            strict_col = f"{tf}_{'pslong' if side == 'long' else 'psshort'}_n1_m2_x2_l{tf}_y1"
            plain_col = f"{tf}_{'plong' if side == 'long' else 'pshort'}_n1_m2_x2"
            labels[(tf, side, True)] = df[strict_col].to_numpy(dtype=float)
            labels[(tf, side, False)] = df[plain_col].to_numpy(dtype=float)
    return labels


def zone_stats(df, cb, mc, labels, params):
    """Winrate + reach-prob of the given shifts on this frame (per cell)."""
    from zplib import _inzone, _open_level, _tgt_level, future_extreme_in_candle
    stats = {}
    for tf in TFS:
        for side in SIDES:
            fut = future_extreme_in_candle(df, tf, side)
            lab = labels[(tf, side, True)]
            for cls in CLASSES:
                s, t = params[(tf, cls, side)]
                level = _open_level(df, cb, tf, side, s)
                inz = _inzone(df, level, side) & ~np.isnan(level) & (mc[tf] == cls)
                pts = inz & ~np.isnan(lab)
                n = int(pts.sum())
                w = int(np.nansum(lab[pts]))
                cls_rows = int(((mc[tf] == cls) & ~np.isnan(lab)).sum())
                cls_wins = int(np.nansum(lab[(mc[tf] == cls) & ~np.isnan(lab)]))
                tlev = _tgt_level(df, cb, tf, side, t)
                atr = df[f"{tf}_atr_14_ma_5"].to_numpy()
                if side == "long":
                    reward = tlev - level
                    reach = fut >= tlev
                else:
                    reward = level - tlev
                    reach = fut <= tlev
                valid = inz & (reward >= 0.5 * 2.0 * atr)
                stats[f"{tf}_{cls}_{side}"] = {
                    "n_points": n, "wins": w, "losses": n - w,
                    "class_rows": cls_rows, "class_wins": cls_wins,
                    "winrate": round(w / n, 4) if n else None,
                    "p_reach": (round(float(reach[valid].mean()), 4)
                                if valid.sum() else None),
                }
    return stats


print("=== phase 1: 2y_az ===", flush=True)
train_df = pd.read_pickle(f"{TRAIN_DIR}/df_with_indicators.pkl")
meta_stored = json.load(open(f"{OOS_DIR}/candle_bounds_meta.json"))

models, cb_train = {}, pd.DataFrame(index=train_df.index)
for tf in TFS:
    for hs in ("high", "low"):
        m = fit_side(train_df, tf, hs)
        stored = meta_stored[f"{tf}_{hs}"]["band_pct"]
        rel = abs(m.band_pct - stored) / stored
        print(f"G1 {tf}_{hs}: refit={m.band_pct:.6f} stored={stored:.6f} rel={rel:.2e}", flush=True)
        assert rel < 1e-3, f"G1 FAIL {tf}_{hs}"
        models[(tf, hs)] = m
        cb_train = cb_train.join(infer_closed(train_df, m))

print("labels ...", flush=True)
labels_train = add_all_labels(train_df)
mc_train = {tf: approx_move_class(train_df, tf) for tf in TFS}

print("search ...", flush=True)
cells = search_cells(train_df, cb_train, mc_train, labels_train,
                     rr_label=LABEL_M / LABEL_X, x_atr=LABEL_X)

params = {(c.tf, c.cls, c.side): (c.open_shift, c.tgt_shift) for c in cells}
print("marking 2y ...", flush=True)
marks_train = mark_zones(train_df, cb_train, mc_train, params)
marks_train.to_pickle(f"{OUT}/df_with_zone_profitability_2y.pkl")
del train_df, cb_train, labels_train, mc_train, marks_train

print("=== phase 2: oos2m ===", flush=True)
df = pd.read_pickle(f"{OOS_DIR}/df_with_indicators.pkl")
cb = pd.read_pickle(f"{OOS_DIR}/df_with_candle_bounds.pkl")
mc_oos = {tf: df[f"{tf}_move_class"].to_numpy(dtype=np.int8) for tf in TFS}
labels_oos = add_all_labels(df)
oos = zone_stats(df, cb, mc_oos, labels_oos, params)
marks = mark_zones(df, cb, mc_oos, params)
marks.to_pickle(f"{OUT}/df_with_zone_profitability_oos2m.pkl")

print("=== artifacts ===", flush=True)
params_out = {
    "experiment": "zone_profitability",
    "base_bounds": "closed cb ({tf}_cb_low/high)",
    "move_class": "7-class frozen sym0 (MOVE_CLASS_CUTS)",
    "shift_unit": "{tf}_atr_14_ma_5",
    "labels": f"n{LABEL_N}_m2_x2_l{{tf}}_y1 (strict primary)",
    "train": "2y_az_link_usdt", "oos": "oos2m_link_usdt",
    "cells": {
        f"{c.tf}_{c.cls}_{c.side}": {
            "open_shift": c.open_shift, "tgt_shift": c.tgt_shift,
            "open_shift_nonstrict": c.open_shift_nonstrict,
        } for c in cells
    },
}
json.dump(params_out, open(f"{OUT}/zone_profitability_params.json", "w"), indent=1)

report = {
    f"{c.tf}_{c.cls}_{c.side}": {
        "train": {
            "open_shift": c.open_shift, "n_points": c.n_points,
            "wins": c.wins, "losses": c.losses, "winrate": c.winrate,
            "base_winrate": c.base_winrate, "lift": c.lift,
            "class_rows": c.class_rows, "class_wins": c.class_wins,
            "open_shift_nonstrict": c.open_shift_nonstrict,
            "winrate_nonstrict": c.winrate_nonstrict,
            "n_points_nonstrict": c.n_points_nonstrict,
            "tgt_shift": c.tgt_shift, "p_reach": c.p_reach, "rr": c.rr,
            "p_breakeven": c.p_breakeven, "margin": c.margin,
            "open_grid": c.open_grid, "tgt_grid": c.tgt_grid,
        },
        "oos": oos[f"{c.tf}_{c.cls}_{c.side}"],
    } for c in cells
}
json.dump(report, open(f"{OUT}/zone_profitability_report.json", "w"), indent=1)
print(f"done -> {OUT}", flush=True)
