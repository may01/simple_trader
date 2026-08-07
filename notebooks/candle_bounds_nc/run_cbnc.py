"""Driver: fit closed-candle bounds models on 2y_az, gate on reproducing the
stored artifacts, then compute non-closed (forming) bounds + hybrid zones for a
target dataset.

Gates (abort on failure):
  G1  refit band_pct  ==  candle_bounds_meta.json band_pct   (per tf/side; only
      when the oos2m meta is present)
  G2  closed inference reproduces stored {tf}_cb_{side}[_up/_dn] — only for
      datasets that carry a stored df_with_candle_bounds.pkl (oos2m)
  G3  nc prediction at each candle's first row == closed prediction (always;
      against the stored artifact when present, else our own closed inference)

Env:
  CBNC_DATASET    dataset dir name under the volume train/ root
                  (default oos2m_link_usdt; e.g. 2y_link_usdt for the NN
                  zone-filtered training set)
  CBNC_STAGE_DIR  host-writable output dir (volume dataset dirs are root-owned;
                  caller docker-cps the artifacts into place)

Outputs (into CBNC_STAGE_DIR):
  df_with_candle_bounds_nc.pkl   — {tf}_cbnc_* bounds/targets/zones + {tf}_cbx_inzone_*
  df_with_candle_bounds.pkl      — closed bounds artifact, ONLY when the target
                                   dataset has none (so viewers/consumers of
                                   cb_* work on that dataset too)
  candle_bounds_nc_meta.json     — band_pct / n_train / config
  candle_bounds_nc_metrics.json  — accuracy by candle-progress bucket + zone stats
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbnc import (  # noqa: E402
    SIDES, TFS, TGT_SHRINK_PCT, ZONE_FRAC, add_zones, fit_side,
    infer_closed, infer_forming,
)

VOL = "/media/om/Alexandria/simple_trader/simple_trader_vol_long/train"
TRAIN_DIR = f"{VOL}/2y_az_link_usdt"
DATASET = os.environ.get("CBNC_DATASET", "oos2m_link_usdt")
OOS_DIR = f"{VOL}/{DATASET}"
STAGE_DIR = os.environ.get("CBNC_STAGE_DIR", OOS_DIR)

print("loading train 2y_az ...", flush=True)
train_df = pd.read_pickle(f"{TRAIN_DIR}/df_with_indicators.pkl")
meta_path = f"{VOL}/oos2m_link_usdt/candle_bounds_meta.json"
meta_stored = json.load(open(meta_path)) if os.path.exists(meta_path) else None

models = {}
meta_out = {}
for tf in TFS:
    for side in SIDES:
        m = fit_side(train_df, tf, side)
        models[(tf, side)] = m
        entry = {"band_pct": m.band_pct, "n_train": m.n_train,
                 "train": "2y_az", "target": DATASET}
        if meta_stored is not None:
            stored = meta_stored[f"{tf}_{side}"]["band_pct"]
            rel = abs(m.band_pct - stored) / stored
            print(f"G1 {tf}_{side}: band_pct refit={m.band_pct:.6f} stored={stored:.6f} rel_diff={rel:.2e}", flush=True)
            assert rel < 1e-3, f"G1 FAIL {tf}_{side}"
            entry["band_pct_stored"] = stored
        meta_out[f"{tf}_{side}"] = entry
del train_df

print(f"loading {DATASET} ...", flush=True)
df = pd.read_pickle(f"{OOS_DIR}/df_with_indicators.pkl")

# --- Closed inference (always computed; doubles as the cb_* artifact for
# datasets that lack one, and as the G2 subject for datasets that have one).
own_cb = pd.DataFrame(index=df.index)
for tf in TFS:
    for side in SIDES:
        own_cb = own_cb.join(infer_closed(df, models[(tf, side)]))
# Closed entry zones, recovered rule (verified exact vs the stored artifact):
for tf in TFS:
    hi, lo = own_cb[f"{tf}_cb_high"], own_cb[f"{tf}_cb_low"]
    span = hi - lo
    z_long = lo + ZONE_FRAC * span
    z_short = hi - ZONE_FRAC * span
    own_cb[f"{tf}_cb_zone_long"] = z_long
    own_cb[f"{tf}_cb_zone_short"] = z_short
    own_cb[f"{tf}_cb_inzone_long"] = (df["1_low"] <= z_long).fillna(False)
    own_cb[f"{tf}_cb_inzone_short"] = (df["1_high"] >= z_short).fillna(False)

stored_cb_path = f"{OOS_DIR}/df_with_candle_bounds.pkl"
if os.path.exists(stored_cb_path):
    cb = pd.read_pickle(stored_cb_path)
    # G2: interior must be machine-exact. Known, accepted edge deltas of the
    # stored artifact: one extra warmup candle at the series start and a stale
    # ffill in the final partial-candle row.
    for tf in TFS:
        for side in SIDES:
            for col in (f"{tf}_cb_{side}", f"{tf}_cb_{side}_up", f"{tf}_cb_{side}_dn"):
                a, b = own_cb[col].to_numpy(), cb[col].to_numpy()
                both = ~(np.isnan(a) | np.isnan(b))
                both[-1] = False
                rel = np.abs(a[both] - b[both]) / np.abs(b[both])
                extra_warmup = int((np.isnan(a) & ~np.isnan(b)).sum())
                print(f"G2 {col}: interior_max_rel={rel.max():.2e} n={both.sum()}", flush=True)
                assert rel.max() < 1e-9, f"G2 FAIL {col}"
                assert extra_warmup == 0, f"G2 FAIL {col}: stored valid where ours NaN"
else:
    cb = own_cb
    print(f"no stored df_with_candle_bounds.pkl in {DATASET}; using own closed inference", flush=True)

# --- NC inference + G3 + hybrid zones
out = pd.DataFrame(index=df.index)
for tf in TFS:
    for side in SIDES:
        nc = infer_forming(df, models[(tf, side)])
        col = f"{tf}_cbnc_{side}"
        firsts = (df[f"{tf}_is_closed"] == True).shift(1).fillna(False).astype(bool).to_numpy().copy()  # noqa: E712
        firsts[-1] = False  # stored artifact ffills a stale bound into the final row
        a = nc[col].to_numpy()[firsts]
        b = cb[f"{tf}_cb_{side}"].reindex(df.index).to_numpy()[firsts]
        mask = ~(np.isnan(a) | np.isnan(b))
        rel = np.abs(a[mask] - b[mask]) / np.abs(b[mask])
        print(f"G3 {col}: first-row max_rel_diff={rel.max():.2e} (n={mask.sum()})", flush=True)
        assert rel.max() < 1e-6, f"G3 FAIL {col}"
        out = out.join(nc)
    add_zones(out, df, cb, tf)

# --- Metrics
metrics = {}
for tf in TFS:
    closed_mask = df[f"{tf}_is_closed"] == True  # noqa: E712
    candle_id = closed_mask.cumsum().shift(1).fillna(0)
    progress = (candle_id.groupby(candle_id).cumcount() + 1) / tf
    for side in SIDES:
        c = df.loc[closed_mask]
        ext = c[f"{tf}_{side}"]
        # realized extreme of the candle after the one containing each row
        ext_next = ext.shift(-1).reindex(df.index).bfill()
        ext_f = df[f"{tf}_{side}"]
        ff_pred = out[f"{tf}_cbnc_{side}"]                # already shifted
        ref = ext_f.shift(1)
        tgt = ext_next.shift(1)
        cb_pred = cb[f"{tf}_cb_{side}"].reindex(df.index)
        cb_up = cb[f"{tf}_cb_{side}_up"].reindex(df.index)
        cb_dn = cb[f"{tf}_cb_{side}_dn"].reindex(df.index)
        ext_cur = ext.reindex(df.index).bfill()

        rows = pd.DataFrame({
            "prog": progress.shift(1), "pred": ff_pred, "ref": ref, "tgt": tgt,
            "up": out[f"{tf}_cbnc_{side}_up"], "dn": out[f"{tf}_cbnc_{side}_dn"],
        }).dropna()
        pred_pct = (rows["pred"] / rows["ref"] - 1) * 100.0
        real_pct = (rows["tgt"] / rows["ref"] - 1) * 100.0
        buckets = pd.cut(rows["prog"], [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
        per_bucket = {}
        for b, g in rows.groupby(buckets, observed=True):
            pp, rp = pred_pct.loc[g.index], real_pct.loc[g.index]
            ss_res = ((rp - pp) ** 2).sum()
            ss_tot = ((rp - rp.mean()) ** 2).sum()
            per_bucket[str(b)] = {
                "n": len(g),
                "r2": float(1 - ss_res / ss_tot),
                "mae_pct": float((pp - rp).abs().mean()),
                "coverage": float(((g["tgt"] >= g["dn"]) & (g["tgt"] <= g["up"])).mean()),
            }
        cbrows = pd.DataFrame({"pred": cb_pred, "tgt": ext_cur, "up": cb_up, "dn": cb_dn}).dropna()
        cb_mae = float(((cbrows["pred"] - cbrows["tgt"]).abs() / cbrows["tgt"] * 100.0).mean())
        cb_cov = float(((cbrows["tgt"] >= cbrows["dn"]) & (cbrows["tgt"] <= cbrows["up"])).mean())
        metrics[f"{tf}_{side}"] = {
            "by_progress": per_bucket,
            "closed_baseline": {"mae_pct": cb_mae, "coverage": cb_cov, "n": len(cbrows)},
        }

    # zone stats + first-touch target reach within the next tf minutes
    fwd_high = df["1_high"].rolling(tf, min_periods=tf).max().shift(-tf)
    fwd_low = df["1_low"].rolling(tf, min_periods=tf).min().shift(-tf)
    zstats = {}
    for dirn, fwd, tcol, ccol in (
        ("long", fwd_high, f"{tf}_cbnc_tgt_long", f"{tf}_cb_high"),
        ("short", fwd_low, f"{tf}_cbnc_tgt_short", f"{tf}_cb_low"),
    ):
        nc_in = out[f"{tf}_cbnc_inzone_{dirn}"].astype(bool)
        cb_in = cb[f"{tf}_cb_inzone_{dirn}"].reindex(df.index).fillna(False).astype(bool)
        both = out[f"{tf}_cbx_inzone_{dirn}"].astype(bool)
        reach = (fwd >= out[tcol]) if dirn == "long" else (fwd <= out[tcol])
        cbc = cb[ccol].reindex(df.index)
        reach_cb = (fwd >= cbc) if dirn == "long" else (fwd <= cbc)
        zstats[dirn] = {
            "inzone_nc": int(nc_in.sum()), "inzone_cb": int(cb_in.sum()),
            "intersection": int(both.sum()),
            "jaccard": float(both.sum() / max((nc_in | cb_in).sum(), 1)),
            "reach_rate_nc_inzone": float(reach[nc_in].mean()),
            "reach_rate_cb_inzone": float(reach_cb[cb_in].mean()),
            "reach_rate_intersection": float(reach[both].mean()),
        }
    metrics[f"{tf}_zones"] = zstats

# --- Write artifacts (staged; caller docker-cps into the root-owned volume dir)
out.to_pickle(f"{STAGE_DIR}/df_with_candle_bounds_nc.pkl")
if not os.path.exists(stored_cb_path):
    own_cb.to_pickle(f"{STAGE_DIR}/df_with_candle_bounds.pkl")
    print(f"WROTE {STAGE_DIR}/df_with_candle_bounds.pkl (closed artifact for {DATASET})", flush=True)
json.dump(
    {"dataset": DATASET, "zone_frac": ZONE_FRAC, "tgt_shrink_pct": TGT_SHRINK_PCT,
     "zone_space": "hybrid: SL=closed bound, target=nc bound - shrink", "models": meta_out},
    open(f"{STAGE_DIR}/candle_bounds_nc_meta.json", "w"), indent=1,
)
json.dump(metrics, open(f"{STAGE_DIR}/candle_bounds_nc_metrics.json", "w"), indent=1)
print(f"WROTE {STAGE_DIR}/df_with_candle_bounds_nc.pkl  shape={out.shape}", flush=True)
print("zone stats:", json.dumps({k: v for k, v in metrics.items() if k.endswith("_zones")}, indent=1), flush=True)
