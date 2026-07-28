"""Driver: fit closed-candle bounds models on 2y_az, gate on reproducing the
stored artifacts, then compute non-closed (forming) bounds + zones on oos2m.

Gates (abort on failure):
  G1  refit band_pct  ==  candle_bounds_meta.json band_pct        (per tf/side)
  G2  closed inference reproduces stored {tf}_cb_{side}[_up/_dn]  (oos2m)
  G3  nc prediction at each candle's first row == closed prediction

Outputs (next to the oos2m dataset):
  df_with_candle_bounds_nc.pkl   — {tf}_cbnc_* bounds/zones + {tf}_cbx_inzone_*
  candle_bounds_nc_meta.json     — band_pct / n_train / gate results
  candle_bounds_nc_metrics.json  — accuracy by candle-progress bucket + zone stats

Run: scratchpad venv python (system pandas + scikit-learn), from this directory.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cbnc import (  # noqa: E402
    SIDES, TFS, TGT_SHRINK_PCT, ZONE_FRAC, add_zones, closed_frame, fit_side,
    forming_features, infer_closed, infer_forming,
)

VOL = "/media/om/Alexandria/simple_trader/simple_trader_vol_long/train"
TRAIN_DIR = f"{VOL}/2y_az_link_usdt"
OOS_DIR = f"{VOL}/oos2m_link_usdt"

print("loading train 2y_az ...", flush=True)
train_df = pd.read_pickle(f"{TRAIN_DIR}/df_with_indicators.pkl")
meta_stored = json.load(open(f"{OOS_DIR}/candle_bounds_meta.json"))

models = {}
meta_out = {}
for tf in TFS:
    for side in SIDES:
        m = fit_side(train_df, tf, side)
        models[(tf, side)] = m
        stored = meta_stored[f"{tf}_{side}"]["band_pct"]
        rel = abs(m.band_pct - stored) / stored
        print(f"G1 {tf}_{side}: band_pct refit={m.band_pct:.6f} stored={stored:.6f} rel_diff={rel:.2e}", flush=True)
        assert rel < 1e-3, f"G1 FAIL {tf}_{side}"
        meta_out[f"{tf}_{side}"] = {
            "band_pct": m.band_pct, "band_pct_stored": stored,
            "n_train": m.n_train, "train": "2y_az", "target": "oos2m",
        }
del train_df

print("loading oos2m ...", flush=True)
df = pd.read_pickle(f"{OOS_DIR}/df_with_indicators.pkl")
cb = pd.read_pickle(f"{OOS_DIR}/df_with_candle_bounds.pkl")

# --- G2: closed reproduction. Known, accepted edge deltas vs the stored artifact:
# the lost original had one extra warmup candle at the series start (its first
# valid row is one candle later than necessary) and ffilled a stale bound into
# the final partial-candle row instead of emitting the fresh prediction. The
# interior must be machine-exact; stored-valid rows must be a subset of ours.
for tf in TFS:
    for side in SIDES:
        rep = infer_closed(df, models[(tf, side)])
        for col in (f"{tf}_cb_{side}", f"{tf}_cb_{side}_up", f"{tf}_cb_{side}_dn"):
            a, b = rep[col].to_numpy(), cb[col].to_numpy()
            both = ~(np.isnan(a) | np.isnan(b))
            both[-1] = False  # final-row ffill delta, see above
            rel = np.abs(a[both] - b[both]) / np.abs(b[both])
            extra_warmup = int((np.isnan(a) & ~np.isnan(b)).sum())
            print(f"G2 {col}: interior_max_rel={rel.max():.2e} n={both.sum()}", flush=True)
            assert rel.max() < 1e-9, f"G2 FAIL {col}"
            assert extra_warmup == 0, f"G2 FAIL {col}: stored valid where ours NaN"

# --- NC inference + G3 + zones
out = pd.DataFrame(index=df.index)
for tf in TFS:
    for side in SIDES:
        nc = infer_forming(df, models[(tf, side)])
        col = f"{tf}_cbnc_{side}"
        firsts = (df[f"{tf}_is_closed"] == True).shift(1).fillna(False).astype(bool).to_numpy().copy()  # noqa: E712
        firsts[-1] = False  # stored artifact ffills a stale bound into the final row
        a = nc[col].to_numpy()[firsts]
        b = cb[f"{tf}_cb_{side}"].to_numpy()[firsts]
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
        # unshifted prediction/reference, then shift(1) together (same convention)
        ff_pred = out[f"{tf}_cbnc_{side}"]                # already shifted
        ref = ext_f.shift(1)
        tgt = ext_next.shift(1)
        band = out[f"{tf}_cbnc_{side}_std"]
        # closed-model baseline on the same rows
        cb_pred = cb[f"{tf}_cb_{side}"]
        cb_up, cb_dn = cb[f"{tf}_cb_{side}_up"], cb[f"{tf}_cb_{side}_dn"]
        # nc targets the *next* candle; the closed columns on row t target the
        # *current* candle — evaluate closed at its own target for reference
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
        # closed baseline vs its own (current-candle) target
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
        cb_in = cb[f"{tf}_cb_inzone_{dirn}"].fillna(False).astype(bool)
        both = out[f"{tf}_cbx_inzone_{dirn}"].astype(bool)
        reach = (fwd >= out[tcol]) if dirn == "long" else (fwd <= out[tcol])
        reach_cb = (fwd >= cb[ccol]) if dirn == "long" else (fwd <= cb[ccol])
        zstats[dirn] = {
            "inzone_nc": int(nc_in.sum()), "inzone_cb": int(cb_in.sum()),
            "intersection": int(both.sum()),
            "jaccard": float(both.sum() / max((nc_in | cb_in).sum(), 1)),
            "reach_rate_nc_inzone": float(reach[nc_in].mean()),
            "reach_rate_cb_inzone": float(reach_cb[cb_in].mean()),
            "reach_rate_intersection": float(reach[both].mean()),
        }
    metrics[f"{tf}_zones"] = zstats

# --- Write artifacts. The volume dataset dir is root-owned (container-written),
# so write to STAGE_DIR (host-writable) and let the caller docker-cp into place.
STAGE_DIR = os.environ.get("CBNC_STAGE_DIR", OOS_DIR)
out.to_pickle(f"{STAGE_DIR}/df_with_candle_bounds_nc.pkl")
json.dump(
    {"zone_frac": ZONE_FRAC, "tgt_shrink_pct": TGT_SHRINK_PCT, "models": meta_out},
    open(f"{STAGE_DIR}/candle_bounds_nc_meta.json", "w"), indent=1,
)
json.dump(metrics, open(f"{STAGE_DIR}/candle_bounds_nc_metrics.json", "w"), indent=1)
print(f"WROTE {STAGE_DIR}/df_with_candle_bounds_nc.pkl  shape={out.shape}", flush=True)
print("metrics:", json.dumps(metrics, indent=1)[:4000], flush=True)
