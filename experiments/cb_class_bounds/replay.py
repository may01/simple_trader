"""Tier B — entry stand-in with the entry zone built from each variant's closed bound.

    python -m experiments.cb_class_bounds.replay --only 2y|oos2m [--hold 60] [--variants a,b,c]

Flow (verbatim from experiments/swing_sinusoid/cb_filter_replay.py of the swing worktree, provenance
2026-10-25): set-C candidate minutes = inzone & overshoot pass (per variant's bound + band), e25 live
trend gate (forming EMA-25 class), first passing minute per tf candle, entry 1_close, candle stop
(forming 15-min extreme ∓ trailing-60 mean 1-min range, placeholder 0.2 %), universal cbnc target
held fixed, hold H minutes, size min(100/dist, 50 000/entry), no fees. Paired Δ net vs the ``uni``
zones per tf candle (bootstrap CI90 over candles) and vs the stored production zones.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np
import pandas as pd
import talib

from indicators.candle_bounds.kernel import to_rows
from indicators.candle_bounds.params import FEATURES, OVERSHOOT_THR, ZONE_FRAC

from . import config
from .score import load_models, predict_all, variant_order

W, X = 105, 0.5
FROZEN25 = {15: (0.0018696074433414492, 0.08854754765129406), 60: (0.007586181744923595, 0.17365581448454287),
            240: (0.03083283482398462, 0.3443346525283253)}
RISK = 100.0; CAP = 100_000 * 0.5; PLACEHOLDER = 0.002; RANGE_N = 60
NBOOT = 1000


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def candle_id(closed):
    return np.concatenate(([0], np.cumsum(closed)[:-1]))


def load(name: str) -> pd.DataFrame:
    t0 = time.time()
    if name == "2y":
        df = pd.read_pickle(config.AZ_FRAME)
        keep = ["1_high", "1_low", "1_close"] + [f"{tf}_{c}" for tf in config.TFS for c in ("close", "is_closed")]
        df = df[keep].copy()
        cb = pd.read_pickle(config.CB_2Y)
        cb = cb[[f"{tf}_{c}" for tf in config.TFS for c in ("cb_low", "cb_high", "cb_low_std", "cb_high_std", "cb_inzone_long", "cb_inzone_short")]].copy()
        nc = pd.read_pickle(config.NC_2Y)
        nc = nc[[f"{tf}_{c}" for tf in config.TFS for c in ("cbnc_high", "cbnc_low")]].copy()
        df = pd.concat([df, cb.reindex(df.index), nc.reindex(df.index)], axis=1)
        del cb, nc
    else:
        df = pd.read_pickle(os.path.join(config.DATASETS["oos2m"]["dir"], "df_with_indicators.pkl"))
        keep = ["1_high", "1_low", "1_close"] + [f"{tf}_{c}" for tf in config.TFS for c in (
            "close", "is_closed", "cb_low", "cb_high", "cb_low_std", "cb_high_std", "cb_inzone_long", "cb_inzone_short", "cbnc_high", "cbnc_low")]
        df = df[keep].copy()
    print(f"[{name}] {df.shape} {df.index[0]} -> {df.index[-1]} loaded {time.time() - t0:.0f}s RSS {rss():.0f} MB", flush=True)
    return df


def ema_closed(closes, Nn):
    n = len(closes); ema = np.full(n, np.nan)
    for j in range(n):
        lo = max(0, j - W + 1)
        if j - lo + 1 >= Nn:
            ema[j] = talib.EMA(closes[lo:j + 1], timeperiod=Nn)[-1]
    return ema


def classes(df, tf):
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    cid = candle_id(closed)
    closes = df.loc[closed, f"{tf}_close"].to_numpy(float)
    c1 = df["1_close"].to_numpy(float)
    ema = ema_closed(closes, 25)
    a = 2.0 / 26
    prev_i = cid - 1
    prev = np.where(prev_i >= 0, ema[np.maximum(prev_i, 0)], np.nan)
    d_f = (a * c1 + (1 - a) * prev - prev) / prev * 100.0
    mu, sd = FROZEN25[tf]
    cl = np.where(d_f < mu - X * sd, -1, np.where(d_f > mu + X * sd, 1, 0)).astype(np.int8); cl[~np.isfinite(d_f)] = 0
    return cl, cid


def simulate_all(df, tf, side, cand, H):
    hi, lo, cl = (df[c].to_numpy(float) for c in ("1_high", "1_low", "1_close"))
    n = len(cl)
    cid15 = candle_id((df["15_is_closed"] == True).to_numpy())  # noqa: E712
    s = pd.Series(lo if side == "long" else hi)
    ext15 = (s.groupby(cid15).cummin() if side == "long" else s.groupby(cid15).cummax()).to_numpy()
    rng = pd.Series(hi - lo).shift(1).rolling(RANGE_N, min_periods=1).mean().to_numpy()
    tgt = df[f"{tf}_cbnc_high" if side == "long" else f"{tf}_cbnc_low"].to_numpy(float)
    idx = np.flatnonzero(cand)
    pnl = np.zeros(len(idx)); reason = np.zeros(len(idx), dtype=np.int8)
    sgn = 1.0 if side == "long" else -1.0
    for k, i0 in enumerate(idx):
        e = cl[i0]
        stop = ext15[i0] - rng[i0] if side == "long" else ext15[i0] + rng[i0]
        if not np.isfinite(stop) or (side == "long" and stop >= e) or (side == "short" and stop <= e):
            stop = e * (1 - PLACEHOLDER) if side == "long" else e * (1 + PLACEHOLDER)
        t = tgt[i0]
        dist = abs(e - stop); size = min(RISK / dist, CAP / e)
        a, b = i0 + 1, min(i0 + 1 + H, n)
        if side == "long":
            hs = lo[a:b] <= stop; ht = hi[a:b] >= t if np.isfinite(t) else np.zeros(b - a, dtype=bool)
        else:
            hs = hi[a:b] >= stop; ht = lo[a:b] <= t if np.isfinite(t) else np.zeros(b - a, dtype=bool)
        js = int(np.argmax(hs)) if hs.any() else 10 ** 9; jt = int(np.argmax(ht)) if ht.any() else 10 ** 9
        if js <= jt and js < 10 ** 9:
            x, r = stop, 1
        elif jt < 10 ** 9:
            x, r = t, 2
        else:
            x, r = cl[b - 1], 3
        pnl[k] = (x - e) * size * sgn; reason[k] = r
    return idx, pnl, reason


# ------------------------------------------------------------------ variant bounds
def variant_bounds(name: str, df: pd.DataFrame, tf: int, variants: list[str]) -> dict[str, dict[str, pd.Series]]:
    """Per variant: minute-level cb_high / cb_low / std_high / std_low (to_rows broadcast of the closed prediction)."""
    out: dict[str, dict[str, pd.Series]] = {v: {} for v in variants}
    models = load_models()
    for side in config.SIDES:
        if name == "2y":
            cf = pd.read_pickle(os.path.join(config.sidecar_dir("2y"), f"closed_tf{tf}_{side}.pkl")).dropna(subset=FEATURES + ["target"])
            pred, bands = predict_all(models["models"][f"{tf}_{side}"]["frozen"], cf)
        else:
            d = config.sidecar_dir("oos2m")
            cf = pd.read_pickle(os.path.join(d, f"closed_tf{tf}_{side}.pkl")).dropna(subset=FEATURES + ["target"])
            pred = pd.read_pickle(os.path.join(d, f"pred_oos2m_tf{tf}_{side}.pkl"))
            bands = pd.read_pickle(os.path.join(d, f"band_oos2m_tf{tf}_{side}.pkl"))
        ext = cf["ext"].astype(float)
        for v in variants:
            if v == "stored":
                out[v][f"cb_{side}"] = df[f"{tf}_cb_{side}"].astype(float)
                out[v][f"std_{side}"] = df[f"{tf}_cb_{side}_std"].astype(float)
                continue
            bound = ext * (1 + pred[v] / 100.0)
            band = ext * bands[v] / 100.0
            out[v][f"cb_{side}"] = to_rows(bound, df.index)
            out[v][f"std_{side}"] = to_rows(band, df.index)
    return out


def candidates(df, b: dict[str, pd.Series], tf: int, side: str) -> np.ndarray:
    hi = b["cb_high"].to_numpy(float); lo = b["cb_low"].to_numpy(float)
    span = hi - lo
    if side == "long":
        zone = lo + ZONE_FRAC * span
        x = df["1_low"].to_numpy(float)
        inz = x <= zone
        over = (lo - x) / b["std_low"].to_numpy(float)
    else:
        zone = hi - ZONE_FRAC * span
        x = df["1_high"].to_numpy(float)
        inz = x >= zone
        over = (x - hi) / b["std_high"].to_numpy(float)
    inz = np.nan_to_num(inz, nan=False) & np.isfinite(zone)
    return inz & ~(over > OVERSHOOT_THR[tf][side])


def first_per_candle(idx, pnl, reason, cid, flag):
    first = np.zeros(len(idx), dtype=bool); seen = set()
    for k in np.flatnonzero(flag):
        c = cid[k]
        if c not in seen:
            seen.add(c); first[k] = True
    return first


def summarize(p, reason_first) -> dict:
    return dict(n=int(len(p)), net=float(p.sum()), pf=float(p[p > 0].sum() / abs(p[p < 0].sum())) if (p < 0).any() else float("inf"),
                wr=float(np.mean(p > 0)) if len(p) else float("nan"), stop=int((reason_first == 1).sum()),
                tgt=int((reason_first == 2).sum()), hold=int((reason_first == 3).sum()))


def paired(per_candle_v: dict, per_candle_u: dict, seed=0):
    keys = sorted(set(per_candle_v) | set(per_candle_u))
    d = np.array([per_candle_v.get(k, 0.0) - per_candle_u.get(k, 0.0) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    sums = d[idx].sum(axis=1)
    return float(d.sum()), float(np.percentile(sums, 5)), float(np.percentile(sums, 95)), float(np.mean(sums < 0))


def tier_b_variants(tf: int) -> list[str]:
    own = list(config.FAMILIES)
    cross = [f"{f}_{H}" for H in config.HIGHER[tf] for f in config.FAMILIES]
    out = ["stored", "uni"] + own + cross + ["avg4"] + (["avg_all"] if cross else [])
    out += [f"int_{f}" for f in own] + [f"shuf_{f}_s0" for f in own] + ["avg4_shuf_s0"]
    return out


def run(name: str, H: int, only_variants: list[str] | None) -> dict:
    df = load(name)
    out = {}
    lines = [f"# Tier B — {name}, hold {H}", "", "Entry stand-in (see module doc). Paired Δ vs `uni` zones per tf candle, bootstrap CI90, P(Δ<0).", ""]
    for tf in config.TFS:
        cl_e25, cid = classes(df, tf)
        variants = only_variants or tier_b_variants(tf)
        B = variant_bounds(name, df, tf, variants)
        for side in config.SIDES:
            direction = "long" if side == "low" else "short"
            bad = -1 if direction == "long" else 1
            res = {}
            for v in variants:
                cand = candidates(df, B[v], tf, direction)
                idx, pnl, reason = simulate_all(df, tf, direction, cand, H)
                e25 = cl_e25[idx] != bad
                first = first_per_candle(idx, pnl, reason, cid[idx], e25)
                p = pnl[first]
                r = summarize(p, reason[first])
                r["cand"] = int(len(idx)); r["pass_e25"] = int(e25.sum())
                r["per_candle"] = dict(zip(cid[idx][first].tolist(), p.tolist()))
                res[v] = r
                print(f"[{name}] tf {tf:3d} {direction:5s} {v:14s}: cand {len(idx):6d} entries {r['n']:5d} net {r['net']:+9.0f} PF {r['pf']:.2f} wr {r['wr']:.3f} | RSS {rss():.0f} MB", flush=True)
            for ref in ("uni", "stored"):
                if ref not in res:
                    continue
                for v in variants:
                    s, lo, hi, pneg = paired(res[v]["per_candle"], res[ref]["per_candle"])
                    res[v][f"d_vs_{ref}"] = (s, lo, hi, pneg)
            lines += [f"## tf {tf} {direction}", "", "| variant | cand | entries | net | PF | wr | stop/tgt/hold | Δ vs uni (CI90) | P(Δ<0) | Δ vs stored |", "|---|---|---|---|---|---|---|---|---|---|"]
            for v in variants:
                r = res[v]
                du = r.get("d_vs_uni", (float("nan"),) * 4); ds = r.get("d_vs_stored", (float("nan"),) * 4)
                lines.append(f"| {v} | {r['cand']} | {r['n']} | {r['net']:+.0f} | {r['pf']:.2f} | {r['wr']:.3f} | {r['stop']}/{r['tgt']}/{r['hold']} | {du[0]:+.0f} [{du[1]:+.0f}, {du[2]:+.0f}] | {du[3]:.2f} | {ds[0]:+.0f} |")
            lines.append("")
            out[f"{tf}_{direction}"] = {v: {k: val for k, val in r.items() if k != "per_candle"} for v, r in res.items()}
        del B
    # totals across tf+side per variant
    allv = sorted({v for r in out.values() for v in r})
    lines += ["## All tf + sides", "", "| variant | entries | net | Σ Δ vs uni | Σ Δ vs stored |", "|---|---|---|---|---|"]
    for v in allv:
        n = sum(r[v]["n"] for r in out.values() if v in r); net = sum(r[v]["net"] for r in out.values() if v in r)
        du = sum(r[v].get("d_vs_uni", (0,))[0] for r in out.values() if v in r); ds = sum(r[v].get("d_vs_stored", (0,))[0] for r in out.values() if v in r)
        lines.append(f"| {v} | {n} | {net:+.0f} | {du:+.0f} | {ds:+.0f} |")
    d = config.sidecar_dir(name)
    with open(os.path.join(d, f"tierB_{name}_h{H}.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(d, f"tierB_{name}_h{H}.json"), "w") as f:
        json.dump(out, f, default=float)
    print(f"[{name}] wrote tierB_{name}_h{H}.md")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["2y", "oos2m"], required=True)
    ap.add_argument("--hold", type=int, default=60)
    ap.add_argument("--variants", default=None)
    a = ap.parse_args()
    run(a.only, a.hold, a.variants.split(",") if a.variants else None)


if __name__ == "__main__":
    main()
