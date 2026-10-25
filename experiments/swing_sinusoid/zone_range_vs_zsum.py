#!/usr/bin/env python3
"""CB zone price range (1-min ATR units) vs the combined swing-zone sum z15 + z60 + z240.

Episode = (tf, side, tf candle): every minute of the candle with {tf}_cb_inzone_{side} True.
dist      = (max 1_high − min 1_low over the episode minutes) / ATR1 at the first in-zone minute
dist_fb   = same over the first contiguous in-zone block of the candle
dist_cdl  = same over every row of the candle (the tf candle's range)
zsum      = z15 + z60 + z240 at the first in-zone minute (causal minute series of the last closed
            candle per tf, built by cb_filter_replay.combined_zone — identical to the rule replay);
            placebos ph5 / ph17 / ph41 (rolled closed rows) and shuf (1-day blocks shuffled).
Tests: Spearman ρ(dist, zsum) with a permutation null, placebo ρ, Kruskal–Wallis over zsum
values (n ≥ 30), partial Spearman given minutes in zone, medians by zsum value.

usage: python3 -m experiments.swing_sinusoid.zone_range_vs_zsum --out /out [--only 2y|oos2m] [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import talib
from scipy import stats

from experiments.swing_sinusoid import cb_filter_replay as rep

TFS = (15, 60, 240)
SIDES = ("long", "short")
ZKEYS = ("z", "ph5", "ph17", "ph41", "shuf")
MIN_CELL = 30


# ------------------------------------------------------------------ pure pieces (tested)
def atr1(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    return talib.ATR(np.asarray(high, float), np.asarray(low, float), np.asarray(close, float), timeperiod=n)


def episodes(inz: np.ndarray, cid: np.ndarray, hi: np.ndarray, lo: np.ndarray) -> pd.DataFrame:
    """One row per tf candle holding ≥ 1 in-zone minute.

    Columns: cid, first (row index of the first in-zone minute), last, n_min, hi, lo (over the
    in-zone minutes), fb_hi, fb_lo, fb_n (first contiguous block), cdl_hi, cdl_lo (all rows of the
    candle), cdl_start (row index of the candle's first row)."""
    idx = np.flatnonzero(inz)
    if len(idx) == 0:
        return pd.DataFrame(columns=["cid", "first", "last", "n_min", "hi", "lo", "fb_hi", "fb_lo", "fb_n",
                                     "cdl_hi", "cdl_lo", "cdl_start"])
    c = cid[idx]
    new_block = np.r_[True, (np.diff(idx) != 1) | (np.diff(c) != 0)]
    block = np.cumsum(new_block) - 1
    f = pd.DataFrame({"idx": idx, "cid": c, "hi": hi[idx], "lo": lo[idx], "block": block})
    g = f.groupby("cid", sort=True)
    ep = g.agg(first=("idx", "min"), last=("idx", "max"), n_min=("idx", "size"), hi=("hi", "max"), lo=("lo", "min"),
               fb=("block", "min")).reset_index()
    fb = f.groupby("block").agg(fb_hi=("hi", "max"), fb_lo=("lo", "min"), fb_n=("idx", "size"))
    ep = ep.join(fb, on="fb").drop(columns="fb")
    allrows = pd.DataFrame({"cid": cid, "hi": hi, "lo": lo, "row": np.arange(len(cid))})
    cdl = allrows.groupby("cid").agg(cdl_hi=("hi", "max"), cdl_lo=("lo", "min"), cdl_start=("row", "min"))
    ep = ep.join(cdl, on="cid")
    # extremes from the first in-zone minute to the candle's end (post-entry path inside the candle)
    rev = slice(None, None, -1)
    post_hi = pd.Series(hi[rev]).groupby(cid[rev]).cummax().to_numpy()[rev]
    post_lo = pd.Series(lo[rev]).groupby(cid[rev]).cummin().to_numpy()[rev]
    ep["post_hi"] = post_hi[ep["first"].to_numpy()]
    ep["post_lo"] = post_lo[ep["first"].to_numpy()]
    return ep


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    """Spearman ρ(x, y) and the permutation distribution of ρ under shuffled y."""
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    rho = float(np.mean(rx * ry))
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = np.mean(rx * rng.permutation(ry))
    return rho, null


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman ρ(x, y) after removing the rank-linear effect of z from both."""
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    Z = np.c_[np.ones(len(rz)), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


# ------------------------------------------------------------------ data
def load(name: str) -> pd.DataFrame:
    t0 = time.time()
    V = rep.V
    if name == "2y":
        df = pd.read_pickle(f"{V}/2y_az_link_usdt/df_with_indicators.pkl")
        df = df[["1_high", "1_low", "1_close"] + [f"{tf}_is_closed" for tf in TFS]].copy()
        cb = pd.read_pickle(f"{V}/2y_link_usdt/df_with_candle_bounds.pkl")
        cb = cb[[f"{tf}_cb_inzone_{s}" for tf in TFS for s in SIDES]].copy()
        df = pd.concat([df, cb.reindex(df.index)], axis=1)
        del cb
    else:
        df = pd.read_pickle(f"{V}/oos2m_link_usdt/df_with_indicators.pkl")
        keep = ["1_high", "1_low", "1_close", "1_atr_14"] + [f"{tf}_is_closed" for tf in TFS] + \
               [f"{tf}_cb_inzone_{s}" for tf in TFS for s in SIDES]
        df = df[keep].copy()
    print(f"[{name}] {df.shape} {df.index[0]} -> {df.index[-1]} loaded {time.time() - t0:.0f}s RSS {rep.rss():.0f} MB", flush=True)
    return df


def build_episode_table(name: str, df: pd.DataFrame, Z: dict) -> pd.DataFrame:
    hi = df["1_high"].to_numpy(float); lo = df["1_low"].to_numpy(float); cl = df["1_close"].to_numpy(float)
    a1 = atr1(hi, lo, cl)
    if "1_atr_14" in df.columns:   # provenance gate against the stored column
        st = df["1_atr_14"].to_numpy(float); ok = np.isfinite(st) & np.isfinite(a1)
        rel = np.abs(a1[ok] - st[ok]) / st[ok]
        print(f"[{name}] ATR gate vs stored 1_atr_14: rows {ok.sum()}, median rel diff {np.median(rel):.2e}, "
              f"share within 1e-6 {np.mean(rel < 1e-6):.3f}, max {rel.max():.2e}", flush=True)
    zsum = {k: sum(Z[tf][k].astype(int) for tf in TFS) for k in ZKEYS}
    parts = []
    for tf in TFS:
        closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
        cid = rep.candle_id(closed)
        for side in SIDES:
            inz = df[f"{tf}_cb_inzone_{side}"].fillna(False).to_numpy(bool)
            ep = episodes(inz, cid, hi, lo)
            fi = ep["first"].to_numpy()
            a = a1[fi]
            ep["atr1"] = a
            ep["dist"] = (ep["hi"] - ep["lo"]) / a
            ep["dist_fb"] = (ep["fb_hi"] - ep["fb_lo"]) / a
            ep["dist_cdl"] = (ep["cdl_hi"] - ep["cdl_lo"]) / a
            ep["progress"] = (fi - ep["cdl_start"].to_numpy()) / tf
            ep["entry"] = cl[fi]
            if side == "long":
                ep["adv"] = (ep["entry"] - ep["post_lo"]) / a; ep["fav"] = (ep["post_hi"] - ep["entry"]) / a
            else:
                ep["adv"] = (ep["post_hi"] - ep["entry"]) / a; ep["fav"] = (ep["entry"] - ep["post_lo"]) / a
            for k in ZKEYS:
                ep[f"zsum_{k}"] = zsum[k][fi]
            for h in TFS:
                ep[f"z{h}"] = Z[h]["z"][fi]
            sgn = 1 if side == "long" else -1
            ep["zsum_aligned"] = sgn * ep["zsum_z"]
            ep["tf"] = tf; ep["side"] = side; ep["dataset"] = name
            ep["t_first"] = df.index[fi]
            parts.append(ep)
            print(f"[{name}] tf {tf:3d} {side:5s}: episodes {len(ep)}, minutes {int(ep['n_min'].sum())}, "
                  f"dist median {ep['dist'].median():.2f} q90 {ep['dist'].quantile(.9):.2f}; n_min median {ep['n_min'].median():.0f}", flush=True)
    out = pd.concat(parts, ignore_index=True)
    return out[np.isfinite(out["dist"])].reset_index(drop=True)


# ------------------------------------------------------------------ stats
def analyse(ep: pd.DataFrame, n_perm: int, seed: int = 0) -> tuple[dict, str]:
    rng = np.random.default_rng(seed)
    res = {}; lines = []
    name = ep["dataset"].iloc[0]
    lines.append(f"\n=== {name}: Spearman ρ(dist, zsum) per tf × side; perm null band (5–95 %), placebo ρ, partial ρ | n_min, Kruskal p ===")
    lines.append(f"{'cell':>12s} {'n':>6s} {'rho':>7s} {'perm5':>7s} {'perm95':>7s} {'P(|null|>=|rho|)':>16s} {'ph5':>7s} {'ph17':>7s} {'ph41':>7s} {'shuf':>7s} {'rho|nmin':>9s} {'rho_aligned':>11s} {'rho_fb':>7s} {'rho_cdl':>8s} {'KW p':>8s} {'rho(dist,nmin)':>14s}")
    for tf in TFS + ("all",):
        for side in SIDES + ("both",):
            m = np.ones(len(ep), bool)
            if tf != "all":
                m &= (ep["tf"] == tf).to_numpy()
            if side != "both":
                m &= (ep["side"] == side).to_numpy()
            e = ep[m]
            if len(e) < MIN_CELL:
                continue
            d = e["dist"].to_numpy(); z = e["zsum_z"].to_numpy(); nm = e["n_min"].to_numpy(float)
            rho, null = spearman_perm(d, z, n_perm, rng)
            p_perm = float(np.mean(np.abs(null) >= abs(rho)))
            pl = {k: float(stats.spearmanr(d, e[f"zsum_{k}"]).statistic) for k in ZKEYS[1:]}
            part = partial_spearman(d, z, nm)
            r_al = float(stats.spearmanr(d, e["zsum_aligned"]).statistic)
            r_fb = float(stats.spearmanr(e["dist_fb"], z).statistic)
            r_cdl = float(stats.spearmanr(e["dist_cdl"], z).statistic)
            r_nm = float(stats.spearmanr(d, nm).statistic)
            groups = [d[z == v] for v in np.unique(z) if (z == v).sum() >= MIN_CELL]
            kw = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else float("nan")
            key = f"{tf}_{side}"
            res[key] = dict(n=int(len(e)), rho=rho, perm5=float(np.percentile(null, 5)), perm95=float(np.percentile(null, 95)),
                            p_perm=p_perm, placebo=pl, rho_partial_nmin=part, rho_aligned=r_al, rho_fb=r_fb, rho_cdl=r_cdl,
                            kw_p=kw, rho_dist_nmin=r_nm)
            lines.append(f"{key:>12s} {len(e):6d} {rho:+7.3f} {res[key]['perm5']:+7.3f} {res[key]['perm95']:+7.3f} {p_perm:16.3f} "
                         f"{pl['ph5']:+7.3f} {pl['ph17']:+7.3f} {pl['ph41']:+7.3f} {pl['shuf']:+7.3f} {part:+9.3f} {r_al:+11.3f} {r_fb:+7.3f} {r_cdl:+8.3f} {kw:8.3f} {r_nm:+14.3f}")
    lines.append(f"\n=== {name}: dist by zsum value: n / median / q25 / q75 / mean (cells n ≥ {MIN_CELL}); last column = whole cell ===")
    byz = {}
    for tf in TFS:
        for side in SIDES:
            e = ep[(ep["tf"] == tf) & (ep["side"] == side)]
            cells = []
            for v in range(-9, 10):
                g = e.loc[e["zsum_z"] == v, "dist"]
                if len(g) >= MIN_CELL:
                    byz[(tf, side, v)] = dict(n=int(len(g)), median=float(g.median()), q25=float(g.quantile(.25)),
                                             q75=float(g.quantile(.75)), mean=float(g.mean()))
                    cells.append(f"{v:+d}:{len(g)}/{g.median():.2f}/{g.quantile(.25):.2f}/{g.quantile(.75):.2f}/{g.mean():.2f}")
            lines.append(f"tf {tf:3d} {side:5s}: " + " ".join(cells) + f" | all {len(e)}/{e['dist'].median():.2f}/{e['dist'].quantile(.25):.2f}/{e['dist'].quantile(.75):.2f}/{e['dist'].mean():.2f}")
    lines.append(f"\n=== {name}: dist by |zsum| (0, 1, 2, 3, 4+) and by aligned sign (against / neutral / with the zone side): n / median / mean ===")
    for tf in TFS:
        for side in SIDES:
            e = ep[(ep["tf"] == tf) & (ep["side"] == side)]
            az = e["zsum_z"].abs().clip(upper=4)
            c1 = " ".join(f"|{v}|:{(az == v).sum()}/{e.loc[az == v, 'dist'].median():.2f}/{e.loc[az == v, 'dist'].mean():.2f}" for v in range(5) if (az == v).sum() >= MIN_CELL)
            al = np.sign(e["zsum_aligned"])
            c2 = " ".join(f"{lab}:{(al == s).sum()}/{e.loc[al == s, 'dist'].median():.2f}/{e.loc[al == s, 'dist'].mean():.2f}" for s, lab in ((-1, "against"), (0, "neutral"), (1, "with")) if (al == s).sum() >= MIN_CELL)
            lines.append(f"tf {tf:3d} {side:5s}: {c1} | {c2}")
    lines.append(f"\n=== {name}: per-tf zone value alone, Spearman ρ(dist, z_h) ===")
    for tf in TFS:
        for side in SIDES:
            e = ep[(ep["tf"] == tf) & (ep["side"] == side)]
            lines.append(f"tf {tf:3d} {side:5s}: " + " ".join(f"z{h} {stats.spearmanr(e['dist'], e[f'z{h}']).statistic:+.3f}" for h in TFS))
    lines.append(f"\n=== {name}: entry = 1_close at the first in-zone minute; adverse / favorable excursion to the candle end (ATR1) by aligned sign ===")
    lines.append(f"{'cell':>14s} {'n':>6s} | adv q50 q75 q90 | fav q50 q75 q90 | P(adv>=0.5) P(adv>=1) P(adv>=2) P(adv>=3) | P(fav>=1) P(fav>=2) P(fav>=3) | P(fav>=2 & adv<1) P(fav>=2 & adv<2)")
    es = {}
    for tf in TFS:
        for side in SIDES:
            e0 = ep[(ep["tf"] == tf) & (ep["side"] == side)]
            al = np.sign(e0["zsum_aligned"])
            for s_, lab in ((-1, "against"), (0, "neutral"), (1, "with"), (None, "all")):
                e = e0 if s_ is None else e0[al == s_]
                if len(e) < MIN_CELL:
                    continue
                av = e["adv"].to_numpy(); fv = e["fav"].to_numpy()
                q = lambda x, k: float(np.quantile(x, k))
                row = dict(n=int(len(e)), adv=[q(av, .5), q(av, .75), q(av, .9)], fav=[q(fv, .5), q(fv, .75), q(fv, .9)],
                           p_adv=[float(np.mean(av >= k)) for k in (0.5, 1, 2, 3)], p_fav=[float(np.mean(fv >= k)) for k in (1, 2, 3)],
                           p_fav2_adv1=float(np.mean((fv >= 2) & (av < 1))), p_fav2_adv2=float(np.mean((fv >= 2) & (av < 2))))
                es[f"{tf}_{side}_{lab}"] = row
                lines.append(f"{tf:>3d} {side:5s} {lab:>7s} {len(e):6d} | {row['adv'][0]:4.2f} {row['adv'][1]:4.2f} {row['adv'][2]:4.2f} | {row['fav'][0]:4.2f} {row['fav'][1]:4.2f} {row['fav'][2]:4.2f} | "
                             + " ".join(f"{v:10.3f}" for v in row["p_adv"]) + " | " + " ".join(f"{v:9.3f}" for v in row["p_fav"]) + f" | {row['p_fav2_adv1']:17.3f} {row['p_fav2_adv2']:17.3f}")
    res["entry_stop"] = es
    res["byz"] = {f"{k[0]}_{k[1]}_{k[2]:+d}": v for k, v in byz.items()}
    return res, "\n".join(lines)


# ------------------------------------------------------------------ plots
def plots(ep: pd.DataFrame, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = ep["dataset"].iloc[0]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    for i, tf in enumerate(TFS):
        for j, side in enumerate(SIDES):
            ax = axes[i, j]
            e = ep[(ep["tf"] == tf) & (ep["side"] == side)]
            vals = [v for v in range(-9, 10) if (e["zsum_z"] == v).sum() >= 20]
            data = [e.loc[e["zsum_z"] == v, "dist"].to_numpy() for v in vals]
            cap = float(e["dist"].quantile(0.99)) if len(e) else 1.0
            ax.boxplot([np.clip(d, 0, cap) for d in data], positions=vals, widths=0.6, showfliers=False)
            for v, d in zip(vals, data):
                ax.scatter(v + rng_jit(len(d)), np.clip(d, 0, cap), s=3, alpha=0.15, color="tab:blue")
                ax.text(v, cap * 1.02, str(len(d)), ha="center", fontsize=7)
            med = [np.median(d) for d in data]
            ax.plot(vals, med, color="tab:red", marker="o", lw=1.5, label="median")
            ax.axhline(float(e["dist"].median()), color="grey", ls="--", lw=1, label="overall median")
            r = stats.spearmanr(e["dist"], e["zsum_z"]).statistic
            ax.set_title(f"{name} tf {tf} {side}: zone range (ATR1) vs zsum — n {len(e)}, Spearman ρ {r:+.3f}")
            ax.set_ylim(0, cap * 1.08); ax.set_xlim(-9.7, 9.7); ax.grid(alpha=0.3)
            if i == 2:
                ax.set_xlabel("zones_sum_value = z15 + z60 + z240 at the first in-zone minute")
            if j == 0:
                ax.set_ylabel("max high − min low over in-zone minutes, in 1-min ATR")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"zone_range_vs_zsum_{name}.png"), dpi=110)
    plt.close(fig)
    # medians by zsum with bootstrap CI, all cells on one axis per tf
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    rng = np.random.default_rng(1)
    for i, tf in enumerate(TFS):
        ax = axes[i]
        for side, col in zip(SIDES, ("tab:green", "tab:red")):
            e = ep[(ep["tf"] == tf) & (ep["side"] == side)]
            vals = [v for v in range(-9, 10) if (e["zsum_z"] == v).sum() >= MIN_CELL]
            med, lo_, hi_ = [], [], []
            for v in vals:
                d = e.loc[e["zsum_z"] == v, "dist"].to_numpy()
                bs = np.array([np.median(rng.choice(d, len(d))) for _ in range(300)])
                med.append(np.median(d)); lo_.append(np.percentile(bs, 5)); hi_.append(np.percentile(bs, 95))
            ax.plot(vals, med, marker="o", color=col, label=f"{side} median (CI90)")
            ax.fill_between(vals, lo_, hi_, color=col, alpha=0.2)
        ax.set_title(f"{name} tf {tf}: median zone range by zsum"); ax.set_xlabel("zsum"); ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("ATR1"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"zone_range_vs_zsum_medians_{name}.png"), dpi=110)
    plt.close(fig)


def rng_jit(n: int) -> np.ndarray:
    return np.random.default_rng(n).uniform(-0.25, 0.25, n)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["2y", "oos2m"])
    ap.add_argument("--perms", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allres = {}
    for name in ("2y", "oos2m"):
        df = load(name)
        series_dir = f"{rep.V}/{name}_link_usdt/swing_sinusoid"
        Z = {tf: rep.combined_zone(name, df, tf, series_dir) for tf in TFS}   # 2y first: freezes thr / const
        if args.only and name != args.only:
            del df
            continue
        if name == "oos2m":   # provenance: sidecar built by causal_zones vs the replay's zones
            p = f"{rep.V}/oos2m_link_usdt/df_with_cb_swing_filter.pkl"
            if os.path.exists(p):
                sw = pd.read_pickle(p)
                for tf in TFS:
                    agree = float(np.mean(sw[f"{tf}_swing_zone"].to_numpy(int) == Z[tf]["z"]))
                    print(f"[oos2m] sidecar {tf}_swing_zone vs replay z: agreement {agree:.3f}", flush=True)
                del sw
        ep = build_episode_table(name, df, Z)
        del df
        ep.to_csv(os.path.join(args.out, f"zone_range_episodes_{name}.csv"), index=False)
        res, txt = analyse(ep, args.perms)
        print(txt, flush=True)
        with open(os.path.join(args.out, f"zone_range_vs_zsum_{name}.md"), "w") as f:
            f.write(txt)
        plots(ep, args.out)
        allres[name] = res
    allres["frozen"] = rep.FROZEN
    with open(os.path.join(args.out, "zone_range_vs_zsum.json"), "w") as f:
        json.dump(allres, f, indent=1, default=float)
    print("frozen:", json.dumps(rep.FROZEN))


if __name__ == "__main__":
    main()
