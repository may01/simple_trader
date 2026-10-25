#!/usr/bin/env python3
"""Combined swing-zone filter on CB zone entries over 2y (2023-01 → 2024-12) and oos2m (2025-01 → 2025-03).

Entry stand-in = the trend-ma-length / higher-tf-trend flow (set-C candidate minutes = inzone and overshoot
pass; first passing minute per tf candle; entry = 1_close; candle stop = forming 15-min extreme ∓ trailing-60
mean 1-min range; cbnc target; max hold H min; size min(100/dist, 50000/entry); no fees).

Zones (causal, from the swing_sinusoid series sidecars):
  cci_ma8 zigzag with threshold 0.5 σ (σ = 2y-train std, frozen): a running extreme becomes an apex when the
  series retraces ≥ threshold from it (confirmation row c); top apex → −1, bottom → +1, active on closed rows
  [max(apex + 3, c), apex + const + 3], const = 2y-train median apex spacing; later apex overrides.
  0_rsi = sym0 move class of rsi_ma8 diff (frozen cuts, classification.MOVE_CLASS_CUTS) with ±3 → 0.
  z_tf = zone + 0_rsi at the last closed candle, usable from the next minute.
Filter = experiments.swing_sinusoid.cb_filter.allowed(tf, side, z15, z60, z240).
Variants: none, e25 (live trend gate), swing, e25+swing, e25 + each level alone, e25 + placebo (z rolled by K
closed rows of each tf), e25 + z shuffled in 1-day blocks.
usage: python -m experiments.swing_sinusoid.cb_filter_replay [--hold=60] [--only=2y|oos2m]
"""
import sys, time, resource, json, os
import numpy as np, pandas as pd, talib
from collections import defaultdict

from experiments.swing_sinusoid.cb_filter import allowed
from indicators.library.classification import MOVE_CLASS_CUTS

V = os.environ.get("SWING_VOL", "/trader_data_long/train")
H = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--hold=")), 60))
ONLY = next((a.split("=")[1] for a in sys.argv if a.startswith("--only=")), None)
TFS = (15, 60, 240); SIDES = ("long", "short"); W = 105; X = 0.5
FROZEN25 = {15: (0.0018696074433414492, 0.08854754765129406), 60: (0.007586181744923595, 0.17365581448454287),
            240: (0.03083283482398462, 0.3443346525283253)}
THR = {15: {"long": 0.3477, "short": 0.3808}, 60: {"long": 0.3756, "short": 0.3796}, 240: {"long": 0.4693, "short": 0.5178}}
RISK = 100.0; CAP = 100_000 * 0.5; PLACEHOLDER = 0.002; RANGE_N = 60
TRAIN_END = pd.Timestamp("2024-06-30 23:59:59+00:00")
CONFIRM, WIN = 3, 3
PLACEBO_K = (5, 17, 41)
FROZEN = {}   # (tf) -> dict(thr, const) from the 2y train rows


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def candle_id(closed):
    return np.concatenate(([0], np.cumsum(closed)[:-1]))


def load(name):
    t0 = time.time()
    if name == "2y":
        df = pd.read_pickle(f"{V}/2y_az_link_usdt/df_with_indicators.pkl")
        keep = ["1_high", "1_low", "1_close"] + [f"{tf}_{c}" for tf in TFS for c in ("close", "is_closed")]
        df = df[keep].copy()
        cb = pd.read_pickle(f"{V}/2y_link_usdt/df_with_candle_bounds.pkl")
        cb = cb[[f"{tf}_{c}" for tf in TFS for c in ("cb_low", "cb_high", "cb_low_std", "cb_high_std", "cb_inzone_long", "cb_inzone_short")]].copy()
        nc = pd.read_pickle(f"{V}/2y_link_usdt/df_with_candle_bounds_nc.pkl")
        nc = nc[[f"{tf}_{c}" for tf in TFS for c in ("cbnc_high", "cbnc_low")]].copy()
        df = pd.concat([df, cb.reindex(df.index), nc.reindex(df.index)], axis=1)
        del cb, nc
    else:
        df = pd.read_pickle(f"{V}/oos2m_link_usdt/df_with_indicators.pkl")
        keep = ["1_high", "1_low", "1_close"] + [f"{tf}_{c}" for tf in TFS for c in ("close", "is_closed", "cb_low", "cb_high", "cb_low_std", "cb_high_std",
                                                                               "cb_inzone_long", "cb_inzone_short", "cbnc_high", "cbnc_low")]
        df = df[keep].copy()
    print(f"[{name}] {df.shape} {df.index[0]} -> {df.index[-1]} loaded {time.time() - t0:.0f}s RSS {rss():.0f} MB", flush=True)
    return df


# ------------------------------------------------------------------ zones
def zigzag(s, thr):
    """Causal alternating apexes: list of (i_apex, sign, i_confirm).

    A running high is confirmed as a top apex when the series falls ≥ thr below it (only after a
    bottom, or at the start); a running low becomes a bottom apex when the series rises ≥ thr above
    it (only after a top, or at the start). After a confirmation the opposite tracker restarts from
    the apex."""
    out = []
    n = len(s)
    i_max = i_min = 0
    last_sign = 0
    for i in range(1, n):
        v = s[i]
        if not np.isfinite(v):
            continue
        if not np.isfinite(s[i_max]) or v > s[i_max]:
            i_max = i
        if not np.isfinite(s[i_min]) or v < s[i_min]:
            i_min = i
        if last_sign <= 0 and np.isfinite(s[i_max]) and s[i_max] - v >= thr:      # top confirmed
            out.append((i_max, 1, i)); last_sign = 1
            seg = s[i_max + 1:i + 1]
            i_min = i_max + 1 + int(np.nanargmin(seg)) if len(seg) else i
        elif last_sign >= 0 and np.isfinite(s[i_min]) and v - s[i_min] >= thr:    # bottom confirmed
            out.append((i_min, -1, i)); last_sign = -1
            seg = s[i_min + 1:i + 1]
            i_max = i_min + 1 + int(np.nanargmax(seg)) if len(seg) else i
    return out


def zone_series(s, apexes, const, n):
    z = np.zeros(n, dtype=np.int8)
    for i_apex, sign, c in apexes:
        a, b = max(i_apex + CONFIRM, c), min(int(round(i_apex + const + WIN)), n - 1)
        if a <= b:
            z[a:b + 1] = -sign
    return z


def combined_zone(name, df, tf, series_dir):
    fr = pd.read_pickle(os.path.join(series_dir, f"series_tf{tf}.pkl"))
    s = fr["cci_ma8"].to_numpy(float)
    if name == "2y":
        tr = fr.index <= TRAIN_END
        thr = 0.5 * float(np.nanstd(s[tr]))
        ap_tr = zigzag(s[tr], thr)
        const = float(np.median(np.diff([a for a, _, _ in ap_tr])))
        FROZEN[tf] = {"thr": thr, "const": const, "n_apex_train": len(ap_tr)}
    thr, const = FROZEN[tf]["thr"], FROZEN[tf]["const"]
    ap = zigzag(s, thr)
    zone = zone_series(s, ap, const, len(s))
    rsi = fr["rsi_ma8"].to_numpy(float)
    diff = np.r_[np.nan, rsi[1:] - rsi[:-1]]
    mc = np.digitize(np.nan_to_num(diff, nan=0.0), MOVE_CLASS_CUTS[tf]) - 3
    rsi0 = np.where(np.abs(mc) == 3, 0, mc).astype(np.int8)
    zc = (zone + rsi0).astype(np.int8)
    lag = np.array([c - a for a, _, c in ap])
    print(f"[{name}] tf{tf} zones: thr {thr:.1f} const {const:.0f} apexes {len(ap)} (train {FROZEN[tf]['n_apex_train']}), "
          f"confirm lag median {np.median(lag):.0f} p75 {np.percentile(lag, 75):.0f}; closed-row shares zone −1/0/+1 "
          f"{np.mean(zone == -1):.2f}/{np.mean(zone == 0):.2f}/{np.mean(zone == 1):.2f}; z {dict(zip(*np.unique(zc, return_counts=True)))}", flush=True)
    # broadcast to 1-min rows: value of the last closed candle, usable from the next minute
    def to_min(arr):
        ser = pd.Series(arr, index=fr.index).reindex(df.index).ffill().shift(1)
        return ser.fillna(0).to_numpy().astype(np.int8)
    out = {"z": to_min(zc)}
    for k in PLACEBO_K:
        out[f"ph{k}"] = to_min(np.roll(zc, k))
    # block-shuffled placebo: shuffle 1-day blocks of the closed-row z series
    rng = np.random.default_rng(42 + tf)
    per_day = 1440 // tf
    blocks = [zc[i:i + per_day] for i in range(0, len(zc), per_day)]
    rng.shuffle(blocks)
    out["shuf"] = to_min(np.concatenate(blocks)[:len(zc)])
    return out


def lut(tf, side):
    t = np.zeros((7, 7, 7), dtype=bool)   # [z15+3, z60+3, z240+3]
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                t[a + 3, b + 3, c + 3] = allowed(tf, side, a, b, c)
    return t


def level_lut(level, tf, side):
    """Flag from one level alone: 240 rule / 60 rule / 15 rule (others neutral = 0)."""
    t = np.zeros((7, 7, 7), dtype=bool)
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                if level == 240:
                    t[a + 3, b + 3, c + 3] = allowed(tf, side, 0, 0, c) if c != 0 else True
                elif level == 60:
                    t[a + 3, b + 3, c + 3] = allowed(tf, side, 0, b, 0)
                else:
                    t[a + 3, b + 3, c + 3] = allowed(tf, side, a, 0, 0) if tf == 15 else True
    return t


# ------------------------------------------------------------------ harness (verbatim from the higher-tf-trend stand-in)
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


def simulate_all(df, tf, side, cand):
    hi, lo, cl = (df[c].to_numpy(float) for c in ("1_high", "1_low", "1_close"))
    n = len(cl)
    cid15 = candle_id((df["15_is_closed"] == True).to_numpy())  # noqa: E712
    s = pd.Series(lo if side == "long" else hi)
    ext15 = (s.groupby(cid15).cummin() if side == "long" else s.groupby(cid15).cummax()).to_numpy()
    rng = pd.Series(hi - lo).shift(1).rolling(RANGE_N, min_periods=1).mean().to_numpy()
    tgt = df[f"{tf}_cbnc_high" if side == "long" else f"{tf}_cbnc_low"].to_numpy(float)
    idx = np.flatnonzero(cand)
    pnl = np.zeros(len(idx)); reason = np.zeros(len(idx), dtype=np.int8)
    size_arr = np.zeros(len(idx)); dist_arr = np.zeros(len(idx))
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
        pnl[k] = (x - e) * size * sgn; reason[k] = r; size_arr[k] = size; dist_arr[k] = dist
    return idx, pnl, reason, size_arr, dist_arr


def run(name):
    df = load(name)
    series_dir = f"{V}/{'2y' if name == '2y' else 'oos2m'}_link_usdt/swing_sinusoid"
    Z = {tf: combined_zone(name, df, tf, series_dir) for tf in TFS}
    CLS = {}; CID = {}
    for tf in TFS:
        CLS[tf], CID[tf] = classes(df, tf)
    out = {}
    for tf in TFS:
        cid = CID[tf]
        for side in SIDES:
            inz = df[f"{tf}_cb_inzone_{side}"].fillna(False).to_numpy(bool)
            b = df[f"{tf}_cb_low" if side == "long" else f"{tf}_cb_high"].to_numpy(float)
            w = df[f"{tf}_cb_low_std" if side == "long" else f"{tf}_cb_high_std"].to_numpy(float)
            x = df["1_low" if side == "long" else "1_high"].to_numpy(float)
            over = (b - x) / w if side == "long" else (x - b) / w
            cand = inz & ~(over > THR[tf][side])
            idx, pnl, reason, size, dist = simulate_all(df, tf, side, cand)
            bad = -1 if side == "long" else 1
            e25 = CLS[tf][idx] != bad
            one = np.ones(len(idx), dtype=bool)
            L = lut(tf, side)
            def flag(key):
                return L[Z[15][key][idx] + 3, Z[60][key][idx] + 3, Z[240][key][idx] + 3]
            sw = flag("z")
            flags = {"none": one, "e25": e25, "swing": sw, "e25+swing": e25 & sw}
            for lv in (240, 60, 15):
                LL = level_lut(lv, tf, side)
                flags[f"e25+r{lv}"] = e25 & LL[Z[15]["z"][idx] + 3, Z[60]["z"][idx] + 3, Z[240]["z"][idx] + 3]
            for k in PLACEBO_K:
                flags[f"e25+ph{k}"] = e25 & flag(f"ph{k}")
            flags["e25+shuf"] = e25 & flag("shuf")
            cid_e = cid[idx]
            res = {}
            for v, fl in flags.items():
                first = np.zeros(len(idx), dtype=bool); seen = set()
                for k in np.flatnonzero(fl):
                    c = cid_e[k]
                    if c not in seen:
                        seen.add(c); first[k] = True
                p = pnl[first]
                res[v] = dict(n=int(first.sum()), net=float(p.sum()), pf=float(p[p > 0].sum() / abs(p[p < 0].sum())) if (p < 0).any() else float("inf"),
                              wr=float(np.mean(p > 0)) if len(p) else float("nan"), stop=int((reason[first] == 1).sum()), tgt=int((reason[first] == 2).sum()),
                              hold=int((reason[first] == 3).sum()), first_idx=np.flatnonzero(first), pnl=p, pass_rate=float(fl.mean()))
            # quality of the e25 entries the swing filter removes / keeps
            fi = res["e25"]["first_idx"]; keep = sw[fi]
            rem = {}
            for lab, m in (("kept", keep), ("removed", ~keep)):
                p = res["e25"]["pnl"][m]
                rem[lab] = dict(n=int(m.sum()), net=float(p.sum()), mean=float(p.mean()) if m.sum() else float("nan"),
                                pf=float(p[p > 0].sum() / abs(p[p < 0].sum())) if (p < 0).any() else float("inf"),
                                stop=float(np.mean(reason[fi][m] == 1)) if m.sum() else float("nan"))
            zz = {h: Z[h]["z"][idx][fi] for h in TFS}
            byz = {}
            for h in TFS:
                for val in range(-3, 4):
                    m = zz[h] == val
                    if m.sum():
                        p = res["e25"]["pnl"][m]
                        byz[(h, val)] = (int(m.sum()), float(p.sum()), float(p.mean()))
            out[(tf, side)] = dict(res=res, idx=idx, pnl=pnl, cid=cid_e, rem=rem, byz=byz)
            r = res["e25"]; q = res["e25+swing"]
            print(f"[{name}] tf {tf:3d} {side:5s}: cand {len(idx)}; e25 {r['n']} net {r['net']:+.0f} PF {r['pf']:.2f} | e25+swing {q['n']} net {q['net']:+.0f} PF {q['pf']:.2f} "
                  f"(pass {sw.mean():.2f}) | removed: n {rem['removed']['n']} net {rem['removed']['net']:+.0f} mean {rem['removed']['mean']:+.2f} | RSS {rss():.0f} MB", flush=True)
    del df
    return out


ORDER = ["none", "e25", "swing", "e25+swing", "e25+r240", "e25+r60", "e25+r15", "e25+ph5", "e25+ph17", "e25+ph41", "e25+shuf"]


def table(out, name):
    print(f"\n=== {name}, hold {H} min: entries, net USDT, PF, win rate, stop/target/hold exits ===")
    print(f"{'variant':>10s} | " + " | ".join(f"{'tf' + str(tf) + ' ' + side:>28s}" for tf in TFS for side in SIDES) + f" | {'ALL n':>7s} {'net':>8s} {'PF':>5s}")
    for v in ORDER:
        cells = []; tot_n = 0; tot = 0.0; w = 0.0; l = 0.0
        for tf in TFS:
            for side in SIDES:
                r = out[(tf, side)]["res"][v]
                cells.append(f"{r['n']:5d} {r['net']:+7.0f} {r['pf']:4.2f} {r['wr']:.2f} {r['stop']}/{r['tgt']}/{r['hold']}")
                tot_n += r["n"]; tot += r["net"]; p = r["pnl"]; w += p[p > 0].sum(); l += abs(p[p < 0].sum())
        print(f"{v:>10s} | " + " | ".join(f"{c:>28s}" for c in cells) + f" | {tot_n:7d} {tot:+8.0f} {w / l if l else float('inf'):5.2f}")
    print(f"--- paired vs e25 by candle (Δnet, bootstrap CI90 over candles, P(Δ<0)); per tf+side and all ---")
    rng = np.random.default_rng(0)
    for v in ORDER:
        if v == "e25":
            continue
        parts = []; all_d = []
        for tf in TFS:
            for side in SIDES:
                o = out[(tf, side)]; r = o["res"][v]; b = o["res"]["e25"]
                dv = defaultdict(float); db = defaultdict(float)
                for k, p in zip(r["first_idx"], r["pnl"]):
                    dv[o["cid"][k]] += p
                for k, p in zip(b["first_idx"], b["pnl"]):
                    db[o["cid"][k]] += p
                keys = set(dv) | set(db)
                d = np.array([dv.get(c, 0.0) - db.get(c, 0.0) for c in keys]) if keys else np.zeros(1)
                boot = np.array([rng.choice(d, len(d), replace=True).sum() for _ in range(2000)])
                parts.append(f"{d.sum():+6.0f} [{np.percentile(boot, 5):+6.0f},{np.percentile(boot, 95):+6.0f}] {np.mean(boot < 0):.2f}")
                all_d.append(d)
        d = np.concatenate(all_d); boot = np.array([rng.choice(d, len(d), replace=True).sum() for _ in range(2000)])
        print(f"{v:>10s} | " + " | ".join(f"{p:>28s}" for p in parts) + f" | {d.sum():+8.0f} [{np.percentile(boot, 5):+7.0f},{np.percentile(boot, 95):+7.0f}] {np.mean(boot < 0):.2f}")
    print(f"--- {name}: pass rate of each flag at the candidate minutes ---")
    for tf in TFS:
        for side in SIDES:
            o = out[(tf, side)]
            print(f"tf {tf:3d} {side:5s}: " + " ".join(f"{v} {o['res'][v]['pass_rate']:.3f}" for v in ORDER[1:]))
    print(f"--- {name}: e25 entries kept / removed by the swing filter: n, net, mean, PF, stop share ---")
    for tf in TFS:
        for side in SIDES:
            o = out[(tf, side)]
            print(f"tf {tf:3d} {side:5s}: " + " | ".join(f"{lab} n {r['n']:6d} net {r['net']:+8.0f} mean {r['mean']:+6.2f} PF {r['pf']:4.2f} stop {r['stop']:.2f}" for lab, r in o["rem"].items()))
    print(f"--- {name}: e25 entries by the combined zone value of each tf at the entry minute: n / net / mean ---")
    for tf in TFS:
        for side in SIDES:
            o = out[(tf, side)]
            for h in TFS:
                cells = " ".join(f"{val:+d}:{o['byz'][(h, val)][0]}/{o['byz'][(h, val)][1]:+.0f}/{o['byz'][(h, val)][2]:+.1f}" for val in range(-3, 4) if (h, val) in o["byz"])
                print(f"tf {tf:3d} {side:5s} by z{h:<3d}: {cells}")


if __name__ == "__main__":
    for name in ("2y", "oos2m"):
        if ONLY and name != ONLY:
            continue
        out = run(name)
        table(out, name)
    print("frozen:", json.dumps(FROZEN))
