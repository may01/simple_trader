"""Driver: quantile-based entry / stop / target armed by the first filtered
cb marker of a candle — fit on 2y, apply to oos2m.

Marker = ``{tf}_cb_inzone_{side}`` minus the extdone veto (run_zone_ext). The
FIRST marker minute t0 of a candle arms one order (long shown; short is the
same code on negated prices):

    m = 1_low[t0]      a = atr1[t0]      b = progress bucket of t0
    E = m - q_ext[b, ENTRY_Q] * a        limit entry   (ENTRY_Q = 0 -> market at close[t0])
    S = m - q_ext[b, STOP_Q]  * a        stop
    T = E + q_mfe[b, tq]      * a        target

q_ext: quantiles of the remaining extension after the first marker,
(m - min low over the rest of the candle) / a, per (tf, side, bucket) on 2y.
q_mfe: quantiles of the favourable excursion of FILLED 2y trades,
(max high from the minute after the fill to candle close - E) / a. The
target quantile tq is picked per (tf, side, variant) by 2y net pnl and
frozen for oos2m.

Replay: fill = first minute after t0 with low <= E (unfilled orders die at
candle close). Stop is checked from the fill minute inclusive, target from
the next minute; both in one minute -> stop. At every candle close with the
position still open the NEW candle's cb bounds (L', H') are read:
    worse = L' < S  (expected low under the stop)  or  H' < T  (target out of reach)
    if worse: T = min(T, H'); if T <= close -> exit at close (market)
The stop never moves. One position per (tf, side); markers firing while it
is open are skipped. After MAX_HOLD candles the position exits at close.
pnl is net of 2 x FEE.

Artifacts: {EV_OUT}/sltgt_report.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evlib import SIDES, TFS, candle_id  # noqa: E402
from run_zone_ext import atr1_ma5, extdone  # noqa: E402

VOL = "/trader_data_long/train"
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
TRAIN_CB = f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl"
OOS_DIR = os.environ.get("EV_DATASET_DIR", f"{VOL}/oos2m_link_usdt")
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
FEE = float(os.environ.get("EV_FEE", 0.001))
MAX_HOLD = int(os.environ.get("SLTGT_MAX_HOLD", 8))
# carry  = rule above; strict = also exit at close when L' < S;
# candle = always exit at the arming candle's close (no carry-over)
MODE = os.environ.get("SLTGT_MODE", "carry")
TAG = os.environ.get("SLTGT_TAG", "")
# ema_25 slope class of the LAST CLOSED tf candle (run_ema_slope convention, x = EMA_X):
# with = no long on fall, no short on rise; against = only the trades "with" removes
EMA = os.environ.get("SLTGT_EMA", "off")
EMA_X = float(os.environ.get("EMA_X", 0.5))
# drop markers whose 1-min extreme is more than CAP_SD band sd past the cb bound (<0 = off)
CAP_SD = float(os.environ.get("SLTGT_CAP_SD", -1))
# same cap as a quantile of the 2y marker overshoot (run_cb_overshoot), e.g. 0.75 (<0 = off)
CAP_Q = float(os.environ.get("SLTGT_CAP_Q", -1))
MIN_BUCKET = 50
EDGES = (0.25, 0.5, 0.75)
TGT_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# (name, entry quantile, stop quantile); entry 0 = market at the marker close
QS = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)             # extension quantiles available to levels()
_DEF = "q50/q90,q50/q95,mkt/q90,mkt/q95"
VARIANTS = tuple((v, 0.0 if v.startswith("mkt") else int(v.split("/")[0][1:]) / 100,
                  int(v.split("/")[1][1:]) / 100)
                 for v in os.environ.get("SLTGT_VARIANTS", _DEF).split(","))
EXITS = ("target", "retarget", "stop", "close", "cap")   # retarget = hit after T was lowered


def frame(df: pd.DataFrame, cb: pd.DataFrame, atr: np.ndarray, tf: int, side: str,
          cls: np.ndarray | None = None, thr: float | None = None) -> dict:
    """Arrays in long-equivalent space (short = negated prices) + armed rows."""
    n = len(df)
    cid = candle_id(df, tf)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    rep = np.diff(np.r_[starts, n])
    prog = (np.arange(n) - np.repeat(starts, rep) + 1) / tf
    cb = cb.reindex(df.index)
    hi, lo, cl = (df[c].to_numpy() for c in ("1_high", "1_low", "1_close"))
    L, H = cb[f"{tf}_cb_low"].to_numpy(), cb[f"{tf}_cb_high"].to_numpy()
    if side == "short":
        hi, lo, cl, L, H = -lo, -hi, -cl, -H, -L
    veto = extdone(df, cid, prog, atr, tf, "high" if side == "long" else "low")
    inz = cb[f"{tf}_cb_inzone_{side}"].fillna(False).to_numpy(dtype=bool)
    m = inz & ~veto & np.isfinite(atr) & (atr > 0)
    if CAP_SD >= 0 or CAP_Q >= 0:
        z = (L - lo) / cb[f"{tf}_cb_{'low' if side == 'long' else 'high'}_std"].to_numpy()
        if CAP_Q >= 0 and thr is None:                # fit on this frame (2y), reuse for oos
            thr = float(np.nanquantile(z[m], CAP_Q))
        m &= z <= (thr if CAP_Q >= 0 else CAP_SD)
    if cls is not None and EMA != "off":
        bad = cls == (-1 if side == "long" else 1)    # long on fall / short on rise
        m &= ~bad if EMA == "with" else bad
    first = m & (pd.Series(m).groupby(cid).cumsum().to_numpy() == 1)
    i0 = np.flatnonzero(first)
    end = np.repeat(starts + rep, rep)[i0]              # exclusive end of the arming candle
    return {"hi": hi, "lo": lo, "cl": cl, "L": L, "H": H, "ends": starts + rep,
            "i0": i0, "end": end, "m": lo[i0], "a": atr[i0],
            "b": np.digitize(prog[i0], EDGES), "n": n, "thr": thr}


def bucket_q(x: np.ndarray, b: np.ndarray, qs) -> np.ndarray:
    """[4 buckets, len(qs)] quantiles; thin buckets fall back to the pooled row."""
    pooled = np.quantile(x, qs) if len(x) else np.zeros(len(qs))
    return np.array([np.quantile(x[b == k], qs) if (b == k).sum() >= MIN_BUCKET else pooled
                     for k in range(4)])


def ext_atr(f: dict) -> np.ndarray:
    out = np.zeros(len(f["i0"]))
    for j, (i, e) in enumerate(zip(f["i0"], f["end"])):
        if i + 1 < e:
            out[j] = max(0.0, f["m"][j] - f["lo"][i + 1:e].min()) / f["a"][j]
    return out


def fills(f: dict, E: np.ndarray | None) -> np.ndarray:
    """Fill row per armed marker (-1 = unfilled); E None = market at t0."""
    if E is None:
        return f["i0"].copy()
    out = np.full(len(E), -1)
    for j, (i, e) in enumerate(zip(f["i0"], f["end"])):
        hit = np.flatnonzero(f["lo"][i + 1:e] <= E[j])
        if len(hit):
            out[j] = i + 1 + hit[0]
    return out


def mfe_atr(f: dict, E: np.ndarray, fl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Favourable excursion of filled trades to candle close, in atr1; + their buckets."""
    x, b = [], []
    for j, k in enumerate(fl):
        if k < 0:
            continue
        e = f["end"][j]
        x.append(max(0.0, f["hi"][k + 1:e].max() - E[j]) / f["a"][j] if k + 1 < e else 0.0)
        b.append(f["b"][j])
    return np.array(x), np.array(b)


def replay(f: dict, E: np.ndarray, S: np.ndarray, T0: np.ndarray, fl: np.ndarray) -> dict:
    hi, lo, cl, L, H, ends, n = (f[k] for k in ("hi", "lo", "cl", "L", "H", "ends", "n"))
    pnl, kind, hold = [], [], []
    armed = busy = 0
    for j, i in enumerate(f["i0"]):
        if i < busy:                                  # position still open
            continue
        armed += 1
        k = fl[j]
        if k < 0:
            continue
        e, s, t, px, ex = f["end"][j], S[j], T0[j], None, None
        low = False                                   # target lowered by a new candle's bounds
        a0, a1 = k, k + 1                             # stop from fill minute, target from next
        if E is None or k == i:                       # market entry at close[t0]
            a0 = k + 1
        for c in range(MAX_HOLD):
            st = np.flatnonzero(lo[a0:e] <= s)
            tg = np.flatnonzero(hi[a1:e] >= t)
            js = a0 + st[0] if len(st) else n
            jt = a1 + tg[0] if len(tg) else n
            if js < n or jt < n:
                px, ex, busy = (s, "stop", js + 1) if js <= jt else (t, "retarget" if low else "target", jt + 1)
                break
            last = cl[e - 1]
            if e >= n or not np.isfinite(L[e]) or c == MAX_HOLD - 1:
                px, ex, busy = last, "cap", e
                break
            if MODE == "candle" or (MODE == "strict" and L[e] < s):
                px, ex, busy = last, "close", e
                break
            if L[e] < s or H[e] < t:                  # new candle's bounds are worse
                low = low or H[e] < t
                t = min(t, H[e])
                if t <= last:
                    px, ex, busy = last, "close", e
                    break
            a0 = a1 = e
            e = ends[np.searchsorted(ends, e, side="right")]
        ent = E[j]
        pnl.append((px - ent) / abs(ent) - 2 * FEE)
        kind.append(ex)
        hold.append(c + 1)
    pnl, kind = np.array(pnl), np.array(kind)
    nf = len(pnl)
    out = {"n_armed": armed, "n_fills": nf, "fill_rate": nf / armed if armed else 0.0,
           "pnl_pct_per_fill": float(100 * pnl.mean()) if nf else 0.0,
           "gross_pct_per_fill": float(100 * (pnl.mean() + 2 * FEE)) if nf else 0.0,
           "sum_pnl_pct": float(100 * pnl.sum()),
           "hold_candles": float(np.mean(hold)) if nf else 0.0}
    for x in EXITS:
        mk = kind == x
        out[f"{x}_rate"] = float(mk.mean()) if nf else 0.0
        out[f"{x}_pnl_pct"] = float(100 * pnl[mk].mean()) if mk.any() else 0.0
    return out


def levels(f: dict, qe: np.ndarray, eq: float, sq: float):
    """E (None for market), entry price array, S. qe columns = QS."""
    col = {q: k for k, q in enumerate(QS)}
    S = f["m"] - qe[f["b"], col[sq]] * f["a"]
    if eq == 0.0:
        return None, f["cl"][f["i0"]], S
    E = f["m"] - qe[f["b"], col[eq]] * f["a"]
    return E, E, S


def closed_slope(df: pd.DataFrame, tf: int) -> pd.Series:
    c = df.loc[df[f"{tf}_is_closed"] == True, f"{tf}_ema_25"]  # noqa: E712
    return ((c - c.shift(1)) / c * 100.0).dropna()


def ema_cls(df: pd.DataFrame, tf: int, mu: float, sd: float) -> np.ndarray:
    """Per 1-min row: class (-1/0/1) of the last closed candle's ema_25 slope."""
    z = ((closed_slope(df, tf) - mu) / sd).reindex(df.index).shift(1).ffill().to_numpy()
    return np.where(z < -EMA_X, -1.0, np.where(z > EMA_X, 1.0, np.where(np.isnan(z), np.nan, 0.0)))


def load(ind: str, cbp: str, rebuild_atr: bool):
    need = ["1_high", "1_low", "1_close"] + [f"{tf}_is_closed" for tf in TFS] + [f"{tf}_ema_25" for tf in TFS]
    df = pd.read_pickle(ind)
    atr = atr1_ma5(df) if rebuild_atr else df["1_atr_14_ma_5"].to_numpy()
    return df[need], pd.read_pickle(cbp), atr


def main() -> None:
    tr = load(TRAIN_IND, TRAIN_CB, True)
    oos = load(os.path.join(OOS_DIR, "df_with_indicators.pkl"),
               os.path.join(OOS_DIR, "df_with_candle_bounds.pkl"), False)
    report = {"fee": FEE, "max_hold": MAX_HOLD, "cells": []}
    for tf in TFS:
        for side in SIDES:
            d = closed_slope(tr[0], tf)
            mu, sd = float(d.mean()), float(d.std())
            ft = frame(*tr, tf, side, ema_cls(tr[0], tf, mu, sd))
            fo = frame(*oos, tf, side, ema_cls(oos[0], tf, mu, sd), ft["thr"])
            qe = bucket_q(ext_atr(ft), ft["b"], QS)
            qe_oos = bucket_q(ext_atr(fo), fo["b"], QS)
            cell = {"tf": tf, "side": side, "q_ext_2y": qe.round(2).tolist(),
                    "q_ext_oos": qe_oos.round(2).tolist(),
                    "n_first_2y": [int((ft["b"] == k).sum()) for k in range(4)],
                    "n_first_oos": [int((fo["b"] == k).sum()) for k in range(4)],
                    "variants": []}
            for name, eq, sq in VARIANTS:
                Et, pt, St = levels(ft, qe, eq, sq)
                Eo, po, So = levels(fo, qe, eq, sq)
                flt, flo = fills(ft, Et), fills(fo, Eo)
                qm = bucket_q(*mfe_atr(ft, pt, flt), TGT_GRID)
                grid = []
                for gi, tq in enumerate(TGT_GRID):
                    r = replay({**ft}, pt, St, pt + qm[ft["b"], gi] * ft["a"], flt)
                    grid.append({"tq": tq, **r})
                best = max(range(len(grid)), key=lambda g: grid[g]["sum_pnl_pct"])
                ro = replay({**fo}, po, So, po + qm[fo["b"], best] * fo["a"], flo)
                risk = float(np.median(100 * (po - So) / np.abs(po)))
                rew = float(np.median(100 * qm[fo["b"], best] * fo["a"] / np.abs(po)))
                v = {"name": name, "tq": TGT_GRID[best], "q_mfe_2y": qm[:, best].round(2).tolist(),
                     "risk_pct_med": risk, "reward_pct_med": rew,
                     "train": grid[best], "oos": ro,
                     "train_grid": [{k: g[k] for k in ("tq", "n_fills", "target_rate", "stop_rate",
                                                       "pnl_pct_per_fill", "sum_pnl_pct")}
                                    for g in grid]}
                cell["variants"].append(v)
                print(tf, side, name, f"tq={v['tq']} R {risk:.2f}% T {rew:.2f}% | 2y fills "
                      f"{grid[best]['n_fills']} pnl/fill {grid[best]['pnl_pct_per_fill']:+.3f} | oos "
                      f"armed {ro['n_armed']} fills {ro['n_fills']} tgt {ro['target_rate']:.2f} "
                      f"stop {ro['stop_rate']:.2f} retgt {ro['retarget_rate']:.2f} close {ro['close_rate']:.2f} cap {ro['cap_rate']:.2f} "
                      f"pnl/fill {ro['pnl_pct_per_fill']:+.3f} sum {ro['sum_pnl_pct']:+.1f}", flush=True)
            report["cells"].append(cell)
    with open(os.path.join(OUT, f"sltgt_report_{MODE}{TAG}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    write_md(report, os.path.join(OUT, f"sltgt_report_{MODE}{TAG}.md"))
    print("wrote", OUT)


def write_md(report: dict, path: str) -> None:
    o = ["# Quantile entry / stop / target on first filtered cb marker — fit 2y, apply oos2m", "",
         f"mode {MODE}, ema filter {EMA}, cap sd {CAP_SD}, cap q {CAP_Q}, fee {report['fee']}/side, max hold {report['max_hold']} candles, target quantile grid {TGT_GRID}",
         "", "## Remaining extension after the FIRST marker, atr1 units " + " / ".join(f"q{int(q*100)}" for q in QS) + "", "",
         "| tf | side | bucket | n 2y | 2y | n oos | oos2m |", "|---|---|---|---|---|---|---|"]
    names = ("0–25 %", "25–50 %", "50–75 %", "75–100 %")
    fmt = lambda r: " / ".join(f"{x:.1f}" for x in r)  # noqa: E731
    for c in report["cells"]:
        for k in range(4):
            o.append(f"| {c['tf']} | {c['side']} | {names[k]} | {c['n_first_2y'][k]} | "
                     f"{fmt(c['q_ext_2y'][k])} | {c['n_first_oos'][k]} | {fmt(c['q_ext_oos'][k])} |")
    for ds in ("train", "oos"):
        o += ["", f"## Replay — {'2y (in-sample)' if ds == 'train' else 'oos2m'}", "",
              "| tf | side | entry/stop | tgt q | risk % | reward % | armed | fills | fill | target | stop | "
              "retarget | close | cap | hold | gross/fill % | net/fill % | Σ net % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in report["cells"]:
            for v in c["variants"]:
                r = v[ds]
                o.append(f"| {c['tf']} | {c['side']} | {v['name']} | {v['tq']} | {v['risk_pct_med']:.2f} | "
                         f"{v['reward_pct_med']:.2f} | {r['n_armed']} | {r['n_fills']} | {r['fill_rate']:.2f} | "
                         f"{r['target_rate']:.2f} | {r['stop_rate']:.2f} | {r['retarget_rate']:.2f} | {r['close_rate']:.2f} | "
                         f"{r['cap_rate']:.2f} | {r['hold_candles']:.1f} | {r['gross_pct_per_fill']:+.3f} | "
                         f"{r['pnl_pct_per_fill']:+.3f} | {r['sum_pnl_pct']:+.1f} |")
    o += ["", "## Mean net pnl % per exit kind (oos2m)", "",
          "| tf | side | entry/stop | target | retarget | stop | close | cap |", "|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        for v in c["variants"]:
            r = v["oos"]
            o.append(f"| {c['tf']} | {c['side']} | {v['name']} | " +
                     " | ".join(f"{r[f'{x}_pnl_pct']:+.3f}" for x in EXITS) + " |")
    with open(path, "w") as fh:
        fh.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
