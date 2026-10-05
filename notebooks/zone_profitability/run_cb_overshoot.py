"""Driver: overshoot of a cb marker past its bound + the p75 marker filter.

Overshoot of a 1-min row, in band sd of the bound the zone hangs on:

    long:  z = (cb_low  - 1_low)  / cb_low_std     (> 0: low is under the predicted low)
    short: z = (1_high - cb_high) / cb_high_std    (> 0: high is over the predicted high)

Threshold per (tf, side) = OVER_Q quantile (default 0.75) of z over the 2y
filtered markers (cb_inzone minus the extdone veto), frozen and applied to
the target dataset: ``{tf}_cbover_{side}`` = z > threshold. The viewer drops
cb_inzone markers on those rows (frontend/data_viewer._MARKER_VETO).

Artifacts:
  {dataset_dir}/df_with_cb_overshoot.pkl   {tf}_cb_overshoot_{side} (z), {tf}_cbover_{side} (bool)
  {EV_OUT}/cb_overshoot_report.{json,md}    thresholds + kept/dropped marker stats
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
DATASET_DIR = os.environ.get("EV_DATASET_DIR", f"{VOL}/oos2m_link_usdt")
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
TRAIN_CB = f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl"
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
OVER_Q = float(os.environ.get("OVER_Q", 0.75))
P = (50, 75, 90, 95)


def overshoot(df: pd.DataFrame, cb: pd.DataFrame, tf: int, side: str) -> np.ndarray:
    if side == "long":
        return ((cb[f"{tf}_cb_low"] - df["1_low"]) / cb[f"{tf}_cb_low_std"]).to_numpy()
    return ((df["1_high"] - cb[f"{tf}_cb_high"]) / cb[f"{tf}_cb_high_std"]).to_numpy()


def markers(df: pd.DataFrame, cb: pd.DataFrame, atr: np.ndarray, tf: int, side: str):
    """Filtered marker mask, candle id and distance (% of price) to the candle's final extreme."""
    n = len(df)
    cid = candle_id(df, tf)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    rep = np.diff(np.r_[starts, n])
    prog = (np.arange(n) - np.repeat(starts, rep) + 1) / tf
    veto = extdone(df, cid, prog, atr, tf, "high" if side == "long" else "low")
    m = cb[f"{tf}_cb_inzone_{side}"].fillna(False).to_numpy(dtype=bool) & ~veto & np.isfinite(atr)
    x = df["1_low" if side == "long" else "1_high"].to_numpy()
    g = pd.Series(x).groupby(cid)
    fin = (g.transform("min") if side == "long" else g.transform("max")).to_numpy()
    return m, cid, 100.0 * np.abs(x - fin) / x


def stats(m: np.ndarray, cid: np.ndarray, dist: np.ndarray) -> dict:
    c = pd.Series(m).groupby(cid).sum()
    c = c[c > 0].to_numpy()
    d = dist[m]
    return {"markers": int(m.sum()), "candles": int(len(c)),
            "per_candle_mean": float(c.mean()) if len(c) else 0.0,
            "per_candle_p50": float(np.median(c)) if len(c) else 0.0,
            "at_ext_pct": float(100 * np.mean(d == 0)) if len(d) else 0.0,
            "dist_mean": float(d.mean()) if len(d) else 0.0, "dist_sd": float(d.std()) if len(d) else 0.0,
            **{f"dist_p{p}": float(np.percentile(d, p)) if len(d) else 0.0 for p in P}}


def main() -> None:
    need = ["1_high", "1_low", "1_close"] + [f"{tf}_is_closed" for tf in TFS]
    tr = pd.read_pickle(TRAIN_IND)
    tr_atr = atr1_ma5(tr)
    tr = tr[need]
    tr_cb = pd.read_pickle(TRAIN_CB).reindex(tr.index)
    oos = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_indicators.pkl"))
    oos_atr = oos["1_atr_14_ma_5"].to_numpy()
    oos = oos[need]
    oos_cb = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_candle_bounds.pkl")).reindex(oos.index)
    cols, report = {}, {"q": OVER_Q, "cells": []}
    for tf in TFS:
        for side in SIDES:
            mt, _, _ = markers(tr, tr_cb, tr_atr, tf, side)
            thr = float(np.nanquantile(overshoot(tr, tr_cb, tf, side)[mt], OVER_Q))
            z = overshoot(oos, oos_cb, tf, side)
            over = z > thr
            cols[f"{tf}_cb_overshoot_{side}"] = z
            cols[f"{tf}_cbover_{side}"] = over
            m, cid, dist = markers(oos, oos_cb, oos_atr, tf, side)
            cell = {"tf": tf, "side": side, "thr_sd": thr,
                    "oos_q_of_thr": float(np.mean(z[m] <= thr)),
                    "all": stats(m, cid, dist), "kept": stats(m & ~over, cid, dist),
                    "dropped": stats(m & over, cid, dist)}
            report["cells"].append(cell)
            k, d = cell["kept"], cell["dropped"]
            print(tf, side, f"thr {thr:+.2f} sd (oos share <= thr {cell['oos_q_of_thr']:.2f}) | kept "
                  f"{k['markers']} mk / {k['candles']} cand, dist p50 {k['dist_p50']:.2f} p90 {k['dist_p90']:.2f} | "
                  f"dropped {d['markers']} mk / {d['candles']} cand, dist p50 {d['dist_p50']:.2f} "
                  f"p90 {d['dist_p90']:.2f}", flush=True)
    pd.DataFrame(cols, index=oos.index).to_pickle(os.path.join(DATASET_DIR, "df_with_cb_overshoot.pkl"))
    with open(os.path.join(OUT, "cb_overshoot_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    o = [f"# cb marker overshoot filter — drop markers above the 2y q{int(OVER_Q * 100)} overshoot", "",
         "| tf | side | threshold (sd) | set | markers | candles | per candle mean / p50 | at extreme % | "
         "dist to extreme % mean | sd | p50 | p75 | p90 | p95 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        for name in ("all", "kept", "dropped"):
            s = c[name]
            o.append(f"| {c['tf']} | {c['side']} | {c['thr_sd']:+.2f} | {name} | {s['markers']} | {s['candles']} | "
                     f"{s['per_candle_mean']:.1f} / {s['per_candle_p50']:.0f} | {s['at_ext_pct']:.1f} | "
                     f"{s['dist_mean']:.2f} | {s['dist_sd']:.2f} | {s['dist_p50']:.2f} | {s['dist_p75']:.2f} | "
                     f"{s['dist_p90']:.2f} | {s['dist_p95']:.2f} |")
    with open(os.path.join(OUT, "cb_overshoot_report.md"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    print("wrote", DATASET_DIR, OUT)


if __name__ == "__main__":
    main()
