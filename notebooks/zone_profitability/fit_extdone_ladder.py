"""Provenance gate for the extdone ladder: fit d(progress) on 2y, score on oos2m.

run_extdone.LADDER was chosen from oos2m event counts (the dataset it is
drawn on) — in-sample for the ladder. This driver applies the same selection
rule on the 2y frame only (smallest grid d per progress bucket whose
event-based P(hold | first trigger in bucket) >= TARGET, with >= MIN_EVENTS
triggers), then replays BOTH ladders as the sticky state on oos2m and
reports P(hold | first trigger) per bucket.

atr1 on 2y is rebuilt from 1-min OHLC (run_zone_ext.atr1_ma5); oos2m uses
the stored 1_atr_14_ma_5.

Artifacts: {EV_OUT}/extdone_ladder_2y.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evlib import TFS, candle_id  # noqa: E402
from run_extdone import LADDER as LADDER_OOS, d_of_progress  # noqa: E402
from run_zone_ext import atr1_ma5  # noqa: E402

VOL = "/trader_data_long/train"
OOS_DIR = os.environ.get("EV_DATASET_DIR", f"{VOL}/oos2m_link_usdt")
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
D_GRID = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12)
BUCKETS = ((0, .25), (.25, .5), (.5, .75), (.75, 1.0))
TARGET = float(os.environ.get("EXTDONE_TARGET", 0.90))
MIN_EVENTS = int(os.environ.get("EXTDONE_MIN_EVENTS", 100))


def prep(df: pd.DataFrame, atr: np.ndarray, tf: int, side: str) -> dict:
    n = len(df)
    cid = candle_id(df, tf)
    g = pd.Series(cid)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    ends = np.r_[starts[1:], n]
    prog = (np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) + 1) / tf
    last = np.zeros(n, bool); last[ends - 1] = True
    x = (df["1_high"] if side == "high" else df["1_low"]).to_numpy()
    s = pd.Series(x)
    run = (s.groupby(g).cummax() if side == "high" else s.groupby(g).cummin()).to_numpy()
    rev = (s[::-1].groupby(g[::-1]).cummax()[::-1] if side == "high"
           else s[::-1].groupby(g[::-1]).cummin()[::-1]).to_numpy()
    fut = np.r_[rev[1:], np.nan]
    again = (fut >= run) if side == "high" else (fut <= run)
    cl = df["1_close"].to_numpy()
    dist = ((run - cl) if side == "high" else (cl - run)) / atr
    return {"g": g, "cid": cid, "prog": prog, "last": last, "again": again,
            "dist": np.where(np.isfinite(dist), dist, -np.inf), "n_candles": len(starts)}


def bucket_grid(p: dict) -> dict:
    """{bucket: {d: (p_hold, n_events)}} — first trigger within the bucket."""
    out = {}
    for a, b in BUCKETS:
        inb = (p["prog"] >= a) & (p["prog"] < b) if b < 1.0 else (p["prog"] >= a)
        row = {}
        for d in D_GRID:
            hit = inb & (p["dist"] >= d)
            first = hit & ~(pd.Series(hit).groupby(p["g"]).cumsum().to_numpy() > 1)
            ev = first & ~p["last"]
            row[d] = (float(1 - p["again"][ev].mean()) if ev.sum() else float("nan"), int(ev.sum()))
        out[f"{a:.2f}-{b:.2f}"] = row
    return out


def pick(grid: dict) -> list:
    ladder = []
    for (a, b), row in zip(BUCKETS, grid.values()):
        ok = [d for d in D_GRID if row[d][1] >= MIN_EVENTS and row[d][0] >= TARGET]
        if ok:
            ladder.append((a, float(ok[0])))
    return ladder


def score(p: dict, ladder: list) -> dict:
    """Sticky state with the full ladder; event stats at the first trigger."""
    d = d_of_progress(p["prog"], ladder)
    done = pd.Series(p["dist"] >= d).groupby(p["g"]).cummax().to_numpy().astype(bool)
    first = done & ~np.r_[False, done[:-1] & (p["cid"][1:] == p["cid"][:-1])]
    trig = first & ~p["last"]
    res = {"fires_pct": float(100 * pd.Series(first).groupby(p["g"]).any().mean()),
           "n_events": int(trig.sum()),
           "p_hold": float(1 - p["again"][trig].mean()) if trig.sum() else float("nan"),
           "by_bucket": {}}
    for a, b in BUCKETS:
        m = trig & (p["prog"] >= a) & ((p["prog"] < b) if b < 1.0 else True)
        res["by_bucket"][f"{a:.2f}-{b:.2f}"] = (
            float(1 - p["again"][m].mean()) if m.sum() else None, int(m.sum()))
    return res


def main() -> None:
    tr = pd.read_pickle(TRAIN_IND)
    oos = pd.read_pickle(os.path.join(OOS_DIR, "df_with_indicators.pkl"))[
        ["1_high", "1_low", "1_close", "1_atr_14_ma_5"] + [f"{tf}_is_closed" for tf in TFS]]
    tr_atr, oos_atr = atr1_ma5(tr), oos["1_atr_14_ma_5"].to_numpy()
    # sanity: the rebuilt atr1 must match the stored one where both exist
    chk = atr1_ma5(oos)
    ok = np.isfinite(chk) & np.isfinite(oos_atr)
    rel = float(np.nanmedian(np.abs(chk[ok] - oos_atr[ok]) / oos_atr[ok]))
    report = {"target": TARGET, "min_events": MIN_EVENTS, "atr1_rebuild_median_rel_diff": rel, "cells": []}
    print("atr1 rebuild vs stored, median rel diff on oos2m: %.2e" % rel, flush=True)
    for tf in TFS:
        for side in ("high", "low"):
            pt, po = prep(tr, tr_atr, tf, side), prep(oos, oos_atr, tf, side)
            grid = bucket_grid(pt)
            lad2y = pick(grid)
            cell = {"tf": tf, "side": side, "ladder_2y": lad2y, "ladder_oos": LADDER_OOS[tf],
                    "grid_2y": {bk: {str(d): v for d, v in row.items()} for bk, row in grid.items()},
                    "train_2y_ladder": score(pt, lad2y),
                    "oos_2y_ladder": score(po, lad2y), "oos_oos_ladder": score(po, LADDER_OOS[tf])}
            report["cells"].append(cell)
            a, b = cell["oos_2y_ladder"], cell["oos_oos_ladder"]
            print(tf, side, "2y ladder", lad2y, "| oos2m P(hold) %.3f fires %.0f%% n=%d  vs oos-picked %s %.3f fires %.0f%%" % (
                a["p_hold"], a["fires_pct"], a["n_events"], LADDER_OOS[tf], b["p_hold"], b["fires_pct"]), flush=True)
    with open(os.path.join(OUT, "extdone_ladder_2y.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    o = [f"# extdone ladder — fit on 2y (target P(hold) ≥ {TARGET}, ≥ {MIN_EVENTS} events), scored on oos2m", "",
         f"atr1 rebuilt from OHLC vs stored `1_atr_14_ma_5` on oos2m: median relative diff {rel:.2e}", "",
         "| tf | side | ladder fit on 2y | 2y P(hold) | oos2m P(hold) [n, fires %] | per bucket oos2m (25–50 / 50–75 / 75–100) | "
         "ladder picked on oos2m | its oos2m P(hold) [fires %] |", "|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        a, b, t = c["oos_2y_ladder"], c["oos_oos_ladder"], c["train_2y_ladder"]
        bb = " / ".join(f"{v[0]:.2f} (n{v[1]})" if v[0] is not None else "—"
                        for k, v in list(a["by_bucket"].items())[1:])
        o.append(f"| {c['tf']} | {c['side']} | {c['ladder_2y']} | {t['p_hold']:.3f} | "
                 f"{a['p_hold']:.3f} [{a['n_events']}, {a['fires_pct']:.0f}%] | {bb} | "
                 f"{c['ladder_oos']} | {b['p_hold']:.3f} [{b['fires_pct']:.0f}%] |")
    with open(os.path.join(OUT, "extdone_ladder_2y.md"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
