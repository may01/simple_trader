"""Driver: "extreme done" state per tf candle — the running high/low has
very likely been set for the rest of the candle.

Measured on oos2m (see external experiment/ev_line.md s9): no fixed level
(cb, cb +/- sigma, cbnc) predicts exhaustion; what does is how far the close has
pulled away from the running extreme (in 1-min ATR) together with candle
progress. The rule is a progress ladder of pull-away thresholds:

    done_high[t] = (run_high[t] - close[t]) / atr1[t] >= d(progress[t])
    done_low[t]  = (close[t] - run_low[t])  / atr1[t] >= d(progress[t])

with d(p) stepping down as the candle matures (LADDER per tf: list of
(progress_from, d); below the first step the state is never set). The state
is sticky within the candle once set (the extreme can still be broken; the
_first marker is the minute it first turned true, the honest trigger).

Artifacts:
  {dataset_dir}/df_with_ext_done.pkl   {tf}_extdone_{high,low}, _first, _lvl
  {EV_OUT}/extdone_report.{json,md}     ladder + event-based accuracy per bucket
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

DATASET_DIR = os.environ.get("EV_DATASET_DIR", "/trader_data_long/train/oos2m_link_usdt")
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
ATR_COL = "1_atr_14_ma_5"
# (progress_from, d in 1-min ATR): per progress bucket the smallest d whose
# EVENT-based P(hold | first trigger in bucket) >= 0.90 on oos2m (scratch
# tune_extdone.py; row-based conditionals overstate it by 5-20 points because
# they count every later minute of an already-done candle). Buckets where no
# d on the grid reaches 0.90 are left out (state never set there).
LADDER = {
    15: [(0.25, 4.0), (0.50, 2.5), (0.75, 1.0)],
    60: [(0.25, 8.0), (0.50, 5.0), (0.75, 3.0)],
    240: [(0.50, 12.0), (0.75, 8.0)],
}
BUCKETS = ((0, .25), (.25, .5), (.5, .75), (.75, 1.0))


def d_of_progress(prog: np.ndarray, ladder: list) -> np.ndarray:
    d = np.full(len(prog), np.inf)
    for p0, dv in ladder:
        d[prog >= p0] = dv
    return d


def main() -> None:
    df = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_indicators.pkl"))
    hi, lo, cl = (df[c].to_numpy() for c in ("1_high", "1_low", "1_close"))
    atr = df[ATR_COL].to_numpy()
    n = len(df)
    cols, report = {}, {"ladder": {str(k): v for k, v in LADDER.items()}, "cells": []}
    for tf in TFS:
        cid = candle_id(df, tf)
        g = pd.Series(cid)
        starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
        ends = np.r_[starts[1:], n]
        pos = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
        prog = (pos + 1) / tf
        last = np.zeros(n, bool); last[ends - 1] = True
        d = d_of_progress(prog, LADDER[tf])
        for side in ("high", "low"):
            x = hi if side == "high" else lo
            s = pd.Series(x)
            run = (s.groupby(g).cummax() if side == "high" else s.groupby(g).cummin()).to_numpy()
            rev = (s[::-1].groupby(g[::-1]).cummax()[::-1] if side == "high"
                   else s[::-1].groupby(g[::-1]).cummin()[::-1]).to_numpy()
            fut = np.r_[rev[1:], np.nan]
            again = (fut >= run) if side == "high" else (fut <= run)   # touched again later
            dist = ((run - cl) if side == "high" else (cl - run)) / atr
            done = (dist >= d) & np.isfinite(atr)
            # sticky within candle
            done_sticky = pd.Series(done).groupby(g).cummax().to_numpy().astype(bool)
            first = done_sticky & ~np.r_[False, done_sticky[:-1] & (cid[1:] == cid[:-1])]
            cols[f"{tf}_extdone_{side}"] = done_sticky
            cols[f"{tf}_extdone_{side}_first"] = first
            cols[f"{tf}_extdone_{side}_lvl"] = np.where(done_sticky, run, np.nan)
            # event-based accuracy: at the FIRST trigger, was the extreme touched again?
            n_cand = len(starts)
            trig = first & ~last
            cell = {"tf": tf, "side": side, "candles": n_cand,
                    "triggered_pct": float(100 * pd.Series(first).groupby(g).any().mean()),
                    "p_hold_first": float(1 - again[trig].mean()),
                    "prog_at_first_med": float(np.median(prog[trig])),
                    "by_bucket": []}
            for a, b in BUCKETS:
                m = trig & (prog >= a) & (prog < b)
                dv = [v for p0, v in LADDER[tf] if p0 <= a + 1e-9]
                cell["by_bucket"].append({
                    "bucket": f"{a:.2f}-{b:.2f}", "d": dv[-1] if dv else None,
                    "n_first": int(m.sum()),
                    "p_hold_first": float(1 - again[m].mean()) if m.sum() else None,
                    "p_hold_rows": float(1 - again[done_sticky & ~last & (prog >= a) & (prog < b)].mean())
                    if (done_sticky & ~last & (prog >= a) & (prog < b)).sum() else None,
                })
            report["cells"].append(cell)
            print(tf, side, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in cell.items() if k != "by_bucket"}), flush=True)
    pd.DataFrame(cols, index=df.index).to_pickle(os.path.join(DATASET_DIR, "df_with_ext_done.pkl"))
    with open(os.path.join(OUT, "extdone_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    o = ["# Extreme-done state — oos2m", ""]
    for c in report["cells"]:
        o += [f"## tf {c['tf']} {c['side']}: ladder {LADDER[c['tf']]} — triggers in "
              f"{c['triggered_pct']:.1f} % of candles, median progress at trigger "
              f"{c['prog_at_first_med']:.2f}, P(hold | first trigger) {c['p_hold_first']:.3f}", "",
              "| progress | d (ATR) | first triggers | P(hold \\| first trigger) | P(hold \\| any done row) |",
              "|---|---|---|---|---|"]
        for b in c["by_bucket"]:
            o.append(f"| {b['bucket']} | {b['d'] if b['d'] is not None else '—'} | {b['n_first']} | "
                     f"{b['p_hold_first']:.3f} | {b['p_hold_rows']:.3f} |"
                     if b["p_hold_first"] is not None else
                     f"| {b['bucket']} | {b['d'] if b['d'] is not None else '—'} | 0 | — | — |")
        o.append("")
    with open(os.path.join(OUT, "extdone_report.md"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    print("wrote", DATASET_DIR, OUT)


if __name__ == "__main__":
    main()
