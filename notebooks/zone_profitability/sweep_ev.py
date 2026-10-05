"""Sweep stop_sigma × tgt_sigma for the EV line (evlib) on one dataset.

For every (stop_sigma, tgt_sigma) pair: compute the EV-best entry per candle
for tf 15/60/240 × long/short, replay it minute by minute (evlib.realized),
and tabulate model-implied vs realized stats. Writes
{EV_OUT}/ev_sweep.json and {EV_OUT}/ev_sweep.md. Same docker invocation as
run_ev.py; grids via EV_STOP_GRID / EV_TGT_GRID (comma-separated).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evlib import SIDES, TFS, EVParams, candle_id, ev_line, realized  # noqa: E402

DATASET_DIR = os.environ.get(
    "EV_DATASET_DIR", "/trader_data_long/train/oos2m_link_usdt")
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
FEE = float(os.environ.get("EV_FEE", 0.001))
STOP_GRID = [float(x) for x in os.environ.get("EV_STOP_GRID", "0.5,1,1.5,2,3").split(",")]
TGT_GRID = [float(x) for x in os.environ.get("EV_TGT_GRID", "-0.5,0,0.5,1,1.5").split(",")]


def main() -> None:
    df = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_indicators.pkl"))
    cb = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_candle_bounds.pkl")).reindex(df.index)
    pre = {}
    for tf in TFS:
        cid = candle_id(df, tf)
        starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
        rep = np.diff(np.r_[starts, len(df)])
        pre[tf] = (cid, starts, rep,
                   cb[f"{tf}_cb_low"].to_numpy()[starts], cb[f"{tf}_cb_low_std"].to_numpy()[starts],
                   cb[f"{tf}_cb_high"].to_numpy()[starts], cb[f"{tf}_cb_high_std"].to_numpy()[starts])
    rows = []
    for ss in STOP_GRID:
        for ts in TGT_GRID:
            p = EVParams(stop_sigma=ss, tgt_sigma=ts, fee=FEE)
            for tf in TFS:
                cid, starts, rep, L, sL, H, sH = pre[tf]
                for side in SIDES:
                    r = ev_line(L, sL, H, sH, side, p)
                    E = np.repeat(r["best"], rep)
                    S = np.repeat(r["S"], rep)
                    T = np.repeat(r["T"], rep)
                    st = realized(df, cid, E, S, T, side, FEE)
                    v = np.isfinite(r["best"])
                    row = {
                        "stop_sigma": ss, "tgt_sigma": ts, "tf": tf, "side": side,
                        "n_valid": int(v.sum()),
                        "m_p_fill": float(np.nanmedian(r["p_fill"])),
                        "m_p_win": float(np.nanmedian(r["p_win"])),
                        "m_ev_candle_pct": float(np.nanmedian(100 * r["ev_candle"] / r["best"])),
                        "m_rr": float(np.nanmedian(
                            np.abs(r["T"] - r["best"]) / np.abs(r["best"] - r["S"]))),
                        **{f"r_{k}": x for k, x in st.items()},
                    }
                    rows.append(row)
                    print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x)
                                      for k, x in row.items()}), flush=True)
    with open(os.path.join(OUT, "ev_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    write_md(rows, os.path.join(OUT, "ev_sweep.md"))


def write_md(rows: list[dict], path: str) -> None:
    out = ["# EV line σ sweep — oos2m", ""]
    # summary: mean realized pnl/candle over the 6 (tf, side) cells per combo
    out += ["## Summary (mean over tf×side)", "",
            "| stop σ | tgt σ | fill | win | pnl/fill % | pnl/candle % | Σ pnl % | model p_win | model EV/candle % |",
            "|---|---|---|---|---|---|---|---|---|"]
    combos = sorted({(r["stop_sigma"], r["tgt_sigma"]) for r in rows})
    for ss, ts in combos:
        rs = [r for r in rows if r["stop_sigma"] == ss and r["tgt_sigma"] == ts]
        m = lambda k: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
        out.append(f"| {ss} | {ts} | {m('r_fill_rate'):.3f} | {m('r_win_rate'):.3f} | "
                   f"{m('r_pnl_pct_per_fill'):+.4f} | {m('r_pnl_pct_per_candle'):+.4f} | "
                   f"{sum(r['r_sum_pnl_pct'] for r in rs):+.1f} | {m('m_p_win'):.3f} | "
                   f"{m('m_ev_candle_pct'):+.4f} |")
    out += ["", "## Per cell", "",
            "| stop σ | tgt σ | tf | side | R:R | fill | win | stop | open | pnl/fill % | pnl/candle % | model p_win |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['stop_sigma']} | {r['tgt_sigma']} | {r['tf']} | {r['side']} | "
                   f"{r['m_rr']:.2f} | {r['r_fill_rate']:.3f} | {r['r_win_rate']:.3f} | "
                   f"{r['r_stop_rate']:.3f} | {r['r_open_at_close_rate']:.3f} | "
                   f"{r['r_pnl_pct_per_fill']:+.4f} | {r['r_pnl_pct_per_candle']:+.4f} | "
                   f"{r['m_p_win']:.3f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
