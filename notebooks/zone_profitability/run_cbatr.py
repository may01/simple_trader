"""Driver: ATR-shifted entry zones vs the 5%-of-span cb_zone, on oos2m.

cb_zone_* places the entry 5% of the stop->target span away from the bound
(cb_low + 0.05*(cb_high - cb_low) for long). Here the shift is k x the
1-minute ATR instead (``1_atr_14_ma_5`` at the row, so the level moves per
minute while the bound stairsteps per candle):

    long:  cbatr_zone_long  = cb_low  + k*atr1
    short: cbatr_zone_short = cb_high - k*atr1

Both rules are replayed minute by minute (evlib.replay) with the SAME
stop/target so only the entry differs: S = cb_low, T = cb_high (the cb_zone
contract), and a second variant with S = cb_low - 1 sigma (cb_low_dn). k is swept;
k = 1 is written to the sidecar + chart.

Artifacts:
  {dataset_dir}/df_with_cb_zone_atr.pkl   {tf}_cbatr_zone_{side}, {tf}_cbatr_inzone_{side}
  {EV_OUT}/cbatr_report.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evlib import SIDES, TFS, candle_id, replay, summarize  # noqa: E402

DATASET_DIR = os.environ.get("EV_DATASET_DIR", "/trader_data_long/train/oos2m_link_usdt")
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
FEE = float(os.environ.get("EV_FEE", 0.001))
ATR_COL = os.environ.get("CBATR_COL", "1_atr_14_ma_5")
K_GRID = [float(x) for x in os.environ.get("CBATR_K_GRID", "0.1,0.2,0.3,0.5,1").split(",")]
K_DRAW = float(os.environ.get("CBATR_K", 0.3))


def main() -> None:
    df = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_indicators.pkl"))
    cb = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_candle_bounds.pkl")).reindex(df.index)
    lo, hi, cl = (df[c].to_numpy() for c in ("1_low", "1_high", "1_close"))
    atr = df[ATR_COL].to_numpy()
    cols, report = {}, {"atr_col": ATR_COL, "fee": FEE, "k_draw": K_DRAW, "cells": []}
    for tf in TFS:
        cid = candle_id(df, tf)
        starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
        L, H = cb[f"{tf}_cb_low"].to_numpy(), cb[f"{tf}_cb_high"].to_numpy()
        Ldn, Hup = cb[f"{tf}_cb_low_dn"].to_numpy(), cb[f"{tf}_cb_high_up"].to_numpy()
        span = H - L
        for side in SIDES:
            z5 = cb[f"{tf}_cb_zone_{side}"].to_numpy()
            if side == "long":
                S1, S2, T = L[starts], Ldn[starts], H[starts]
                zatr = {k: L + k * atr for k in K_GRID}
                inz = lambda z: lo <= z  # noqa: E731
            else:
                S1, S2, T = H[starts], Hup[starts], L[starts]
                zatr = {k: H - k * atr for k in K_GRID}
                inz = lambda z: hi >= z  # noqa: E731
            rules = {"cb_zone_5pct": z5, **{f"atr_k{k:g}": z for k, z in zatr.items()}}
            cell = {"tf": tf, "side": side, "rules": {}}
            for name, z in rules.items():
                shift_pct = 100 * np.abs(z - (L if side == "long" else H)) / z
                entry = {
                    "shift_pct_of_price_med": float(np.nanmedian(shift_pct)),
                    "shift_frac_of_span_med": float(np.nanmedian(
                        np.abs(z - (L if side == "long" else H)) / span)),
                    "inzone_rows_pct": float(100 * np.nanmean(inz(z) & np.isfinite(z))),
                }
                for sname, S in (("S=bound", S1), ("S=bound-1sd", S2)):
                    r = replay(lo, hi, cl, starts, z, S, T, side, FEE)
                    st = summarize(r, r["E"])
                    st["n_fills"] = int(r["filled"].sum())
                    entry[sname] = st
                cell["rules"][name] = entry
            report["cells"].append(cell)
            print(tf, side, json.dumps({n: (round(e["shift_pct_of_price_med"], 3),
                                          round(e["inzone_rows_pct"], 1),
                                          round(e["S=bound"]["pnl_pct_per_candle"], 4),
                                          round(e["S=bound-1sd"]["pnl_pct_per_candle"], 4))
                                      for n, e in cell["rules"].items()}), flush=True)
            z = zatr[K_DRAW]
            cols[f"{tf}_cbatr_zone_{side}"] = z
            cols[f"{tf}_cbatr_inzone_{side}"] = inz(z) & np.isfinite(z)
    pd.DataFrame(cols, index=df.index).to_pickle(
        os.path.join(DATASET_DIR, "df_with_cb_zone_atr.pkl"))
    with open(os.path.join(OUT, "cbatr_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    write_md(report, os.path.join(OUT, "cbatr_report.md"))
    print("wrote", DATASET_DIR, OUT)


def write_md(report: dict, path: str) -> None:
    o = [f"# ATR-shifted entry zone vs cb_zone (5% of span) — oos2m", "",
         f"atr col `{report['atr_col']}`, fee {report['fee']}/side, chart k = {report['k_draw']}", ""]
    for sname in ("S=bound", "S=bound-1sd"):
        o += [f"## stop {sname}, target = opposite bound", "",
              "| tf | side | rule | shift % price | shift/span | in-zone rows % | fills | fill | "
              "win | stop | open | pnl/fill % | pnl/candle % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in report["cells"]:
            for n, e in c["rules"].items():
                s = e[sname]
                o.append(f"| {c['tf']} | {c['side']} | {n} | {e['shift_pct_of_price_med']:.3f} | "
                         f"{e['shift_frac_of_span_med']:.2f} | {e['inzone_rows_pct']:.1f} | "
                         f"{s['n_fills']} | {s['fill_rate']:.3f} | {s['win_rate']:.3f} | "
                         f"{s['stop_rate']:.3f} | {s['open_at_close_rate']:.3f} | "
                         f"{s['pnl_pct_per_fill']:+.4f} | {s['pnl_pct_per_candle']:+.4f} |")
        o.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
