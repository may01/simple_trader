"""Driver: EV-line levels on oos2m + comparison with cb_zone / cb levels.

Reads the dataset's df_with_indicators.pkl + df_with_candle_bounds.pkl,
computes per-candle EV-optimal entries (evlib.ev_line) for tf 15/60/240,
long and short, then replays four entry rules minute by minute
(evlib.realized):

  ev        E = EV-best,       S = L - 1σ_L,  T = H - 0.5σ_H   (long; mirror short)
  ev_be     E = EV-breakeven,  same S/T      (outer edge of the EV zone)
  cb_zone   E = cb_zone_*,     S = cb_low,   T = cb_high       (current 5% rule)
  cb_level  E = cb_low,        S = cb_low_dn, T = cb_high      (entry at the bound)

Artifacts:
  {dataset_dir}/df_with_ev_line.pkl   sidecar: {tf}_ev_{side}[,_be,_pwin,_evpc]
  {EV_OUT}/ev_line_report.json        per (tf, side, rule) model + realized stats
  {EV_OUT}/ev_line_report.md          same, as tables

Run inside the simple_trader image (volume is root-owned):
  docker run --rm -v <worktree>:/code -v simple_trader_vol_long:/trader_data_long \
    -v <scratch>:/out -e EV_OUT=/out -w /code simple_trader \
    python3 notebooks/zone_profitability/run_ev.py
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
PARAMS = EVParams(
    stop_sigma=float(os.environ.get("EV_STOP_SIGMA", 1.0)),
    tgt_sigma=float(os.environ.get("EV_TGT_SIGMA", 0.5)),
    fee=float(os.environ.get("EV_FEE", 0.001)),
)


def main() -> None:
    df = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_indicators.pkl"))
    cb = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_candle_bounds.pkl"))
    cb = cb.reindex(df.index)
    side_cols = {}
    report = {"params": PARAMS.__dict__, "dataset": DATASET_DIR, "cells": []}

    for tf in TFS:
        cid = candle_id(df, tf)
        starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
        L = cb[f"{tf}_cb_low"].to_numpy(); sL = cb[f"{tf}_cb_low_std"].to_numpy()
        H = cb[f"{tf}_cb_high"].to_numpy(); sH = cb[f"{tf}_cb_high_std"].to_numpy()
        for side in SIDES:
            # per-candle inputs (constant within a candle) -> broadcast back
            r = ev_line(L[starts], sL[starts], H[starts], sH[starts], side, PARAMS)
            rep = np.diff(np.r_[starts, len(df)])
            best = np.repeat(r["best"], rep)
            be = np.repeat(r["breakeven"], rep)
            S = np.repeat(r["S"], rep)
            T = np.repeat(r["T"], rep)
            side_cols[f"{tf}_ev_{side}"] = best
            side_cols[f"{tf}_ev_{side}_be"] = be
            side_cols[f"{tf}_ev_{side}_pwin"] = np.repeat(r["p_win"], rep)
            side_cols[f"{tf}_ev_{side}_evpc"] = np.repeat(
                100.0 * r["ev_candle"] / r["best"], rep)

            z = cb[f"{tf}_cb_zone_{side}"].to_numpy()
            if side == "long":
                cb_S, cb_T, lvl, lvl_S = L, H, L, cb[f"{tf}_cb_low_dn"].to_numpy()
            else:
                cb_S, cb_T, lvl, lvl_S = H, L, H, cb[f"{tf}_cb_high_up"].to_numpy()
            rules = {
                "ev": (best, S, T),
                "ev_be": (be, S, T),
                "cb_zone": (z, cb_S, cb_T),
                "cb_level": (lvl, lvl_S, cb_T),
            }
            v = np.isfinite(r["best"])
            span = (H - L)[starts][v]
            model = {
                "n_candles_valid": int(v.sum()),
                "p_fill_med": float(np.nanmedian(r["p_fill"])),
                "p_win_med": float(np.nanmedian(r["p_win"])),
                "ev_pct_med": float(np.nanmedian(100 * r["ev"] / r["best"])),
                "ev_candle_pct_med": float(np.nanmedian(100 * r["ev_candle"] / r["best"])),
                "best_frac_of_cb_span_med": float(np.median(
                    ((r["best"] - L[starts]) / (H - L)[starts])[v])) if side == "long"
                else float(np.median(((H[starts] - r["best"]) / (H - L)[starts])[v])),
                "be_frac_of_cb_span_med": float(np.nanmedian(
                    ((r["breakeven"] - L[starts]) / (H - L)[starts])[v])) if side == "long"
                else float(np.nanmedian(((H[starts] - r["breakeven"]) / (H - L)[starts])[v])),
                "best_minus_cb_zone_pct_med": float(np.nanmedian(
                    100 * (r["best"] - z[starts]) / r["best"])),
                "best_minus_cb_level_pct_med": float(np.nanmedian(
                    100 * (r["best"] - lvl[starts]) / r["best"])),
                "cb_span_pct_med": float(np.median(100 * span / L[starts][v])),
            }
            cell = {"tf": tf, "side": side, "model": model, "realized": {}}
            for name, (E_, S_, T_) in rules.items():
                cell["realized"][name] = realized(df, cid, E_, S_, T_, side, PARAMS.fee)
            report["cells"].append(cell)
            print(tf, side, json.dumps(model), flush=True)
            for name, st in cell["realized"].items():
                print("   ", name, json.dumps({k: round(x, 4) for k, x in st.items()}),
                      flush=True)

    out = pd.DataFrame(side_cols, index=df.index)
    out.to_pickle(os.path.join(DATASET_DIR, "df_with_ev_line.pkl"))
    with open(os.path.join(OUT, "ev_line_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    write_md(report, os.path.join(OUT, "ev_line_report.md"))
    print("wrote", os.path.join(DATASET_DIR, "df_with_ev_line.pkl"), OUT)


def write_md(report: dict, path: str) -> None:
    lines = ["# EV line — oos2m", "", f"params: `{report['params']}`", ""]
    lines += ["## Model-implied (median per candle)", "",
              "| tf | side | n | p_fill | p_win | EV/fill % | EV/candle % | "
              "best @ span | be @ span | best−cb_zone % | best−cb_level % | span % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        m = c["model"]
        lines.append(
            f"| {c['tf']} | {c['side']} | {m['n_candles_valid']} | {m['p_fill_med']:.3f} | "
            f"{m['p_win_med']:.3f} | {m['ev_pct_med']:+.3f} | {m['ev_candle_pct_med']:+.3f} | "
            f"{m['best_frac_of_cb_span_med']:.2f} | {m['be_frac_of_cb_span_med']:.2f} | "
            f"{m['best_minus_cb_zone_pct_med']:+.3f} | {m['best_minus_cb_level_pct_med']:+.3f} | "
            f"{m['cb_span_pct_med']:.2f} |")
    lines += ["", "## Realized (minute replay, net of 2×fee)", "",
              "| tf | side | rule | n | fill | win | stop | open@close | "
              "pnl/fill % | pnl/candle % | Σ pnl % |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        for name, s in c["realized"].items():
            lines.append(
                f"| {c['tf']} | {c['side']} | {name} | {s['n_candles']} | {s['fill_rate']:.3f} | "
                f"{s['win_rate']:.3f} | {s['stop_rate']:.3f} | {s['open_at_close_rate']:.3f} | "
                f"{s['pnl_pct_per_fill']:+.4f} | {s['pnl_pct_per_candle']:+.4f} | "
                f"{s['sum_pnl_pct']:+.1f} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
