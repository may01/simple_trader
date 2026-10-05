"""Driver: empirical-reach EV line — fit on 2y, apply to oos2m.

Replaces the Gaussian-independence prior of evlib.ev_line with what the 2y
frame actually did. Entry/stop/target are parametrized in band units:

    long:  E = L + a·σ_L     S = L − s·σ_L     T = H − t·σ_H
    short: E = H − a·σ_H     S = H + s·σ_H     T = L + t·σ_L

(a < 0: entry beyond the predicted extreme, into the stop side; t < 0:
target beyond the predicted extreme.) For every (tf, side, a, s, t) the
rule is replayed minute by minute on 2y (evlib.replay) giving empirical
p_fill, p_win|fill, p_stop|fill and net pnl per candle — the conditional
reach the Gaussian model cannot see. Selection per (tf, side): argmax of
2y net pnl per candle with >= MIN_FILLS fills. The chosen (a, s, t) is then
replayed on oos2m (true out-of-sample for the rule; the bound models were
fitted on 2y_az so the 2y bounds themselves are in-sample).

Artifacts:
  {oos_dir}/df_with_ev_reach.pkl       {tf}_evr_{side}[,_sl,_tgt] per 1-min row
  {EV_OUT}/ev_reach_params.json        chosen (a, s, t) per (tf, side) + top-5
  {EV_OUT}/ev_reach_grid.json          full 2y grid (empirical + Gaussian per cell)
  {EV_OUT}/ev_reach_report.md          tables

Run inside the simple_trader image (see run_ev.py docstring).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evlib import SIDES, TFS, Phi, candle_id, replay, summarize  # noqa: E402

VOL = "/trader_data_long/train"
TRAIN_IND = os.environ.get("EVR_TRAIN_IND", f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl")
TRAIN_CB = os.environ.get("EVR_TRAIN_CB", f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl")
OOS_DIR = os.environ.get("EV_DATASET_DIR", f"{VOL}/oos2m_link_usdt")
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
FEE = float(os.environ.get("EV_FEE", 0.001))
MIN_FILLS = int(os.environ.get("EVR_MIN_FILLS", 100))
A_GRID = [float(x) for x in os.environ.get(
    "EVR_A_GRID", ",".join(str(round(x, 2)) for x in np.arange(-3.0, 1.01, 0.25))).split(",")]
S_GRID = [float(x) for x in os.environ.get("EVR_S_GRID", "0.5,1,1.5,2,3").split(",")]
T_GRID = [float(x) for x in os.environ.get("EVR_T_GRID", "-0.5,0,0.5,1,1.5").split(",")]

NEED = ["1_low", "1_high", "1_close"] + [f"{tf}_is_closed" for tf in TFS]


def load(ind_path: str, cb_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_pickle(ind_path)[NEED]
    cb = pd.read_pickle(cb_path).reindex(df.index)
    return df, cb


def frame(df: pd.DataFrame, cb: pd.DataFrame, tf: int) -> dict:
    cid = candle_id(df, tf)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    return {
        "starts": starts, "rep": np.diff(np.r_[starts, len(df)]),
        "lo": df["1_low"].to_numpy(), "hi": df["1_high"].to_numpy(),
        "cl": df["1_close"].to_numpy(),
        "L": cb[f"{tf}_cb_low"].to_numpy()[starts], "sL": cb[f"{tf}_cb_low_std"].to_numpy()[starts],
        "H": cb[f"{tf}_cb_high"].to_numpy()[starts], "sH": cb[f"{tf}_cb_high_std"].to_numpy()[starts],
    }


def levels(f: dict, side: str, a: float, s: float, t: float):
    if side == "long":
        E, S, T = f["L"] + a * f["sL"], f["L"] - s * f["sL"], f["H"] - t * f["sH"]
    else:
        E, S, T = f["H"] - a * f["sH"], f["H"] + s * f["sH"], f["L"] + t * f["sL"]
    return E, S, T


def gaussian(a: float, s: float, t: float) -> dict[str, float]:
    p_fill = float(Phi(np.array([a]))[0])
    p_stop = float(Phi(np.array([-s]))[0]) / p_fill
    p_tgt = float(Phi(np.array([t]))[0])
    return {"g_p_fill": p_fill, "g_p_stop": p_stop, "g_p_win": p_tgt * (1 - p_stop)}


def run_cell(f: dict, side: str, a: float, s: float, t: float) -> dict:
    E, S, T = levels(f, side, a, s, t)
    ok = (T > E) & (E > S) if side == "long" else (T < E) & (E < S)
    E = np.where(ok, E, np.nan)
    st = summarize(replay(f["lo"], f["hi"], f["cl"], f["starts"], E, S, T, side, FEE), E)
    st["n_fills"] = int(round(st["fill_rate"] * st["n_candles"])) if st["n_candles"] else 0
    return st


def main() -> None:
    tr_df, tr_cb = load(TRAIN_IND, TRAIN_CB)
    grid, params = [], {}
    for tf in TFS:
        f = frame(tr_df, tr_cb, tf)
        for side in SIDES:
            cells = []
            for a in A_GRID:
                for s in S_GRID:
                    if a <= -s:          # entry would sit at/beyond the stop
                        continue
                    for t in T_GRID:
                        st = run_cell(f, side, a, s, t)
                        cell = {"tf": tf, "side": side, "a": a, "s": s, "t": t,
                                **gaussian(a, s, t), **{f"e_{k}": v for k, v in st.items()}}
                        cells.append(cell)
            grid += cells
            elig = [c for c in cells if c["e_n_fills"] >= MIN_FILLS]
            elig.sort(key=lambda c: -c["e_pnl_pct_per_candle"])
            best = elig[0]
            params[f"{tf}_{side}"] = {
                "a": best["a"], "s": best["s"], "t": best["t"],
                "train": {k: best[k] for k in best if k.startswith("e_") or k.startswith("g_")},
                "top5": [{k: c[k] for k in ("a", "s", "t", "e_n_fills", "e_win_rate",
                                             "e_pnl_pct_per_fill", "e_pnl_pct_per_candle")}
                         for c in elig[:5]],
            }
            print(tf, side, "best", best["a"], best["s"], best["t"],
                  "train pnl/candle %.4f win %.3f fills %d" % (
                      best["e_pnl_pct_per_candle"], best["e_win_rate"], best["e_n_fills"]),
                  flush=True)
    del tr_df, tr_cb

    oos_df, oos_cb = load(os.path.join(OOS_DIR, "df_with_indicators.pkl"),
                          os.path.join(OOS_DIR, "df_with_candle_bounds.pkl"))
    cols = {}
    for tf in TFS:
        f = frame(oos_df, oos_cb, tf)
        for side in SIDES:
            p = params[f"{tf}_{side}"]
            E, S, T = levels(f, side, p["a"], p["s"], p["t"])
            p["oos"] = run_cell(f, side, p["a"], p["s"], p["t"])
            # the Gaussian-EV line's own cell (a from its median, s=1, t=0.5) for reference
            print(tf, side, "oos", json.dumps({k: round(v, 4) for k, v in p["oos"].items()}),
                  flush=True)
            cols[f"{tf}_evr_{side}"] = np.repeat(E, f["rep"])
            cols[f"{tf}_evr_{side}_sl"] = np.repeat(S, f["rep"])
            cols[f"{tf}_evr_{side}_tgt"] = np.repeat(T, f["rep"])
    pd.DataFrame(cols, index=oos_df.index).to_pickle(os.path.join(OOS_DIR, "df_with_ev_reach.pkl"))
    with open(os.path.join(OUT, "ev_reach_params.json"), "w") as fh:
        json.dump(params, fh, indent=1)
    with open(os.path.join(OUT, "ev_reach_grid.json"), "w") as fh:
        json.dump(grid, fh)
    write_md(params, grid, os.path.join(OUT, "ev_reach_report.md"))
    print("wrote", OOS_DIR, OUT)


def write_md(params: dict, grid: list[dict], path: str) -> None:
    o = ["# Empirical-reach EV line — fit 2y, apply oos2m", "",
         f"grid: a {A_GRID[0]}..{A_GRID[-1]} step {A_GRID[1]-A_GRID[0]}, s {S_GRID}, t {T_GRID}; "
         f"min fills {MIN_FILLS}; fee {FEE}/side", "",
         "## Selected per (tf, side): 2y in-sample -> oos2m", "",
         "| tf | side | a | s | t | 2y fills | 2y win | 2y pnl/fill % | 2y pnl/candle % | "
         "oos fills | oos win | oos stop | oos open | oos pnl/fill % | oos pnl/candle % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, p in params.items():
        tf, side = k.split("_")
        tr, oo = p["train"], p["oos"]
        o.append(f"| {tf} | {side} | {p['a']} | {p['s']} | {p['t']} | {tr['e_n_fills']} | "
                 f"{tr['e_win_rate']:.3f} | {tr['e_pnl_pct_per_fill']:+.4f} | "
                 f"{tr['e_pnl_pct_per_candle']:+.4f} | {oo['n_fills']} | {oo['win_rate']:.3f} | "
                 f"{oo['stop_rate']:.3f} | {oo['open_at_close_rate']:.3f} | "
                 f"{oo['pnl_pct_per_fill']:+.4f} | {oo['pnl_pct_per_candle']:+.4f} |")
    o += ["", "## Gaussian vs empirical reach on 2y (s = 1.0, t = 0.5, by entry offset a)", "",
          "| tf | side | a | G p_fill | E p_fill | G p_win | E p_win|fill | E stop|fill | "
          "E open|fill | E pnl/fill % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in grid:
        if c["s"] == 1.0 and c["t"] == 0.5 and c["a"] in (-0.75, -0.5, -0.25, 0.0, 0.5, 1.0):
            o.append(f"| {c['tf']} | {c['side']} | {c['a']} | {c['g_p_fill']:.3f} | "
                     f"{c['e_fill_rate']:.3f} | {c['g_p_win']:.3f} | {c['e_win_rate']:.3f} | "
                     f"{c['e_stop_rate']:.3f} | {c['e_open_at_close_rate']:.3f} | "
                     f"{c['e_pnl_pct_per_fill']:+.4f} |")
    o += ["", "## Top-5 per (tf, side) on 2y", ""]
    for k, p in params.items():
        o.append(f"- **{k}**: " + "; ".join(
            f"a={c['a']} s={c['s']} t={c['t']} fills={c['e_n_fills']} win={c['e_win_rate']:.3f} "
            f"pnl/fill={c['e_pnl_pct_per_fill']:+.4f} pnl/candle={c['e_pnl_pct_per_candle']:+.4f}"
            for c in p["top5"]))
    pos = sum(1 for c in grid if c["e_pnl_pct_per_candle"] > 0 and c["e_n_fills"] >= MIN_FILLS)
    o += ["", f"2y cells net-positive (>= {MIN_FILLS} fills): {pos} of "
          f"{sum(1 for c in grid if c['e_n_fills'] >= MIN_FILLS)}"]
    with open(path, "w") as fh:
        fh.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
