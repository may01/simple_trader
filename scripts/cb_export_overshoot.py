#!/usr/bin/env python3
"""One-off: refit the marker-overshoot thresholds and print them against the
frozen ``params.OVERSHOOT_THR`` (phase 19, gate G1b). Writes nothing.

Threshold per (tf, side) = quantile ``--q`` (default ``OVERSHOOT_Q``) of the
overshoot z over the 2y **filtered markers**. Unlike the sidecar producer
(``notebooks/zone_profitability/run_cb_overshoot.py``, whose veto was the
oos2m-picked ladder), the marker set here is the main-flow one:
``cb_inzone_{side}`` and not ``extdone_{high|low}`` (``kernel.extdone`` with the
frozen 2y ``EXTDONE_LADDER``) and a finite ATR. Runs in the ``simple_trader``
image (no scikit-learn):

    docker compose -p cbtest run --rm ohlc_gen python3 scripts/cb_export_overshoot.py \
        [--train-ind /trader_data_long/train/2y_az_link_usdt/df_with_indicators.pkl] \
        [--train-cb  /trader_data_long/train/2y_link_usdt/df_with_candle_bounds.pkl] [--q 0.75]

``--train-cb`` is the stored 2y closed-bounds sidecar (``{tf}_cb_{side}``,
``_std``, ``cb_inzone_{side}``); pass ``--recompute-cb`` to run
``stage.compute_cb_columns`` on the training frame instead (slow).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from indicators.candle_bounds import kernel  # noqa: E402
from indicators.candle_bounds.params import (  # noqa: E402
    ATR_COL, DIRECTIONS, EXTDONE_LADDER, OVERSHOOT_Q, OVERSHOOT_THR, TFS,
)

VOL = "/trader_data_long/train"
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
TRAIN_CB = f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl"


def filtered_markers(df: pd.DataFrame, cb: pd.DataFrame, tf: int, side: str) -> np.ndarray:
    """Main-flow marker set: in zone, extdone veto off (2y ladder), finite ATR."""
    veto_side = "high" if side == "long" else "low"
    ext = kernel.extdone(df, tf, EXTDONE_LADDER[tf])[f"{tf}_extdone_{veto_side}"].to_numpy(dtype=bool)
    inzone = cb[f"{tf}_cb_inzone_{side}"].fillna(False).to_numpy(dtype=bool)
    atr = df[ATR_COL].to_numpy(dtype=float)
    return inzone & ~ext & np.isfinite(atr)


def refit_thresholds(df: pd.DataFrame, cb: pd.DataFrame, q: float) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for tf in TFS:
        z = kernel.overshoot(cb, df, tf)
        out[tf] = {}
        for side in DIRECTIONS:
            m = filtered_markers(df, cb, tf, side)
            zz = z[f"{tf}_cb_overshoot_{side}"].to_numpy(dtype=float)[m]
            out[tf][side] = float(np.nanquantile(zz, q)) if np.isfinite(zz).any() else float("nan")
    return out


def _needed(tf: int) -> list[str]:
    return [f"{tf}_high", f"{tf}_low", f"{tf}_is_closed"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ind", default=TRAIN_IND)
    ap.add_argument("--train-cb", default=TRAIN_CB)
    ap.add_argument("--recompute-cb", action="store_true")
    ap.add_argument("--q", type=float, default=OVERSHOOT_Q)
    ap.add_argument("--pair", default=os.environ.get("PAIR", "link_usdt"))
    a = ap.parse_args(argv)

    need = ["1_high", "1_low", "1_close", ATR_COL] + [c for tf in TFS for c in _needed(tf)]
    tr = pd.read_pickle(a.train_ind)
    if a.recompute_cb:
        from indicators.candle_bounds.stage import compute_cb_columns, required_columns
        cb = compute_cb_columns(tr[required_columns()], a.pair)
        if cb is None:
            print(f"no candle-bounds model for pair {a.pair!r}", flush=True)
            return 2
    else:
        cb = pd.read_pickle(a.train_cb).reindex(tr.index)
    tr = tr[need]
    print(f"[overshoot] train {tr.shape} {tr.index[0]} -> {tr.index[-1]}; q={a.q}", flush=True)

    refit = refit_thresholds(tr, cb, a.q)
    print("\n| tf | side | refit (main-flow veto) | frozen (sidecar) | diff | n markers |")
    print("|---|---|---|---|---|---|")
    for tf in TFS:
        for side in DIRECTIONS:
            n = int(filtered_markers(tr, cb, tf, side).sum())
            r, f = refit[tf][side], OVERSHOOT_THR[tf][side]
            print(f"| {tf} | {side} | {r:.4f} | {f:.4f} | {r - f:+.4f} | {n} |", flush=True)
    print("\nparams.py not written; freeze by hand if the refit is wanted (spec D1).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
