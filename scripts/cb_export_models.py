#!/usr/bin/env python3
"""One-off: freeze the six candle-bound Ridge models into JSON (gate G1).

Fits each (tf, side) with ``cbnc.fit_side`` on the 2y_az training frame, checks
the refit against the stored ``candle_bounds_meta.json`` and the numpy predict
against sklearn, then writes ``indicators/candle_bounds/models_{pair}.json``.
Needs scikit-learn (not in the simple_trader image):

    python scripts/cb_export_models.py \
        --train <2y_az_link_usdt>/df_with_indicators.pkl \
        --meta  <oos2m_link_usdt>/candle_bounds_meta.json \
        --out   indicators/candle_bounds/models_link_usdt.json

Exits non-zero (and writes nothing) when any model misses a G1 threshold.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from indicators.candle_bounds.models import BoundModel  # noqa: E402
from indicators.candle_bounds.params import FEATURES, SIDES, TFS  # noqa: E402
from notebooks.candle_bounds_nc import cbnc  # noqa: E402

BAND_REL_TOL = 1e-3
PRED_ABS_TOL = 1e-12


def _needed_columns(tf: int) -> list[str]:
    return [f"{tf}_{c}" for c in (
        "is_closed", "high", "low", "open", "close",
        "rsi_ma8", "rsi_14", "macd_12_26_9", "macd_hist_12_26_9", "ema_25")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", default="link_usdt")
    ap.add_argument("--train-name", default="2y_az")
    args = ap.parse_args(argv)

    assert list(cbnc.FEATURES) == list(FEATURES), "cbnc.FEATURES drifted from params.FEATURES"
    with open(args.meta) as f:
        meta = json.load(f)

    cols = [c for tf in TFS for c in _needed_columns(tf)]
    df = pd.read_pickle(args.train)
    df = df[cols]  # drop everything else right away; the full frame is large

    models, ok = {}, True
    for tf in TFS:
        for side in SIDES:
            key = f"{tf}_{side}"
            sm = cbnc.fit_side(df, tf, side)
            scaler, ridge = sm.pipe[0], sm.pipe[-1]
            bm = BoundModel(
                tf=tf, side=side,
                mean=np.asarray(scaler.mean_, dtype=float),
                scale=np.asarray(scaler.scale_, dtype=float),
                coef=np.asarray(ridge.coef_, dtype=float),
                intercept=float(ridge.intercept_),
                band_pct=sm.band_pct, n_train=sm.n_train,
            )
            rows = cbnc.closed_frame(df, tf, side).dropna(subset=FEATURES + ["target"])
            X = rows[FEATURES].to_numpy()
            pred_diff = float(np.max(np.abs(bm.predict_pct(X) - sm.pipe.predict(X))))
            stored = float(meta[key]["band_pct"])
            band_rel = abs(sm.band_pct - stored) / stored
            good = band_rel < BAND_REL_TOL and pred_diff < PRED_ABS_TOL
            ok &= good
            print(f"{key:8s} n_train={sm.n_train:7d} band_pct={sm.band_pct:.12f} stored={stored:.12f} "
                  f"band_rel={band_rel:.3e} max_pred_diff={pred_diff:.3e} {'OK' if good else 'FAIL'}")
            models[key] = {
                "mean": bm.mean.tolist(), "scale": bm.scale.tolist(), "coef": bm.coef.tolist(),
                "intercept": bm.intercept, "band_pct": bm.band_pct, "n_train": bm.n_train,
            }

    if not ok:
        print("G1 FAILED — not writing", args.out, file=sys.stderr)
        return 1
    payload = {"pair": args.pair, "train": args.train_name, "features": list(FEATURES), "models": models}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
