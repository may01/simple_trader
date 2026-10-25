"""Reproduction gate: closed-form Ridge (fit.ridge_fit) on 2y_az must reproduce the production
``indicators/candle_bounds/models_link_usdt.json`` (band_pct rel < 1e-3, coef max abs diff < 1e-6).

    python -m experiments.cb_class_bounds.gate_az
"""
import json

import numpy as np
import pandas as pd

from indicators.candle_bounds import kernel
from indicators.candle_bounds.models import models_path
from indicators.candle_bounds.params import FEATURES, SIDES, TFS

from . import config
from .fit import ridge_fit, ridge_predict

cols = [f"{tf}_{c}" for tf in TFS for c in ("is_closed", "high", "low", "open", "close", "rsi_ma8", "rsi_14",
                                               "macd_12_26_9", "macd_hist_12_26_9", "ema_25")]
df = pd.read_pickle(config.AZ_FRAME)[cols]
with open(models_path("link_usdt")) as f:
    prod = json.load(f)["models"]
ok = True
for tf in TFS:
    for side in SIDES:
        cf = kernel.closed_frame(df, tf, side)
        cf["target"] = cf["d"].shift(-1)
        rows = cf.dropna(subset=FEATURES + ["target"])
        X, y = rows[FEATURES].to_numpy(float), rows["target"].to_numpy(float)
        p = ridge_fit(X, y, 1.0)
        b = float(np.std(y - ridge_predict(p, X), ddof=1))
        q = prod[f"{tf}_{side}"]
        rel = abs(b - q["band_pct"]) / q["band_pct"]
        dc = float(np.max(np.abs(p["coef"] - np.asarray(q["coef"]))))
        di = abs(p["intercept"] - q["intercept"])
        good = rel < 1e-3 and dc < 1e-6 and di < 1e-9 and p["n_train"] == q["n_train"]
        ok &= good
        print(f"[gate_az] {tf}_{side}: n {p['n_train']} vs {q['n_train']} band {b:.6f} vs {q['band_pct']:.6f} rel {rel:.1e} | max|Δcoef| {dc:.1e} Δint {di:.1e} -> {'OK' if good else 'FAIL'}")
print("[gate_az]", "ALL OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
