"""Candle-bounds prediction — closed-candle models + non-closed (forming) inference.

Re-implements the frozen algorithm of
``external/docs/superpowers/experiment/candle_bounds_algorithm.md`` (the original
producer scripts were scratchpad-only and lost) and extends it to per-minute
inference on the *forming* candle: at every 1-minute row the forming candle plays
the "latest candle" role, so the prediction is the extreme of the next
~tf-minute window starting at that row.

Two invariants pin the re-implementation to the stored artifact:

1. ``fit_side`` on 2y_az must reproduce the ``band_pct`` values persisted in
   ``candle_bounds_meta.json``.
2. Closed-candle inference on oos2m must reproduce the stored
   ``{tf}_cb_{side}`` / ``_up`` / ``_dn`` columns of ``df_with_candle_bounds.pkl``.

At a candle's closing minute the forming features equal the closed features, so
the non-closed prediction converges row-exactly to the closed prediction for the
next candle — asserted by the driver as a third gate.

Production implementation: ``indicators/candle_bounds/`` (kernel + params, numpy
models exported by ``scripts/cb_export_models.py``). The constants, the feature
builders (``closed_frame`` without ``target``, ``forming_features``) and
``_to_rows`` come from there; this module keeps the sklearn fitting
(``fit_side``/``SideModel``) and the experiment-side inference wrappers.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from indicators.candle_bounds import kernel  # noqa: E402
# Re-exported frozen constants. TGT_SHRINK_PCT: target haircut, in % of price,
# applied toward the loss side (long target sits 0.15% below the predicted
# high, short target 0.15% above the predicted low); ZONE_FRAC: entry-zone span
# fraction, matches the closed-zone artifact.
from indicators.candle_bounds.params import (  # noqa: E402,F401
    FEATURES, SIDES, TFS, TGT_SHRINK_PCT, ZONE_FRAC,
)


def closed_frame(df: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Per-closed-candle frame indexed by closing timestamp.

    ``kernel.closed_frame`` (candle k's input features, ``d``, ``ext``) plus
    ``target`` = d[k+1], which the fitting needs.
    """
    out = kernel.closed_frame(df, tf, side)
    out["target"] = out["d"].shift(-1)
    return out


@dataclass
class SideModel:
    tf: int
    side: str
    pipe: object
    band_pct: float
    n_train: int

    def predict_pct(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X[FEATURES].to_numpy())


def fit_side(train_df: pd.DataFrame, tf: int, side: str) -> SideModel:
    cf = closed_frame(train_df, tf, side)
    rows = cf.dropna(subset=FEATURES + ["target"])
    X = rows[FEATURES].to_numpy()
    y = rows["target"].to_numpy()
    # sklearn is not in the simple_trader image; imported lazily so the
    # feature builders stay importable there (fit_side needs the zpenv venv).
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(X, y)
    band_pct = float(np.std(y - pipe.predict(X), ddof=1))
    return SideModel(tf=tf, side=side, pipe=pipe, band_pct=band_pct, n_train=len(rows))


_to_rows = kernel.to_rows  # row t carries the last candle closed strictly before t


def infer_closed(df: pd.DataFrame, model: SideModel) -> pd.DataFrame:
    """Stored-artifact reproduction: prediction made at each candle's close,
    shifted one row and forward-filled through the following candle."""
    tf, side = model.tf, model.side
    cf = closed_frame(df, tf, side)
    valid = cf.dropna(subset=FEATURES)
    pred = pd.Series(np.nan, index=cf.index)
    pred.loc[valid.index] = model.predict_pct(valid)
    bound = cf["ext"] * (1 + pred / 100.0)
    band = cf["ext"] * model.band_pct / 100.0

    out = pd.DataFrame(index=df.index)
    b = _to_rows(bound, df.index)
    w = _to_rows(band, df.index)
    out[f"{tf}_cb_{side}"] = b
    out[f"{tf}_cb_{side}_std"] = w
    out[f"{tf}_cb_{side}_up"] = b + w
    out[f"{tf}_cb_{side}_dn"] = b - w
    return out


forming_features = kernel.forming_features  # forming candle as the input candle


def infer_forming(df: pd.DataFrame, model: SideModel) -> pd.DataFrame:
    """Non-closed bounds per 1-minute row.

    The prediction uses data through row t and is stored at t+1 (shift(1)), the
    same no-same-row-look-ahead convention as the closed broadcast. Columns:
    ``{tf}_cbnc_{side}[,_std,_up,_dn]``.
    """
    tf, side = model.tf, model.side
    ff = forming_features(df, tf, side)
    valid = ff.dropna(subset=FEATURES)
    pred = pd.Series(np.nan, index=ff.index)
    pred.loc[valid.index] = model.predict_pct(valid)
    bound = (ff["ext_f"] * (1 + pred / 100.0)).shift(1)
    band = (ff["ext_f"] * model.band_pct / 100.0).shift(1)

    out = pd.DataFrame(index=df.index)
    out[f"{tf}_cbnc_{side}"] = bound
    out[f"{tf}_cbnc_{side}_std"] = band
    out[f"{tf}_cbnc_{side}_up"] = bound + band
    out[f"{tf}_cbnc_{side}_dn"] = bound - band
    return out


def add_zones(out: pd.DataFrame, df: pd.DataFrame, cb: pd.DataFrame, tf: int) -> None:
    """Entry zones on a HYBRID level space, plus intersection with the stored
    closed-zone markers.

    The closed zone (cb artifact) spans its own bound pair — long: SL =
    cb_low, target = cb_high; short mirrored. The non-closed zone here keeps
    the closed bound as the stop-loss side but takes the per-minute forming
    prediction as the target side:

        long:  SL = cb_low  (closed), target = cbnc_high (non-closed) - TGT_SHRINK_PCT
        short: SL = cb_high (closed), target = cbnc_low  (non-closed) + TGT_SHRINK_PCT

    The operative target is the nc bound hair-cut TGT_SHRINK_PCT% of price
    toward the loss side (stored as ``{tf}_cbnc_tgt_{long,short}``). Zone level
    sits ZONE_FRAC of the SL→target span away from the SL, same rule as the
    closed zones. A row where the hybrid span is inverted (target on the wrong
    side of the SL) has no valid zone -> inzone False.
    """
    shrink = TGT_SHRINK_PCT / 100.0
    tgt_long = out[f"{tf}_cbnc_high"] * (1.0 - shrink)
    tgt_short = out[f"{tf}_cbnc_low"] * (1.0 + shrink)
    sl_long = cb[f"{tf}_cb_low"].reindex(out.index)
    sl_short = cb[f"{tf}_cb_high"].reindex(out.index)
    z_long = sl_long + ZONE_FRAC * (tgt_long - sl_long)
    z_short = sl_short - ZONE_FRAC * (sl_short - tgt_short)
    ok_long = (tgt_long - sl_long) > 0
    ok_short = (sl_short - tgt_short) > 0
    low1 = df["1_low"].reindex(out.index)
    high1 = df["1_high"].reindex(out.index)
    out[f"{tf}_cbnc_tgt_long"] = tgt_long
    out[f"{tf}_cbnc_tgt_short"] = tgt_short
    out[f"{tf}_cbnc_zone_long"] = z_long
    out[f"{tf}_cbnc_zone_short"] = z_short
    out[f"{tf}_cbnc_inzone_long"] = ((low1 <= z_long) & ok_long).fillna(False)
    out[f"{tf}_cbnc_inzone_short"] = ((high1 >= z_short) & ok_short).fillna(False)
    cb_long = cb[f"{tf}_cb_inzone_long"].reindex(out.index).fillna(False).astype(bool)
    cb_short = cb[f"{tf}_cb_inzone_short"].reindex(out.index).fillna(False).astype(bool)
    out[f"{tf}_cbx_inzone_long"] = out[f"{tf}_cbnc_inzone_long"] & cb_long
    out[f"{tf}_cbx_inzone_short"] = out[f"{tf}_cbnc_inzone_short"] & cb_short
