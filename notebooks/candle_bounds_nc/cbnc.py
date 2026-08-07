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
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TFS = (15, 60, 240)
SIDES = ("high", "low")
FEATURES = [
    "R_position", "R_slope", "R_distance",
    "M_position", "M_slope", "M_distance",
    "A_position", "A_slope",
    "V_std12", "V_absmean12", "V_range", "V_body",
    "P_z1", "P_z2", "P_rm",
]
ZONE_FRAC = 0.05  # entry-zone span fraction, matches the closed-zone artifact
# Target haircut, in % of price, applied toward the loss side: the long target
# sits below the predicted high, the short target above the predicted low.
# Compensates the bound being an *extreme* estimate — a fill needs the target
# strictly inside the reached range. History: 0.15 (first haircut) → 0.35
# (user widened by a further 0.2 on 2026-07-28).
TGT_SHRINK_PCT = 0.35


def closed_frame(df: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Per-closed-candle frame indexed by closing timestamp.

    Row k holds candle k's *input* features (candle k viewed as the latest
    candle) plus ``d`` (its own extension) and ``target`` = d[k+1].
    """
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712
    c = df.loc[closed]
    ext = c[f"{tf}_{side}"]
    d = ext.pct_change() * 100.0
    rm = d.rolling(6).mean()

    out = pd.DataFrame(index=c.index)
    out["R_position"] = c[f"{tf}_rsi_ma8"]
    out["R_slope"] = c[f"{tf}_rsi_ma8"].diff()
    out["R_distance"] = c[f"{tf}_rsi_14"] - c[f"{tf}_rsi_ma8"]
    out["M_position"] = c[f"{tf}_macd_12_26_9"]
    out["M_slope"] = c[f"{tf}_macd_12_26_9"].diff()
    out["M_distance"] = c[f"{tf}_macd_hist_12_26_9"]
    out["A_position"] = (c[f"{tf}_close"] - c[f"{tf}_ema_25"]) / c[f"{tf}_close"] * 100.0
    out["A_slope"] = c[f"{tf}_ema_25"].diff()
    out["V_std12"] = d.rolling(12).std()
    out["V_absmean12"] = d.abs().rolling(12).mean()
    out["V_range"] = (c[f"{tf}_high"] - c[f"{tf}_low"]) / c[f"{tf}_low"] * 100.0
    out["V_body"] = (c[f"{tf}_close"] - c[f"{tf}_open"]) / c[f"{tf}_open"] * 100.0
    out["P_z1"] = d - rm.shift(1)
    out["P_z2"] = (d - rm.shift(1)).shift(1)
    out["P_rm"] = rm
    out["d"] = d
    out["ext"] = ext
    out["target"] = d.shift(-1)
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
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(X, y)
    band_pct = float(np.std(y - pipe.predict(X), ddof=1))
    return SideModel(tf=tf, side=side, pipe=pipe, band_pct=band_pct, n_train=len(rows))


def _to_rows(closed_series: pd.Series, index: pd.Index) -> pd.Series:
    """Broadcast a per-closed-candle series onto the 1-minute index so that row t
    carries the last candle closed *strictly before* t (azlib space.py convention)."""
    return closed_series.reindex(index).shift(1).ffill()


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


def forming_features(df: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Per-1-minute-row features with the forming candle as the input candle.

    Rolling blocks decompose into (closed history before the forming candle) +
    (the forming candle's own running value), so everything vectorizes with
    shift(1)-ffill broadcasts of the closed sequence.
    """
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712
    c = df.loc[closed]
    ext = c[f"{tf}_{side}"]
    d = ext.pct_change() * 100.0
    rm = d.rolling(6).mean()

    idx = df.index
    E1 = _to_rows(ext, idx)                              # ext[k-1]
    d_f = (df[f"{tf}_{side}"] - E1) / E1 * 100.0         # forming candle's own d

    S11 = _to_rows(d.rolling(11).sum(), idx)
    Q11 = _to_rows((d ** 2).rolling(11).sum(), idx)
    A11 = _to_rows(d.abs().rolling(11).sum(), idx)
    S5 = _to_rows(d.rolling(5).sum(), idx)
    rm_prev = _to_rows(rm, idx)                          # rm[k-1]
    z1_prev = _to_rows(d - rm.shift(1), idx)             # P_z1 of candle k-1
    rsi_prev = _to_rows(c[f"{tf}_rsi_ma8"], idx)
    macd_prev = _to_rows(c[f"{tf}_macd_12_26_9"], idx)
    ema_prev = _to_rows(c[f"{tf}_ema_25"], idx)

    out = pd.DataFrame(index=idx)
    out["R_position"] = df[f"{tf}_rsi_ma8"]
    out["R_slope"] = df[f"{tf}_rsi_ma8"] - rsi_prev
    out["R_distance"] = df[f"{tf}_rsi_14"] - df[f"{tf}_rsi_ma8"]
    out["M_position"] = df[f"{tf}_macd_12_26_9"]
    out["M_slope"] = df[f"{tf}_macd_12_26_9"] - macd_prev
    out["M_distance"] = df[f"{tf}_macd_hist_12_26_9"]
    out["A_position"] = (df[f"{tf}_close"] - df[f"{tf}_ema_25"]) / df[f"{tf}_close"] * 100.0
    out["A_slope"] = df[f"{tf}_ema_25"] - ema_prev
    S12, n = S11 + d_f, 12
    out["V_std12"] = np.sqrt((Q11 + d_f ** 2 - S12 ** 2 / n) / (n - 1))
    out["V_absmean12"] = (A11 + d_f.abs()) / n
    out["V_range"] = (df[f"{tf}_high"] - df[f"{tf}_low"]) / df[f"{tf}_low"] * 100.0
    out["V_body"] = (df[f"{tf}_close"] - df[f"{tf}_open"]) / df[f"{tf}_open"] * 100.0
    out["P_z1"] = d_f - rm_prev
    out["P_z2"] = z1_prev
    out["P_rm"] = (S5 + d_f) / 6.0
    out["ext_f"] = df[f"{tf}_{side}"]
    return out


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
