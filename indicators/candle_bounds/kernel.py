"""Candle-bounds kernel — pure functions over the 1-minute wide frame.

Ported from ``notebooks/candle_bounds_nc/cbnc.py`` (``closed_frame``,
``_to_rows``, ``infer_closed``, ``forming_features``, ``infer_forming``,
``add_zones``), numerically identical except that the prediction comes from the
frozen numpy ``BoundModel``. Adds the closed entry zone and the raw marker
(``{tf}_cb_inzone_{long,short}``). ``infer_forming`` is split into an unshifted
``forming_prediction`` (live takes its last row) and ``forming_bounds`` (the
one-row shift used by prep).

Wide-frame contract (``data._build_wide_df``): one row per minute; ``{tf}_high``
/ ``{tf}_low`` are the candle's running extremes; ``{tf}_is_closed`` is True only
on the candle's last minute; indicator columns are final on the closed row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.candle_bounds.models import BoundModel
from indicators.candle_bounds.params import ATR_COL, FEATURES, OVERSHOOT_THR, TGT_SHRINK_PCT, ZONE_FRAC


def to_rows(closed_series: pd.Series, index: pd.Index) -> pd.Series:
    """Broadcast a per-closed-candle series onto the 1-minute index so that row t
    carries the last candle closed *strictly before* t."""
    return closed_series.reindex(index).shift(1).ffill()


def closed_frame(df: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Per-closed-candle frame indexed by closing timestamp.

    Row k holds candle k's input features (candle k viewed as the latest
    candle), ``d`` (its own extension, %) and ``ext`` (its extreme)."""
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712 — column may be object dtype
    c = df.loc[closed]
    ext = c[f"{tf}_{side}"]
    d = ext.pct_change() * 100.0
    rm = d.rolling(6).mean()

    cols: dict[str, object] = {}
    cols["R_position"] = c[f"{tf}_rsi_ma8"]
    cols["R_slope"] = c[f"{tf}_rsi_ma8"].diff()
    cols["R_distance"] = c[f"{tf}_rsi_14"] - c[f"{tf}_rsi_ma8"]
    cols["M_position"] = c[f"{tf}_macd_12_26_9"]
    cols["M_slope"] = c[f"{tf}_macd_12_26_9"].diff()
    cols["M_distance"] = c[f"{tf}_macd_hist_12_26_9"]
    cols["A_position"] = (c[f"{tf}_close"] - c[f"{tf}_ema_25"]) / c[f"{tf}_close"] * 100.0
    cols["A_slope"] = c[f"{tf}_ema_25"].diff()
    cols["V_std12"] = d.rolling(12).std()
    cols["V_absmean12"] = d.abs().rolling(12).mean()
    cols["V_range"] = (c[f"{tf}_high"] - c[f"{tf}_low"]) / c[f"{tf}_low"] * 100.0
    cols["V_body"] = (c[f"{tf}_close"] - c[f"{tf}_open"]) / c[f"{tf}_open"] * 100.0
    cols["P_z1"] = d - rm.shift(1)
    cols["P_z2"] = (d - rm.shift(1)).shift(1)
    cols["P_rm"] = rm
    cols["d"] = d
    cols["ext"] = ext
    return pd.DataFrame(cols, index=c.index)


def closed_bounds(df: pd.DataFrame, model: BoundModel) -> pd.DataFrame:
    """Closed-candle bounds: prediction made at each candle's close, broadcast
    to the rows of the following candle (``to_rows``).

    Columns ``{tf}_cb_{side}``, ``_std``, ``_up``, ``_dn``; index = ``df.index``."""
    tf, side = model.tf, model.side
    cf = closed_frame(df, tf, side)
    valid = cf.dropna(subset=FEATURES)
    pred = pd.Series(np.nan, index=cf.index, dtype=float)
    if len(valid):
        pred.loc[valid.index] = model.predict_pct(valid[FEATURES].to_numpy(dtype=float))
    ext = cf["ext"].astype(float)
    bound = ext * (1 + pred / 100.0)
    band = ext * model.band_pct / 100.0

    b = to_rows(bound, df.index)
    w = to_rows(band, df.index)
    cols: dict[str, object] = {}
    cols[f"{tf}_cb_{side}"] = b
    cols[f"{tf}_cb_{side}_std"] = w
    cols[f"{tf}_cb_{side}_up"] = b + w
    cols[f"{tf}_cb_{side}_dn"] = b - w
    return pd.DataFrame(cols, index=df.index)


def closed_zones(cb: pd.DataFrame, df: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Closed entry zones and the raw marker for ``tf``.

    ``cb`` must hold ``{tf}_cb_high`` / ``{tf}_cb_low``; ``df`` supplies
    ``1_low`` / ``1_high``. No span-validity guard (matches the stored artifact).
    A NaN level makes the marker False."""
    hi = cb[f"{tf}_cb_high"].astype(float)
    lo = cb[f"{tf}_cb_low"].astype(float)
    span = hi - lo
    zone_long = lo + ZONE_FRAC * span
    zone_short = hi - ZONE_FRAC * span

    cols: dict[str, object] = {}
    cols[f"{tf}_cb_zone_long"] = zone_long
    cols[f"{tf}_cb_zone_short"] = zone_short
    cols[f"{tf}_cb_inzone_long"] = (df["1_low"] <= zone_long).fillna(False).astype(bool)
    cols[f"{tf}_cb_inzone_short"] = (df["1_high"] >= zone_short).fillna(False).astype(bool)
    return pd.DataFrame(cols, index=cb.index)


HISTORY_COLS: tuple[str, ...] = (
    "E1", "S11", "Q11", "A11", "S5", "rm_prev", "z1_prev", "rsi_prev", "macd_prev", "ema_prev",
)


def forming_history(df: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Closed-history aggregates the forming features need, per 1-minute row
    (``to_rows``: from the candles closed strictly before the row). Constant
    through a candle, so the live stage can cache the current candle's row.
    Columns ``HISTORY_COLS``; index = ``df.index``."""
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712 — column may be object dtype
    c = df.loc[closed]
    ext = c[f"{tf}_{side}"]
    d = ext.pct_change() * 100.0
    rm = d.rolling(6).mean()

    idx = df.index
    cols: dict[str, object] = {}
    cols["E1"] = to_rows(ext, idx)                            # ext[k-1]
    cols["S11"] = to_rows(d.rolling(11).sum(), idx)
    cols["Q11"] = to_rows((d ** 2).rolling(11).sum(), idx)
    cols["A11"] = to_rows(d.abs().rolling(11).sum(), idx)
    cols["S5"] = to_rows(d.rolling(5).sum(), idx)
    cols["rm_prev"] = to_rows(rm, idx)                        # rm[k-1]
    cols["z1_prev"] = to_rows(d - rm.shift(1), idx)           # P_z1 of candle k-1
    cols["rsi_prev"] = to_rows(c[f"{tf}_rsi_ma8"], idx)
    cols["macd_prev"] = to_rows(c[f"{tf}_macd_12_26_9"], idx)
    cols["ema_prev"] = to_rows(c[f"{tf}_ema_25"], idx)
    return pd.DataFrame(cols, index=idx)


def forming_row_features(df: pd.DataFrame, hist: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Per-row forming features from the row's own (forming) values and its
    ``forming_history`` row ``hist`` (same index). Columns FEATURES + ``ext_f``."""
    E1 = hist["E1"]
    d_f = (df[f"{tf}_{side}"] - E1) / E1 * 100.0         # forming candle's own d

    cols: dict[str, object] = {}
    cols["R_position"] = df[f"{tf}_rsi_ma8"]
    cols["R_slope"] = df[f"{tf}_rsi_ma8"] - hist["rsi_prev"]
    cols["R_distance"] = df[f"{tf}_rsi_14"] - df[f"{tf}_rsi_ma8"]
    cols["M_position"] = df[f"{tf}_macd_12_26_9"]
    cols["M_slope"] = df[f"{tf}_macd_12_26_9"] - hist["macd_prev"]
    cols["M_distance"] = df[f"{tf}_macd_hist_12_26_9"]
    cols["A_position"] = (df[f"{tf}_close"] - df[f"{tf}_ema_25"]) / df[f"{tf}_close"] * 100.0
    cols["A_slope"] = df[f"{tf}_ema_25"] - hist["ema_prev"]
    S12, n = hist["S11"] + d_f, 12
    cols["V_std12"] = np.sqrt((hist["Q11"] + d_f ** 2 - S12 ** 2 / n) / (n - 1))
    cols["V_absmean12"] = (hist["A11"] + d_f.abs()) / n
    cols["V_range"] = (df[f"{tf}_high"] - df[f"{tf}_low"]) / df[f"{tf}_low"] * 100.0
    cols["V_body"] = (df[f"{tf}_close"] - df[f"{tf}_open"]) / df[f"{tf}_open"] * 100.0
    cols["P_z1"] = d_f - hist["rm_prev"]
    cols["P_z2"] = hist["z1_prev"]
    cols["P_rm"] = (hist["S5"] + d_f) / 6.0
    cols["ext_f"] = df[f"{tf}_{side}"]
    return pd.DataFrame(cols, index=df.index)


def forming_features(df: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Per-1-minute-row features with the forming candle as the input candle.

    Rolling blocks decompose into (closed history before the forming candle,
    ``forming_history``) + (the forming candle's own running value,
    ``forming_row_features``); on a closing row the result equals
    ``closed_frame`` for that candle. Columns FEATURES + ``ext_f``."""
    return forming_row_features(df, forming_history(df, tf, side), tf, side)


def forming_prediction(df: pd.DataFrame, model: BoundModel,
                       hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """UNSHIFTED forming-candle prediction per 1-minute row (price units).

    ``bound`` is NaN where any feature is NaN; ``band`` = ext_f * band_pct / 100
    is NaN only where ext_f is (as in ``cbnc.infer_forming``). ``hist``: a
    precomputed ``forming_history`` for ``df``'s rows (live cache); default
    computes it from ``df``."""
    tf, side = model.tf, model.side
    if hist is None:
        ff = forming_features(df, tf, side)
    else:
        ff = forming_row_features(df, hist, tf, side)
    valid = ff.dropna(subset=FEATURES)
    pred = pd.Series(np.nan, index=ff.index, dtype=float)
    if len(valid):
        pred.loc[valid.index] = model.predict_pct(valid[FEATURES].to_numpy(dtype=float))
    ext_f = ff["ext_f"].astype(float)
    cols: dict[str, object] = {}
    cols["bound"] = ext_f * (1 + pred / 100.0)
    cols["band"] = ext_f * model.band_pct / 100.0
    return pd.DataFrame(cols, index=df.index)


def forming_bounds(pred: pd.DataFrame, tf: int, side: str) -> pd.DataFrame:
    """Forming bounds: ``forming_prediction`` shifted one row, so row t carries
    the prediction made on row t-1. Columns ``{tf}_cbnc_{side}[,_std,_up,_dn]``."""
    b = pred["bound"].shift(1)
    w = pred["band"].shift(1)
    cols: dict[str, object] = {}
    cols[f"{tf}_cbnc_{side}"] = b
    cols[f"{tf}_cbnc_{side}_std"] = w
    cols[f"{tf}_cbnc_{side}_up"] = b + w
    cols[f"{tf}_cbnc_{side}_dn"] = b - w
    return pd.DataFrame(cols, index=pred.index)


def forming_zones(cbnc: pd.DataFrame, cb: pd.DataFrame, df: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Hybrid entry zones and the forming/closed intersection for ``tf``.

    Long: SL = closed ``cb_low``, target = ``cbnc_high`` hair-cut
    TGT_SHRINK_PCT % toward the loss side; short mirrored. Zone level sits
    ZONE_FRAC of the SL->target span away from the SL. An inverted span or a
    NaN level makes the marker False. ``cbx_inzone`` = ``cbnc_inzone`` AND the
    closed ``cb_inzone``."""
    idx = cbnc.index
    shrink = TGT_SHRINK_PCT / 100.0
    tgt_long = cbnc[f"{tf}_cbnc_high"].astype(float) * (1.0 - shrink)
    tgt_short = cbnc[f"{tf}_cbnc_low"].astype(float) * (1.0 + shrink)
    sl_long = cb[f"{tf}_cb_low"].reindex(idx).astype(float)
    sl_short = cb[f"{tf}_cb_high"].reindex(idx).astype(float)
    z_long = sl_long + ZONE_FRAC * (tgt_long - sl_long)
    z_short = sl_short - ZONE_FRAC * (sl_short - tgt_short)
    ok_long = (tgt_long - sl_long) > 0
    ok_short = (sl_short - tgt_short) > 0
    low1 = df["1_low"].reindex(idx)
    high1 = df["1_high"].reindex(idx)

    cols: dict[str, object] = {}
    cols[f"{tf}_cbnc_tgt_long"] = tgt_long
    cols[f"{tf}_cbnc_tgt_short"] = tgt_short
    cols[f"{tf}_cbnc_zone_long"] = z_long
    cols[f"{tf}_cbnc_zone_short"] = z_short
    nc_long = ((low1 <= z_long) & ok_long).fillna(False).astype(bool)
    nc_short = ((high1 >= z_short) & ok_short).fillna(False).astype(bool)
    cols[f"{tf}_cbnc_inzone_long"] = nc_long
    cols[f"{tf}_cbnc_inzone_short"] = nc_short
    cb_long = cb[f"{tf}_cb_inzone_long"].reindex(idx).fillna(False).astype(bool)
    cb_short = cb[f"{tf}_cb_inzone_short"].reindex(idx).fillna(False).astype(bool)
    cols[f"{tf}_cbx_inzone_long"] = nc_long & cb_long
    cols[f"{tf}_cbx_inzone_short"] = nc_short & cb_short
    return pd.DataFrame(cols, index=idx)


def candle_progress(df: pd.DataFrame, tf: int, *, by_rows: bool = False) -> np.ndarray:
    """Progress of the forming ``tf`` candle at each row, in (0, 1].

    ``by_rows=False`` (production): wall-clock minutes since the candle's open,
    counting the row itself, over ``tf`` — robust to missing minutes.
    ``by_rows=True`` (experiment convention, only to reproduce the stored
    sidecar): row position inside the candle + 1 over ``tf``, where rows after a
    ``{tf}_is_closed`` row through the next closing row share a candle."""
    if not by_rows:
        idx = pd.DatetimeIndex(df.index)
        minutes = (idx - idx.floor(f"{tf}min")) / pd.Timedelta(minutes=1)
        return (np.asarray(minutes, dtype=float) + 1.0) / tf
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712 — column may be object dtype
    n = len(closed)
    if n == 0:
        return np.empty(0, dtype=float)
    cid = np.concatenate(([0], np.cumsum(closed)[:-1]))
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    pos = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    return (pos + 1.0) / tf


def ladder_threshold(progress: np.ndarray, ladder: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Pull-away threshold d(progress) in 1-min ATR: each ``(p0, d)`` step
    applies for ``progress >= p0`` until the next step; ``+inf`` (never set)
    below the first step."""
    progress = np.asarray(progress, dtype=float)
    d = np.full(progress.shape, np.inf)
    for p0, dv in ladder:
        d[progress >= p0] = dv
    return d


def extdone(df: pd.DataFrame, tf: int, ladder: dict[str, tuple[tuple[float, float], ...]], *,
            by_rows: bool = False) -> pd.DataFrame:
    """Per-minute extreme-done state of the forming ``tf`` candle (NOT sticky).

    ``extdone_high[t] = ({tf}_high - 1_close) / atr >= d_high(progress)``; low
    mirrored with ``(1_close - {tf}_low)``. False where ATR is NaN / non-finite /
    non-positive or the extreme is NaN. ``_lvl`` = the running extreme where the
    state is True, NaN elsewhere. ``ladder`` = ``{"high": steps, "low": steps}``."""
    prog = candle_progress(df, tf, by_rows=by_rows)
    close = df["1_close"].to_numpy(dtype=float)
    atr = df[ATR_COL].to_numpy(dtype=float)
    ok_atr = np.isfinite(atr) & (atr > 0)
    safe_atr = np.where(ok_atr, atr, 1.0)

    cols: dict[str, object] = {}
    lvls = {}
    for side in ("high", "low"):
        run = df[f"{tf}_{side}"].to_numpy(dtype=float)
        pull = (run - close) if side == "high" else (close - run)
        pull = pull / safe_atr
        with np.errstate(invalid="ignore"):
            state = ok_atr & (pull >= ladder_threshold(prog, ladder[side]))
        cols[f"{tf}_extdone_{side}"] = state.astype(bool)
        lvls[side] = np.where(state, run, np.nan)
    for side in ("high", "low"):
        cols[f"{tf}_extdone_{side}_lvl"] = lvls[side].astype(float)
    return pd.DataFrame(cols, index=df.index)


def overshoot(cb: pd.DataFrame, df: pd.DataFrame, tf: int) -> pd.DataFrame:
    """How far the minute's extreme is past the closed bound, in band sd.

    ``{tf}_cb_overshoot_long = (cb_low - 1_low) / cb_low_std``,
    ``{tf}_cb_overshoot_short = (1_high - cb_high) / cb_high_std``. Defined on
    every row (negative when price is away from the bound). NaN bound, std or
    extreme, and std <= 0, give NaN -- never +-inf (the sidecar did)."""
    cols: dict[str, object] = {}
    for side, bound_side, ext_col in (("long", "low", "1_low"), ("short", "high", "1_high")):
        b = cb[f"{tf}_cb_{bound_side}"].to_numpy(dtype=float)
        w = cb[f"{tf}_cb_{bound_side}_std"].to_numpy(dtype=float)
        x = df[ext_col].to_numpy(dtype=float)
        num = (b - x) if side == "long" else (x - b)
        ok = np.isfinite(num) & np.isfinite(w) & (w > 0)
        z = np.full(len(df), np.nan)
        z[ok] = num[ok] / w[ok]
        cols[f"{tf}_cb_overshoot_{side}"] = z
    return pd.DataFrame(cols, index=df.index)


def filters(df: pd.DataFrame, ext: pd.DataFrame, over: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Pass flags of the row's state (not applied to any marker).

    extdone: long = not ``extdone_high``, short = not ``extdone_low``.
    trend: long = ``ema_25_slope_class != -1``, short = ``!= +1`` (NaN -> both True).
    mcx: long = ``move_class == -3``, short = ``== +3`` (NaN -> both False).
    overshoot: not ``cb_overshoot_{side} > OVERSHOOT_THR[tf][side]`` (NaN -> True)."""
    trend = pd.to_numeric(df[f"{tf}_ema_25_slope_class"], errors="coerce").to_numpy(dtype=float)
    mcx = pd.to_numeric(df[f"{tf}_move_class"], errors="coerce").to_numpy(dtype=float)
    hi = ext[f"{tf}_extdone_high"].reindex(df.index).fillna(False).to_numpy(dtype=bool)
    lo = ext[f"{tf}_extdone_low"].reindex(df.index).fillna(False).to_numpy(dtype=bool)

    cols: dict[str, object] = {}
    cols[f"{tf}_cb_f_extdone_long"] = ~hi
    cols[f"{tf}_cb_f_extdone_short"] = ~lo
    cols[f"{tf}_cb_f_trend_long"] = trend != -1    # NaN != -1 -> True
    cols[f"{tf}_cb_f_trend_short"] = trend != 1
    cols[f"{tf}_cb_f_mcx_long"] = mcx == -3        # NaN == -3 -> False
    cols[f"{tf}_cb_f_mcx_short"] = mcx == 3
    for side in ("long", "short"):
        z = over[f"{tf}_cb_overshoot_{side}"].reindex(df.index).to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            cols[f"{tf}_cb_f_overshoot_{side}"] = ~(z > OVERSHOOT_THR[tf][side])   # NaN > thr -> False -> pass
    return pd.DataFrame(cols, index=df.index)
