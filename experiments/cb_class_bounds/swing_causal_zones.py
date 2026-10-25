# Copied 2026-10-25 from worktrees/swing-sinusoid-experiment/experiments/swing_sinusoid/causal_zones.py (branch swing-sinusoid-experiment, uncommitted) — provenance for the cb_class_bounds experiment; unchanged except the relative import.
"""Causal cci_ma8 swing zones + combined zone + CB entry filter columns.

Closed-row logic per tf (nothing after row j is used at row j):
  * y = SMA-8 of cci_14 on closed candles.
  * Zig-zag apex detection with prominence p: after the last confirmed apex the running
    extreme in the opposite direction is tracked; when y retraces ≥ p from it (row j) the
    running extreme (row i) becomes a confirmed apex, known at row j.
  * cci zone: +1 after a confirmed LOW, −1 after a confirmed HIGH, from max(i + confirm, j)
    to i + const + window; a later confirmation overrides. 0 elsewhere.
  * combined = cci zone + 0_rsi, where 0_rsi = closed-row move_class_sym0 with ±3 → 0.
Minute rows take the value of the last closed candle (ffill).
Marker validity gates (user rule 2026-10-25): the 5-min and the 15-min cci_14 at the minute
(wide-frame per-minute values) must both lie in [−CCI_LIMIT, +CCI_LIMIT] for either side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .swing_cb_filter import allowed

CONST = {15: 11.0, 60: 11.0, 240: 10.0}   # train-median cci_ma8 apex spacing (apex_spacing_p05.csv)
STD = {15: 85.0, 60: 86.2, 240: 87.0}      # 2y cci_ma8 std (swing_sinusoid_params_p05.json apex_meta)
TFS = (15, 60, 240)
CCI5_LIMIT = 140.0
CCI_GATE_TFS = (5, 15)       # tfs whose per-minute cci_14 must be inside ±CCI5_LIMIT


def causal_apexes(y: np.ndarray, prom: float) -> list[tuple[int, int, int]]:
    """Confirmed apexes as (apex_row, sign, confirm_row), causal zig-zag.

    After a confirmed high the running low since it is tracked and becomes the next
    apex once y rises ≥ prom above it (and vice versa). Before the first apex both
    running extremes are tracked and whichever retraces first wins.
    """
    out: list[tuple[int, int, int]] = []
    n = len(y)
    valid = np.where(~np.isnan(y))[0]
    if len(valid) < 2:
        return out
    hi_i = lo_i = int(valid[0])
    s_last = 0
    for j in valid[1:]:
        j = int(j)
        if y[j] > y[hi_i]:
            hi_i = j
        if y[j] < y[lo_i]:
            lo_i = j
        if s_last <= 0 and hi_i < j and y[hi_i] - y[j] >= prom:
            out.append((hi_i, 1, j))
            s_last = 1
            seg = y[hi_i + 1: j + 1]
            lo_i = hi_i + 1 + int(np.nanargmin(seg))     # running low since the high
            continue
        if s_last >= 0 and lo_i < j and y[j] - y[lo_i] >= prom:
            out.append((lo_i, -1, j))
            s_last = -1
            seg = y[lo_i + 1: j + 1]
            hi_i = lo_i + 1 + int(np.nanargmax(seg))     # running high since the low
    return out


def cci_zone(y: np.ndarray, prom: float, const: float, confirm: int = 3, window: int = 3) -> tuple[np.ndarray, list]:
    n = len(y)
    z = np.zeros(n, dtype=int)
    apexes = causal_apexes(y, prom)
    for i, s, j in apexes:
        a = max(i + confirm, j)
        b = int(min(round(i + const + window), n - 1))
        if a > b:
            continue
        z[a:b + 1] = -1 if s > 0 else 1
    return z, apexes


def closed_rows_combined(df: pd.DataFrame, tf: int, confirm: int = 3, window: int = 3,
                         prom_frac: float = 0.5) -> pd.DataFrame:
    """Per closed row of tf: cci_ma8, cci zone, 0_rsi, combined."""
    m = df[f"{tf}_is_closed"].astype(bool).to_numpy()
    idx = df.index[m]
    cci = df[f"{tf}_cci_14"].to_numpy(dtype=float)[m]
    y = pd.Series(cci).rolling(8).mean().to_numpy()
    z, apexes = cci_zone(y, prom_frac * STD[tf], CONST[tf], confirm, window)
    mc = df[f"{tf}_move_class_sym0"].to_numpy(dtype=float)[m]
    rsi0 = np.where(np.abs(mc) == 3, 0, np.nan_to_num(mc, nan=0.0)).astype(int)
    out = pd.DataFrame({"cci_ma8": y, "cci_zone": z, "rsi0": rsi0, "combined": z + rsi0}, index=idx)
    out.attrs["apexes"] = apexes
    return out


def build_filter_frame(df: pd.DataFrame, confirm: int = 3, window: int = 3,
                       cci5_limit: float | None = CCI5_LIMIT) -> pd.DataFrame:
    """Minute-level frame: {tf}_swing_zone, {tf}_swing_cci_zone, {tf}_cb_f_swing_{long,short}.

    `cci5_limit`: the flags are additionally False where |{g}_cci_14| > limit for any
    g in CCI_GATE_TFS (None = no gate; a missing column is skipped)."""
    out = pd.DataFrame(index=df.index)
    cci5_ok = np.ones(len(df), dtype=bool)
    if cci5_limit is not None:
        for g in CCI_GATE_TFS:
            col = f"{g}_cci_14"
            if col in df.columns:
                c = df[col].to_numpy(dtype=float)
                ok = np.isfinite(c) & (np.abs(c) <= cci5_limit)
                out[f"cci{g}_ok"] = ok
                cci5_ok &= ok
        out["cci_gate_ok"] = cci5_ok
    comb = {}
    for tf in TFS:
        c = closed_rows_combined(df, tf, confirm, window)
        comb[tf] = c["combined"].reindex(df.index).ffill().fillna(0).astype(int).to_numpy()
        out[f"{tf}_swing_zone"] = comb[tf]
        out[f"{tf}_swing_cci_zone"] = c["cci_zone"].reindex(df.index).ffill().fillna(0).astype(int).to_numpy()
    # vectorised rule evaluation via a lookup over the 343 combinations
    key = (comb[240] + 3) * 49 + (comb[60] + 3) * 7 + (comb[15] + 3)
    for tf in TFS:
        for side in ("long", "short"):
            lut = np.array([allowed(tf, side, z15, z60, z240)
                            for z240 in range(-3, 4) for z60 in range(-3, 4) for z15 in range(-3, 4)], dtype=bool)
            out[f"{tf}_cb_f_swing_{side}"] = lut[key] & cci5_ok
    return out
