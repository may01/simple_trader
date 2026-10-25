"""Closed-row series extraction + derivation for one tf.

Two stages so the 2y part loop can concatenate RAW rows across parts first and
derive the moving averages once over the continuous series:

    raw = extract_raw(df_wide, tf)      # per part
    full = derive(pd.concat(raws))      # once

Derivations mirror the production recipes: rsi_ma{L} = talib.EMA(rsi_14, L)
(indicators.library.momentum.RSI_MAField), cci_ma{L} = rolling SMA of cci_14
(indicators.library.oscillators.CCI_MAField).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from . import config


def extract_raw(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Closed-candle rows of one tf, short column names, labels if present."""
    mask = df[f"{tf}_is_closed"].astype(bool).to_numpy()
    cols = {short: f"{tf}_{short}" for short in config.RAW_COLS}
    for short, wide in config.label_cols(tf).items():
        if wide in df.columns:
            cols[short] = wide
    out = pd.DataFrame(
        {short: df[wide].to_numpy()[mask] for short, wide in cols.items()},
        index=df.index[mask],
    )
    out.index.name = "ts"
    return out


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """Append the swing series to a raw closed-row frame (returns a copy)."""
    out = raw.copy()
    rsi = out["rsi_14"].to_numpy(dtype=float)
    for length in config.RSI_MA_LENGTHS:
        out[f"rsi_ma{length}"] = talib.EMA(rsi, timeperiod=length)
    for length in config.CCI_MA_LENGTHS:
        out[f"cci_ma{length}"] = out["cci_14"].rolling(length).mean().to_numpy()
    out["close_m_ema25"] = out["close"].to_numpy(dtype=float) - out["ema_25"].to_numpy(dtype=float)
    return out


def series_columns() -> list[str]:
    return list(config.SERIES)
