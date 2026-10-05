"""Shared synthetic fixtures for the candle-bounds (CB) kernel/stage tests.

``make_wide`` builds a real ``data._build_wide_df`` frame from a seeded 1-min
random walk and adds the CB input columns as deterministic, causal, smooth
functions of ``{tf}_close`` (TA-Lib is not needed for kernel tests).
``make_models`` returns six small random ``BoundModel``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.candle_bounds.models import BoundModel, ModelSet
from indicators.candle_bounds.params import ATR_COL, FEATURES, SIDES, TFS


def _raw_ohlc(n_minutes: int, seed: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n_minutes)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick_hi = np.abs(rng.normal(0.0, 0.0005, n_minutes))
    wick_lo = np.abs(rng.normal(0.0, 0.0005, n_minutes))
    high = np.maximum(open_, close) * (1.0 + wick_hi)
    low = np.minimum(open_, close) * (1.0 - wick_lo)
    volume = rng.uniform(100.0, 200.0, n_minutes)
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "taker_base_vol": volume * rng.uniform(0.3, 0.7, n_minutes),
        },
        index=idx,
    )


def make_wide(n_minutes: int = 6000, seed: int = 0,
              start: str = "2025-01-01 00:00") -> pd.DataFrame:
    from data import _build_wide_df

    raw = _raw_ohlc(n_minutes, seed, start)
    wide = _build_wide_df(raw)
    rng = np.random.default_rng(seed + 1)
    cols: dict[str, object] = {}
    for tf in TFS:
        close = wide[f"{tf}_close"]
        ema25 = close.ewm(span=25 * tf, adjust=False).mean()
        ema8 = close.ewm(span=8 * tf, adjust=False).mean()
        ema12 = close.ewm(span=12 * tf, adjust=False).mean()
        ema26 = close.ewm(span=26 * tf, adjust=False).mean()
        rsi14 = 50.0 + 40.0 * np.tanh((close - ema25) / ema25 * 100.0)
        macd = ema12 - ema26
        cols[f"{tf}_rsi_14"] = rsi14
        cols[f"{tf}_rsi_ma8"] = rsi14.ewm(span=8 * tf, adjust=False).mean()
        cols[f"{tf}_macd_12_26_9"] = macd
        cols[f"{tf}_macd_hist_12_26_9"] = macd - macd.ewm(span=9 * tf, adjust=False).mean()
        cols[f"{tf}_ema_25"] = ema25
        cols[f"{tf}_move_class"] = rng.integers(-3, 4, len(wide))
        cols[f"{tf}_ema_25_slope_class"] = np.sign(ema8 - ema25).astype(int)
    cols[ATR_COL] = (raw["high"] - raw["low"]).rolling(14, min_periods=1).mean() \
        .rolling(5, min_periods=1).mean() + 1e-6
    return pd.concat([wide, pd.DataFrame(cols, index=wide.index)], axis=1)


def make_models(seed: int = 0) -> ModelSet:
    rng = np.random.default_rng(seed)
    n = len(FEATURES)
    out: ModelSet = {}
    for tf in TFS:
        for side in SIDES:
            out[(tf, side)] = BoundModel(
                tf=tf, side=side,
                mean=np.zeros(n), scale=np.ones(n),
                coef=rng.normal(0.0, 0.01, n),
                intercept=float(rng.normal(0.0, 0.1)),
                band_pct=0.3, n_train=100,
            )
    return out
