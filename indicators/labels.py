# indicators/labels.py — lookahead profit labels for training/backtest targets.
#
# Labels look into the future by construction. They are NEVER features:
# this module must not be registered in indicators_config.yaml or the field
# registry, and must never be imported by the live path (pybtctr.py / Robot).
# It operates on the FULL wide DataFrame only — never on DataPoint.get_df()
# slices, which end at "now" and have no future rows.

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

# Cap on elements per materialized window chunk (~160 MB per float array):
# fancy-indexing the sliding window view copies chunk_rows × window cells.
_CHUNK_ELEMS = 20_000_000


def _fmt(v) -> str:
    """Number formatting for column names: ints verbatim, floats dot→p."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).replace(".", "p")


def _entry_state(
    wide_df: pd.DataFrame, tf: int, atr_period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-row positions, entry prices, and per-entry ATR for one tf."""
    closed = wide_df[f"{tf}_is_closed"].to_numpy(dtype=bool)
    pos = np.flatnonzero(closed)
    high = wide_df[f"{tf}_high"].to_numpy(dtype=float)
    low = wide_df[f"{tf}_low"].to_numpy(dtype=float)
    close = wide_df[f"{tf}_close"].to_numpy(dtype=float)
    atr = talib.ATR(high[pos], low[pos], close[pos], timeperiod=atr_period)
    return pos, close[pos], atr


def _forward_labels(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    atr_period: int, direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-entry forward labels.

    Returns (pos, entry, atr, labels) — all aligned to closed rows of tf.
    labels: 1.0 target touched strictly first, 0.0 otherwise, NaN when ATR
    is warming up or the future window is incomplete.
    """
    one_high = wide_df["1_high"].to_numpy(dtype=float)
    one_low = wide_df["1_low"].to_numpy(dtype=float)
    pos, entry, atr = _entry_state(wide_df, tf, atr_period)
    n_rows = len(wide_df)
    win = n * tf

    if direction == "long":
        target = entry + m * atr
        stop = entry - x * atr
    else:
        target = entry - m * atr
        stop = entry + x * atr

    labels = np.full(len(pos), np.nan)
    valid = np.flatnonzero(~np.isnan(atr) & (pos + win <= n_rows - 1))
    if len(valid) == 0:
        return pos, entry, atr, labels

    # sw_*[i] is the window of `win` 1-min rows starting at row i; the window
    # for entry at row p is rows p+1 .. p+win, i.e. sw_*[p+1].
    sw_high = np.lib.stride_tricks.sliding_window_view(one_high, win)
    sw_low = np.lib.stride_tricks.sliding_window_view(one_low, win)

    chunk_rows = max(1, _CHUNK_ELEMS // win)
    for start in range(0, len(valid), chunk_rows):
        idx = valid[start:start + chunk_rows]
        wh = sw_high[pos[idx] + 1]
        wl = sw_low[pos[idx] + 1]
        if direction == "long":
            target_hit = wh > target[idx, None]
            stop_hit = wl < stop[idx, None]
        else:
            target_hit = wl < target[idx, None]
            stop_hit = wh > stop[idx, None]
        # First-touch minute per entry; `win` = never touched. Strict `<`
        # encodes the pessimistic tie rule: same minute → stop wins → 0.
        t_first = np.where(target_hit.any(axis=1),
                           target_hit.argmax(axis=1), win)
        s_first = np.where(stop_hit.any(axis=1), stop_hit.argmax(axis=1), win)
        labels[idx] = (t_first < s_first).astype(float)

    return pos, entry, atr, labels


def _past_clean(
    wide_df: pd.DataFrame, tf: int, l: int, y: float, direction: str,
    pos: np.ndarray, entry: np.ndarray, atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward clean-entry condition over the last l 1-min rows (entry incl).

    Returns (clean, insufficient) per entry; insufficient marks entries with
    fewer than l rows of history (rolling min/max with min_periods=l is NaN).
    """
    if direction == "long":
        roll = (pd.Series(wide_df["1_low"].to_numpy(dtype=float))
                .rolling(l, min_periods=l).min().to_numpy())
        dirty = roll[pos] < entry - y * atr
    else:
        roll = (pd.Series(wide_df["1_high"].to_numpy(dtype=float))
                .rolling(l, min_periods=l).max().to_numpy())
        dirty = roll[pos] > entry + y * atr
    return ~dirty, np.isnan(roll[pos])


def _to_series(wide_df: pd.DataFrame, pos: np.ndarray,
               values: np.ndarray) -> pd.Series:
    out = np.full(len(wide_df), np.nan)
    out[pos] = values
    return pd.Series(out, index=wide_df.index)


def _profit(wide_df, tf, n, m, x, atr_period, direction) -> pd.Series:
    pos, _entry, _atr, labels = _forward_labels(
        wide_df, tf, n, m, x, atr_period, direction)
    return _to_series(wide_df, pos, labels)


def _profit_strict(wide_df, tf, n, m, x, l, y, atr_period,
                   direction) -> pd.Series:
    pos, entry, atr, labels = _forward_labels(
        wide_df, tf, n, m, x, atr_period, direction)
    clean, insufficient = _past_clean(
        wide_df, tf, l, y, direction, pos, entry, atr)
    ok = ~np.isnan(labels)
    labels[ok] = labels[ok] * clean[ok]
    labels[insufficient] = np.nan
    return _to_series(wide_df, pos, labels)


def profit_long(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    atr_period: int = 14,
) -> pd.Series:
    """1 where a long entry at a closed tf candle reaches entry + m*atr
    within the next n*tf minutes before touching entry - x*atr."""
    return _profit(wide_df, tf, n, m, x, atr_period, "long")


def profit_short(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    atr_period: int = 14,
) -> pd.Series:
    """Short mirror of profit_long: target below entry, stop above."""
    return _profit(wide_df, tf, n, m, x, atr_period, "short")


def add_profit_labels(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    atr_period: int = 14,
) -> None:
    """Append {tf}_plong_... and {tf}_pshort_... columns in place."""
    suffix = f"n{_fmt(n)}_m{_fmt(m)}_x{_fmt(x)}"
    wide_df[f"{tf}_plong_{suffix}"] = profit_long(
        wide_df, tf, n, m, x, atr_period)
    wide_df[f"{tf}_pshort_{suffix}"] = profit_short(
        wide_df, tf, n, m, x, atr_period)


def profit_strict_long(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    l: int, y: float, atr_period: int = 14,
) -> pd.Series:
    """profit_long AND no dip below entry - y*atr in the last l 1-min rows
    (entry row inclusive) — labels bottoms, not rising slopes."""
    return _profit_strict(wide_df, tf, n, m, x, l, y, atr_period, "long")


def profit_strict_short(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    l: int, y: float, atr_period: int = 14,
) -> pd.Series:
    """Short mirror of profit_strict_long: no spike above entry + y*atr."""
    return _profit_strict(wide_df, tf, n, m, x, l, y, atr_period, "short")


def add_profit_strict_labels(
    wide_df: pd.DataFrame, tf: int, n: int, m: float, x: float,
    l: int, y: float, atr_period: int = 14,
) -> None:
    """Append {tf}_pslong_... and {tf}_psshort_... columns in place."""
    suffix = f"n{_fmt(n)}_m{_fmt(m)}_x{_fmt(x)}_l{_fmt(l)}_y{_fmt(y)}"
    wide_df[f"{tf}_pslong_{suffix}"] = profit_strict_long(
        wide_df, tf, n, m, x, l, y, atr_period)
    wide_df[f"{tf}_psshort_{suffix}"] = profit_strict_short(
        wide_df, tf, n, m, x, l, y, atr_period)
