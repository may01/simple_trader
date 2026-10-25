"""Market-state classes at a candle's close — closed (own tf) and forming (higher tf).

All functions are pure numpy over closed-row sequences. Frozen parameters come from
``indicators.library.classification`` (move cuts, EMA-25 slope stats) and from the copied
swing zig-zag module (cci_ma8 prominence σ, apex spacing CONST, confirm 3, window 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.library.classification import EMA_SLOPE_STATS, EMA_SLOPE_X, MOVE_CLASS_CUTS, _five_tiers

from .swing_causal_zones import CONST, STD, causal_apexes, cci_zone

CONFIRM, WINDOW = 3, 3
CCI_MA = 8


# ------------------------------------------------------------------ RSI move class (mc)
def move_class(rsi_ma8: np.ndarray, tf: int) -> np.ndarray:
    """7-class sym0 move class of the closed rsi_ma8 sequence (diff vs the previous closed)."""
    d = np.diff(np.asarray(rsi_ma8, dtype=float), prepend=np.nan)
    return move_class_from_diff(d, tf)


def move_class_from_diff(d: np.ndarray, tf: int) -> np.ndarray:
    return (np.digitize(np.nan_to_num(np.asarray(d, dtype=float), nan=0.0), MOVE_CLASS_CUTS[tf]) - 3).astype(int)


# ------------------------------------------------------------------ RSI zone class (zc)
def zone_stats(rsi_ma8_closed: np.ndarray) -> tuple[float, float]:
    x = np.asarray(rsi_ma8_closed, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()), float(x.std(ddof=1))


def zone_class(rsi_ma8: np.ndarray, mean: float, std: float) -> np.ndarray:
    return _five_tiers(pd.Series(np.asarray(rsi_ma8, dtype=float)), mean, std, 0).astype(int)


# ------------------------------------------------------------------ EMA-25 slope class (ec)
def ema_slope_class_from(ema_now: np.ndarray, ema_prev: np.ndarray, tf: int) -> np.ndarray:
    """-1 / 0 / 1 from d = (ema_now - ema_prev) / ema_prev * 100 (repo ema_25_diff_prc convention)."""
    now = np.asarray(ema_now, dtype=float)
    prev = np.asarray(ema_prev, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (now - prev) / prev * 100.0
    mean, std = EMA_SLOPE_STATS[tf]
    cl = np.where(d < mean - EMA_SLOPE_X * std, -1, np.where(d > mean + EMA_SLOPE_X * std, 1, 0)).astype(int)
    cl[~np.isfinite(d)] = 0
    return cl


def ema_slope_class_closed(ema_closed: np.ndarray, tf: int) -> np.ndarray:
    e = np.asarray(ema_closed, dtype=float)
    prev = np.concatenate(([np.nan], e[:-1]))
    return ema_slope_class_from(e, prev, tf)


# ------------------------------------------------------------------ CCI swing zone (cz)
def cci_ma8_closed(cci_closed: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(cci_closed, dtype=float)).rolling(CCI_MA).mean().to_numpy()


def cci_zone_closed(y: np.ndarray, tf: int) -> np.ndarray:
    z, _ = cci_zone(y, 0.5 * STD[tf], CONST[tf], CONFIRM, WINDOW)
    return z.astype(int)


class ZigZag:
    """Incremental twin of ``swing_causal_zones.causal_apexes``.

    ``feed(j, v)`` processes index j (value v) and returns the apex confirmed at j, if any,
    as (apex_row, sign, j); ``peek(j, v)`` returns what ``feed`` would return without changing
    the state. NaN values are skipped (as the batch version does).
    """

    def __init__(self, prom: float):
        self.prom = prom
        self.y: list[float] = []
        self.hi_i = self.lo_i = -1
        self.s_last = 0
        self.started = False

    def _check(self, j: int, v: float, hi_i: int, lo_i: int):
        """Apex confirmed at (j, v) given running extremes hi_i / lo_i (already updated with v)?"""
        if self.s_last <= 0 and hi_i < j and self.y[hi_i] - v >= self.prom:
            return (hi_i, 1, j)
        if self.s_last >= 0 and lo_i < j and v - self.y[lo_i] >= self.prom:
            return (lo_i, -1, j)
        return None

    def _extremes_with(self, j: int, v: float) -> tuple[int, int]:
        hi_i, lo_i = self.hi_i, self.lo_i
        if v > self.y[hi_i]:
            hi_i = j
        if v < self.y[lo_i]:
            lo_i = j
        return hi_i, lo_i

    def peek(self, j: int, v: float):
        if not self.started or not np.isfinite(v):
            return None
        hi_i, lo_i = self._extremes_with(j, v)
        return self._check(j, v, hi_i, lo_i)

    def feed(self, j: int, v: float):
        assert j == len(self.y), "feed indices must be consecutive"
        self.y.append(float(v))
        if not np.isfinite(v):
            return None
        if not self.started:
            self.started = True
            self.hi_i = self.lo_i = j
            return None
        self.hi_i, self.lo_i = self._extremes_with(j, v)
        apex = self._check(j, v, self.hi_i, self.lo_i)
        if apex is None:
            return None
        i, s, _ = apex
        y = np.asarray(self.y, dtype=float)
        seg = y[i + 1: j + 1]
        if s > 0:
            self.s_last = 1
            self.lo_i = i + 1 + int(np.nanargmin(seg))
        else:
            self.s_last = -1
            self.hi_i = i + 1 + int(np.nanargmax(seg))
        return apex


def zone_from_apexes(apexes, n: int, const: float) -> np.ndarray:
    """``cci_zone`` window semantics over indices 0..n-1 (n may exceed the series length)."""
    z = np.zeros(n, dtype=int)
    for i, s, j in apexes:
        a = max(i + CONFIRM, j)
        b = int(min(round(i + const + WINDOW), n - 1))
        if a > b:
            continue
        z[a:b + 1] = -1 if s > 0 else 1
    return z


def cci_zone_forming(y_closed: np.ndarray, last_closed: np.ndarray, y_forming: np.ndarray, tf: int) -> np.ndarray:
    """Zone value for rows observing a forming higher-tf candle.

    ``y_closed``: closed cci_ma8 sequence of the higher tf. For row r, ``last_closed[r]`` = index of
    the last higher candle closed strictly before the row (-1 = none) and ``y_forming[r]`` = the
    forming cci_ma8 (SMA of the last 7 closed cci + the forming cci). The forming candle plays index
    ``last_closed + 1``: an apex it would confirm is active from that index (if its window has
    started), otherwise the existing apexes' window value at that index applies.
    """
    y = np.asarray(y_closed, dtype=float)
    n = len(y)
    prom, const = 0.5 * STD[tf], CONST[tf]
    zz = ZigZag(prom)
    apexes = []
    # state snapshot after feeding index k, for every k (shallow: counts + extremes)
    snap_apex_count = np.zeros(n, dtype=int)
    snaps = []
    for k in range(n):
        a = zz.feed(k, y[k])
        if a is not None:
            apexes.append(a)
        snap_apex_count[k] = len(apexes)
        snaps.append((zz.hi_i, zz.lo_i, zz.s_last, zz.started))
    assert apexes == causal_apexes(y, prom), "incremental zig-zag drifted from causal_apexes"
    z_ext = zone_from_apexes(apexes, n + 1, const)

    out = np.zeros(len(last_closed), dtype=int)
    lc = np.asarray(last_closed, dtype=int)
    yf = np.asarray(y_forming, dtype=float)
    for r in range(len(lc)):
        k = lc[r]
        if k < 0 or not np.isfinite(yf[r]):
            continue
        j = k + 1
        hi_i, lo_i, s_last, started = snaps[k]
        val = 0
        if started:
            # peek with the snapshot state
            zz.hi_i, zz.lo_i, zz.s_last, zz.started = hi_i, lo_i, s_last, started
            apex = zz.peek(j, yf[r])
            if apex is not None:
                i, s, _ = apex
                a = max(i + CONFIRM, j)
                b = int(round(i + const + WINDOW))
                if a <= j <= b:
                    val = -1 if s > 0 else 1
                    out[r] = val
                    continue
        # existing apexes confirmed up to k, their window at j
        zj = zone_from_apexes(apexes[:snap_apex_count[k]], j + 1, const)[j] if snap_apex_count[k] else 0
        out[r] = zj
    return out
