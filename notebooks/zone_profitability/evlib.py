"""EV-line experiment — probability-split entry level between the two bands.

Given the closed-candle bound predictions ``{tf}_cb_low`` / ``{tf}_cb_high``
and their frozen ±1σ bands (``{tf}_cb_{side}_std``), treat each band as a
Gaussian over where that candle's extreme lands and pick, per candle and
side, the entry level whose win odds cover the risk/reward the trade needs.

Long (short is the mirror):

    S = L - stop_sigma * σ_L            stop:   beyond the low band
    T = H - tgt_sigma  * σ_H            target: inside the high band
    E in (S, T)                         candidate entry (resting limit)

    p_fill   = Φ((E - L) / σ_L)                 low gets down to E
    p_stop   = Φ((S - L) / σ_L) / p_fill        given filled, low also reaches S
    p_target = 1 - Φ((T - H) / σ_H)             high reaches T
    p_win    = p_target * (1 - p_stop)          independence, no ordering

    EV(E)    = p_win * (T - E) - (1 - p_win) * (E - S) - 2 * fee * E
    EV/candle = p_fill * EV(E)                  value of placing the order

``best``      = argmax_E p_fill * EV(E)   (the drawn line, ``{tf}_ev_{side}``)
``breakeven`` = outermost E with EV(E) >= 0 (``{tf}_ev_{side}_be``): beyond
it, toward the target, every entry has negative EV.

The model side is a prior only: it assumes Gaussian bands (±1σ coverage
on oos2m is 0.66-0.78, fat-tailed) and ignores which of stop/target is hit
first. ``realized`` replays the rule minute by minute inside each candle
(pessimistic: a fill minute whose low also crosses the stop is a stop;
the target counts only from the next minute) and reports what actually
happened, next to the model's p_fill/p_win/EV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TFS = (15, 60, 240)
SIDES = ("long", "short")

_erf = np.vectorize(math.erf, otypes=[float])
_SQRT2 = math.sqrt(2.0)


def Phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, elementwise (no scipy in the image)."""
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / _SQRT2))


@dataclass(frozen=True)
class EVParams:
    stop_sigma: float = 1.0   # stop = L - stop_sigma*σ_L (long)
    tgt_sigma: float = 0.5    # target = H - tgt_sigma*σ_H (long)
    fee: float = 0.001        # per side, fraction of price (round trip = 2*fee)
    grid: int = 201           # entry candidates between S and T


def ev_line(
    L: np.ndarray, sL: np.ndarray, H: np.ndarray, sH: np.ndarray,
    side: str, p: EVParams = EVParams(),
) -> dict[str, np.ndarray]:
    """Per-candle EV-optimal entry. Inputs are (n,) arrays of the predicted
    low/high and their band half-widths in price units. Returns (n,) arrays:
    ``S``, ``T``, ``best``, ``breakeven`` plus ``p_fill``/``p_win``/``ev``/
    ``ev_candle`` at ``best``. Rows with a NaN input or an inverted span
    (S on the wrong side of T) come back NaN.
    """
    L, sL, H, sH = (np.asarray(a, dtype=float) for a in (L, sL, H, sH))
    if side == "long":
        S = L - p.stop_sigma * sL
        T = H - p.tgt_sigma * sH
    else:
        S = H + p.stop_sigma * sH
        T = L + p.tgt_sigma * sL
    u = np.linspace(0.0, 1.0, p.grid)[1:-1]            # strictly inside (S, T)
    E = S[:, None] + u[None, :] * (T - S)[:, None]      # (n, g)

    if side == "long":
        p_fill = Phi((E - L[:, None]) / sL[:, None])
        p_stop = Phi((S - L) / sL)[:, None] / p_fill
        p_target = (1.0 - Phi((T - H) / sH))[:, None]
        reward, risk = T[:, None] - E, E - S[:, None]
    else:
        p_fill = 1.0 - Phi((E - H[:, None]) / sH[:, None])
        p_stop = (1.0 - Phi((S - H) / sH))[:, None] / p_fill
        p_target = Phi((T - L) / sL)[:, None]
        reward, risk = E - T[:, None], S[:, None] - E
    p_win = p_target * (1.0 - p_stop)
    ev = p_win * reward - (1.0 - p_win) * risk - 2.0 * p.fee * E
    ev_candle = p_fill * ev

    valid = np.isfinite(ev_candle).all(axis=1) & (
        (T > S) if side == "long" else (T < S)
    )
    out = {k: np.full(len(L), np.nan) for k in
           ("best", "breakeven", "p_fill", "p_win", "ev", "ev_candle")}
    out["S"], out["T"] = S, T
    if not valid.any():
        return out
    evc = np.where(valid[:, None], ev_candle, -np.inf)
    j = evc.argmax(axis=1)
    rows = np.arange(len(L))
    out["best"][valid] = E[rows, j][valid]
    out["p_fill"][valid] = p_fill[rows, j][valid]
    out["p_win"][valid] = p_win[rows, j][valid]
    out["ev"][valid] = ev[rows, j][valid]
    out["ev_candle"][valid] = ev_candle[rows, j][valid]
    # breakeven: last grid point (walking S -> T) with EV >= 0
    ok = (ev >= 0.0) & valid[:, None]
    has = ok.any(axis=1)
    jb = p.grid - 3 - np.argmax(ok[:, ::-1], axis=1)
    out["breakeven"][has] = E[rows, jb][has]
    return out


def candle_id(df: pd.DataFrame, tf: int) -> np.ndarray:
    """Group id per 1-min row (zplib convention): rows after a close through
    the next closing row share an id."""
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    return np.concatenate(([0], np.cumsum(closed)[:-1]))


def replay(
    lo: np.ndarray, hi: np.ndarray, cl: np.ndarray, starts: np.ndarray,
    E: np.ndarray, S: np.ndarray, T: np.ndarray, side: str, fee: float,
) -> dict[str, np.ndarray]:
    """Vectorized per-candle replay. ``S``/``T`` are per CANDLE (len =
    len(starts)); ``E`` is per candle, or per 1-min ROW (len = len(lo)) for a
    level that moves within the candle — the fill price is then the row's
    level at the fill minute. ``lo``/``hi``/``cl`` per row. Same semantics as the
    loop in ``realized``: first fill minute, stop from that minute inclusive,
    target from the next minute, otherwise exit at the candle's last close.
    Returns per-candle ``valid``, ``filled``, ``win``, ``stop``, ``open`` and
    ``pnl`` (price units, net of round-trip fee; NaN when not filled).
    """
    n_rows = len(lo)
    rep = np.diff(np.r_[starts, n_rows])
    idx = np.arange(n_rows)
    big = n_rows + 1
    per_row = len(E) == n_rows
    Er = E if per_row else np.repeat(E, rep)
    Sr, Tr = (np.repeat(x, rep) for x in (S, T))
    if side == "long":
        c_fill, c_stop, c_tgt = lo <= Er, lo <= Sr, hi >= Tr
    else:
        c_fill, c_stop, c_tgt = hi >= Er, hi >= Sr, lo <= Tr
    first = lambda cond: np.minimum.reduceat(np.where(cond, idx, big), starts)  # noqa: E731
    i_fill = first(c_fill)
    i_stop = first(c_stop)              # S beyond E on the fill side -> >= i_fill
    i_tgt = first(c_tgt & (idx > np.repeat(i_fill, rep)))
    if per_row:
        E = Er[np.minimum(i_fill, n_rows - 1)]      # level at the fill minute
        E = np.where(i_fill < big, E, Er[starts])
    valid = np.isfinite(E) & np.isfinite(S) & np.isfinite(T)
    filled = valid & (i_fill < big)
    win = filled & (i_tgt < i_stop)
    stop = filled & ~win & (i_stop < big)
    opn = filled & ~win & ~stop
    last_close = cl[np.r_[starts[1:], n_rows] - 1]
    exit_px = np.where(win, T, np.where(stop, S, last_close))
    g = (exit_px - E) if side == "long" else (E - exit_px)
    pnl = np.where(filled, g - 2.0 * fee * E, np.nan)
    return {"valid": valid, "filled": filled, "win": win, "stop": stop,
            "open": opn, "pnl": pnl, "E": E}


def summarize(r: dict[str, np.ndarray], E: np.ndarray) -> dict[str, float]:
    """Rates + mean pnl from a ``replay`` result (same keys as ``realized``)."""
    n = int(r["valid"].sum()); f = int(r["filled"].sum())
    pct = 100.0 * r["pnl"] / E
    mean_pct = float(np.nanmean(pct)) if f else float("nan")
    return {
        "n_candles": n,
        "fill_rate": f / n if n else float("nan"),
        "win_rate": r["win"].sum() / f if f else float("nan"),
        "stop_rate": r["stop"].sum() / f if f else float("nan"),
        "open_at_close_rate": r["open"].sum() / f if f else float("nan"),
        "pnl_per_fill": float(np.nanmean(r["pnl"])) if f else float("nan"),
        "pnl_pct_per_fill": mean_pct,
        "pnl_pct_per_candle": mean_pct * f / n if n else float("nan"),
        "sum_pnl_pct": float(np.nansum(pct)),
    }


def realized(
    df: pd.DataFrame, cid: np.ndarray, E: np.ndarray, S: np.ndarray,
    T: np.ndarray, side: str, fee: float,
) -> dict[str, float]:
    """Replay a (E, S, T) rule inside each candle. ``E``/``S``/``T`` are per
    1-min row (constant within a candle). Returns fill/win/stop/open-at-close
    rates over candles with a valid rule, and the mean pnl per filled trade
    (price units, and % of entry) net of round-trip fee.
    """
    lo = df["1_low"].to_numpy(); hi = df["1_high"].to_numpy()
    cl = df["1_close"].to_numpy()
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    ends = np.r_[starts[1:], len(cid)]
    n = fills = wins = stops = opens = 0
    pnl = []
    pnl_pct = []
    for a, b in zip(starts, ends):
        e, s, t = E[a], S[a], T[a]
        if not (np.isfinite(e) and np.isfinite(s) and np.isfinite(t)):
            continue
        n += 1
        l_, h_ = lo[a:b], hi[a:b]
        if side == "long":
            fill = np.flatnonzero(l_ <= e)
        else:
            fill = np.flatnonzero(h_ >= e)
        if len(fill) == 0:
            continue
        fills += 1
        f = fill[0]
        if side == "long":
            stop_hits = np.flatnonzero(l_[f:] <= s)
            tgt_hits = np.flatnonzero(h_[f + 1:] >= t) + 1
        else:
            stop_hits = np.flatnonzero(h_[f:] >= s)
            tgt_hits = np.flatnonzero(l_[f + 1:] <= t) + 1
        i_stop = stop_hits[0] if len(stop_hits) else np.inf
        i_tgt = tgt_hits[0] if len(tgt_hits) else np.inf
        if i_tgt < i_stop:
            wins += 1
            exit_px = t
        elif np.isfinite(i_stop):
            stops += 1
            exit_px = s
        else:
            opens += 1
            exit_px = cl[b - 1]
        g = (exit_px - e) if side == "long" else (e - exit_px)
        g -= 2.0 * fee * e
        pnl.append(g)
        pnl_pct.append(100.0 * g / e)
    pnl = np.asarray(pnl)
    pnl_pct = np.asarray(pnl_pct)
    mean_pnl = float(pnl.mean()) if len(pnl) else float("nan")
    mean_pct = float(pnl_pct.mean()) if len(pnl) else float("nan")
    return {
        "n_candles": n,
        "fill_rate": fills / n if n else float("nan"),
        "win_rate": wins / fills if fills else float("nan"),
        "stop_rate": stops / fills if fills else float("nan"),
        "open_at_close_rate": opens / fills if fills else float("nan"),
        "pnl_per_fill": mean_pnl,
        "pnl_pct_per_fill": mean_pct,
        "pnl_pct_per_candle": mean_pct * fills / n if n else float("nan"),
        "sum_pnl_pct": float(pnl_pct.sum()) if len(pnl) else 0.0,
    }
