"""Zone-profitability experiment — per-move-class shifted entry/target bounds.

Spec: external/docs/superpowers/experiment/zone_profitability.md.

For each (tf, rsi move class, side) the closed-candle bound ({tf}_cb_low for
long entries, {tf}_cb_high for short) is shifted by a coefficient measured in
units of {tf}_atr_14_ma_5. A 1-min row is "in the open zone" when price
traded at or beyond the shifted level: long ``1_low <= cb_low + s·atr``,
short ``1_high >= cb_high + s·atr`` (a resting limit at the level fills that
minute). One-sided, so the zone count is monotone in the shift. Profitability
of the zone's rows is read from the existing lookahead profit labels
(indicators.labels; entry price = 1-min extreme, pessimistic fill), strict
and non-strict variants both. Best shift per cell: the zone is sized off the
class's own positive count — pick the shift whose zone row count n_zone is
closest to ZONE_SIZE_MULT × wins_class, where wins_class is the number of
label==1 rows the class has in total (argmin |n_zone − mult·wins_class|).
With the one-sided zone rule n_zone rises monotonically in s, so the match
is a single crossing, not two symmetric tails, and the level always stays on
its own side of the corridor. At mult=1 a perfectly concentrating shift
would make the zone all-positive; mult=2 (the current setting) targets a
zone twice that size — half the theoretical ceiling in winrate, but a wider,
more tradeable zone with more OOS rows per cell. Winrate/lift measure how
far from the ceiling it lands, instead of driving the choice (earlier
criteria — EV, wins·winrate, plain winrate — each degenerated toward empty
or floor-sized zones; grid rows keep n/wins/EV so any rule can be re-derived
from the report).

The target bound (opposite side, same shift mechanics) is then chosen by
empirical reach probability: from each open-zone row, does the 1-min price
reach the target level before the current tf candle closes (the bound is that
candle's expected extreme, so its natural validity horizon is the candle).
Best target shift maximizes p_reach / p_breakeven, where p_breakeven =
risk/(risk+reward), risk = x·atr (the label stop distance), reward =
|target − open level| — the target whose achieved reach probability comes
closest to (or exceeds, ratio > 1 = EV-positive) the probability its
risk/reward ratio requires (spec line 16). The ratio is used instead of the
margin p − p_be because when every candidate is below breakeven the margin's
argmax lands on far targets with p_reach ≈ 0 (low p_be, no reach) rather
than on genuine correspondence. Candidates with reward < 0.5·risk are
excluded: as reward → 0 both p_reach and p_breakeven → 1 and any criterion
degenerates to a zero-width target. Margin and per-point EV are recorded
per grid row for reference.

Move class here is the 7-class frozen-sym0 classification (classes −3..3,
MOVE_CLASS_CUTS on {tf}_rsi_ma8_diff — the move_class7 experiment winner that
is materialized as {tf}_move_class in oos2m). The 2y_az frame has neither
move_class nor rsi_ma8_diff; approx_move_class() rebuilds the diff as
rsi_ma8(row) − rsi_ma8(prev closed candle) which reproduces the stored oos2m
classes at 99.2% (the residual is numeric drift of the prev-candle reference
near cut boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TFS = (15, 60, 240)
SIDES = ("long", "short")
CLASSES = (-3, -2, -1, 0, 1, 2, 3)

# Frozen 2y sym0 cuts, single source: indicators.library.classification
# MOVE_CLASS_CUTS on the experimental_imp_2 line (0 ± {0.3,1,2}·diff_std).
MOVE_CLASS_CUTS = {
    15: (-2.728564, -1.364282, -0.409285, 0.409285, 1.364282, 2.728564),
    60: (-2.819219, -1.409609, -0.422883, 0.422883, 1.409609, 2.819219),
    240: (-2.919142, -1.459571, -0.437871, 0.437871, 1.459571, 2.919142),
}

SHIFT_GRID = tuple(np.round(np.arange(-2.0, 2.01, 0.1), 2))
MIN_POINTS = 30  # cells with fewer open-zone points keep shift 0 (no fit)
X_ATR = 2.0      # label stop distance in atr units (LabelParams.x) = the risk leg
ZONE_SIZE_MULT = 2.0  # zone target size = this × the class's label==1 count


def approx_move_class(df: pd.DataFrame, tf: int) -> np.ndarray:
    """7-class frozen-sym0 move class from wide rsi_ma8 (no stored diff col).

    diff(row) = rsi_ma8(row) − rsi_ma8 of the last candle closed strictly
    before the row; NaN → class 0, matching MoveClassField.
    """
    rsi = df[f"{tf}_rsi_ma8"]
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712
    prev = rsi.where(closed).ffill().shift(1)
    x = np.nan_to_num((rsi - prev).to_numpy(dtype=float), nan=0.0)
    return (np.digitize(x, MOVE_CLASS_CUTS[tf]) - 3).astype(np.int8)


def candle_id(df: pd.DataFrame, tf: int) -> np.ndarray:
    """Group id per 1-min row: rows share an id from the row after a close
    through the next closing row (the closing row belongs to its candle)."""
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    cid = np.concatenate(([0], np.cumsum(closed)[:-1]))
    return cid


def future_extreme_in_candle(df: pd.DataFrame, tf: int, side: str) -> np.ndarray:
    """Per row: max 1_high (side='long') / min 1_low ('short') from this row
    to the end of its tf candle, inclusive."""
    cid = candle_id(df, tf)
    if side == "long":
        s = pd.Series(df["1_high"].to_numpy(), index=np.arange(len(df)))
        rev = s[::-1].groupby(cid[::-1]).cummax()[::-1]
    else:
        s = pd.Series(df["1_low"].to_numpy(), index=np.arange(len(df)))
        rev = s[::-1].groupby(cid[::-1]).cummin()[::-1]
    return rev.to_numpy()


@dataclass
class Cell:
    """Search result for one (tf, move class, side)."""

    tf: int
    cls: int
    side: str
    open_shift: float
    n_points: int
    wins: int
    losses: int
    winrate: float
    base_winrate: float  # class-conditional winrate over ALL labeled rows
    lift: float          # winrate / base_winrate
    class_rows: int      # labeled rows of this class (zone-size denominator)
    class_wins: int      # label==1 rows of this class = the zone-size target
    open_shift_nonstrict: float
    winrate_nonstrict: float
    n_points_nonstrict: int
    tgt_shift: float
    p_reach: float
    rr: float
    p_breakeven: float
    margin: float  # p_reach − p_breakeven at the selected target shift
    open_grid: list  # [(shift, n, wins, losses, ev_strict), ...]
    tgt_grid: list   # [(shift, p_reach, rr, p_be, ev_per_point), ...]


def _open_level(df: pd.DataFrame, cb: pd.DataFrame, tf: int, side: str,
                shift: float) -> np.ndarray:
    base = cb[f"{tf}_cb_low"] if side == "long" else cb[f"{tf}_cb_high"]
    atr = df[f"{tf}_atr_14_ma_5"]
    return (base + shift * atr).to_numpy()


def _tgt_level(df: pd.DataFrame, cb: pd.DataFrame, tf: int, side: str,
               shift: float) -> np.ndarray:
    base = cb[f"{tf}_cb_high"] if side == "long" else cb[f"{tf}_cb_low"]
    atr = df[f"{tf}_atr_14_ma_5"]
    return (base + shift * atr).to_numpy()


def _inzone(df: pd.DataFrame, level: np.ndarray, side: str) -> np.ndarray:
    """Row is in the open zone when price traded at or beyond the level.

    long:  ``1_low <= level``   (the minute dipped to/through the level)
    short: ``1_high >= level``  (the minute spiked to/through it)

    One-sided, not "level inside the row's range": the count is then monotone
    in the shift (raising a long level can only add rows), so the size-match
    search below has a single crossing point instead of two symmetric tails —
    a long zone can no longer land above the corridor and still "match".
    """
    if side == "long":
        return df["1_low"].to_numpy() <= level
    return df["1_high"].to_numpy() >= level


def _reached(df: pd.DataFrame, level: np.ndarray, side: str) -> np.ndarray:
    """Row where the target level was traded through: long ``1_high >=``,
    short ``1_low <=`` — the mirror of ``_inzone``, on the profit side."""
    if side == "long":
        return df["1_high"].to_numpy() >= level
    return df["1_low"].to_numpy() <= level


def search_cells(
    df: pd.DataFrame, cb: pd.DataFrame, mc: dict[int, np.ndarray],
    labels: dict[tuple[int, str, bool], np.ndarray],
    rr_label: float = 1.0, x_atr: float = X_ATR,
) -> list[Cell]:
    """Grid search open + target shifts for every (tf, class, side) cell.

    ``labels[(tf, side, strict)]`` are float arrays (NaN = undefined tail).
    ``rr_label`` = m/x of the label params; ``x_atr`` = stop distance in atr
    units used as the risk leg of the target EV.
    """
    cells: list[Cell] = []
    for tf in TFS:
        atr = df[f"{tf}_atr_14_ma_5"].to_numpy()
        for side in SIDES:
            fut = future_extreme_in_candle(df, tf, side)
            lab_s = labels[(tf, side, True)]
            lab_n = labels[(tf, side, False)]
            # class-conditional base rates over all labeled rows, per label
            # variant: selection target (zone fraction ≈ base rate) + lift denom
            base = {}
            for strict, lab in ((True, lab_s), (False, lab_n)):
                ok_all = ~np.isnan(lab)
                b_wins = np.bincount(mc[tf][ok_all] + 3,
                                     weights=lab[ok_all], minlength=7)
                b_tot = np.bincount(mc[tf][ok_all] + 3, minlength=7)
                base[strict] = (np.divide(b_wins, b_tot, out=np.zeros(7),
                                          where=b_tot > 0), b_tot, b_wins)
            base_rate = base[True][0]
            class_wins = base[True][2]
            # per-shift stats for all classes at once via bincount over mc+3
            open_stats: dict[float, dict] = {}
            for s in SHIFT_GRID:
                level = _open_level(df, cb, tf, side, s)
                inz = _inzone(df, level, side) & ~np.isnan(level)
                st = {}
                for strict, lab in ((True, lab_s), (False, lab_n)):
                    ok = inz & ~np.isnan(lab)
                    cls_idx = mc[tf][ok] + 3
                    wins = np.bincount(cls_idx, weights=lab[ok], minlength=7)
                    tot = np.bincount(cls_idx, minlength=7)
                    st[strict] = (wins, tot - wins)
                open_stats[s] = st
            for cls in CLASSES:
                k = cls + 3
                best: dict[bool, tuple] = {}
                grid_rows = []
                for s in SHIFT_GRID:
                    w_s, l_s = (open_stats[s][True][0][k],
                                open_stats[s][True][1][k])
                    w_n, l_n = (open_stats[s][False][0][k],
                                open_stats[s][False][1][k])
                    ev_s = w_s * rr_label - l_s
                    grid_rows.append((float(s), int(w_s + l_s), int(w_s),
                                      int(l_s), float(ev_s)))
                    for strict, w, l in ((True, w_s, l_s), (False, w_n, l_n)):
                        n = w + l
                        if n < MIN_POINTS:
                            continue
                        _, tot, cls_wins = base[strict]
                        if tot[k] == 0:
                            continue
                        # zone sized to ZONE_SIZE_MULT × the class's positive
                        # count, see module doc
                        crit = abs(n - ZONE_SIZE_MULT * cls_wins[k])
                        if strict not in best or crit < best[strict][0]:
                            best[strict] = (crit, float(s), int(w), int(l))
                if True not in best:
                    best[True] = (np.inf, 0.0, 0, 0)
                if False not in best:
                    best[False] = (np.inf, 0.0, 0, 0)
                _, s_star, w, l = best[True]
                _, s_star_n, w_n_, l_n_ = best[False]

                # target search on the strict-best open zone
                level = _open_level(df, cb, tf, side, s_star)
                inz = _inzone(df, level, side) & ~np.isnan(level)
                pts = inz & (mc[tf] == cls)
                tgt_grid = []
                t_best = (-np.inf, 0.0, 0.0, 0.0, 0.0)
                n_pts = int(pts.sum())
                if n_pts >= MIN_POINTS:
                    lev_p = level[pts]
                    atr_p = atr[pts]
                    fut_p = fut[pts]
                    for t in SHIFT_GRID:
                        tlev = _tgt_level(df, cb, tf, side, t)[pts]
                        if side == "long":
                            reward = tlev - lev_p
                            reach = fut_p >= tlev
                        else:
                            reward = lev_p - tlev
                            reach = fut_p <= tlev
                        risk_all = x_atr * atr_p
                        valid = reward >= 0.5 * risk_all
                        if valid.sum() < MIN_POINTS:
                            continue
                        p = float(reach[valid].mean())
                        risk = risk_all[valid]
                        rew = reward[valid]
                        ev = float(p * rew.mean() - (1 - p) * risk.mean())
                        rr = float(rew.mean() / risk.mean())
                        p_be = float(risk.mean() / (risk.mean() + rew.mean()))
                        crit = p / p_be if p_be > 0 else 0.0
                        tgt_grid.append((float(t), round(p, 4), round(rr, 3),
                                         round(p_be, 4), round(ev, 6)))
                        if crit > t_best[0]:
                            t_best = (crit, float(t), p, rr, p_be)
                crit, t_star, p_reach, rr, p_be = t_best
                if not np.isfinite(crit):
                    t_star, p_reach, rr, p_be = 0.0, 0.0, 0.0, 0.0
                marg = p_reach - p_be
                br = float(base_rate[k])
                wr = w / (w + l) if w + l else 0.0
                cells.append(Cell(
                    tf=tf, cls=cls, side=side,
                    open_shift=s_star, n_points=w + l, wins=w, losses=l,
                    winrate=round(wr, 4),
                    base_winrate=round(br, 4),
                    lift=round(wr / br, 3) if br else 0.0,
                    class_rows=int(base[True][1][k]),
                    class_wins=int(class_wins[k]),
                    open_shift_nonstrict=s_star_n,
                    winrate_nonstrict=(round(w_n_ / (w_n_ + l_n_), 4)
                                       if w_n_ + l_n_ else 0.0),
                    n_points_nonstrict=w_n_ + l_n_,
                    tgt_shift=t_star, p_reach=round(p_reach, 4),
                    rr=round(rr, 3), p_breakeven=round(p_be, 4),
                    margin=round(marg, 4),
                    open_grid=grid_rows, tgt_grid=tgt_grid,
                ))
    return cells


def mark_zones(df: pd.DataFrame, cb: pd.DataFrame, mc: dict[int, np.ndarray],
               params: dict) -> pd.DataFrame:
    """Marker frame: {tf}_zp_open/tgt_{side} levels + {tf}_zp_inzone/intgt_{side}.

    ``params[(tf, cls, side)]`` = (open_shift, tgt_shift). Levels are the
    per-row shifted bounds selected by the row's own move class.

    ``inzone`` uses the one-sided entry rule (``_inzone``) the search
    optimized. ``intgt`` instead marks rows whose 1-min range *contains* the
    target level: one-sided would be true for ~96% of rows (any row trading
    above a target that sits below price), which flags nothing. A row is also
    excluded from ``intgt``, and its target level blanked, when the target is
    not at least 0.5·(x·atr) away from the entry level on the profit side —
    the same validity filter the target search applied, so the drawn line
    never shows an inverted or degenerate span.
    """
    out = pd.DataFrame(index=df.index)
    for tf in TFS:
        atr = df[f"{tf}_atr_14_ma_5"].to_numpy()
        for side in SIDES:
            base_o = (cb[f"{tf}_cb_low"] if side == "long"
                      else cb[f"{tf}_cb_high"]).to_numpy()
            base_t = (cb[f"{tf}_cb_high"] if side == "long"
                      else cb[f"{tf}_cb_low"]).to_numpy()
            s_row = np.zeros(len(df))
            t_row = np.zeros(len(df))
            for cls in CLASSES:
                s, t = params[(tf, cls, side)]
                sel = mc[tf] == cls
                s_row[sel] = s
                t_row[sel] = t
            open_lev = base_o + s_row * atr
            tgt_lev = base_t + t_row * atr
            reward = (tgt_lev - open_lev if side == "long"
                      else open_lev - tgt_lev)
            valid = reward >= 0.5 * X_ATR * atr
            tgt_lev = np.where(valid, tgt_lev, np.nan)
            lo, hi = df["1_low"].to_numpy(), df["1_high"].to_numpy()
            out[f"{tf}_zp_open_{side}"] = open_lev
            out[f"{tf}_zp_tgt_{side}"] = tgt_lev
            out[f"{tf}_zp_inzone_{side}"] = _inzone(df, open_lev, side)
            out[f"{tf}_zp_intgt_{side}"] = (
                (lo <= tgt_lev) & (tgt_lev <= hi) & valid)
    return out
