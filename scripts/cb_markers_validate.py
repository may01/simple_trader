#!/usr/bin/env python3
"""Gate report: main-flow CB columns of a materialized/prepared dataset against
the stored experiment sidecars next to it (phase 18, gates G2-G5; phase 19, G4-o).

Reads ``df_with_indicators.pkl`` (with the CB columns) and the never-modified
sidecars ``df_with_candle_bounds.pkl``, ``df_with_candle_bounds_nc.pkl``,
``df_with_ext_done.pkl``. Prints one line per check plus a P(hold) table and a
marker-count table. Exit code 0 only if every gate passes.

    TRAIN_ENV=configs/oos2m_dataset.env docker compose -p cbtest run --rm ohlc_gen \\
        python3 scripts/cb_markers_validate.py

Known edge deltas (``notebooks/candle_bounds_nc/run_cbnc.py``): the stored
artifacts have one extra warm-up candle at the series start (stored NaN where
ours is valid) and a stale bound ffilled into the final row. Values are compared
on rows where both sides are non-NaN, final row excluded; every row where only
one side is valid, and every bool mismatch off that mask, must lie in the
leading warm-up candle of the tf or be the final row, else the gate fails.

Hybrid columns (``cbnc_tgt/zone/inzone``, ``cbx_inzone``): when the nc
sidecar's ``candle_bounds_nc_meta.json`` records a ``tgt_shrink_pct`` other than
``TGT_SHRINK_PCT`` the sidecar is a stale reference — the gate becomes kernel
parity at the sidecar's value and a WARN line (not a failure) states that the
materialized hybrid columns are not comparable to it. G5c checks the
materialized extdone columns against the kernel with the production ladder.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import wide_df_path  # noqa: E402
from indicators.candle_bounds import kernel  # noqa: E402
from indicators.candle_bounds.params import (  # noqa: E402
    DIRECTIONS, EXTDONE_LADDER, OVERSHOOT_THR, SIDES, TFS, TGT_SHRINK_PCT,
)

REL_TOL = 1e-9
# width of the bracket the oos2m sidecar's rows put around each frozen overshoot threshold (spec D1)
OVERSHOOT_BRACKET = 5e-4
PHOLD_TOL = 0.01
# oos2m-picked ladder of the stored sidecar (notebooks/zone_profitability/run_extdone.py)
LADDER_OOS = {15: ((0.25, 4.0), (0.50, 2.5), (0.75, 1.0)),
              60: ((0.25, 8.0), (0.50, 5.0), (0.75, 3.0)),
              240: ((0.50, 12.0), (0.75, 8.0))}
# oos2m P(hold | first trigger) of the 2y ladder (extdone_zone_filter_results.md §1.3)
PHOLD_REF = {(15, "high"): 0.938, (15, "low"): 0.936, (60, "high"): 0.951,
             (60, "low"): 0.952, (240, "high"): 0.931, (240, "low"): 0.955}


class Report:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, gate: str, name: str, ok: bool, detail: str) -> None:
        print(f"{gate} {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if not ok:
            self.failed.append(f"{gate} {name}")


def _not_last(n: int) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    if n:
        m[-1] = False
    return m


def _rows(idx: np.ndarray, index: pd.Index, k: int = 3) -> str:
    """Count plus the first few timestamps of offending rows."""
    if len(idx) == 0:
        return "0"
    return f"{len(idx)} (first: {[str(t) for t in index[idx[:k]]]})"


def warmup_allowed(df: pd.DataFrame, tf: int) -> np.ndarray:
    """Rows where an edge delta vs the stored artifacts is accepted: the leading
    warm-up candle of ``tf`` (the candle holding our first valid closed bound —
    the stored artifacts start one candle later) and the final row."""
    n = len(df)
    allowed = np.zeros(n, dtype=bool)
    if n == 0:
        return allowed
    allowed[-1] = True
    valid = _valid(df[f"{tf}_cb_high"]) | _valid(df[f"{tf}_cb_low"])
    if valid.any():
        cid = candle_ids(df, tf)
        allowed |= cid == cid[int(np.flatnonzero(valid)[0])]
    return allowed


def _float_cmp(rep: Report, gate: str, col: str, ours: pd.Series, stored: pd.Series,
               allowed: np.ndarray, mask: np.ndarray | None = None) -> None:
    """max rel on ``mask`` (default: both non-NaN, final row excluded); rows where
    exactly one side is NaN must all lie in ``allowed``."""
    a, b = ours.to_numpy(dtype=float), stored.to_numpy(dtype=float)
    na, nb = np.isnan(a), np.isnan(b)
    both = ~(na | nb) & _not_last(len(a)) if mask is None else mask & ~(na | nb)
    rel = np.abs(a[both] - b[both]) / np.abs(b[both])
    mx = float(rel.max()) if both.any() else float("nan")
    rep.check(gate, f"{col} max_rel", bool(both.any()) and mx < REL_TOL,
              f"{mx:.2e} (n={int(both.sum())})")
    lost = np.flatnonzero(na & ~nb)
    extra = np.flatnonzero(~na & nb)
    lost_bad = lost[~allowed[lost]]
    extra_bad = extra[~allowed[extra]]
    rep.check(gate, f"{col} stored-valid/ours-NaN outside warm-up candle+final row",
              len(lost_bad) == 0, f"{_rows(lost_bad, ours.index)} (total {len(lost)})")
    rep.check(gate, f"{col} ours-valid/stored-NaN outside warm-up candle+final row",
              len(extra_bad) == 0, f"{_rows(extra_bad, ours.index)} (total {len(extra)})")


def _bool_cmp(rep: Report, gate: str, col: str, ours: pd.Series, stored: pd.Series,
              mask: np.ndarray, allowed: np.ndarray) -> None:
    """0 mismatches on ``mask``; mismatches off the mask only in ``allowed``."""
    a = ours.fillna(False).to_numpy(dtype=bool)
    b = stored.fillna(False).to_numpy(dtype=bool)
    diff = a != b
    mism = int((diff & mask).sum())
    off = np.flatnonzero(diff & ~mask)
    off_bad = off[~allowed[off]]
    rep.check(gate, f"{col} mismatches", mism == 0,
              f"{mism} (n={int(mask.sum())}, true ours/stored={int(a[mask].sum())}/{int(b[mask].sum())})")
    rep.check(gate, f"{col} off-mask mismatches outside warm-up candle+final row",
              len(off_bad) == 0, f"{_rows(off_bad, ours.index)} (off-mask total {len(off)})")


def _valid(*series: pd.Series) -> np.ndarray:
    m = np.ones(len(series[0]), dtype=bool)
    for s in series:
        m &= ~np.isnan(s.to_numpy(dtype=float))
    return m


def candle_ids(df: pd.DataFrame, tf: int) -> np.ndarray:
    """Rows after a ``{tf}_is_closed`` row through the next closing row share an id
    (the delimitation of ``kernel.candle_progress(by_rows=True)``)."""
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    return np.concatenate(([0], np.cumsum(closed)[:-1]))


def _g2_mask(df: pd.DataFrame, cb: pd.DataFrame, tf: int) -> np.ndarray:
    return _valid(df[f"{tf}_cb_high"], df[f"{tf}_cb_low"],
                  cb[f"{tf}_cb_high"], cb[f"{tf}_cb_low"]) & _not_last(len(df))


def gate_g2_g3(rep: Report, df: pd.DataFrame, cb: pd.DataFrame) -> None:
    for tf in TFS:
        allowed = warmup_allowed(df, tf)
        for side in SIDES:
            for suf in ("", "_up", "_dn"):
                col = f"{tf}_cb_{side}{suf}"
                _float_cmp(rep, "G2", col, df[col], cb[col], allowed)
        mask = _g2_mask(df, cb, tf)                     # G3 uses the G2 row mask
        for d in DIRECTIONS:
            col = f"{tf}_cb_zone_{d}"
            _float_cmp(rep, "G3", col, df[col], cb[col], allowed, mask)
        for d in DIRECTIONS:
            col = f"{tf}_cb_inzone_{d}"
            _bool_cmp(rep, "G3", col, df[col], cb[col], mask, allowed)


def read_sidecar_shrink(ddir: str) -> float | None:
    meta_path = os.path.join(ddir, "candle_bounds_nc_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as fh:
        v = json.load(fh).get("tgt_shrink_pct")
    return None if v is None else float(v)


def _hybrid_zones(df: pd.DataFrame, tf: int, shrink: float) -> pd.DataFrame:
    """Hybrid zones recomputed by the kernel from the frame's own cbnc/cb columns
    at hair-cut ``shrink`` (%)."""
    saved = kernel.TGT_SHRINK_PCT
    kernel.TGT_SHRINK_PCT = shrink
    try:
        return kernel.forming_zones(
            df[[f"{tf}_cbnc_high", f"{tf}_cbnc_low"]],
            df[[f"{tf}_cb_high", f"{tf}_cb_low", f"{tf}_cb_inzone_long", f"{tf}_cb_inzone_short"]],
            df, tf)
    finally:
        kernel.TGT_SHRINK_PCT = saved


_HYBRID_FLOAT = ("cbnc_tgt", "cbnc_zone")
_HYBRID_BOOL = ("cbnc_inzone", "cbx_inzone")


def gate_g4(rep: Report, df: pd.DataFrame, cb: pd.DataFrame, nc: pd.DataFrame,
            stored_shrink: float | None) -> None:
    """Forming bounds vs the sidecar directly. Hybrid columns (tgt/zone/inzone/cbx)
    vs the sidecar directly when the sidecar's hair-cut equals TGT_SHRINK_PCT;
    otherwise (ruling R1) kernel parity: hybrid zones recomputed from our columns
    at the sidecar's hair-cut vs the sidecar, plus a WARN (not a failure)."""
    parity = stored_shrink is not None and stored_shrink != TGT_SHRINK_PCT
    if stored_shrink is None:
        print("G4 WARN: sidecar shrink unknown — parity-at-sidecar-shrink skipped", flush=True)
    if parity:
        print(f"G4 WARN sidecar tgt_shrink_pct={stored_shrink} != main flow TGT_SHRINK_PCT="
              f"{TGT_SHRINK_PCT}: the materialized hybrid columns (cbnc_tgt/cbnc_zone/"
              f"cbnc_inzone/cbx_inzone) are NOT comparable to the sidecar; gating kernel "
              f"parity recomputed at {stored_shrink}", flush=True)
    for tf in TFS:
        allowed = warmup_allowed(df, tf)
        for side in SIDES:
            for suf in ("", "_up", "_dn"):
                col = f"{tf}_cbnc_{side}{suf}"
                _float_cmp(rep, "G4", col, df[col], nc[col], allowed)
        # self-consistency: stored hybrid columns == kernel at the production hair-cut
        prod = _hybrid_zones(df, tf, TGT_SHRINK_PCT)
        for kind in _HYBRID_FLOAT + _HYBRID_BOOL:
            for d in DIRECTIONS:
                col = f"{tf}_{kind}_{d}"
                eq = prod[col].equals(df[col].astype(prod[col].dtype))
                rep.check("G4", f"{col} == kernel at TGT_SHRINK_PCT={TGT_SHRINK_PCT}", eq,
                          "identical" if eq else "differs")
        ref = _hybrid_zones(df, tf, stored_shrink) if parity else df
        tag = f" @shrink {stored_shrink}" if parity else ""
        mask = _valid(df[f"{tf}_cbnc_high"], df[f"{tf}_cbnc_low"],
                      nc[f"{tf}_cbnc_high"], nc[f"{tf}_cbnc_low"],
                      df[f"{tf}_cb_high"], df[f"{tf}_cb_low"],
                      cb[f"{tf}_cb_high"], cb[f"{tf}_cb_low"]) & _not_last(len(df))
        for kind in _HYBRID_FLOAT:
            for d in DIRECTIONS:
                col = f"{tf}_{kind}_{d}"
                _float_cmp(rep, "G4", col + tag, ref[col], nc[col], allowed)
        for kind in _HYBRID_BOOL:
            for d in DIRECTIONS:
                col = f"{tf}_{kind}_{d}"
                _bool_cmp(rep, "G4", col + tag, ref[col], nc[col], mask, allowed)


def gate_g5c(rep: Report, df: pd.DataFrame) -> None:
    """Materialized extdone columns == kernel.extdone with the production 2y
    ladder and wall-clock progress, recomputed from the frame."""
    for tf in TFS:
        ext = kernel.extdone(df, tf, EXTDONE_LADDER[tf])
        for side in SIDES:
            col = f"{tf}_extdone_{side}"
            mism = int((ext[col].to_numpy(dtype=bool) != df[col].to_numpy(dtype=bool)).sum())
            rep.check("G5c", f"{col} == kernel.extdone(EXTDONE_LADDER)", mism == 0,
                      f"{mism} mismatches (n={len(df)})")
            lvl = f"{col}_lvl"
            a, b = ext[lvl].to_numpy(dtype=float), df[lvl].to_numpy(dtype=float)
            same = (np.isnan(a) & np.isnan(b)) | (a == b)
            rep.check("G5c", f"{lvl} == kernel", bool(same.all()),
                      f"{int((~same).sum())} mismatches")


def gate_g5a(rep: Report, df: pd.DataFrame, ext: pd.DataFrame) -> None:
    for tf in TFS:
        lad = {"high": LADDER_OOS[tf], "low": LADDER_OOS[tf]}
        state = kernel.extdone(df, tf, lad, by_rows=True)
        g = pd.Series(candle_ids(df, tf), index=df.index)
        for side in SIDES:
            col = f"{tf}_extdone_{side}"
            sticky = state[col].groupby(g).cummax().to_numpy(dtype=bool)
            stored = ext[col].fillna(False).to_numpy(dtype=bool)
            mism = int((sticky != stored).sum())
            rep.check("G5a", f"{col} sticky(oos ladder, by_rows) mismatches", mism == 0,
                      f"{mism} (n={len(df)}, true ours/stored={int(sticky.sum())}/{int(stored.sum())})")


def p_hold(df: pd.DataFrame, tf: int, side: str) -> dict:
    """P(running extreme not touched again before the close | first True minute
    of the candle, candle's last row excluded) — fit_extdone_ladder.score()."""
    n = len(df)
    cid = candle_ids(df, tf)
    g = pd.Series(cid)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    ends = np.r_[starts[1:], n]
    last = np.zeros(n, dtype=bool)
    last[ends - 1] = True
    x = pd.Series(df["1_high" if side == "high" else "1_low"].to_numpy(dtype=float))
    if side == "high":
        run = x.groupby(g).cummax().to_numpy()
        rev = x[::-1].groupby(g[::-1]).cummax()[::-1].to_numpy()
    else:
        run = x.groupby(g).cummin().to_numpy()
        rev = x[::-1].groupby(g[::-1]).cummin()[::-1].to_numpy()
    fut = np.r_[rev[1:], np.nan]
    same = np.r_[cid[1:] == cid[:-1], False]
    fut = np.where(same, fut, np.nan)                 # nothing left in the candle
    with np.errstate(invalid="ignore"):
        again = (fut >= run) if side == "high" else (fut <= run)
    state = df[f"{tf}_extdone_{side}"].fillna(False).to_numpy(dtype=bool)
    first = state & (pd.Series(state).groupby(g).cumsum().to_numpy() == 1)
    trig = first & ~last
    return {"p_hold": float(1 - again[trig].mean()) if trig.any() else float("nan"),
            "n": int(trig.sum()),
            "fires_pct": float(100 * pd.Series(first).groupby(g).any().mean()),
            "share_true": float(state.mean())}


def gate_g5b(rep: Report, df: pd.DataFrame) -> dict:
    out = {}
    for tf in TFS:
        for side in SIDES:
            r = p_hold(df, tf, side)
            out[(tf, side)] = r
            ref = PHOLD_REF[(tf, side)]
            ok = abs(r["p_hold"] - ref) <= PHOLD_TOL
            rep.check("G5b", f"{tf}_extdone_{side} P(hold|first) 2y ladder", ok,
                      f"{r['p_hold']:.3f} vs ref {ref:.3f} ±{PHOLD_TOL} (n={r['n']}, "
                      f"fires {r['fires_pct']:.0f}% of candles, true minutes {100 * r['share_true']:.1f}%)")
    return out


def gate_g4o(rep: Report, df: pd.DataFrame, ov: pd.DataFrame | None) -> None:
    """Phase 19: overshoot z and pass flag vs the stored ``df_with_cb_overshoot.pkl``.

    On rows where both z are finite: |ours - sidecar| < 1e-9 and
    ``cb_f_overshoot == ~cbover`` with 0 mismatches (gating). Rows where the
    sidecar is +-inf (zero/NaN band): ours NaN and the flag True -- counted and
    printed, not gating. Absent sidecar -> one SKIP line."""
    if ov is None:
        print("G4-o SKIP: df_with_cb_overshoot.pkl absent", flush=True)
        return
    for tf in TFS:
        for d in DIRECTIONS:
            zc, fc, sc = f"{tf}_cb_overshoot_{d}", f"{tf}_cb_f_overshoot_{d}", f"{tf}_cbover_{d}"
            if zc not in df.columns or fc not in df.columns:
                rep.check("G4-o", f"{zc}/{fc} present", False, "column absent (run cb_markers_materialize.py)")
                continue
            ours = df[zc].to_numpy(dtype=float)
            theirs = ov[zc].reindex(df.index).to_numpy(dtype=float)
            flag = df[fc].fillna(False).to_numpy(dtype=bool)
            over = ov[sc].reindex(df.index).fillna(False).to_numpy(dtype=bool)
            # final row excluded like every other gate (the stored sidecars' last
            # row holds a bound ours does not reproduce — phase 18 edge rule)
            fin = np.isfinite(ours) & np.isfinite(theirs) & _not_last(len(ours))
            mx = float(np.abs(ours[fin] - theirs[fin]).max()) if fin.any() else float("nan")
            rep.check("G4-o", f"{zc} max_abs vs sidecar z", bool(fin.any()) and mx < 1e-9,
                      f"{mx:.2e} (n={int(fin.sum())})")
            thr = OVERSHOOT_THR[tf][d]
            mism = np.flatnonzero(fin & (flag != ~over))
            # the frozen threshold is the sidecar's to within the bracket its own
            # rows allow; a z inside that bracket may legitimately fall on the
            # other side — reported, not gating
            in_bracket = np.abs(ours[mism] - thr) <= OVERSHOOT_BRACKET
            rep.check("G4-o", f"{fc} == ~{sc} on finite rows (|z-thr| > {OVERSHOOT_BRACKET:g})",
                      int((~in_bracket).sum()) == 0,
                      f"{_rows(mism[~in_bracket], df.index)}; in-bracket flips {int(in_bracket.sum())} "
                      f"z={[round(float(v), 5) for v in ours[mism[in_bracket]][:4]]} thr={thr} "
                      f"(dropped ours/sidecar={int((~flag[fin]).sum())}/{int(over[fin].sum())})")
            inf = np.isinf(theirs)
            print(f"G4-o INFO {zc}: sidecar +-inf rows {int(inf.sum())}; ours NaN there "
                  f"{bool(np.isnan(ours[inf]).all()) if inf.any() else 'n/a'}; flag True there "
                  f"{bool(flag[inf].all()) if inf.any() else 'n/a'}; sidecar-finite/ours-NaN "
                  f"{int((np.isfinite(theirs) & np.isnan(ours)).sum())}; ours-finite/sidecar-NaN "
                  f"{int((np.isfinite(ours) & np.isnan(theirs)).sum())}", flush=True)


def marker_table(df: pd.DataFrame) -> None:
    print("\nMarker counts (minutes, oos2m):", flush=True)
    has_over = f"{TFS[0]}_cb_f_overshoot_long" in df.columns
    print("| tf | dir | cb_inzone | ∧ cb_f_extdone | ∧ cb_f_trend | ∧ cb_f_mcx | ∧ cb_f_overshoot | cbx_inzone "
          "| extdone share |", flush=True)
    print("|---|---|---|---|---|---|---|---|---|", flush=True)
    for tf in TFS:
        for d in DIRECTIONS:
            side = "high" if d == "long" else "low"     # long filtered by extdone_high
            m = df[f"{tf}_cb_inzone_{d}"].to_numpy(dtype=bool)
            n0 = int(m.sum())

            def _and(kind: str) -> str:
                k = int((m & df[f"{tf}_{kind}_{d}"].to_numpy(dtype=bool)).sum())
                return f"{k} ({100 * k / n0:.0f} %)" if n0 else "0"

            cbx = int(df[f"{tf}_cbx_inzone_{d}"].to_numpy(dtype=bool).sum())
            share = 100 * df[f"{tf}_extdone_{side}"].to_numpy(dtype=bool).mean()
            over = _and("cb_f_overshoot") if has_over else "n/a"
            print(f"| {tf} | {d} | {n0} | {_and('cb_f_extdone')} | {_and('cb_f_trend')} | "
                  f"{_and('cb_f_mcx')} | {over} | {cbx} | extdone_{side} {share:.1f} % |", flush=True)


def main() -> int:
    path = wide_df_path()
    ddir = os.path.dirname(path)
    df = pd.read_pickle(path)
    print(f"[validate] {path} shape={df.shape} {df.index[0]} -> {df.index[-1]}", flush=True)
    cb = pd.read_pickle(os.path.join(ddir, "df_with_candle_bounds.pkl"))
    nc = pd.read_pickle(os.path.join(ddir, "df_with_candle_bounds_nc.pkl"))
    ext = pd.read_pickle(os.path.join(ddir, "df_with_ext_done.pkl"))
    ov_path = os.path.join(ddir, "df_with_cb_overshoot.pkl")
    ov = pd.read_pickle(ov_path) if os.path.exists(ov_path) else None
    rep = Report()
    for name, side in (("df_with_candle_bounds", cb), ("df_with_candle_bounds_nc", nc),
                       ("df_with_ext_done", ext)) + ((("df_with_cb_overshoot", ov),) if ov is not None else ()):
        rep.check("IDX", f"{name} index == dataset index", side.index.equals(df.index),
                  f"{len(side)} rows")
    gaps = int((df.index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).sum())
    print(f"[validate] non-1-min steps in index: {gaps}", flush=True)

    gate_g2_g3(rep, df, cb)
    gate_g4(rep, df, cb, nc, read_sidecar_shrink(ddir))
    gate_g5a(rep, df, ext)
    gate_g5c(rep, df)
    gate_g5b(rep, df)
    gate_g4o(rep, df, ov)
    marker_table(df)

    if rep.failed:
        print(f"\n[validate] FAILED {len(rep.failed)} check(s): {rep.failed}", flush=True)
        return 1
    print("\n[validate] all gates PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
