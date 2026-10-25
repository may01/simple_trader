"""Phase → up / down / sideways class with time-left, and its forward scoring.

Features come ONLY from the fit at row t (amp, resid_std, period_fit, phase,
dt_pred, sign_pred). Truth columns come from apex.label_series (future) and the
profit labels stored in the series frame.

    python -m experiments.swing_sinusoid.zoning --params <params json>   # thresholds on 2y val
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import config
from .apex import label_series
from .provenance import stamp
from .sweep import run_tf

A_MIN_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PERIOD_BAND = (0.5, 2.0)        # × train median period
MIN_CLASS_N = 300
SELECT_K = 2                    # horizon used for threshold selection
N_PERM = 200
TIME_LEFT_EDGES = ((0, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 4.5, "3-4"), (4.5, np.inf, "5+"))


def time_left_bucket(dt: np.ndarray) -> np.ndarray:
    out = np.full(len(dt), "", dtype=object)
    for lo, hi, lab in TIME_LEFT_EDGES:
        out[(dt > lo) & (dt <= hi)] = lab
    return out


def classify(pred: pd.DataFrame, a_min: float, p_lo: float, p_hi: float) -> np.ndarray:
    """+1 up (heading to a high), -1 down, 0 neutral / sideways."""
    ok = pred["ok"].to_numpy(dtype=bool)
    snr = (pred["amp"] / pred["resid_std"]).to_numpy(dtype=float)
    per = pred["period_fit"].to_numpy(dtype=float)
    moving = ok & np.isfinite(snr) & (snr >= a_min) & (per >= p_lo) & (per <= p_hi)
    cls = np.where(moving, pred["sign_pred"].to_numpy(dtype=float), 0.0)
    return cls.astype(int)


def _perm_p95(hits: np.ndarray, mask: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """95th pct of the hit rate when the class mask is shuffled over the same rows."""
    k = int(mask.sum())
    if k == 0:
        return np.nan
    vals = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(len(hits), size=k, replace=False)
        vals[i] = np.nanmean(hits[idx])
    return float(np.percentile(vals, 95))


def score_classes(pred: pd.DataFrame, cls: np.ndarray, labels: pd.DataFrame, frame: pd.DataFrame,
                  tf: int, horizons=config.HORIZONS, seed: int = 0) -> pd.DataFrame:
    """Truth-ladder table per (class, k): indicator dir → price sign / return → plain → strict."""
    rng = np.random.default_rng(seed)
    rows_idx = pred["row"].to_numpy()
    lab = labels.iloc[rows_idx]
    fr = frame.iloc[rows_idx]
    lcols = config.label_cols(tf)
    out = []
    for k in horizons:
        fdir = lab[f"fwd_dir_{k}"].to_numpy(dtype=float)
        fret = lab[f"fwd_ret_{k}"].to_numpy(dtype=float)
        valid = np.isfinite(fdir) & np.isfinite(fret)
        for c in (1, -1, 0):
            m = (cls == c) & valid
            n = int(m.sum())
            rec = {"tf": tf, "k": k, "class": {1: "up", -1: "down", 0: "neutral"}[c], "n": n,
                   "share": n / max(int(valid.sum()), 1)}
            if c != 0:
                ind_hit = (fdir == c).astype(float)
                px_hit = (np.sign(fret) == c).astype(float)
                rec.update({
                    "ind_hit": float(ind_hit[m].mean()) if n else np.nan,
                    "ind_base": float(ind_hit[valid].mean()),
                    "ind_perm95": _perm_p95(ind_hit[valid], m[valid], rng),
                    "px_hit": float(px_hit[m].mean()) if n else np.nan,
                    "px_base": float(px_hit[valid].mean()),
                    "px_perm95": _perm_p95(px_hit[valid], m[valid], rng),
                    "ret_signed_bp": float(1e4 * (c * fret[m]).mean()) if n else np.nan,
                    "ret_abs_base_bp": float(1e4 * np.abs(fret[valid]).mean()),
                })
                side = "long" if c == 1 else "short"
                for kind in ("plain", "strict"):
                    col = f"{kind}_n1_{side}"
                    if col in fr.columns:
                        lv = fr[col].to_numpy(dtype=float)
                        lvalid = valid & np.isfinite(lv)
                        rec[f"{kind}_rate"] = float(lv[m & lvalid].mean()) if (m & lvalid).sum() else np.nan
                        rec[f"{kind}_base"] = float(lv[lvalid].mean())
                        rec[f"{kind}_perm95"] = _perm_p95(lv[lvalid], m[lvalid], rng)
            else:
                rec.update({"ind_hit": np.nan, "ind_base": np.nan,
                            "ret_abs_bp": float(1e4 * np.abs(fret[m]).mean()) if n else np.nan,
                            "ret_abs_base_bp": float(1e4 * np.abs(fret[valid]).mean())})
            out.append(rec)
    df = pd.DataFrame(out)
    for a, b in (("ind_hit", "ind_base"), ("px_hit", "px_base"), ("plain_rate", "plain_base"), ("strict_rate", "strict_base")):
        if a in df.columns:
            df[a.replace("hit", "lift").replace("rate", "lift")] = df[a] / df[b]
    return df


def score_time_left(pred: pd.DataFrame, cls: np.ndarray, labels: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Does time_left mean anything: realised next-apex distance per predicted bucket (moving rows)."""
    rows_idx = pred["row"].to_numpy()
    lab = labels.iloc[rows_idx]
    tl = time_left_bucket(pred["dt_pred"].to_numpy(dtype=float))
    true_dt = lab["next_apex_dt"].to_numpy(dtype=float)
    out = []
    for b in ("1", "2", "3-4", "5+"):
        m = (cls != 0) & (tl == b) & np.isfinite(true_dt)
        if m.sum() == 0:
            continue
        out.append({"tf": tf, "time_left": b, "n": int(m.sum()),
                    "true_dt_median": float(np.median(true_dt[m])),
                    "true_dt_q25": float(np.percentile(true_dt[m], 25)),
                    "true_dt_q75": float(np.percentile(true_dt[m], 75)),
                    "hit_t": float((np.abs(pred["dt_pred"].to_numpy()[m] - true_dt[m]) <= 1.0).mean())})
    base = {"tf": tf, "time_left": "all moving", "n": int((cls != 0).sum()),
            "true_dt_median": float(np.nanmedian(true_dt[cls != 0])) if (cls != 0).any() else np.nan}
    out.append(base)
    return pd.DataFrame(out)


def load_winner_predictions(sdir: str, tf: int, ptag: str, winners: dict) -> dict[str, pd.DataFrame]:
    pred = pd.read_pickle(os.path.join(sdir, f"predictions_tf{tf}_{ptag}.pkl"))
    out = {}
    for key, w in winners.items():
        if "method" not in w or int(w["tf"]) != tf:
            continue
        m = (pred["series"] == w["series"]) & (pred["method"] == w["method"]) & (pred["W"] == w["W"])
        out[w["series"]] = pred[m].reset_index(drop=True)
    return out


def fit_thresholds(params_path: str, dataset: str = "2y", workers: int = 8) -> dict:
    params = json.load(open(params_path))
    ptag = f"p{params['prominence_frac']:g}".replace(".", "")
    sdir = config.sidecar_dir(dataset)
    winners = params["winners"]
    zoning, tables = {}, []
    for tf in config.TFS:
        frame = pd.read_pickle(os.path.join(sdir, f"series_tf{tf}.pkl"))
        plan = {w["series"]: [(w["method"], w["W"])] for w in winners.values()
                if "method" in w and int(w["tf"]) == tf}
        med = {s_: params["apex_meta"][str(tf)][s_]["median_period_train"] for s_ in plan}
        if not plan:
            continue
        allpred, _, _ = run_tf(tf, dataset, params["prominence_frac"], 1, workers, None, plan=plan,
                               median_periods=med, min_period=params.get("min_period", 4.0))
        wp = {s_: allpred[allpred["series"] == s_].reset_index(drop=True) for s_ in plan}
        for series, pred in wp.items():
            labels, _ = label_series(frame, series, prominence_frac=params["prominence_frac"])
            med_p = params["apex_meta"][str(tf)][series]["median_period_train"]
            p_lo, p_hi = PERIOD_BAND[0] * med_p, PERIOD_BAND[1] * med_p
            val = pred[pred["split"] == "val"].reset_index(drop=True)
            best = None
            grid = []
            for a_min in A_MIN_GRID:
                cls = classify(val, a_min, p_lo, p_hi)
                tab = score_classes(val, cls, labels, frame, tf, horizons=(SELECT_K,), seed=1)
                mv = tab[(tab["k"] == SELECT_K) & (tab["class"] != "neutral")]
                n_min = int(mv["n"].min()) if len(mv) else 0
                lift = float(np.nanmean(mv["ind_lift"])) if len(mv) else np.nan
                share = float((cls != 0).mean())
                grid.append({"tf": tf, "series": series, "a_min": a_min, "n_min_class": n_min,
                             "moving_share": share, "ind_lift_k2": lift,
                             "px_lift_k2": float(np.nanmean(mv["px_lift"])) if len(mv) else np.nan})
                if n_min >= MIN_CLASS_N and (best is None or lift > best[1]):
                    best = (a_min, lift)
            tables.extend(grid)
            a_sel = best[0] if best else None
            zoning[f"{tf}:{series}"] = {
                "tf": tf, "series": series, "a_min": a_sel, "p_lo": p_lo, "p_hi": p_hi,
                "median_period_train": med_p, "val_ind_lift_k2": best[1] if best else None,
                "grid_edge": (a_sel in (A_MIN_GRID[0], A_MIN_GRID[-1])) if a_sel is not None else None,
            }
            print(f"[tf{tf}] {series}: a_min={a_sel} lift_k2={best[1] if best else None:.3f}" if best else
                  f"[tf{tf}] {series}: no a_min with n ≥ {MIN_CLASS_N}", flush=True)
    params["zoning"] = zoning
    params["zoning_meta"] = {"a_min_grid": list(A_MIN_GRID), "period_band": list(PERIOD_BAND),
                             "min_class_n": MIN_CLASS_N, "select_k": SELECT_K, "selected_on": "2y val split",
                             **stamp(dataset)}
    with open(params_path, "w") as fh:
        json.dump(params, fh, indent=2, default=float)
    pd.DataFrame(tables).to_csv(os.path.join(sdir, f"zoning_grid_{ptag}{params.get('tag', '')}.csv"), index=False)
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fit_thresholds(args.params, workers=args.workers)


if __name__ == "__main__":
    main()
