"""Statistical "most likely time to the next apex" from the fitted state.

Instead of projecting the sine's phase deterministically, bin the rows by the
fitted state at t (phase at the window end × fitted-period tercile) and take the
empirical distribution of the TRUE distance to the next apex inside each bin on
2y train. Prediction for a row = the mode of its bin. Scored on 2y val and oos2m
against (a) the unconditional mode (no state at all) and (b) the raw projection.

    python -m experiments.swing_sinusoid.calibrate --params <json> [--n-phase 8]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import config
from .sweep import run_tf

D_MAX = 12          # dt_true bins 1..12, 13 = "13+"
MIN_BIN = 100


def _bins(pred: pd.DataFrame, n_phase: int, per_cuts: np.ndarray) -> np.ndarray:
    ph = np.floor(pred["phase"].to_numpy(dtype=float) / (2 * np.pi) * n_phase).astype(int) % n_phase
    pt = np.searchsorted(per_cuts, pred["period_fit"].to_numpy(dtype=float))
    return ph * (len(per_cuts) + 1) + pt


def _dist(dt: np.ndarray) -> np.ndarray:
    d = np.clip(dt, 1, D_MAX + 1).astype(int)
    return np.bincount(d, minlength=D_MAX + 2)[1:] / len(d)


def calibrate_cell(tr: pd.DataFrame, n_phase: int) -> tuple[dict, np.ndarray, pd.DataFrame]:
    per_cuts = np.quantile(tr["period_fit"].to_numpy(dtype=float), [1 / 3, 2 / 3])
    b = _bins(tr, n_phase, per_cuts)
    dt = tr["dt_true"].to_numpy(dtype=float)
    table, rows = {}, []
    uncond = _dist(dt)
    for k in np.unique(b):
        m = b == k
        if m.sum() < MIN_BIN:
            continue
        p = _dist(dt[m])
        mode = int(np.argmax(p)) + 1
        table[int(k)] = mode
        rows.append({"bin": int(k), "phase_bin": int(k) // 3, "period_tercile": int(k) % 3, "n": int(m.sum()),
                     "mode": mode, "p_mode": float(p.max()), "p_mode_pm1": float(p[max(mode - 2, 0): mode + 1].sum()),
                     "median_dt": float(np.median(dt[m])), "p_far": float(p[-1])})
    um = int(np.argmax(uncond)) + 1
    meta = {"uncond_mode": um, "uncond_p_mode": float(uncond.max()),
            "uncond_p_mode_pm1": float(uncond[max(um - 2, 0): um + 1].sum()), "uncond_p_far": float(uncond[-1])}
    return table, per_cuts, pd.DataFrame(rows), meta


def score(df: pd.DataFrame, table: dict, per_cuts: np.ndarray, n_phase: int, uncond_mode: int) -> dict:
    b = _bins(df, n_phase, per_cuts)
    dt = df["dt_true"].to_numpy(dtype=float)
    pred = np.array([table.get(int(k), np.nan) for k in b], dtype=float)
    ok = np.isfinite(pred)
    raw = df["dt_pred"].to_numpy(dtype=float)
    near = dt <= 8
    return {
        "n": int(ok.sum()),
        "hit_calib": float((np.abs(pred[ok] - dt[ok]) <= 1).mean()),
        "hit_uncond": float((np.abs(uncond_mode - dt[ok]) <= 1).mean()),
        "hit_raw": float((np.abs(raw[ok] - dt[ok]) <= 1).mean()),
        "hit_calib_lead8": float((np.abs(pred[ok & near] - dt[ok & near]) <= 1).mean()),
        "hit_uncond_lead8": float((np.abs(uncond_mode - dt[ok & near]) <= 1).mean()),
        "hit_raw_lead8": float((np.abs(raw[ok & near] - dt[ok & near]) <= 1).mean()),
        "exact_calib": float((pred[ok] == dt[ok]).mean()),
        "exact_uncond": float((uncond_mode == dt[ok]).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n-phase", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    params = json.load(open(args.params))
    ptag = f"p{params['prominence_frac']:g}".replace(".", "") + params.get("tag", "")
    sdir2y = config.sidecar_dir("2y")
    out_rows, bin_rows = [], []
    for tf in config.TFS:
        plan, med = {}, {}
        for key, w in params["winners"].items():
            if "method" in w and int(w["tf"]) == tf and w["series"] != "close_diff_prc":
                plan[w["series"]] = [(w["method"], w["W"])]
                med[w["series"]] = params["apex_meta"][str(tf)][w["series"]]["median_period_train"]
        oos, _, _ = run_tf(tf, "oos2m", params["prominence_frac"], 1, args.workers, None, plan=plan,
                           median_periods=med, min_period=params.get("min_period", 4.0))
        pred2y, _, _ = run_tf(tf, "2y", params["prominence_frac"], 1, args.workers, None, plan=plan,
                              median_periods=med, min_period=params.get("min_period", 4.0))
        for series, cells in plan.items():
            m, W = cells[0]
            cell = pred2y[(pred2y["series"] == series) & (pred2y["method"] == m) & (pred2y["W"] == W) & pred2y["ok"]]
            cell = cell[np.isfinite(cell["dt_true"])]
            tr, va = cell[cell["split"] == "train"], cell[cell["split"] == "val"]
            oo = oos[(oos["series"] == series) & (oos["method"] == m) & oos["ok"] & np.isfinite(oos["dt_true"])]
            table, cuts, bins, meta = calibrate_cell(tr, args.n_phase)
            bins.insert(0, "series", series); bins.insert(0, "tf", tf)
            bin_rows.append(bins)
            best = bins.sort_values("p_mode", ascending=False).iloc[0] if len(bins) else None
            for name, df in (("2y_val", va), ("oos2m", oo)):
                r = score(df, table, cuts, args.n_phase, meta["uncond_mode"])
                r.update({"tf": tf, "series": series, "cell": f"{m} W{W}", "split": name, **meta,
                          "n_bins": len(table), "best_bin_p_mode": float(best["p_mode"]) if best is not None else np.nan,
                          "best_bin_mode": int(best["mode"]) if best is not None else -1,
                          "best_bin_n": int(best["n"]) if best is not None else 0})
                out_rows.append(r)
            print(f"[tf{tf}] {series} {m} W{W}: uncond mode {meta['uncond_mode']} p {meta['uncond_p_mode']:.2f} | "
                  f"best bin mode {r['best_bin_mode']} p {r['best_bin_p_mode']:.2f} (n {r['best_bin_n']})", flush=True)
    res = pd.DataFrame(out_rows)
    bins = pd.concat(bin_rows, ignore_index=True)
    res.to_csv(os.path.join(sdir2y, f"calibration_{ptag}.csv"), index=False)
    bins.to_csv(os.path.join(sdir2y, f"calibration_bins_{ptag}.csv"), index=False)
    pd.set_option("display.width", 300)
    print(res[["tf", "series", "cell", "split", "n", "uncond_mode", "uncond_p_mode", "best_bin_p_mode", "hit_calib",
               "hit_uncond", "hit_raw", "hit_calib_lead8", "hit_uncond_lead8", "hit_raw_lead8", "exact_calib",
               "exact_uncond"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
