"""Next apex from the spacing of the last three apexes (user request 2026-10-24).

Per series and tf: apexes = alternating highs / lows (prominence p·σ, ≥ 2 apart). For
every apex i ≥ 2 with spacings s1 = t_{i-1} − t_{i-2}, s2 = t_i − t_{i-1}, predict the
next apex time t̂_{i+1} = t_i + f(s1, s2):
  last      f = s2                       (next spacing = last spacing)
  mean2     f = (s1 + s2) / 2            (mean of the two spacings)
  trend     f = max(2·s2 − s1, 1)        (linear extrapolation of the spacing)
  median    f = train median spacing     (null: no apex information beyond "an apex just happened")
Error e = t_{i+1} − t̂_{i+1} in candles (positive = real apex later than predicted).

Two views:
  oracle  — the apex is used as soon as it occurs (t_i), i.e. perfect detection;
  causal  — the apex is only known once the series has moved ≥ prominence away from it
            (confirmation time c_i); a prediction is "usable" only if t̂_{i+1} > c_i.

    python -m experiments.swing_sinusoid.apex_spacing [--prominence 0.5]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import config
from .apex import find_apexes

METHODS = ("last", "mean2", "trend", "median")


def _confirm_times(y: np.ndarray, idx: np.ndarray, sgn: np.ndarray, prom: float) -> np.ndarray:
    """First t > t_i where the series has moved ≥ prom against the apex (apex becomes knowable)."""
    out = np.full(len(idx), -1, dtype=int)
    n = len(y)
    for j, (i, s) in enumerate(zip(idx, sgn)):
        seg = y[i + 1:]
        hit = np.where((y[i] - seg >= prom) if s > 0 else (seg - y[i] >= prom))[0]
        out[j] = i + 1 + hit[0] if len(hit) else n  # never confirmed → end of series
    return out


def evaluate(frame: pd.DataFrame, series: str, prom_frac: float, median_spacing: float | None,
             mask_rows: np.ndarray | None = None) -> tuple[pd.DataFrame, float]:
    y = frame[series].to_numpy(dtype=float)
    std = float(np.nanstd(y))
    idx, sgn = find_apexes(y, prominence=prom_frac * std, distance=2)
    conf = _confirm_times(y, idx, sgn, prom_frac * std)
    sp = np.diff(idx).astype(float)
    rows = []
    for i in range(2, len(idx) - 1):
        if mask_rows is not None and not mask_rows[idx[i]]:
            continue
        s1, s2 = sp[i - 2], sp[i - 1]
        real = idx[i + 1]
        preds = {"last": s2, "mean2": (s1 + s2) / 2.0, "trend": max(2 * s2 - s1, 1.0)}
        if median_spacing is not None:
            preds["median"] = median_spacing
        for m, f in preds.items():
            t_hat = idx[i] + f
            rows.append({"series": series, "method": m, "apex_row": int(idx[i]), "sign": int(sgn[i]),
                         "s1": s1, "s2": s2, "real_spacing": float(real - idx[i]), "pred_spacing": float(f),
                         "err": float(real - t_hat), "confirm_lag": int(conf[i] - idx[i]),
                         "usable": bool(t_hat > conf[i])})
    med_sp = float(np.median(sp)) if len(sp) else np.nan
    return pd.DataFrame(rows), med_sp


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (tf, series, m), g in df.groupby(["tf", "series", "method"], sort=False):
        e = g["err"].to_numpy()
        u = g[g["usable"]]
        eu = u["err"].to_numpy()
        out.append({
            "tf": tf, "series": series, "method": m, "n_apex": len(g),
            "median_real_spacing": float(g["real_spacing"].median()),
            "median_abs_err": float(np.median(np.abs(e))), "mean_abs_err": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "p_le1": float((np.abs(e) <= 1).mean()), "p_le2": float((np.abs(e) <= 2).mean()),
            "p_le3": float((np.abs(e) <= 3).mean()),
            "median_rel_err": float(np.median(np.abs(e) / g["real_spacing"].to_numpy())),
            "usable_share": float(g["usable"].mean()), "median_confirm_lag": float(g["confirm_lag"].median()),
            "usable_median_abs_err": float(np.median(np.abs(eu))) if len(eu) else np.nan,
            "usable_p_le1": float((np.abs(eu) <= 1).mean()) if len(eu) else np.nan,
            "usable_p_le2": float((np.abs(eu) <= 2).mean()) if len(eu) else np.nan,
        })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prominence", type=float, default=0.5)
    args = ap.parse_args()
    ptag = f"p{args.prominence:g}".replace(".", "")
    series_list = [s for s in config.SERIES if s != "close_diff_prc"]
    results = {}
    med = {}
    for dataset in ("2y", "oos2m"):
        sdir = config.sidecar_dir(dataset)
        parts = []
        for tf in config.TFS:
            frame = pd.read_pickle(os.path.join(sdir, f"series_tf{tf}.pkl"))
            train_mask = np.asarray(frame.index <= pd.Timestamp(config.TRAIN_END))
            for series in series_list:
                if dataset == "2y":
                    # median spacing from train apexes only
                    _, m_sp = evaluate(frame.loc[train_mask], series, args.prominence, None)
                    med[(tf, series)] = m_sp
                    for split, mask in (("train", train_mask), ("val", ~train_mask)):
                        df, _ = evaluate(frame, series, args.prominence, med[(tf, series)], mask)
                        df["tf"], df["split"] = tf, split
                        parts.append(df)
                else:
                    df, _ = evaluate(frame, series, args.prominence, med[(tf, series)])
                    df["tf"], df["split"] = tf, "oos2m"
                    parts.append(df)
        results[dataset] = pd.concat(parts, ignore_index=True)
    allp = pd.concat(results.values(), ignore_index=True)
    out_dir = config.sidecar_dir("2y")
    allp.to_pickle(os.path.join(out_dir, f"apex_spacing_errors_{ptag}.pkl"))
    summ = []
    for split in ("train", "val", "oos2m"):
        s = summarize(allp[allp["split"] == split]); s.insert(0, "split", split); summ.append(s)
    summ = pd.concat(summ, ignore_index=True)
    summ.to_csv(os.path.join(out_dir, f"apex_spacing_{ptag}.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 500)
    cols = ["split", "tf", "series", "method", "n_apex", "median_real_spacing", "median_abs_err", "bias", "p_le1", "p_le2", "p_le3",
            "median_rel_err", "usable_share", "median_confirm_lag", "usable_median_abs_err", "usable_p_le1"]
    print(summ[summ.split != "train"][cols].round(3).to_string(index=False))
    # error distribution of the real spacing itself (how irregular are swings?)
    g = allp[(allp.split == "val") & (allp.method == "last")].groupby(["tf", "series"])["real_spacing"]
    print("\nreal spacing quantiles (2y val): q10 / q25 / q50 / q75 / q90")
    print(g.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack().round(1).to_string())


if __name__ == "__main__":
    main()
