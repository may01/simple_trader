"""Driver: EMA-25 slope distribution + rise / fall / neutral classes, and the
chart sidecar that draws ema_25 coloured by class.

Slope = the in-code indicator ``ema_25_diff_prc`` on closed candles:
    d[k] = (ema_25[k] - ema_25[k-1]) / ema_25[k-1] * 100    per closed tf candle
    z[k] = (d[k] - mean_2y) / std_2y                         mean/std over 2y closed candles
mean/std and the class rule are the frozen constants of
indicators.library.classification (EMA_SLOPE_STATS, EMA_SLOPE_X,
ema_slope_class — the ``ema_25_slope_class`` field); this driver recomputes
them from 2y and asserts they match (provenance gate) before writing.
Class of every 1-min row = class of the LAST CLOSED candle's slope (shift(1)
+ ffill, no forming input):   fall z < -X,  neutral |z| <= X,  rise z > X.

Artifacts:
  {dataset_dir}/df_with_ema_slope.pkl   {tf}_ema25_slope_z, {tf}_ema25_cls (-1/0/1),
                                        {tf}_ema_25_{rise,fall,neutral} (ema_25 where the
                                        class holds, NaN elsewhere; drawn with the
                                        unshifted class, segments joined at closing rows)
  {EV_OUT}/ema_slope_report.{json,md}   2y distribution + class shares/run lengths (2y, oos2m)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from evlib import TFS  # noqa: E402
from indicators.library.classification import (  # noqa: E402
    EMA_SLOPE_STATS, EMA_SLOPE_X, ema_slope_class,
)

VOL = "/trader_data_long/train"
DATASET_DIR = os.environ.get("EV_DATASET_DIR", f"{VOL}/oos2m_link_usdt")
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
X = EMA_SLOPE_X
PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
CLS = {-1: "fall", 0: "neutral", 1: "rise"}


def closed_slope(df: pd.DataFrame, tf: int) -> pd.Series:
    """d[k] in % of price, indexed by each closed candle's closing row."""
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712
    c = df.loc[closed, f"{tf}_ema_25"]
    return ((c - c.shift(1)) / c.shift(1) * 100.0).dropna()


def run_lengths(cls: np.ndarray) -> dict:
    """Mean/median run length (in candles) per class."""
    if len(cls) == 0:
        return {}
    cut = np.flatnonzero(np.r_[True, cls[1:] != cls[:-1], True])
    lens, vals = np.diff(cut), cls[cut[:-1]]
    return {CLS[v]: {"runs": int((vals == v).sum()),
                     "mean": float(lens[vals == v].mean()) if (vals == v).any() else 0.0,
                     "median": float(np.median(lens[vals == v])) if (vals == v).any() else 0.0}
            for v in (-1, 0, 1)}


def main() -> None:
    tr = pd.read_pickle(TRAIN_IND)
    oos = pd.read_pickle(os.path.join(DATASET_DIR, "df_with_indicators.pkl"))
    cols, report = {}, {"x": X, "tf": {}}
    for tf in TFS:
        d_tr, d_oo = closed_slope(tr, tf), closed_slope(oos, tf)
        mu, sd = EMA_SLOPE_STATS[tf]
        fresh = (float(d_tr.mean()), float(d_tr.std()))
        assert np.allclose(fresh, (mu, sd), rtol=1e-9, atol=0), (
            f"tf {tf}: frozen EMA_SLOPE_STATS {(mu, sd)} != 2y recompute {fresh}")
        z_tr, z_oo = (d_tr - mu) / sd, (d_oo - mu) / sd
        c_tr, c_oo = ema_slope_class(d_tr, tf), ema_slope_class(d_oo, tf)
        m4 = float(((d_tr - mu) ** 4).mean() / sd ** 4)
        report["tf"][tf] = {
            "n_2y": int(len(d_tr)), "n_oos": int(len(d_oo)),
            "mean": mu, "std": sd,
            "skew": float(((d_tr - mu) ** 3).mean() / sd ** 3), "kurtosis": m4,
            "pct_2y": {p: float(np.percentile(d_tr, p)) for p in PCTS},
            "pct_oos": {p: float(np.percentile(d_oo, p)) for p in PCTS},
            "cut_pct": {"fall_below": mu - X * sd, "rise_above": mu + X * sd},
            "share_2y": {CLS[v]: float((c_tr == v).mean()) for v in (-1, 0, 1)},
            "share_oos": {CLS[v]: float((c_oo == v).mean()) for v in (-1, 0, 1)},
            "share_gauss": {"fall": 0.3085, "neutral": 0.3829, "rise": 0.3085} if X == 0.5 else None,
            "runs_2y": run_lengths(c_tr), "runs_oos": run_lengths(c_oo),
            "oos_mean": float(d_oo.mean()), "oos_std": float(d_oo.std()),
        }
        r = report["tf"][tf]
        print(tf, "2y n=%d mean %+.4f std %.4f skew %+.2f kurt %.1f | cuts %+.4f / %+.4f | "
              "share 2y %s oos %s" % (
                  r["n_2y"], mu, sd, r["skew"], m4, r["cut_pct"]["fall_below"],
                  r["cut_pct"]["rise_above"],
                  {k: round(v, 2) for k, v in r["share_2y"].items()},
                  {k: round(v, 2) for k, v in r["share_oos"].items()}), flush=True)
        # per-row sidecar on the target dataset: last closed candle's z / class
        z_row = z_oo.reindex(oos.index).shift(1).ffill()
        cls_row = np.where(z_row < -X, -1, np.where(z_row > X, 1, 0)).astype(float)
        cls_row[z_row.isna().to_numpy()] = np.nan
        ema = oos[f"{tf}_ema_25"].to_numpy()
        cols[f"{tf}_ema25_slope_z"] = z_row.to_numpy()
        cols[f"{tf}_ema25_cls"] = cls_row
        # Drawing columns use the UNSHIFTED class: at candle k's closing row
        # the colour is d[k]'s own class (known at that close), so on the tf
        # chart — which plots closing rows — the segment ending at candle k is
        # coloured by the slope of that same candle, not the previous one.
        # _slope_z/_cls above keep the shift(1) no-same-row convention.
        z_draw = z_oo.reindex(oos.index).ffill().to_numpy()
        cls_draw = np.where(z_draw < -X, -1.0, np.where(z_draw > X, 1.0, 0.0))
        cls_draw[np.isnan(z_draw)] = np.nan
        closed = (oos[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
        for v, name in CLS.items():
            m = cls_draw == v
            # extend each segment to the next closing row so consecutive
            # segments share a point and the coloured line is continuous
            nxt = pd.Series(np.where(closed, m, np.nan)).shift(1).ffill().fillna(0).to_numpy(dtype=bool)
            cols[f"{tf}_ema_25_{name}"] = np.where(m | nxt, ema, np.nan)
    pd.DataFrame(cols, index=oos.index).to_pickle(os.path.join(DATASET_DIR, "df_with_ema_slope.pkl"))
    with open(os.path.join(OUT, "ema_slope_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    o = [f"# EMA-25 slope distribution (closed candles) and classes at x = {X}", "",
         "slope d[k] = (ema_25[k] − ema_25[k−1]) / ema_25[k] · 100, % of price per candle; "
         "z = (d − mean)/std from 2y; fall z < −x, rise z > x.", "",
         "| tf | n 2y | mean | std | skew | kurtosis | p1 | p5 | p25 | p50 | p75 | p95 | p99 | fall cut | rise cut |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tf, r in report["tf"].items():
        p = r["pct_2y"]
        o.append(f"| {tf} | {r['n_2y']} | {r['mean']:+.4f} | {r['std']:.4f} | {r['skew']:+.2f} | "
                 f"{r['kurtosis']:.1f} | {p[1]:+.3f} | {p[5]:+.3f} | {p[25]:+.3f} | {p[50]:+.3f} | "
                 f"{p[75]:+.3f} | {p[95]:+.3f} | {p[99]:+.3f} | {r['cut_pct']['fall_below']:+.4f} | "
                 f"{r['cut_pct']['rise_above']:+.4f} |")
    o += ["", "| tf | share 2y fall / neutral / rise | share oos2m | mean run (candles) 2y f / n / r | oos2m mean / std |",
          "|---|---|---|---|---|"]
    for tf, r in report["tf"].items():
        s, so, rl = r["share_2y"], r["share_oos"], r["runs_2y"]
        o.append(f"| {tf} | {s['fall']:.2f} / {s['neutral']:.2f} / {s['rise']:.2f} | "
                 f"{so['fall']:.2f} / {so['neutral']:.2f} / {so['rise']:.2f} | "
                 f"{rl['fall']['mean']:.1f} / {rl['neutral']['mean']:.1f} / {rl['rise']['mean']:.1f} | "
                 f"{r['oos_mean']:+.4f} / {r['oos_std']:.4f} |")
    with open(os.path.join(OUT, "ema_slope_report.md"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    print("wrote", DATASET_DIR, OUT)


if __name__ == "__main__":
    main()
