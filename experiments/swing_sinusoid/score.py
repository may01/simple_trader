"""Score the frozen configuration on a dataset (2y val split, or oos2m ONCE).

    python -m experiments.swing_sinusoid.score --params <json> --dataset oos2m

Runs only the frozen winner (and its best baseline) per (tf, series), applies the
frozen zoning thresholds, and writes:
  <dataset sidecar>/scores_{tag}.csv         truth-ladder table per (tf, series, class, k)
  <dataset sidecar>/time_left_{tag}.csv      realised apex distance per predicted time-left bucket
  <dataset sidecar>/deviation_{tag}.csv      winner vs baseline apex deviation on this dataset
  <dataset sidecar>/gates_{tag}.json         validity-gate numbers (provenance, reproduction, …)
  <dataset sidecar>/plots/*.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config, fitters  # noqa: E402
from .apex import label_series  # noqa: E402
from .provenance import stamp  # noqa: E402
from .sweep import run_tf, summarize  # noqa: E402
from .zoning import classify, score_classes, score_time_left, time_left_bucket  # noqa: E402


def _plan(params: dict, tf: int) -> tuple[dict, dict]:
    plan, med = {}, {}
    for key, w in params["winners"].items():
        if "method" not in w or int(w["tf"]) != tf:
            continue
        cells = [(w["method"], w["W"])]
        if w.get("best_baseline"):
            cells.append((w["best_baseline"]["method"], w["best_baseline"]["W"]))
        cells.append(("quad", 8))  # no-sinusoid timing null, always scored alongside
        plan[w["series"]] = cells
        med[w["series"]] = params["apex_meta"][str(tf)][w["series"]]["median_period_train"]
    return plan, med


def _reproduce(pred: pd.DataFrame, frame: pd.DataFrame, series: str, method: str, W: int,
               med_period: float, n: int = 200, seed: int = 0, min_period: float = 4.0) -> float:
    """Recompute n random stored predictions in-process; max abs diff of dt_pred."""
    rng = np.random.default_rng(seed)
    sub = pred[(pred["series"] == series) & (pred["method"] == method) & (pred["W"] == W) & pred["ok"]]
    if sub.empty:
        return np.nan
    take = sub.sample(min(n, len(sub)), random_state=int(rng.integers(1 << 30)))
    y = frame[series].to_numpy(dtype=float)
    diffs = []
    for _, r in take.iterrows():
        t = int(r["row"])
        kw = {"period": med_period} if method == "median_period" else {}
        p = fitters.predict(method, y[t - W + 1: t + 1], min_period=min_period, **kw)
        diffs.append(abs(p.dt - r["dt_pred"]) + abs(p.val - r["val_pred"]))
    return float(np.max(diffs))


def _plot_series(frame, series, pred, labels, tf, out_dir, n_show=240):
    win = pred[pred["method"].isin(fitters.FITTED)]
    if win.empty:
        return
    rows = win["row"].to_numpy()
    end = int(rows.max())
    start = max(end - n_show, int(rows.min()))
    seg = win[(win["row"] >= start) & (win["row"] <= end) & win["ok"]]
    y = frame[series].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(start, end + 1), y[start:end + 1], color="#333", lw=1, label=series)
    apex_rows = np.where(labels["prev_apex_dt"].to_numpy() == 0)[0]
    apex_rows = apex_rows[(apex_rows >= start) & (apex_rows <= end)]
    ax.scatter(apex_rows, y[apex_rows], color="#1f77b4", s=25, zorder=3, label="true apex")
    ax.scatter(seg["row"] + seg["dt_pred"], seg["val_pred"], color="#d62728", s=6, alpha=0.5,
               label=f"predicted apex ({seg['method'].iloc[0]} W{int(seg['W'].iloc[0])})")
    ax.set_title(f"tf{tf} {series}: real series, true apexes, predicted next apex from each row")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"series_tf{tf}_{series}.png"), dpi=110)
    plt.close(fig)


def _plot_dev(summary_by_lead: pd.DataFrame, tf: int, series: str, out_dir: str):
    sub = summary_by_lead[(summary_by_lead["tf"] == tf) & (summary_by_lead["series"] == series)]
    if sub.empty:
        return
    order = ["1", "2", "3-4", "5-8", "9+"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for (m, W), g in sub.groupby(["method", "W"]):
        g = g.set_index("lead").reindex(order)
        ax[0].plot(order, g["med_abs_dt"], marker="o", label=f"{m} W{W}")
        ax[1].plot(order, g["hit_t"], marker="o", label=f"{m} W{W}")
    ax[0].set_title(f"tf{tf} {series}: median |Δt| by true lead")
    ax[0].set_ylabel("candles")
    ax[1].set_title("P(|Δt| ≤ 1) by true lead")
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"deviation_tf{tf}_{series}.png"), dpi=110)
    plt.close(fig)


def _by_lead(pred: pd.DataFrame) -> pd.DataFrame:
    ok = pred[pred["ok"]].copy()
    ok["sign_hit"] = (ok["sign_pred"].to_numpy() == ok["sign_true"].to_numpy()).astype(float)
    g = ok.groupby(["tf", "series", "method", "W", "lead"], sort=False, observed=True)
    return g.agg(n=("d_t", "size"), med_abs_dt=("d_t", lambda s: s.abs().median()),
                 hit_t=("d_t", lambda s: (s.abs() <= 1.0).mean()),
                 sign_acc=("sign_hit", "mean"),
                 med_abs_dv=("d_v", lambda s: s.abs().median())).reset_index()


def run(params_path: str, dataset: str, workers: int, split: str | None) -> None:
    params = json.load(open(params_path))
    prom = params["prominence_frac"]
    tag = f"{dataset}_{split or 'all'}_p{prom:g}".replace(".", "") + params.get("tag", "")
    sdir = config.sidecar_dir(dataset)
    plot_dir = os.path.join(sdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    scores, tls, devs, leads = [], [], [], []
    gates = {"reproduction_max_abs_diff": {}, "class_share": {}, "zoning_grid_edge": {},
             "params_sha256": hashlib.sha256(open(params_path, "rb").read()).hexdigest(),
             "params_path": params_path, **stamp(dataset)}
    for tf in config.TFS:
        plan, med = _plan(params, tf)
        if not plan:
            continue
        pred, _, _ = run_tf(tf, dataset, prom, 1, workers, None, plan=plan, median_periods=med,
                            min_period=params.get("min_period", 4.0))
        if split:
            pred = pred[pred["split"] == split].reset_index(drop=True)
        frame = pd.read_pickle(os.path.join(sdir, f"series_tf{tf}.pkl"))
        devs.append(summarize(pred))
        bl = _by_lead(pred)
        leads.append(bl)
        for series, cells in plan.items():
            z = params["zoning"].get(f"{tf}:{series}")
            if not z or z["a_min"] is None:
                continue
            labels, _ = label_series(frame, series, prominence_frac=prom)
            wp = pred[(pred["series"] == series) & (pred["method"] == cells[0][0]) & (pred["W"] == cells[0][1])]
            wp = wp.reset_index(drop=True)
            cls = classify(wp, z["a_min"], z["p_lo"], z["p_hi"])
            tab = score_classes(wp, cls, labels, frame, tf)
            tab.insert(1, "series", series)
            tab.insert(2, "cell", f"{cells[0][0]} W{cells[0][1]} a{z['a_min']}")
            scores.append(tab)
            # controls: trivial slope-sign classes on the same rows (no fit at all)
            y = frame[series].to_numpy(dtype=float)
            r = wp["row"].to_numpy()
            for lag in (1, 3):
                ctrl = np.sign(y[r] - y[r - lag]).astype(int)
                ctab = score_classes(wp, ctrl, labels, frame, tf)
                ctab.insert(1, "series", series)
                ctab.insert(2, "cell", f"ctrl_slope{lag}")
                scores.append(ctab)
            # control 2: the frozen class restricted to the same share, but direction from the slope
            ctrl = np.where(cls != 0, np.sign(y[r] - y[r - 1]), 0).astype(int)
            ctab = score_classes(wp, ctrl, labels, frame, tf)
            ctab.insert(1, "series", series)
            ctab.insert(2, "cell", "ctrl_moving_slope1")
            scores.append(ctab)
            tl = score_time_left(wp, cls, labels, tf)
            tl.insert(1, "series", series)
            tls.append(tl)
            gates["class_share"][f"{tf}:{series}"] = {
                "up": float((cls == 1).mean()), "down": float((cls == -1).mean()), "neutral": float((cls == 0).mean())}
            gates["zoning_grid_edge"][f"{tf}:{series}"] = z.get("grid_edge")
            gates["reproduction_max_abs_diff"][f"{tf}:{series}"] = _reproduce(
                pred, frame, series, cells[0][0], cells[0][1], med[series],
                min_period=params.get("min_period", 4.0))
            _plot_series(frame, series, pred[pred["series"] == series], labels, tf, plot_dir)
            _plot_dev(bl, tf, series, plot_dir)
        print(f"[{dataset}] tf{tf} scored", flush=True)
    pd.concat(scores, ignore_index=True).to_csv(os.path.join(sdir, f"scores_{tag}.csv"), index=False)
    pd.concat(tls, ignore_index=True).to_csv(os.path.join(sdir, f"time_left_{tag}.csv"), index=False)
    pd.concat(devs, ignore_index=True).to_csv(os.path.join(sdir, f"deviation_{tag}.csv"), index=False)
    pd.concat(leads, ignore_index=True).to_csv(os.path.join(sdir, f"deviation_by_lead_{tag}.csv"), index=False)
    with open(os.path.join(sdir, f"gates_{tag}.json"), "w") as fh:
        json.dump(gates, fh, indent=2, default=float)
    print(json.dumps(gates["reproduction_max_abs_diff"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--dataset", default="oos2m", choices=list(config.DATASETS))
    ap.add_argument("--split", default=None, help="2y only: train | val")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    run(args.params, args.dataset, args.workers, args.split)


if __name__ == "__main__":
    main()
