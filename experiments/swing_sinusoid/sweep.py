"""Deviation sweep on the 2y TRAIN split: method × window × series × tf.

    python -m experiments.swing_sinusoid.sweep [--tfs 240] [--stride15 3] [--workers 12]
                                               [--prominence 0.5] [--rows-limit N]

For every closed row t (train split, window available, next apex known) each
method predicts the next apex from y[t-W+1 .. t]; the prediction is compared to
the ground-truth apex (apex.py, full-series extrema). Writes per tf:
  <2y sidecar>/predictions_tf{tf}_p{prom}.pkl    long table (one row per prediction)
  <2y sidecar>/deviation_tf{tf}_p{prom}.csv       summary per (series, method, W)
and, over all tfs, swing_sinusoid_params.json with the frozen winner per (tf, series).
Selection uses train rows only; the val split is written to the same prediction
table (split column) for the zoning task but never used for selection here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from . import config, fitters
from .apex import label_series
from .provenance import stamp

LEAD_BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)]
LEAD_MAX_SELECT = 8      # selection looks at rows whose true apex is ≤ 8 candles ahead
MIN_OK = 200             # minimum successful predictions for a cell to be eligible
HIT_T = 1.0              # |Δt| ≤ 1 candle counts as a time hit (locked decision 6)
HIT_V = 0.5              # |Δv| ≤ 0.5 series-std counts as a level hit

# ---------------------------------------------------------------- worker state
_Y: np.ndarray | None = None
_PERIOD: float = np.nan
_MIN_PERIOD: float = 4.0
MIN_COVERAGE = 0.7       # a cell must predict on ≥ 70 % of its rows to be eligible


def _init(y: np.ndarray, period: float, min_period: float = 4.0) -> None:
    global _Y, _PERIOD, _MIN_PERIOD
    _Y, _PERIOD, _MIN_PERIOD = y, period, min_period


PRED_COLS = ["row", "W", "method_i", "ok", "dt_pred", "val_pred", "sign_pred", "amp", "period_fit",
             "phase", "resid_std", "slope"]


def _work(args) -> np.ndarray:
    """Predictions for `rows` × `windows` × `methods` as a (n, 12) float array (PRED_COLS)."""
    rows, windows, methods = args
    out = []
    for t in rows:
        for W in windows:
            if t - W + 1 < 0:
                continue
            win = _Y[t - W + 1: t + 1]
            if np.isnan(win).any():
                continue
            for m in methods:
                kw = {"period": _PERIOD, "min_period": _MIN_PERIOD} if m == "median_period" else {"min_period": _MIN_PERIOD}
                p = fitters.predict(m, win, **kw)
                out.append((t, W, fitters.METHODS.index(m), p.ok, p.dt, p.val, p.sign, p.amp, p.period,
                            p.phase, p.resid_std, p.slope))
    return np.asarray(out, dtype=float).reshape(-1, len(PRED_COLS))


def _lead_bucket(lead: np.ndarray) -> np.ndarray:
    out = np.full(len(lead), "", dtype=object)
    for lo, hi in LEAD_BUCKETS:
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 8 else f"{lo}+")
        out[(lead >= lo) & (lead <= hi)] = lab
    return out


def run_tf(tf: int, dataset: str, prominence: float, stride: int, workers: int,
           rows_limit: int | None, windows=config.WINDOWS, methods=fitters.METHODS,
           plan: dict | None = None, median_periods: dict | None = None, min_period: float = 4.0,
           ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Predictions for every (series, method, W) — or, with `plan`, only the frozen
    {series: (method, W)} cells (scoring mode). `median_periods` overrides the
    train-derived baseline period (needed when scoring a dataset ≠ 2y)."""
    sdir = config.sidecar_dir(dataset)
    frame = pd.read_pickle(os.path.join(sdir, f"series_tf{tf}.pkl"))
    train_mask = frame.index <= pd.Timestamp(config.TRAIN_END)
    preds, metas, summaries = [], {}, []
    for series in config.SERIES:
        if plan is not None and series not in plan:
            continue
        s_windows, s_methods = windows, methods
        pairs = None
        if plan is not None:  # list of (method, W) cells to compute for this series
            pairs = [(m, int(W)) for m, W in plan[series]]
            s_windows = sorted({W for _, W in pairs})
        labels, meta = label_series(frame, series, prominence_frac=prominence)
        y = frame[series].to_numpy(dtype=float)
        std = meta["std"]
        # median FULL period on train rows only (baseline parameter, frozen)
        # half-swing lengths on train: distance between consecutive apexes ≤ TRAIN_END
        apex_rows = np.where((labels["prev_apex_dt"].to_numpy() == 0) & train_mask)[0]
        med_period = 2.0 * float(np.median(np.diff(apex_rows))) if len(apex_rows) > 2 else np.nan
        if median_periods is not None:
            med_period = float(median_periods[series])
        meta["median_period_train"] = med_period
        metas[series] = meta
        valid = ~labels["next_apex_dt"].isna().to_numpy()
        rows = np.where(valid)[0]
        rows = rows[rows >= min(s_windows) - 1]
        if stride > 1:
            rows = rows[::stride]
        if rows_limit:
            rows = rows[:rows_limit]
        chunks = np.array_split(rows, max(1, min(len(rows) // 50, workers * 8)))
        t0 = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init, initargs=(y, med_period, min_period)) as pool:
            if pairs is None:
                jobs = [(c, s_windows, list(s_methods)) for c in chunks if len(c)]
            else:
                jobs = [(c, [W], [m]) for c in chunks if len(c) for m, W in pairs]
            parts = pool.map(_work, jobs)
        arr = np.vstack([p for p in parts if len(p)])
        del parts
        df = pd.DataFrame(arr, columns=PRED_COLS)
        del arr
        df["row"] = df["row"].astype(np.int32)
        df["W"] = df["W"].astype(np.int16)
        df["ok"] = df["ok"].astype(bool)
        df["method"] = pd.Categorical.from_codes(df.pop("method_i").astype(int), list(fitters.METHODS))
        for col in ("dt_pred", "val_pred", "amp", "period_fit", "phase", "resid_std", "slope"):
            df[col] = df[col].astype(np.float32)
        df["sign_pred"] = df["sign_pred"].astype(np.int8)
        df["tf"], df["series"] = np.int16(tf), series
        df["ts"] = frame.index.to_numpy()[df["row"].to_numpy()]
        df["split"] = np.where(train_mask[df["row"].to_numpy()], "train", "val")
        lab = labels.iloc[df["row"].to_numpy()]
        df["dt_true"] = lab["next_apex_dt"].to_numpy()
        df["val_true"] = lab["next_apex_val"].to_numpy()
        df["sign_true"] = lab["next_apex_sign"].to_numpy()
        df["y_now"] = y[df["row"].to_numpy()]
        df["d_t"] = df["dt_pred"] - df["dt_true"]
        df["d_v"] = (df["val_pred"] - df["val_true"]) / std
        df["lead"] = _lead_bucket(df["dt_true"].to_numpy())
        if plan is None:
            # selection is per (tf, series) on train rows → keep only the winner + best baseline cells
            s_sum = summarize(df[df["split"] == "train"])
            summaries.append(s_sum)
            win = select(s_sum).get(f"{tf}:{series}", {})
            keep = np.zeros(len(df), dtype=bool)
            cells = []
            if "method" in win:
                cells.append((win["method"], win["W"]))
                if win.get("best_baseline"):
                    cells.append((win["best_baseline"]["method"], win["best_baseline"]["W"]))
            for m, W in cells:
                keep |= (df["method"].to_numpy() == m) & (df["W"].to_numpy() == W)
            df = df[keep].reset_index(drop=True)
        preds.append(df)
        print(f"[tf{tf}] {series}: rows={len(rows)} kept={len(df)} apexes={meta['n_apex']} "
              f"medP={med_period:.1f} {time.time() - t0:.0f}s", flush=True)
    pred = pd.concat(preds, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True) if summaries else summarize(pred[pred["split"] == "train"])
    return pred, summary, metas


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    """Per (series, method, W) deviation summary; selection rows have lead ≤ LEAD_MAX_SELECT."""
    g = pred.groupby(["tf", "series", "method", "W"], sort=False, observed=True)
    rows = []
    for key, grp in g:
        ok = grp[grp["ok"]]
        sel = ok[ok["dt_true"] <= LEAD_MAX_SELECT]
        rows.append({
            "tf": key[0], "series": key[1], "method": key[2], "W": key[3],
            "n": len(grp), "n_ok": len(ok), "coverage": len(ok) / max(len(grp), 1),
            "n_sel": len(sel),
            "med_abs_dt": float(sel["d_t"].abs().median()) if len(sel) else np.nan,
            "mean_abs_dt": float(sel["d_t"].abs().mean()) if len(sel) else np.nan,
            "bias_dt": float(sel["d_t"].mean()) if len(sel) else np.nan,
            "hit_t": float((sel["d_t"].abs() <= HIT_T).mean()) if len(sel) else np.nan,
            "sign_acc": float((sel["sign_pred"] == sel["sign_true"]).mean()) if len(sel) else np.nan,
            "med_abs_dv": float(sel["d_v"].abs().median()) if len(sel) else np.nan,
            "hit_v": float((sel["d_v"].abs() <= HIT_V).mean()) if len(sel) else np.nan,
            "med_abs_dt_all": float(ok["d_t"].abs().median()) if len(ok) else np.nan,
            "hit_t_all": float((ok["d_t"].abs() <= HIT_T).mean()) if len(ok) else np.nan,
        })
    return pd.DataFrame(rows)


def by_lead(pred: pd.DataFrame) -> pd.DataFrame:
    ok = pred[pred["ok"]].copy()
    ok["sign_hit"] = (ok["sign_pred"].to_numpy() == ok["sign_true"].to_numpy()).astype(float)
    g = ok.groupby(["tf", "series", "method", "W", "lead"], sort=False, observed=True)
    return g.agg(n=("d_t", "size"), med_abs_dt=("d_t", lambda s: s.abs().median()),
                 hit_t=("d_t", lambda s: (s.abs() <= HIT_T).mean()),
                 sign_acc=("sign_hit", "mean"),
                 med_abs_dv=("d_v", lambda s: s.abs().median())).reset_index()


def select(summary: pd.DataFrame) -> dict:
    """Winner per (tf, series): best fitted cell by med_abs_dt, tie-break hit_t; margin vs baselines."""
    out = {}
    for (tf, series), grp in summary.groupby(["tf", "series"]):
        elig = grp[(grp["n_sel"] >= MIN_OK) & (grp["coverage"] >= MIN_COVERAGE)]
        fitted = elig[elig["method"].isin(fitters.FITTED)]
        base = elig[elig["method"].isin(fitters.BASELINES)]
        if fitted.empty:
            out[f"{tf}:{series}"] = {"status": "no eligible fitted cell"}
            continue
        f = fitted.sort_values(["med_abs_dt", "hit_t"], ascending=[True, False]).iloc[0]
        b = base.sort_values(["med_abs_dt", "hit_t"], ascending=[True, False]).iloc[0] if not base.empty else None
        out[f"{tf}:{series}"] = {
            "tf": int(tf), "series": series, "method": f["method"], "W": int(f["W"]),
            "n_sel": int(f["n_sel"]), "coverage": float(f["coverage"]),
            "med_abs_dt": float(f["med_abs_dt"]), "hit_t": float(f["hit_t"]),
            "sign_acc": float(f["sign_acc"]), "med_abs_dv": float(f["med_abs_dv"]), "hit_v": float(f["hit_v"]),
            "best_baseline": None if b is None else {
                "method": b["method"], "W": int(b["W"]), "med_abs_dt": float(b["med_abs_dt"]),
                "hit_t": float(b["hit_t"]), "sign_acc": float(b["sign_acc"]), "hit_v": float(b["hit_v"])},
            "margin_med_abs_dt": None if b is None else float(b["med_abs_dt"] - f["med_abs_dt"]),
            "margin_hit_t": None if b is None else float(f["hit_t"] - b["hit_t"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2y")
    ap.add_argument("--tfs", type=int, nargs="*", default=config.TFS)
    ap.add_argument("--stride15", type=int, default=2, help="row stride for tf 15 (compute budget)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prominence", type=float, default=0.5)
    ap.add_argument("--rows-limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-period", type=float, default=4.0, help="period floor for the fitted methods (candles)")
    args = ap.parse_args()
    sdir = config.sidecar_dir(args.dataset)
    ptag = f"p{args.prominence:g}".replace(".", "")
    if args.min_period != 4.0:
        args.tag = f"_mp{args.min_period:g}{args.tag}"
    all_summary, all_meta = [], {}
    for tf in args.tfs:
        stride = args.stride15 if tf == 15 else 1
        pred, summary, metas = run_tf(tf, args.dataset, args.prominence, stride, args.workers, args.rows_limit,
                                      min_period=args.min_period)
        pred.to_pickle(os.path.join(sdir, f"predictions_tf{tf}_{ptag}{args.tag}.pkl"))
        summary.to_csv(os.path.join(sdir, f"deviation_tf{tf}_{ptag}{args.tag}.csv"), index=False)
        by_lead(pred[pred["split"] == "train"]).to_csv(
            os.path.join(sdir, f"deviation_by_lead_tf{tf}_{ptag}{args.tag}.csv"), index=False)
        all_summary.append(summary)
        all_meta[str(tf)] = metas
    summary = pd.concat(all_summary, ignore_index=True)
    winners = select(summary)
    params = stamp(args.dataset, prominence_frac=args.prominence, distance=2, windows=list(config.WINDOWS),
                   train_end=config.TRAIN_END, stride15=args.stride15, lead_max_select=LEAD_MAX_SELECT,
                   min_ok=MIN_OK, min_coverage=MIN_COVERAGE, min_period=args.min_period, hit_t=HIT_T, hit_v=HIT_V,
                   tag=args.tag, apex_meta=all_meta, winners=winners)
    path = os.path.join(sdir, f"swing_sinusoid_params_{ptag}{args.tag}.json")
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2, default=float)
    print(f"params → {path}")
    print(json.dumps(winners, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
