"""Tier A — bound-prediction accuracy of every variant vs the universal model.

    python -m experiments.cb_class_bounds.score [--skip-oos2m]

2y-val tables from ``val_pred_tf*.pkl`` (train-half fit); oos2m tables by predicting the oos2m
closed frames with the frozen full-2y models (scored once). Per variant: r², MAE, ΔMAE % vs uni,
paired bootstrap CI90 of the per-candle |error| difference, band, ±band coverage. Per class cell:
n, MAE uni / class, Δ %. Writes ``tierA_*.md`` + ``tierA.json`` and ``pred_oos2m_tf{tf}_{side}.pkl``
(predictions + per-row bands, consumed by the replay) under the respective sidecar dirs.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from indicators.candle_bounds.params import FEATURES

from . import config
from .fit import block_shuffle, predict_class_model, ridge_predict

NBOOT = 1000


def load_models() -> dict:
    with open(os.path.join(config.sidecar_dir("2y"), "models.json")) as f:
        m = json.load(f)
    for key, res in m["models"].items():
        for v, fz in res["frozen"].items():
            if "cells" in fz:
                fz["cells"] = {int(k): _arr(p) for k, p in fz["cells"].items()}
            elif "coef" in fz:
                _arr(fz)
    return m


def _arr(p: dict) -> dict:
    for k in ("mean", "scale", "coef"):
        if k in p:
            p[k] = np.asarray(p[k], dtype=float)
    return p


def class_col(variant: str) -> str | None:
    v = variant
    if v.startswith("shuf_"):
        v = v[5:].rsplit("_s", 1)[0]
    if v.startswith("int_"):
        v = v[4:]
    if v in ("uni",) or v.startswith("avg"):
        return None
    return f"cls_{v}"


def variant_classes(variant: str, cf: pd.DataFrame) -> np.ndarray | None:
    col = class_col(variant)
    if col is None:
        return None
    c = cf[col].to_numpy(int)
    if variant.startswith("shuf_"):
        c = block_shuffle(c, cf.index, int(variant.rsplit("_s", 1)[1]))
    return c


def predict_all(frozen: dict, cf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions and per-row bands for every frozen variant on a closed frame."""
    X = cf[FEATURES].to_numpy(float)
    pred = pd.DataFrame({"y": cf["target"].to_numpy(float)}, index=cf.index)
    bands = pd.DataFrame(index=cf.index)
    for v, fz in frozen.items():
        if "members" in fz:
            continue
        if v == "uni":
            pred[v] = ridge_predict(fz, X)
            bands[v] = fz["band_pct"]
        else:
            c = variant_classes(v, cf)
            pred[v] = predict_class_model(fz, X, c)
            b = np.full(len(cf), np.nan)
            for val, p in fz["cells"].items():
                b[c == val] = p["band_pct"]
            bands[v] = b
    for v, fz in frozen.items():
        if "members" in fz:
            pred[v] = pred[fz["members"]].mean(axis=1)
            bands[v] = fz["band_pct"]
    return pred, bands


def paired_boot(d: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """mean, CI90 of the mean, P(mean < 0) by bootstrap over candles."""
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 5)), float(np.percentile(means, 95)), float(np.mean(means < 0))


def metrics(pred: pd.DataFrame, bands: pd.DataFrame | None) -> dict:
    y = pred["y"].to_numpy(float)
    eu = np.abs(y - pred["uni"].to_numpy(float))
    out = {}
    sst = float(np.sum((y - y.mean()) ** 2))
    for v in pred.columns:
        if v == "y":
            continue
        p = pred[v].to_numpy(float)
        e = np.abs(y - p)
        row = {"r2": 1 - float(np.sum((y - p) ** 2)) / sst, "mae": float(e.mean()),
               "d_mae_pct": float((e.mean() / eu.mean() - 1) * 100), "n": int(len(y))}
        if v != "uni":
            m, lo, hi, pneg = paired_boot(e - eu)
            row.update({"d_mae": m, "ci_lo": lo, "ci_hi": hi, "p_better": pneg})
        if bands is not None and v in bands:
            b = bands[v].to_numpy(float)
            row["band"] = float(np.nanmean(b))
            row["cov"] = float(np.mean(e <= b))
        out[v] = row
    return out


def cell_table(pred: pd.DataFrame, cf: pd.DataFrame, variants: list[str]) -> list[dict]:
    y = pred["y"].to_numpy(float)
    eu = np.abs(y - pred["uni"].to_numpy(float))
    rows = []
    for v in variants:
        c = variant_classes(v, cf)
        if c is None:
            continue
        e = np.abs(y - pred[v].to_numpy(float))
        for val in np.unique(c):
            m = c == val
            rows.append({"variant": v, "class": int(val), "n": int(m.sum()), "mae_uni": float(eu[m].mean()),
                         "mae": float(e[m].mean()), "d_pct": float((e[m].mean() / eu[m].mean() - 1) * 100),
                         "y_mean": float(y[m].mean()), "uni_mean": float(pred["uni"].to_numpy()[m].mean()),
                         "pred_mean": float(pred[v].to_numpy()[m].mean())})
    return rows


ORDER_OWN = ["uni", "mc", "zc", "cz", "ec", "avg4"]


def variant_order(tf: int) -> list[str]:
    cross = [f"{f}_{H}" for H in config.HIGHER[tf] for f in config.FAMILIES]
    out = ORDER_OWN + cross + (["avg_all"] if cross else [])
    out += [f"int_{f}" for f in config.FAMILIES]
    out += [f"shuf_{f}_s{k}" for f in config.FAMILIES for k in config.SHUFFLE_SEEDS]
    out += [f"avg4_shuf_s{k}" for k in config.SHUFFLE_SEEDS]
    return out


def md_table(res: dict, tf: int, side: str) -> list[str]:
    lines = [f"### tf {tf} {side}", "", "| variant | n | r² | MAE | ΔMAE % vs uni | ΔMAE CI90 | P(better) | band | cov ±band |", "|---|---|---|---|---|---|---|---|---|"]
    for v in variant_order(tf):
        if v not in res:
            continue
        r = res[v]
        ci = f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]" if "ci_lo" in r else "—"
        pb = f"{r['p_better']:.2f}" if "p_better" in r else "—"
        band = f"{r['band']:.4f}" if "band" in r else "—"
        cov = f"{r['cov']:.3f}" if "cov" in r else "—"
        lines.append(f"| {v} | {r['n']} | {r['r2']:.4f} | {r['mae']:.5f} | {r['d_mae_pct']:+.2f} | {ci} | {pb} | {band} | {cov} |")
    lines.append("")
    return lines


def md_cells(rows: list[dict], tf: int, side: str) -> list[str]:
    lines = [f"### tf {tf} {side}", "", "| variant | class | n | MAE uni | MAE class | Δ % | mean y | mean uni | mean class |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['class']} | {r['n']} | {r['mae_uni']:.5f} | {r['mae']:.5f} | {r['d_pct']:+.2f} | {r['y_mean']:+.4f} | {r['uni_mean']:+.4f} | {r['pred_mean']:+.4f} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-oos2m", action="store_true")
    a = ap.parse_args()
    models = load_models()
    out2y = config.sidecar_dir("2y")
    outoos = config.sidecar_dir("oos2m")
    allres = {"2yval": {}, "oos2m": {}, "cells_2yval": {}, "cells_oos2m": {}}
    md_val, md_oos, md_cv, md_co = ["# Tier A — 2y validation half (fit on 2023-01 → 2024-06, scored 2024-07 → 2024-12)", ""], \
        ["# Tier A — oos2m (frozen full-2y models, scored once)", ""], ["# Cells — 2y validation half", ""], ["# Cells — oos2m", ""]
    for tf in config.TFS:
        for side in config.SIDES:
            key = f"{tf}_{side}"
            frozen = models["models"][key]["frozen"]
            vp = pd.read_pickle(os.path.join(out2y, f"val_pred_tf{tf}_{side}.pkl"))
            vb = pd.DataFrame({v: b for v, b in vp.attrs.get("band", {}).items() if np.isfinite(b)}, index=vp.index)
            cf_val = pd.read_pickle(os.path.join(out2y, f"closed_tf{tf}_{side}.pkl")).dropna(subset=FEATURES + ["target"])
            cf_val = cf_val.loc[vp.index]
            r = metrics(vp, vb if len(vb.columns) else None)
            allres["2yval"][key] = r
            md_val += md_table(r, tf, side)
            cells = cell_table(vp, cf_val, [v for v in variant_order(tf) if class_col(v) and not v.startswith(("shuf_", "int_"))])
            allres["cells_2yval"][key] = cells
            md_cv += md_cells(cells, tf, side)
            print(f"[2yval] {key}: " + " ".join(f"{v} {r[v]['d_mae_pct']:+.2f}%" for v in ("mc", "zc", "cz", "ec", "avg4") if v in r), flush=True)
            if a.skip_oos2m:
                continue
            cf = pd.read_pickle(os.path.join(outoos, f"closed_tf{tf}_{side}.pkl")).dropna(subset=FEATURES + ["target"])
            pred, bands = predict_all(frozen, cf)
            pred.to_pickle(os.path.join(outoos, f"pred_oos2m_tf{tf}_{side}.pkl"))
            bands.to_pickle(os.path.join(outoos, f"band_oos2m_tf{tf}_{side}.pkl"))
            r = metrics(pred, bands)
            allres["oos2m"][key] = r
            md_oos += md_table(r, tf, side)
            cells = cell_table(pred, cf, [v for v in variant_order(tf) if class_col(v) and not v.startswith(("shuf_", "int_"))])
            allres["cells_oos2m"][key] = cells
            md_co += md_cells(cells, tf, side)
            print(f"[oos2m] {key}: n {r['uni']['n']} uni r² {r['uni']['r2']:.4f} | " + " ".join(f"{v} {r[v]['d_mae_pct']:+.2f}% (P {r[v]['p_better']:.2f})" for v in ("mc", "zc", "cz", "ec", "avg4", "avg_all") if v in r), flush=True)
    with open(os.path.join(out2y, "tierA_2yval.md"), "w") as f:
        f.write("\n".join(md_val))
    with open(os.path.join(out2y, "cells_2yval.md"), "w") as f:
        f.write("\n".join(md_cv))
    if not a.skip_oos2m:
        with open(os.path.join(outoos, "tierA_oos2m.md"), "w") as f:
            f.write("\n".join(md_oos))
        with open(os.path.join(outoos, "cells_oos2m.md"), "w") as f:
            f.write("\n".join(md_co))
    with open(os.path.join(out2y, "tierA.json"), "w") as f:
        json.dump(allres, f, default=float)
    print("[score] done")


if __name__ == "__main__":
    main()
