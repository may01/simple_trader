"""Fit the universal and the class-conditional candle-bound Ridge models on 2y.

    python -m experiments.cb_class_bounds.fit [--tfs 15,60,240]

Per (tf, side): hyper-parameters (MIN_N, α) selected on the internal 2y split (train → val MAE),
then every variant refit on the full 2y and frozen to ``<2y dir>/cb_class_bounds/models.json``.
Also writes ``val_pred_tf{tf}_{side}.pkl`` (val predictions of the train-half fit, for the 2y-val
tier-A tables) and ``pred_2y_tf{tf}_{side}.pkl`` (full-2y in-sample predictions, for the replay).
Variants: uni · families mc/zc/cz/ec (own tf) · cross families {fam}_{H} (forming higher class) ·
int_{fam} (universal slopes + per-class intercept) · shuf_{fam}_s{k} (classes block-shuffled) ·
avg4 / avg_all / avg4_shuf_s{k} (averages of family predictions).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

from indicators.candle_bounds.params import FEATURES

from . import config

NF = len(FEATURES)


# ------------------------------------------------------------------ ridge (closed form, == sklearn StandardScaler+Ridge)
def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    Z = (X - mean) / scale
    yc = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (y - yc))
    return {"mean": mean, "scale": scale, "coef": coef, "intercept": float(yc), "n_train": int(len(y))}


def ridge_predict(p: dict, X: np.ndarray) -> np.ndarray:
    return ((X - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"]


# ------------------------------------------------------------------ class models
def fit_class_model(X, y, c, uni: dict, min_n: int, alpha: float, values) -> dict:
    cells = {}
    for v in values:
        m = c == v
        n = int(m.sum())
        if n >= min_n:
            p = ridge_fit(X[m], y[m], alpha)
            p["fallback"] = False
        else:
            p = dict(uni)
            p["fallback"] = True
            p["n_train"] = n
        cells[int(v)] = p
    return {"min_n": min_n, "alpha": alpha, "cells": cells}


def predict_class_model(model: dict, X, c) -> np.ndarray:
    out = np.full(len(X), np.nan)
    for v, p in model["cells"].items():
        m = c == int(v)
        if m.any():
            out[m] = ridge_predict(p, X[m])
    return out


def fit_intercept_model(X, y, c, uni: dict, min_n: int, values) -> dict:
    base = ridge_predict(uni, X)
    cells = {}
    for v in values:
        m = c == v
        n = int(m.sum())
        p = dict(uni)
        if n >= min_n:
            p["intercept"] = uni["intercept"] + float(np.mean(y[m] - base[m]))
            p["fallback"] = False
        else:
            p["fallback"] = True
        p["n_train"] = n
        cells[int(v)] = p
    return {"min_n": min_n, "alpha": uni["alpha"], "cells": cells}


def block_shuffle(c: np.ndarray, index: pd.DatetimeIndex, seed: int) -> np.ndarray:
    days = index.floor(config.BLOCK).to_numpy()
    _, starts = np.unique(days, return_index=True)
    starts = np.sort(starts)
    chunks = np.split(np.asarray(c), starts[1:])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    return np.concatenate([chunks[i] for i in order])


def band(y, pred) -> float:
    r = y - pred
    r = r[np.isfinite(r)]
    return float(np.std(r, ddof=1))


def variant_specs(tf: int) -> list[tuple[str, str, str]]:
    """(variant, kind, class column) in analysis order. kind ∈ class / int / shuf."""
    fams = [(f, f"cls_{f}") for f in config.FAMILIES]
    for H in config.HIGHER[tf]:
        fams += [(f"{f}_{H}", f"cls_{f}_{H}") for f in config.FAMILIES]
    out = [(v, "class", col) for v, col in fams]
    out += [(f"int_{v}", "int", col) for v, col in fams if "_" not in v]
    for k in config.SHUFFLE_SEEDS:
        out += [(f"shuf_{v}_s{k}", "shuf", col) for v, col in fams if "_" not in v]
    return out


def family_values(col: str):
    fam = col.split("_")[1]
    return config.CLASS_VALUES[fam]


# ------------------------------------------------------------------ driver per (tf, side)
def load_closed(dataset: str, tf: int, side: str) -> pd.DataFrame:
    cf = pd.read_pickle(os.path.join(config.sidecar_dir(dataset), f"closed_tf{tf}_{side}.pkl"))
    return cf.dropna(subset=FEATURES + ["target"])


def fit_tf_side(tf: int, side: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cf = load_closed("2y", tf, side)
    tr = cf.index <= pd.Timestamp(config.TRAIN_END)
    X, y = cf[FEATURES].to_numpy(float), cf["target"].to_numpy(float)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[~tr], y[~tr]
    sel: dict = {}
    frozen: dict = {}
    val_pred = pd.DataFrame({"y": yva}, index=cf.index[~tr])
    val_band: dict = {}
    full_pred = pd.DataFrame({"y": y}, index=cf.index)

    # universal
    grid = {a: float(np.mean(np.abs(yva - ridge_predict(ridge_fit(Xtr, ytr, a), Xva)))) for a in config.ALPHA_GRID}
    a_uni = min(grid, key=grid.get)
    sel["uni"] = {"grid": {str(k): v for k, v in grid.items()}, "alpha": a_uni}
    uni_tr = ridge_fit(Xtr, ytr, a_uni); uni_tr["alpha"] = a_uni
    val_pred["uni"] = ridge_predict(uni_tr, Xva); val_band["uni"] = band(ytr, ridge_predict(uni_tr, Xtr))
    uni = ridge_fit(X, y, a_uni); uni["alpha"] = a_uni
    uni["band_pct"] = band(y, ridge_predict(uni, X))
    frozen["uni"] = uni
    full_pred["uni"] = ridge_predict(uni, X)

    cls_cols = {}
    for variant, kind, col in variant_specs(tf):
        c_all = cf[col].to_numpy(int)
        if kind == "shuf":
            seed = int(variant.rsplit("_s", 1)[1])
            c_all = block_shuffle(c_all, cf.index, seed)
        cls_cols[variant] = c_all
        ctr, cva = c_all[tr], c_all[~tr]
        values = family_values(col)
        table = {}
        if kind == "int":
            for mn in config.MIN_N_GRID:
                m = fit_intercept_model(Xtr, ytr, ctr, uni_tr, mn, values)
                table[(mn, a_uni)] = float(np.mean(np.abs(yva - predict_class_model(m, Xva, cva))))
        else:
            for mn in config.MIN_N_GRID:
                for a in config.ALPHA_GRID:
                    m = fit_class_model(Xtr, ytr, ctr, uni_tr, mn, a, values)
                    table[(mn, a)] = float(np.mean(np.abs(yva - predict_class_model(m, Xva, cva))))
        best = min(table, key=table.get)
        mn, a = best
        sel[variant] = {"grid": {f"{k[0]}/{k[1]}": v for k, v in table.items()}, "min_n": mn, "alpha": a}
        if kind == "int":
            m_tr = fit_intercept_model(Xtr, ytr, ctr, uni_tr, mn, values)
            m_full = fit_intercept_model(X, y, c_all, uni, mn, values)
        else:
            m_tr = fit_class_model(Xtr, ytr, ctr, uni_tr, mn, a, values)
            m_full = fit_class_model(X, y, c_all, uni, mn, a, values)
        val_pred[variant] = predict_class_model(m_tr, Xva, cva)
        val_band[variant] = band(ytr, predict_class_model(m_tr, Xtr, ctr))
        pf = predict_class_model(m_full, X, c_all)
        for v, p in m_full["cells"].items():
            mm = c_all == v
            p["band_pct"] = band(y[mm], pf[mm]) if mm.sum() > 2 and not p["fallback"] else uni["band_pct"]
        m_full["band_pct"] = band(y, pf)
        frozen[variant] = m_full
        full_pred[variant] = pf

    # averages
    own = list(config.FAMILIES)
    cross = [f"{f}_{H}" for H in config.HIGHER[tf] for f in config.FAMILIES]
    avgs = {"avg4": own, "avg_all": own + cross}
    for k in config.SHUFFLE_SEEDS:
        avgs[f"avg4_shuf_s{k}"] = [f"shuf_{f}_s{k}" for f in own]
    for name, members in avgs.items():
        val_pred[name] = val_pred[members].mean(axis=1)
        val_band[name] = float("nan")   # train-half band of an average not stored (val tables use coverage of members)
        full_pred[name] = full_pred[members].mean(axis=1)
        frozen[name] = {"members": members, "band_pct": band(y, full_pred[name].to_numpy())}
    val_pred.attrs["band"] = val_band
    return {"selection": sel, "frozen": frozen, "n_train_half": int(tr.sum()), "n_val": int((~tr).sum()), "n_full": int(len(y))}, val_pred, full_pred


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/code", text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="15,60,240")
    a = ap.parse_args()
    tfs = [int(t) for t in a.tfs.split(",")]
    out_dir = config.sidecar_dir("2y")
    models = {"features": list(FEATURES), "train": "2y_link_usdt parts", "train_end_internal": config.TRAIN_END,
              "grids": {"min_n": config.MIN_N_GRID, "alpha": config.ALPHA_GRID, "seeds": config.SHUFFLE_SEEDS},
              "provenance": {"git_sha": git_sha(), "fitted": pd.Timestamp.utcnow().isoformat(),
                             "spec": "external/docs/superpowers/experiment/cb_class_bounds.md"},
              "models": {}}
    for tf in tfs:
        for side in config.SIDES:
            res, val_pred, full_pred = fit_tf_side(tf, side)
            key = f"{tf}_{side}"
            models["models"][key] = res
            val_pred.to_pickle(os.path.join(out_dir, f"val_pred_tf{tf}_{side}.pkl"))
            full_pred.to_pickle(os.path.join(out_dir, f"pred_2y_tf{tf}_{side}.pkl"))
            u = res["frozen"]["uni"]
            line = " ".join(f"{v}:{res['selection'][v]['min_n']}/{res['selection'][v]['alpha']}"
                            for v in ("mc", "zc", "cz", "ec"))
            print(f"[fit] {key}: n {res['n_full']} (train {res['n_train_half']} val {res['n_val']}) uni α {u['alpha']} band {u['band_pct']:.4f} | sel {line}", flush=True)
    with open(os.path.join(out_dir, "models.json"), "w") as f:
        json.dump(_jsonable(models), f)
    print(f"[fit] wrote {out_dir}/models.json")


if __name__ == "__main__":
    main()
