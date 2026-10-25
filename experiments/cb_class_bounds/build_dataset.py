"""Build per-(tf, side) closed-row frames with CB features, target and market classes.

    python -m experiments.cb_class_bounds.build_dataset --dataset 2y [--limit 2]
    python -m experiments.cb_class_bounds.build_dataset --dataset oos2m      # needs the 2y meta (zone stats)

Reads the wide frames one part at a time (read-only), keeps the closed rows of each tf with
the raw inputs of ``kernel.closed_frame`` plus the class inputs, and the FORMING values of
every higher tf at those minutes. After concatenation (continuous across parts) it derives
features, target d[k+1], own-tf classes (mc / zc / cz / ec) and forming higher-tf classes, and
writes ``<dataset dir>/cb_class_bounds/closed_tf{tf}_{side}.pkl`` + ``meta.json`` with the
reproduction gates against the stored class columns.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from indicators.candle_bounds import kernel
from indicators.candle_bounds.params import FEATURES

from . import classes as C
from . import config


def part_paths(dataset: str, limit: int | None = None) -> list[str]:
    d = config.DATASETS[dataset]
    if d["parts"]:
        paths = sorted(glob.glob(os.path.join(d["dir"], "df_with_indicators.part_*.pkl")))
    else:
        paths = [os.path.join(d["dir"], "df_with_indicators.pkl")]
    if not paths:
        raise FileNotFoundError(f"no frames under {d['dir']}")
    return paths[:limit] if limit else paths


def extract_part(df: pd.DataFrame, tf: int) -> pd.DataFrame:
    m = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    cols: dict[str, np.ndarray] = {}
    for c in config.RAW_OWN:
        cols[f"{tf}_{c}"] = df[f"{tf}_{c}"].to_numpy()[m]
    for c in config.REF_OWN:
        name = f"{tf}_{c}"
        if name in df.columns:
            cols[f"ref_{c}"] = df[name].to_numpy()[m]
    for H in config.HIGHER[tf]:
        for c in config.RAW_HIGHER:
            cols[f"{H}_{c}_f"] = df[f"{H}_{c}"].to_numpy()[m]
        for c in config.REF_HIGHER:
            name = f"{H}_{c}"
            if name in df.columns:
                cols[f"{H}_{c}_fref"] = df[name].to_numpy()[m]
    out = pd.DataFrame(cols, index=df.index[m])
    out[f"{tf}_is_closed"] = True
    return out


def _agree(a: np.ndarray, b: np.ndarray) -> float | None:
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(b)
    if ok.sum() == 0:
        return None
    return float(np.mean(np.asarray(a)[ok] == b[ok].astype(int)))


def own_classes(raw: pd.DataFrame, tf: int, zstats: dict) -> dict[str, np.ndarray]:
    rsi = raw[f"{tf}_rsi_ma8"].to_numpy(dtype=float)
    out = {
        "cls_mc": C.move_class(rsi, tf),
        "cls_zc": C.zone_class(rsi, *zstats[str(tf)]),
        "cls_ec": C.ema_slope_class_closed(raw[f"{tf}_ema_25"].to_numpy(dtype=float), tf),
        "cls_cz": C.cci_zone_closed(C.cci_ma8_closed(raw[f"{tf}_cci_14"].to_numpy(dtype=float)), tf),
    }
    return out


def forming_classes(raw: pd.DataFrame, tf: int, raw_h: pd.DataFrame, H: int, zstats: dict) -> dict[str, np.ndarray]:
    """Classes of the forming higher candle H at tf's closing minutes."""
    t = raw.index.to_numpy()
    ht = raw_h.index.to_numpy()
    lc = np.searchsorted(ht, t, side="left") - 1          # last H candle closed strictly before t
    valid = lc >= 0
    lci = np.maximum(lc, 0)
    rsi_f = raw[f"{H}_rsi_ma8_f"].to_numpy(dtype=float)
    rsi_c = raw_h[f"{H}_rsi_ma8"].to_numpy(dtype=float)
    d = np.where(valid, rsi_f - rsi_c[lci], np.nan)
    ema_f = raw[f"{H}_ema_25_f"].to_numpy(dtype=float)
    ema_c = raw_h[f"{H}_ema_25"].to_numpy(dtype=float)
    ema_prev = np.where(valid, ema_c[lci], np.nan)
    cci_c = raw_h[f"{H}_cci_14"].to_numpy(dtype=float)
    cci_f = raw[f"{H}_cci_14_f"].to_numpy(dtype=float)
    # SMA-8 forming = (sum of the 7 closed cci before + forming cci) / 8; NaN when fewer than 7 closed
    s7_all = pd.Series(cci_c).rolling(C.CCI_MA - 1).sum().to_numpy()      # sum cci[k-6..k] at k
    y_f = np.where(valid, (s7_all[lci] + cci_f) / C.CCI_MA, np.nan)
    y_closed = C.cci_ma8_closed(cci_c)
    return {
        f"cls_mc_{H}": C.move_class_from_diff(d, H),
        f"cls_zc_{H}": C.zone_class(rsi_f, *zstats[str(H)]),
        f"cls_ec_{H}": C.ema_slope_class_from(ema_f, ema_prev, H),
        f"cls_cz_{H}": C.cci_zone_forming(y_closed, lc, y_f, H),
        f"lc_{H}": lc,
    }


def finalize(raws: dict[int, pd.DataFrame], dataset: str, zstats: dict) -> dict:
    meta: dict = {"dataset": dataset, "tfs": {}, "zone_stats": zstats}
    out_dir = config.sidecar_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    for tf in config.TFS:
        raw = raws[tf]
        assert raw.index.is_monotonic_increasing and raw.index.is_unique, f"tf {tf}: closed index not clean"
        cls = own_classes(raw, tf, zstats)
        for H in config.HIGHER[tf]:
            cls.update(forming_classes(raw, tf, raws[H], H, zstats))
        info: dict = {"n_closed": int(len(raw)), "classes": {}, "agree": {}}
        for k, v in cls.items():
            if k.startswith("cls_"):
                vals, cnt = np.unique(v, return_counts=True)
                info["classes"][k] = {int(a): int(b) for a, b in zip(vals, cnt)}
        # reproduction gates against stored columns
        if "ref_move_class" in raw:
            info["agree"]["mc_vs_stored_move_class"] = _agree(cls["cls_mc"], raw["ref_move_class"].to_numpy())
        if "ref_ema_25_slope_class" in raw:
            info["agree"]["ec_vs_stored_ema_25_slope_class"] = _agree(cls["cls_ec"], raw["ref_ema_25_slope_class"].to_numpy())
        if "ref_zone_class" in raw:
            info["agree"]["zc_vs_stored_zone_class(stale)"] = _agree(cls["cls_zc"], raw["ref_zone_class"].to_numpy())
        for H in config.HIGHER[tf]:
            if f"{H}_move_class_fref" in raw:
                info["agree"][f"mc_{H}_forming_vs_stored_minute_move_class"] = _agree(cls[f"cls_mc_{H}"], raw[f"{H}_move_class_fref"].to_numpy())
            if f"{H}_ema_25_slope_class_fref" in raw:
                info["agree"][f"ec_{H}_forming_vs_stored_minute_slope_class"] = _agree(cls[f"cls_ec_{H}"], raw[f"{H}_ema_25_slope_class_fref"].to_numpy())
        for side in config.SIDES:
            cf = kernel.closed_frame(raw, tf, side)
            cf["target"] = cf["d"].shift(-1)
            for k, v in cls.items():
                cf[k] = v
            for c in raw.columns:
                if c.startswith("ref_"):
                    cf[c] = raw[c].to_numpy()
            cf.to_pickle(os.path.join(out_dir, f"closed_tf{tf}_{side}.pkl"))
            info[f"n_valid_{side}"] = int(cf.dropna(subset=FEATURES + ["target"]).shape[0])
        meta["tfs"][str(tf)] = info
        print(f"[{dataset}] tf {tf}: closed {len(raw)} valid {info['n_valid_high']} agree {info['agree']}", flush=True)
    return meta


def swing_sidecar_gate(dataset: str, meta: dict) -> None:
    """cz (closed) vs the swing-filter sidecar's {tf}_swing_cci_zone at the closed rows (oos2m only)."""
    p = os.path.join(config.DATASETS[dataset]["dir"], "df_with_cb_swing_filter.pkl")
    if not os.path.exists(p):
        return
    sc = pd.read_pickle(p)
    for tf in config.TFS:
        col = f"{tf}_swing_cci_zone"
        if col not in sc.columns:
            continue
        cf = pd.read_pickle(config.closed_path(dataset, tf).replace(".pkl", "_high.pkl"))
        ref = sc[col].reindex(cf.index)
        meta["tfs"][str(tf)]["agree"]["cz_vs_swing_sidecar_cci_zone"] = _agree(cf["cls_cz"].to_numpy(), ref.to_numpy())
    del sc


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/code", text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build(dataset: str, limit: int | None = None) -> dict:
    raws: dict[int, list] = {tf: [] for tf in config.TFS}
    t0 = time.time()
    for i, path in enumerate(part_paths(dataset, limit)):
        df = pd.read_pickle(path)
        for tf in config.TFS:
            raws[tf].append(extract_part(df, tf))
        print(f"[{dataset}] part {i} {os.path.basename(path)} {df.shape} {time.time() - t0:.0f}s", flush=True)
        del df
        gc.collect()
    raw = {tf: pd.concat(raws[tf]) for tf in config.TFS}
    if dataset == "2y":
        zstats = {str(tf): list(C.zone_stats(raw[tf][f"{tf}_rsi_ma8"].to_numpy())) for tf in config.TFS}
    else:
        with open(os.path.join(config.sidecar_dir("2y"), "meta.json")) as f:
            zstats = json.load(f)["zone_stats"]
    meta = finalize(raw, dataset, zstats)
    swing_sidecar_gate(dataset, meta)
    meta["provenance"] = {"git_sha": git_sha(), "built": pd.Timestamp.utcnow().isoformat(), "parts": limit or "all",
                          "spec": "external/docs/superpowers/experiment/cb_class_bounds.md"}
    with open(os.path.join(config.sidecar_dir(dataset), "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(config.DATASETS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    meta = build(a.dataset, a.limit)
    print(json.dumps({k: v for k, v in meta["tfs"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
