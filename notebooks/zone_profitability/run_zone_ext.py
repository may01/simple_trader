"""Driver: how far is a (vetoed) cb-zone marker from the candle's eventual
low/high, and is a Ridge-on-market-state regression (the candle_bounds
recipe) able to predict that remaining extension?

Marker set = the chart's current cbzone markers: ``{tf}_cb_inzone_long`` with
``{tf}_extdone_high`` False (long), ``{tf}_cb_inzone_short`` with
``{tf}_extdone_low`` False (short). Per marker minute t (long side; short
mirrored on highs):

    ext_pct = (1_low[t] - candle_low) / 1_low[t] * 100      remaining extension
    ext_atr = (1_low[t] - candle_low) / atr1[t]              same in 1-min ATR
    zone_pct = (cb_zone_long[t] - candle_low) / zone * 100   from the zone line
    at_low   = ext == 0 (this minute IS the candle low, or the low was set
               earlier and never broken -> running-low distance also reported)

Regression: target ext_pct, features = cbnc.forming_features (the 15 bound
features on the forming candle) + progress + pull-away of close from the
running extreme (ATR) + distance of 1_low below cb_low in band sd + atr1 %
of price. Fit on 2y (markers rebuilt there: cb sidecar on 2y_link_usdt,
extdone ladder recomputed, atr1 rebuilt from 1-min OHLC), test on oos2m;
plus an oos2m-internal time split as a sanity check. Baseline = predict the
train mean.

Artifacts: {EV_OUT}/zone_ext_report.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "notebooks", "candle_bounds_nc"))

from cbnc import FEATURES, forming_features  # noqa: E402
from evlib import TFS, candle_id  # noqa: E402
from run_extdone import LADDER, d_of_progress  # noqa: E402

VOL = "/trader_data_long/train"
OOS_DIR = os.environ.get("EV_DATASET_DIR", f"{VOL}/oos2m_link_usdt")
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
TRAIN_CB = f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl"
OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
SIDES = {"long": "low", "short": "high"}
EXTRA = ["progress", "pull_atr", "below_cb_sd", "atr1_pct"]


def atr1_ma5(df: pd.DataFrame) -> np.ndarray:
    """1_atr_14_ma_5 rebuilt from 1-min OHLC (Wilder ATR14, then MA5)."""
    h, l, c = df["1_high"], df["1_low"], df["1_close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return atr.rolling(5).mean().to_numpy()


def extdone(df: pd.DataFrame, cid: np.ndarray, prog: np.ndarray, atr: np.ndarray,
            tf: int, side: str) -> np.ndarray:
    x = (df["1_high"] if side == "high" else df["1_low"]).to_numpy()
    g = pd.Series(cid); s = pd.Series(x)
    run = (s.groupby(g).cummax() if side == "high" else s.groupby(g).cummin()).to_numpy()
    cl = df["1_close"].to_numpy()
    dist = ((run - cl) if side == "high" else (cl - run)) / atr
    done = (dist >= d_of_progress(prog, LADDER[tf])) & np.isfinite(atr)
    return pd.Series(done).groupby(g).cummax().to_numpy().astype(bool)


def build(df: pd.DataFrame, cb: pd.DataFrame, atr: np.ndarray, tf: int, side: str,
          ext_done: dict | None = None) -> pd.DataFrame:
    """Marker-row frame for one (tf, side): targets + features."""
    ext = SIDES[side]                      # extreme the entry aims at ("low" for long)
    opp = "high" if ext == "low" else "low"
    n = len(df)
    cid = candle_id(df, tf)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    rep = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, rep)
    prog = (pos + 1) / tf
    g = pd.Series(cid)
    x = df[f"1_{ext}"].to_numpy()
    s = pd.Series(x)
    final = (s.groupby(g).transform("min") if ext == "low" else s.groupby(g).transform("max")).to_numpy()
    run = (s.groupby(g).cummin() if ext == "low" else s.groupby(g).cummax()).to_numpy()
    veto = (ext_done[opp] if ext_done is not None
            else extdone(df, cid, prog, atr, tf, opp))
    inz = cb[f"{tf}_cb_inzone_{side}"].reindex(df.index).fillna(False).to_numpy(dtype=bool)
    m = inz & ~veto & np.isfinite(atr)
    sign = 1.0 if ext == "low" else -1.0
    zone = cb[f"{tf}_cb_zone_{side}"].reindex(df.index).to_numpy()
    bound = cb[f"{tf}_cb_{ext}"].reindex(df.index).to_numpy()
    bsd = cb[f"{tf}_cb_{ext}_std"].reindex(df.index).to_numpy()
    cl = df["1_close"].to_numpy()
    ff = forming_features(df, tf, ext)
    out = pd.DataFrame(index=df.index[m])
    out["ext_pct"] = (sign * (x - final) / x * 100.0)[m]
    out["ext_atr"] = (sign * (x - final) / atr)[m]
    out["ext_tfatr"] = (sign * (x - final) / df[f"{tf}_atr_14_ma_5"].to_numpy())[m]
    out["zone_pct"] = (sign * (zone - final) / zone * 100.0)[m]
    out["run_pct"] = (sign * (run - final) / run * 100.0)[m]   # from running extreme
    out["at_ext"] = (x == final)[m]
    out["run_is_final"] = (run == final)[m]
    out["progress"] = prog[m]
    out["pull_atr"] = (sign * (cl - run) / atr)[m]             # close pulled away from run extreme
    out["below_cb_sd"] = (sign * (bound - x) / bsd)[m]          # how far 1_low sits below cb_low
    out["atr1_pct"] = (atr / x * 100.0)[m]
    out["cid"] = cid[m]
    for f in FEATURES:
        out[f] = ff[f].to_numpy()[m]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def describe(fr: pd.DataFrame) -> dict:
    q = lambda c, p: float(np.percentile(fr[c], p))  # noqa: E731
    return {
        "n_markers": int(len(fr)), "n_candles": int(fr["cid"].nunique()),
        "at_ext_pct": float(100 * fr["at_ext"].mean()),
        "run_is_final_pct": float(100 * fr["run_is_final"].mean()),
        **{f"{c}_{k}": v for c in ("ext_pct", "ext_atr", "ext_tfatr", "zone_pct", "run_pct")
           for k, v in (("mean", float(fr[c].mean())), ("median", float(fr[c].median())),
                        ("std", float(fr[c].std())), ("p75", q(c, 75)), ("p90", q(c, 90)))},
        "by_progress": [
            {"bucket": f"{a:.2f}-{b:.2f}", "n": int(mm.sum()),
             "ext_pct_mean": float(fr.loc[mm, "ext_pct"].mean()),
             "ext_pct_std": float(fr.loc[mm, "ext_pct"].std()),
             "at_ext_pct": float(100 * fr.loc[mm, "at_ext"].mean())}
            for a, b in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))
            for mm in [(fr["progress"] >= a) & (fr["progress"] < b)] if mm.sum() > 0
        ],
    }


class Ridge:
    """StandardScaler + Ridge(alpha) in numpy (no sklearn in the image).
    Same estimator as cbnc.fit_side: standardized X, L2 on the slopes only."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        self.y0 = y.mean()
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (y - self.y0))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.coef_ + self.y0


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_eval(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> dict:
    y_tr, y_te = tr["ext_pct"].to_numpy(), te["ext_pct"].to_numpy()
    X_tr, X_te = tr[feats].to_numpy(), te[feats].to_numpy()
    model = Ridge(1.0).fit(X_tr, y_tr)
    p = model.predict(X_te)
    base = np.full(len(te), y_tr.mean())
    top = sorted(zip(feats, model.coef_), key=lambda kv: -abs(kv[1]))[:6]
    return {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "r2_test": r2(y_te, p),
        "r2_train": r2(y_tr, model.predict(X_tr)),
        "mae_model": float(np.abs(y_te - p).mean()),
        "mae_baseline": float(np.abs(y_te - base).mean()),
        "resid_std_model": float((y_te - p).std()),
        "resid_std_baseline": float(y_te.std()),
        "top_coef": [(f, round(float(c), 4)) for f, c in top],
    }


def main() -> None:
    need = ["1_open", "1_high", "1_low", "1_close", "1_atr_14_ma_5"]
    per_tf = ["open", "high", "low", "close", "is_closed", "atr_14_ma_5", "rsi_14",
              "rsi_ma8", "macd_12_26_9", "macd_hist_12_26_9", "ema_25"]
    cols = need + [f"{tf}_{c}" for tf in TFS for c in per_tf]
    oos = pd.read_pickle(os.path.join(OOS_DIR, "df_with_indicators.pkl"))[cols]
    oos_cb = pd.read_pickle(os.path.join(OOS_DIR, "df_with_candle_bounds.pkl"))
    oos_ed = pd.read_pickle(os.path.join(OOS_DIR, "df_with_ext_done.pkl")).reindex(oos.index)
    oos_atr = oos["1_atr_14_ma_5"].to_numpy()
    tr = pd.read_pickle(TRAIN_IND)
    tr_cb = pd.read_pickle(TRAIN_CB)
    tr_atr = atr1_ma5(tr)
    feats = FEATURES + EXTRA
    report = {"cells": []}
    split = oos.index[int(len(oos) * 0.75)]
    for tf in TFS:
        for side in SIDES:
            ed = {s: oos_ed[f"{tf}_extdone_{s}"].fillna(False).to_numpy(dtype=bool)
                  for s in ("high", "low")}
            fo = build(oos, oos_cb, oos_atr, tf, side, ed)
            ft = build(tr, tr_cb, tr_atr, tf, side)
            cell = {"tf": tf, "side": side, "oos": describe(fo), "train": describe(ft),
                    "ridge_2y_to_oos": fit_eval(ft, fo, feats),
                    "ridge_oos_timesplit": fit_eval(fo[fo.index < split], fo[fo.index >= split], feats),
                    "ridge_2y_to_oos_extra_only": fit_eval(ft, fo, EXTRA)}
            report["cells"].append(cell)
            d, r = cell["oos"], cell["ridge_2y_to_oos"]
            print(tf, side, f"n={d['n_markers']} at_ext={d['at_ext_pct']:.1f}% "
                  f"ext% mean={d['ext_pct_mean']:.3f} med={d['ext_pct_median']:.3f} "
                  f"sd={d['ext_pct_std']:.3f} | ridge r2={r['r2_test']:.3f} "
                  f"mae {r['mae_model']:.3f} vs base {r['mae_baseline']:.3f} "
                  f"resid sd {r['resid_std_model']:.3f} vs {r['resid_std_baseline']:.3f}", flush=True)
    with open(os.path.join(OUT, "zone_ext_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    write_md(report, os.path.join(OUT, "zone_ext_report.md"))
    print("wrote", OUT)


def write_md(report: dict, path: str) -> None:
    o = ["# cb-zone marker -> candle extreme: remaining extension (oos2m)", "",
         "Markers = cb_inzone_{side} minus the extdone veto. Long: distance from the marker's "
         "1-min low down to the candle's final low (short: high up to final high).", "",
         "## Distribution per marker minute (oos2m)", "",
         "| tf | side | markers | candles | at ext % | run=final % | ext % mean | median | sd | p75 | p90 | "
         "ext (1m ATR) mean | sd | ext (tf ATR) mean | from zone line % mean | sd |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        d = c["oos"]
        o.append(f"| {c['tf']} | {c['side']} | {d['n_markers']} | {d['n_candles']} | {d['at_ext_pct']:.1f} | "
                 f"{d['run_is_final_pct']:.1f} | {d['ext_pct_mean']:.3f} | {d['ext_pct_median']:.3f} | "
                 f"{d['ext_pct_std']:.3f} | {d['ext_pct_p75']:.3f} | {d['ext_pct_p90']:.3f} | "
                 f"{d['ext_atr_mean']:.2f} | {d['ext_atr_std']:.2f} | {d['ext_tfatr_mean']:.2f} | "
                 f"{d['zone_pct_mean']:.3f} | {d['zone_pct_std']:.3f} |")
    o += ["", "## By candle progress (oos2m, ext % mean / sd / at-ext %)", "",
          "| tf | side | 0-25 % | 25-50 % | 50-75 % | 75-100 % |", "|---|---|---|---|---|---|"]
    for c in report["cells"]:
        cells = {b["bucket"]: b for b in c["oos"]["by_progress"]}
        row = [f"{b['ext_pct_mean']:.3f} / {b['ext_pct_std']:.3f} / {b['at_ext_pct']:.0f}% (n{b['n']})"
               if (b := cells.get(k)) else "—"
               for k in ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.01")]
        o.append(f"| {c['tf']} | {c['side']} | " + " | ".join(row) + " |")
    o += ["", "## Ridge regression of ext % on market state", "",
          "| tf | side | fit -> test | n train | n test | r² train | r² test | MAE model | MAE baseline | "
          "resid sd model | resid sd baseline | top coefficients |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in report["cells"]:
        for k, lab in (("ridge_2y_to_oos", "2y -> oos2m (15 bound feats + 4 state)"),
                       ("ridge_2y_to_oos_extra_only", "2y -> oos2m (4 state feats only)"),
                       ("ridge_oos_timesplit", "oos2m first 75 % -> last 25 %")):
            r = c[k]
            o.append(f"| {c['tf']} | {c['side']} | {lab} | {r['n_train']} | {r['n_test']} | "
                     f"{r['r2_train']:.3f} | {r['r2_test']:.3f} | {r['mae_model']:.3f} | "
                     f"{r['mae_baseline']:.3f} | {r['resid_std_model']:.3f} | {r['resid_std_baseline']:.3f} | "
                     f"{', '.join(f'{f}={v:+.3f}' for f, v in r['top_coef'][:4])} |")
    with open(path, "w") as fh:
        fh.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
