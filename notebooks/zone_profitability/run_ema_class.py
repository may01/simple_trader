"""Driver: EMA-25 slope market classes (fall / neutral / rise) as a conditioner
for the zone-marker remaining-extension quantiles.

Steps (spec from the 2026-10-03 session):
  1. {tf}_ema_25 per 1-min row (forming value).
  2. ema_dif = change of ema_25 over one tf candle, per row:
         ema_dif_pct[t] = (ema_25[t] - ema_25[prev closed candle]) / ema_25[t] * 100
     (= cbnc A_slope, in % of price so 2y's 5x price range does not dominate).
  3. z = (ema_dif_pct - mean) / std, mean/std over ALL 2y 1-min rows, per tf.
  4. classes: fall  z < -x,  neutral |z| <= x,  rise  z > x.  x from X_GRID: the
     value minimizing the out-of-time pinball loss (q75 + q90) of the
     class-conditional quantile table on the LAST 25 % of 2y (table fit on
     the first 75 %), summed over tf/side/progress.
  5. quantile table q50/q75/q90 of ext_atr per (tf, side, progress bucket,
     class) on full 2y with the chosen x; evaluated on oos2m against the
     unconditional (no-class) table: pinball loss, coverage, and the q values
     per class so the shift is visible.

Reuses run_zone_ext.build (marker rows = cb_inzone minus the extdone veto,
ext_atr = remaining extension in 1-min ATR). Artifacts:
  {EV_OUT}/ema_class_report.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_zone_ext as z  # noqa: E402
from evlib import TFS  # noqa: E402

OUT = os.environ.get("EV_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
X_GRID = [float(v) for v in os.environ.get("EMA_X_GRID", "0.25,0.5,0.75,1,1.25,1.5,2").split(",")]
QS = (0.5, 0.75, 0.9)
BUCKETS = ((0, .25), (.25, .5), (.5, .75))      # 75-100 % too thin (20-70 markers)
CLASSES = ("fall", "neutral", "rise")
MIN_CELL = 30


EMA_MODE = os.environ.get("EMA_MODE", "closed")   # closed | forming


def ema_dif_pct(df: pd.DataFrame, tf: int) -> np.ndarray:
    """Per 1-min row, in % of price.

    closed  (default): candle-to-candle change of ema_25 over the CLOSED
            sequence, d[k] = (ema[k] - ema[k-1]) / ema[k] * 100 at candle k's
            closing row, broadcast to the following candle's rows (shift(1)
            + ffill, the candle_bounds convention) -> constant within a
            candle, no forming-candle input. mean/std are then also closed-
            candle statistics (one value per candle, not per minute).
    forming: ema_25[t] (forming) vs the last closed candle's ema, per minute.
    """
    closed = df[f"{tf}_is_closed"] == True  # noqa: E712
    ema = df[f"{tf}_ema_25"]
    if EMA_MODE == "forming":
        prev = ema.where(closed).shift(1).ffill()
        return ((ema - prev) / ema * 100.0).to_numpy()
    c = ema[closed]
    d = (c - c.shift(1)) / c * 100.0
    return d.reindex(df.index).shift(1).ffill().to_numpy()


def ema_dif_norm(df: pd.DataFrame, tf: int, d: np.ndarray) -> tuple[float, float]:
    """mean/std of the dif: over closed candles (one value each) in closed
    mode, over all rows in forming mode."""
    if EMA_MODE == "forming":
        return float(np.nanmean(d)), float(np.nanstd(d))
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    # the closing row of candle k carries d[k-1] after the shift; take the
    # per-candle values directly from the closed sequence instead
    ema = df[f"{tf}_ema_25"][closed]
    dc = ((ema - ema.shift(1)) / ema * 100.0).to_numpy()
    return float(np.nanmean(dc)), float(np.nanstd(dc))


def classify(zv: np.ndarray, x: float) -> np.ndarray:
    return np.where(zv < -x, "fall", np.where(zv > x, "rise", "neutral"))


def bucket_of(prog: np.ndarray) -> np.ndarray:
    out = np.full(len(prog), "", dtype=object)
    for a, b in BUCKETS:
        out[(prog >= a) & (prog < b)] = f"{a:.2f}-{b:.2f}"
    return out


def qtable(fr: pd.DataFrame, keys: list[str]) -> dict:
    """{key tuple: (q50, q75, q90)} over groups with >= MIN_CELL rows."""
    tab = {}
    for k, g in fr.groupby(keys):
        if len(g) >= MIN_CELL:
            tab[k if isinstance(k, tuple) else (k,)] = tuple(float(np.percentile(g["ext_atr"], 100 * q)) for q in QS)
    return tab


def pinball(y: np.ndarray, p: np.ndarray, q: float) -> float:
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def evaluate(te: pd.DataFrame, tab_u: dict, tab_c: dict | None) -> dict:
    """Pinball per q + coverage for unconditional vs class-conditional tables.
    Class table falls back to the unconditional cell when the class cell is thin."""
    keys_u = list(zip(te["tf"], te["side"], te["bucket"]))
    pu = np.array([tab_u.get(k, (np.nan,) * 3) for k in keys_u])
    if tab_c is None:
        pc = pu
    else:
        keys_c = list(zip(te["tf"], te["side"], te["bucket"], te["cls"]))
        pc = np.array([tab_c.get(kc, tab_u.get(ku, (np.nan,) * 3)) for kc, ku in zip(keys_c, keys_u)])
    ok = np.isfinite(pu).all(1) & np.isfinite(pc).all(1)
    y = te["ext_atr"].to_numpy()[ok]
    res = {"n": int(ok.sum())}
    for i, q in enumerate(QS):
        res[f"pin_u_q{int(q*100)}"] = pinball(y, pu[ok, i], q)
        res[f"pin_c_q{int(q*100)}"] = pinball(y, pc[ok, i], q)
        res[f"cov_u_q{int(q*100)}"] = float((y <= pu[ok, i]).mean())
        res[f"cov_c_q{int(q*100)}"] = float((y <= pc[ok, i]).mean())
    res["pin_u_7590"] = res["pin_u_q75"] + res["pin_u_q90"]
    res["pin_c_7590"] = res["pin_c_q75"] + res["pin_c_q90"]
    res["improve_pct"] = 100 * (1 - res["pin_c_7590"] / res["pin_u_7590"])
    return res


def main() -> None:
    cols = ["1_open", "1_high", "1_low", "1_close", "1_atr_14_ma_5"] + [
        f"{tf}_{c}" for tf in TFS for c in
        ["open", "high", "low", "close", "is_closed", "atr_14_ma_5", "rsi_14", "rsi_ma8",
         "macd_12_26_9", "macd_hist_12_26_9", "ema_25"]]
    oos = pd.read_pickle(os.path.join(z.OOS_DIR, "df_with_indicators.pkl"))[cols]
    oos_cb = pd.read_pickle(os.path.join(z.OOS_DIR, "df_with_candle_bounds.pkl"))
    oos_ed = pd.read_pickle(os.path.join(z.OOS_DIR, "df_with_ext_done.pkl")).reindex(oos.index)
    tr = pd.read_pickle(z.TRAIN_IND)
    tr_cb = pd.read_pickle(z.TRAIN_CB)
    tr_atr = z.atr1_ma5(tr)
    oos_atr = oos["1_atr_14_ma_5"].to_numpy()

    # step 2-3: ema_dif + 2y normalisation per tf
    norm, frames_tr, frames_oos = {}, [], []
    for tf in TFS:
        d_tr, d_oos = ema_dif_pct(tr, tf), ema_dif_pct(oos, tf)
        mu, sd = ema_dif_norm(tr, tf, d_tr)
        mu_o, sd_o = ema_dif_norm(oos, tf, d_oos)
        norm[tf] = {"mean": mu, "std": sd, "oos_mean": mu_o, "oos_std": sd_o, "mode": EMA_MODE}
        z_tr = pd.Series((d_tr - mu) / sd, index=tr.index)
        z_oos = pd.Series((d_oos - mu) / sd, index=oos.index)
        for side in ("long", "short"):
            ed = {s: oos_ed[f"{tf}_extdone_{s}"].fillna(False).to_numpy(dtype=bool) for s in ("high", "low")}
            fo = z.build(oos, oos_cb, oos_atr, tf, side, ed)
            ft = z.build(tr, tr_cb, tr_atr, tf, side)
            for fr, zz in ((fo, z_oos), (ft, z_tr)):
                fr["z"] = zz.reindex(fr.index).to_numpy()
                fr["tf"], fr["side"] = tf, side
                fr["bucket"] = bucket_of(fr["progress"].to_numpy())
            frames_tr.append(ft); frames_oos.append(fo)
        print(tf, EMA_MODE, "ema_dif_pct 2y mean %.4f std %.4f | oos mean %.4f std %.4f" % (
            mu, sd, mu_o, sd_o), flush=True)
    TR = pd.concat(frames_tr); OO = pd.concat(frames_oos)
    TR = TR[(TR["bucket"] != "") & np.isfinite(TR["z"])]
    OO = OO[(OO["bucket"] != "") & np.isfinite(OO["z"])]
    base_keys = ["tf", "side", "bucket"]

    # step 4: choose x out-of-time inside 2y
    cut = TR.index[int(len(TR) * 0.75)]
    a, b = TR[TR.index < cut], TR[TR.index >= cut]
    tab_u_a = qtable(a, base_keys)
    sel = []
    for x in X_GRID:
        a["cls"], b["cls"] = classify(a["z"].to_numpy(), x), classify(b["z"].to_numpy(), x)
        r = evaluate(b, tab_u_a, qtable(a, base_keys + ["cls"]))
        share = {c: float((b["cls"] == c).mean()) for c in CLASSES}
        sel.append({"x": x, **r, "share": share})
        print("x=%.2f  2y-oot pinball(q75+q90) uncond %.4f -> class %.4f (%+.2f %%)  shares %s" % (
            x, r["pin_u_7590"], r["pin_c_7590"], r["improve_pct"],
            {k: round(v, 2) for k, v in share.items()}), flush=True)
    best = max(sel, key=lambda r: r["improve_pct"])
    x_star = best["x"]

    # step 5: full-2y tables, oos2m evaluation, for every x (chosen one highlighted)
    tab_u = qtable(TR, base_keys)
    oos_eval = []
    for x in X_GRID:
        TR["cls"], OO["cls"] = classify(TR["z"].to_numpy(), x), classify(OO["z"].to_numpy(), x)
        tab_c = qtable(TR, base_keys + ["cls"])
        r = evaluate(OO, tab_u, tab_c)
        per_cell = []
        for (tf, side, bk), g in OO.groupby(base_keys):
            rc = evaluate(g, tab_u, tab_c)
            per_cell.append({"tf": tf, "side": side, "bucket": bk, **rc})
        oos_eval.append({"x": x, **r, "per_cell": per_cell,
                         "share_oos": {c: float((OO["cls"] == c).mean()) for c in CLASSES}})
        print("x=%.2f  oos2m pinball(q75+q90) uncond %.4f -> class %.4f (%+.2f %%)  cov q90 %.3f -> %.3f" % (
            x, r["pin_u_7590"], r["pin_c_7590"], r["improve_pct"], r["cov_u_q90"], r["cov_c_q90"]), flush=True)
    TR["cls"], OO["cls"] = classify(TR["z"].to_numpy(), x_star), classify(OO["z"].to_numpy(), x_star)
    tab_c = qtable(TR, base_keys + ["cls"])
    rows = []
    for (tf, side, bk), qu in sorted(tab_u.items()):
        for c in CLASSES:
            qc = tab_c.get((tf, side, bk, c))
            m = (OO["tf"] == tf) & (OO["side"] == side) & (OO["bucket"] == bk) & (OO["cls"] == c)
            n_tr = int(((TR["tf"] == tf) & (TR["side"] == side) & (TR["bucket"] == bk) & (TR["cls"] == c)).sum())
            rows.append({"tf": tf, "side": side, "bucket": bk, "cls": c, "n_2y": n_tr, "n_oos": int(m.sum()),
                         "q_uncond": qu, "q_class": qc,
                         "oos_q": [float(np.percentile(OO.loc[m, "ext_atr"], 100 * q)) for q in QS] if m.sum() >= MIN_CELL else None,
                         "cov_c_q90": float((OO.loc[m, "ext_atr"] <= qc[2]).mean()) if (qc and m.sum() >= MIN_CELL) else None})
    report = {"norm": norm, "x_grid": X_GRID, "x_star": x_star, "selection_2y_oot": sel,
              "oos_eval": oos_eval, "table_x_star": rows}
    with open(os.path.join(OUT, f"ema_class_{EMA_MODE}_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    write_md(report, os.path.join(OUT, f"ema_class_{EMA_MODE}_report.md"))
    print("x* =", x_star, "wrote", OUT)


def write_md(rep: dict, path: str) -> None:
    mode = next(iter(rep["norm"].values()))["mode"]
    o = [f"# EMA-25 slope classes × zone-marker extension quantiles — mode {mode}", "",
         ("ema_dif_pct = closed-candle change (ema[k] − ema[k−1]) / ema[k] · 100, held constant over "
          "the next candle's rows; z = (dif − mean)/std with mean/std over 2y closed candles per tf."
          if mode == "closed" else
          "ema_dif_pct = (ema_25[t] − ema_25[last closed candle]) / ema_25[t] · 100, per 1-min row; "
          "z = (ema_dif_pct − mean) / std with mean/std over all 2y rows per tf."), "",
         "| tf | 2y mean | 2y std | oos mean | oos std |", "|---|---|---|---|---|"]
    o += [f"| {tf} | {v['mean']:+.4f} | {v['std']:.4f} | {v['oos_mean']:+.4f} | {v['oos_std']:.4f} |"
          for tf, v in rep["norm"].items()]
    o += ["", "## x selection — 2y out-of-time (fit first 75 %, score last 25 %), pinball q75+q90 over all cells", "",
          "| x | uncond | class | improvement % | share fall / neutral / rise |", "|---|---|---|---|---|"]
    for r in rep["selection_2y_oot"]:
        s = r["share"]
        o.append(f"| {r['x']} | {r['pin_u_7590']:.4f} | {r['pin_c_7590']:.4f} | {r['improve_pct']:+.2f} | "
                 f"{s['fall']:.2f} / {s['neutral']:.2f} / {s['rise']:.2f} |")
    o += ["", f"**x\\* = {rep['x_star']}**", "",
          "## oos2m — class table (full 2y fit) vs unconditional, all x", "",
          "| x | pinball q50 u→c | q75 u→c | q90 u→c | q75+q90 improvement % | cov q75 u→c | cov q90 u→c | oos share fall/neutral/rise |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rep["oos_eval"]:
        s = r["share_oos"]
        o.append(f"| {r['x']} | {r['pin_u_q50']:.4f}→{r['pin_c_q50']:.4f} | {r['pin_u_q75']:.4f}→{r['pin_c_q75']:.4f} | "
                 f"{r['pin_u_q90']:.4f}→{r['pin_c_q90']:.4f} | {r['improve_pct']:+.2f} | "
                 f"{r['cov_u_q75']:.3f}→{r['cov_c_q75']:.3f} | {r['cov_u_q90']:.3f}→{r['cov_c_q90']:.3f} | "
                 f"{s['fall']:.2f}/{s['neutral']:.2f}/{s['rise']:.2f} |")
    best = [r for r in rep["oos_eval"] if r["x"] == rep["x_star"]][0]
    o += ["", f"## oos2m per cell at x* = {rep['x_star']} (pinball q75+q90 improvement %, cov q90 u→c)", "",
          "| tf | side | bucket | n | improvement % | cov q90 u→c |", "|---|---|---|---|---|---|"]
    for c in best["per_cell"]:
        o.append(f"| {c['tf']} | {c['side']} | {c['bucket']} | {c['n']} | {c['improve_pct']:+.2f} | "
                 f"{c['cov_u_q90']:.3f}→{c['cov_c_q90']:.3f} |")
    o += ["", f"## Quantile table at x* = {rep['x_star']}: ext_atr q50 / q75 / q90 — unconditional vs per class (2y fit), oos2m actual", "",
          "| tf | side | bucket | class | n 2y | n oos | uncond (2y) | class (2y) | oos actual | oos cov of class q90 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda t: " / ".join(f"{v:.1f}" for v in t) if t else "—"  # noqa: E731
    for r in rep["table_x_star"]:
        o.append(f"| {r['tf']} | {r['side']} | {r['bucket']} | {r['cls']} | {r['n_2y']} | {r['n_oos']} | "
                 f"{f3(r['q_uncond'])} | {f3(r['q_class'])} | {f3(r['oos_q'])} | "
                 f"{r['cov_c_q90']:.2f} |" if r["cov_c_q90"] is not None else
                 f"| {r['tf']} | {r['side']} | {r['bucket']} | {r['cls']} | {r['n_2y']} | {r['n_oos']} | "
                 f"{f3(r['q_uncond'])} | {f3(r['q_class'])} | {f3(r['oos_q'])} | — |")
    with open(path, "w") as fh:
        fh.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
