"""Charts for the three-apex spacing predictor (RSI-MA / CCI-MA).

    python -m experiments.swing_sinusoid.plot_apex_spacing [--tf 15] [--dataset oos2m]

Writes <2y sidecar>/plots/apex_spacing_*.png:
  timeline_<series>      series + true apexes + predicted next-apex times per rule
  errors                 error distributions (real − predicted) per rule, one panel per series
  cumulative             P(|err| ≤ k) by k per rule, one panel per series
  spacing_scatter        predicted vs real spacing (rule `last`), one panel per series
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .apex import find_apexes  # noqa: E402

SERIES = ["rsi_ma6", "rsi_ma8", "cci_ma6", "cci_ma8"]
COL = {"last": "#eb6834", "mean2": "#1baf7a", "median": "#eda100", "trend": "#6250d6"}
LABEL = {"last": "last spacing", "mean2": "mean of 2 spacings", "median": "train median (const)",
         "trend": "linear trend"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURF = "#fcfcfb"


def _style(ax, title):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def timeline(frame, series, tf, med_sp, out, start=None, n_show=260):
    y = frame[series].to_numpy(dtype=float)
    std = float(np.nanstd(y))
    idx, sgn = find_apexes(y, prominence=0.5 * std, distance=2)
    end = len(y) - 1 if start is None else start + n_show
    start = end - n_show if start is None else start
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.32})
    fig.patch.set_facecolor(SURF)
    x = np.arange(start, end + 1)
    ax.plot(x, y[start:end + 1], color=INK, lw=1.2, label=series)
    a = idx[(idx >= start) & (idx <= end)]
    ax.scatter(a, y[a], s=36, facecolor=SURF, edgecolor="#2a78d6", lw=1.6, zorder=4, label="true apex")
    for t in a:
        ax.axvline(t, color="#2a78d6", lw=0.6, alpha=0.35)
        ax2.axvline(t, color="#2a78d6", lw=0.6, alpha=0.35)
    rules = ["last", "mean2", "median"]
    for r_i, rule in enumerate(rules):
        yy = len(rules) - r_i
        for i in range(2, len(idx) - 1):
            if idx[i] < start or idx[i] > end:
                continue
            s1, s2 = idx[i - 1] - idx[i - 2], idx[i] - idx[i - 1]
            f = {"last": s2, "mean2": (s1 + s2) / 2, "median": med_sp}[rule]
            t_hat = idx[i] + f
            ax2.plot([idx[i], t_hat], [yy, yy], color=COL[rule], lw=1.2, alpha=0.6)
            ax2.scatter([t_hat], [yy], s=22, color=COL[rule], zorder=3)
    ax2.set_yticks(range(1, len(rules) + 1))
    ax2.set_yticklabels([LABEL[r] for r in rules[::-1]], fontsize=8, color=INK2)
    ax2.set_ylim(0.4, len(rules) + 0.6)
    _style(ax, f"tf{tf} {series}: true apexes (rings / vertical lines) and the next apex predicted from the last three apexes")
    _style(ax2, "predicted next-apex time per rule (dot, line from the apex it was issued at); dot on a vertical line = exact hit")
    ax2.title.set_fontsize(8.5)
    ax2.set_xlabel("closed-candle index", color=INK2, fontsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def errors(err: pd.DataFrame, tf, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    fig.patch.set_facecolor(SURF)
    bins = np.arange(-20.5, 21.5, 1.0)
    for ax, series in zip(axes, SERIES):
        sub = err[(err.series == series) & (err.tf == tf)]
        for rule in ("last", "mean2", "median"):
            e = sub[sub.method == rule]["err"].to_numpy()
            ax.hist(np.clip(e, -20, 20), bins=bins, histtype="step", lw=1.8, color=COL[rule],
                    density=True, label=f"{LABEL[rule]}  median |err| {np.median(np.abs(e)):.1f}")
        ax.axvline(0, color=INK2, lw=0.8, ls="--")
        _style(ax, f"tf{tf} {series}  (n apex {len(sub[sub.method == 'last'])})")
        ax.set_xlabel("real − predicted apex time (candles)", fontsize=8, color=INK2)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("density", fontsize=8, color=INK2)
    fig.suptitle("Error of the next-apex time predicted from the last three apexes — 2y val + oos2m", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURF)
    plt.close(fig)


def cumulative(err: pd.DataFrame, tf, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    fig.patch.set_facecolor(SURF)
    ks = np.arange(0, 11)
    for ax, series in zip(axes, SERIES):
        sub = err[(err.series == series) & (err.tf == tf)]
        for rule in ("last", "mean2", "trend", "median"):
            e = np.abs(sub[sub.method == rule]["err"].to_numpy())
            ax.plot(ks, [(e <= k).mean() for k in ks], color=COL[rule], lw=2, marker="o", ms=4, label=LABEL[rule])
        med = sub[sub.method == "last"]["real_spacing"].median()
        ax.axvline(med / 2, color=INK2, lw=0.8, ls="--")
        ax.text(med / 2 + 0.2, 0.05, f"half the median spacing ({med / 2:.0f})", fontsize=7, color=INK2)
        _style(ax, f"tf{tf} {series}")
        ax.set_xlabel("tolerance k (candles)", fontsize=8, color=INK2)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("P(|real − predicted| ≤ k)", fontsize=8, color=INK2)
    fig.suptitle("How often the real apex lands within k candles of the prediction — 2y val + oos2m", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURF)
    plt.close(fig)


def scatter(err: pd.DataFrame, tf, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    fig.patch.set_facecolor(SURF)
    rng = np.random.default_rng(0)
    for ax, series in zip(axes, SERIES):
        sub = err[(err.series == series) & (err.tf == tf) & (err.method == "last")]
        p = sub["pred_spacing"].to_numpy() + rng.uniform(-0.3, 0.3, len(sub))
        r = sub["real_spacing"].to_numpy() + rng.uniform(-0.3, 0.3, len(sub))
        lim = np.percentile(np.concatenate([p, r]), 98)
        ax.scatter(p, r, s=8, color="#2a78d6", alpha=0.35, lw=0)
        ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--")
        rho = np.corrcoef(sub["pred_spacing"], sub["real_spacing"])[0, 1]
        _style(ax, f"tf{tf} {series}: next spacing vs last spacing, r = {rho:.2f}")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("last spacing (candles)", fontsize=8, color=INK2)
    axes[0].set_ylabel("real next spacing (candles)", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=15)
    ap.add_argument("--dataset", default="oos2m")
    args = ap.parse_args()
    sdir = config.sidecar_dir("2y")
    pdir = os.path.join(sdir, "plots")
    os.makedirs(pdir, exist_ok=True)
    err = pd.read_pickle(os.path.join(sdir, "apex_spacing_errors_p05.pkl"))
    err = err[err.split != "train"]
    frame = pd.read_pickle(os.path.join(config.sidecar_dir(args.dataset), f"series_tf{args.tf}.pkl"))
    summ = pd.read_csv(os.path.join(sdir, "apex_spacing_p05.csv"))
    for series in SERIES:
        row = summ[(summ.split == "train") & (summ.tf == args.tf) & (summ.series == series) & (summ.method == "median")]
        med_sp = float(row["median_real_spacing"].iloc[0]) if len(row) else 10.0
        # train-median spacing used by the rule = median of train spacings (stored as pred_spacing of 'median')
        e = err[(err.series == series) & (err.tf == args.tf) & (err.method == "median")]
        med_sp = float(e["pred_spacing"].iloc[0]) if len(e) else med_sp
        timeline(frame, series, args.tf, med_sp, os.path.join(pdir, f"apex_spacing_timeline_tf{args.tf}_{series}.png"))
    errors(err, args.tf, os.path.join(pdir, f"apex_spacing_errors_tf{args.tf}.png"))
    cumulative(err, args.tf, os.path.join(pdir, f"apex_spacing_cumulative_tf{args.tf}.png"))
    scatter(err, args.tf, os.path.join(pdir, f"apex_spacing_scatter_tf{args.tf}.png"))
    print("plots →", pdir)


if __name__ == "__main__":
    main()
