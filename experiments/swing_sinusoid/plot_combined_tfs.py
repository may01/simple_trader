"""OHLC for tf 15 / 60 / 240 over one common time window, each shaded by its own
combined zone = cci_ma8 zone (+1 buy / −1 sell / 0) + 0_rsi (RSI move class, ±3 → 0).

    python -m experiments.swing_sinusoid.plot_combined_tfs --dataset oos2m --days 5 [--end 2025-02-28]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .apex import find_apexes  # noqa: E402
from .plot_price_apex import INK, INK2, SURF, UP, DOWN, _style, build_zones, load_ohlc  # noqa: E402

CMB = {-3: "#b2182b", -2: "#ef8a62", -1: "#fddbc7", 0: None, 1: "#d1e5f0", 2: "#67a9cf", 3: "#2166ac"}


def combined_for_tf(dataset: str, tf: int, confirm: int, window: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_pickle(os.path.join(config.sidecar_dir(dataset), f"series_tf{tf}.pkl"))
    ohlc = load_ohlc(dataset, tf)
    assert len(ohlc) == len(frame)
    y = frame["cci_ma8"].to_numpy(dtype=float)
    idx, sgn = find_apexes(y, prominence=0.5 * float(np.nanstd(y)), distance=2)
    errs = pd.read_pickle(os.path.join(config.sidecar_dir("2y"), "apex_spacing_errors_p05.pkl"))
    const = float(errs[(errs.series == "cci_ma8") & (errs.tf == tf) & (errs.method == "median")]["pred_spacing"].iloc[0])
    z = build_zones(idx, sgn, const, confirm, window, len(y))
    mc = ohlc["move_class_sym0"].to_numpy(dtype=float)
    rsi0 = np.where(np.abs(mc) == 3, 0, np.nan_to_num(mc, nan=0.0)).astype(int)
    return ohlc, z + rsi0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="oos2m")
    ap.add_argument("--days", type=float, default=5)
    ap.add_argument("--end", default=None, help="window end (UTC), default = dataset end")
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--window", type=int, default=3)
    args = ap.parse_args()
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"hspace": 0.3})
    fig.patch.set_facecolor(SURF)
    t_end = None
    for ax, tf in zip(axes, config.TFS):
        ohlc, comb = combined_for_tf(args.dataset, tf, args.confirm, args.window)
        ts = ohlc.index
        if t_end is None:
            t_end = pd.Timestamp(args.end, tz="UTC") if args.end else ts[-1]
        t_start = t_end - pd.Timedelta(days=args.days)
        m = (ts > t_start) & (ts <= t_end)
        sub, c = ohlc[m], comb[m]
        x = mdates.date2num(sub.index.tz_convert(None).to_pydatetime())
        w = tf / (24 * 60)
        o, h, l, cl = (sub[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        up = cl >= o
        # zones as background, per run of equal value
        i = 0
        while i < len(c):
            j = i
            while j + 1 < len(c) and c[j + 1] == c[i]:
                j += 1
            if CMB.get(int(c[i])):
                ax.axvspan(x[i] - w, x[j], color=CMB[int(c[i])], alpha=0.35, lw=0)
            i = j + 1
        ax.vlines(x - w / 2, l, h, color=np.where(up, UP, DOWN), lw=0.9)
        ax.bar(x - w / 2, np.abs(cl - o), bottom=np.minimum(o, cl), width=w * 0.75, color=np.where(up, UP, DOWN), lw=0)
        n_pos, n_neg = int((c > 0).sum()), int((c < 0).sum())
        _style(ax, f"tf{tf}: OHLC, background = combined zone (cci_ma8 zone ±1 + 0_rsi), {len(c)} candles, "
                   f"positive {n_pos} / negative {n_neg} / zero {len(c) - n_pos - n_neg}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for k, col in CMB.items():
        if col:
            axes[0].fill_between([], [], [], color=col, alpha=0.6, label=f"{k:+d}")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=6, title="combined value", title_fontsize=8)
    fig.suptitle(f"{args.dataset}: {t_start:%Y-%m-%d %H:%M} → {t_end:%Y-%m-%d %H:%M} UTC — combined = cci_ma8 zone (+1 buy / −1 sell, confirm +{args.confirm}, window ±{args.window}) + 0_rsi (RSI move class with ±3 → 0)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    out_dir = os.path.join(config.sidecar_dir("2y"), "plots")
    out = os.path.join(out_dir, f"combined_tfs_{args.dataset}_{t_end:%Y%m%d}_{args.days:g}d.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
