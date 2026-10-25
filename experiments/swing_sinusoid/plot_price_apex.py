"""Price (OHLC) + cci_14 + cci_ma8 with real apexes and the next apex predicted by the
train-median constant, with a ±3-candle window around every prediction.

    python -m experiments.swing_sinusoid.plot_price_apex --tf 15 --dataset oos2m --n-show 200 [--start N]
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

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
UP, DOWN, APEX, PRED, WIN, CONF = "#1baf7a", "#e34948", "#2a78d6", "#eda100", "#eda100", "#6250d6"


def _style(ax, title):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)


def build_zones(idx, sgn, const, confirm, window, n):
    """Zones: after a top apex confirmed at apex+confirm → sell until apex+const+window;
    after a bottom apex → buy. A later confirmation overrides an open zone. 0 = neutral."""
    z = np.zeros(n, dtype=int)
    for t, s in zip(idx, sgn):
        a, b = int(t + confirm), int(min(round(t + const + window), n - 1))
        if a >= n or a > b:
            continue
        z[a:b + 1] = -1 if s > 0 else 1
    return z


def zone_stats(z, close, high, low, tf):
    """Per zone: signed close-to-close return start→end, best/worst excursion, hit rate."""
    rows = []
    n = len(z)
    i = 0
    while i < n:
        if z[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and z[j + 1] == z[i]:
            j += 1
        d = z[i]
        if j < n - 1:
            ret = d * (close[j] / close[i] - 1)
            fav = d * ((high[i + 1:j + 1].max() if d > 0 else low[i + 1:j + 1].min()) / close[i] - 1) if j > i else 0.0
            adv = d * ((low[i + 1:j + 1].min() if d > 0 else high[i + 1:j + 1].max()) / close[i] - 1) if j > i else 0.0
            rows.append({"tf": tf, "zone": "buy" if d > 0 else "sell", "start": i, "len": j - i + 1,
                         "ret_bp": 1e4 * ret, "mfe_bp": 1e4 * fav, "mae_bp": 1e4 * adv})
        i = j + 1
    return pd.DataFrame(rows)


def load_ohlc(dataset: str, tf: int) -> pd.DataFrame:
    d = config.DATASETS[dataset]
    path = os.path.join(d["dir"], "df_with_indicators.pkl" if not d["parts"] else "df_with_indicators.part_24.pkl")
    df = pd.read_pickle(path)
    m = df[f"{tf}_is_closed"].astype(bool).to_numpy()
    cols = ["open", "high", "low", "close"]
    if f"{tf}_move_class_sym0" in df.columns:
        cols.append("move_class_sym0")
    out = pd.DataFrame({c: df[f"{tf}_{c}"].to_numpy()[m] for c in cols}, index=df.index[m])
    out.index.name = "ts"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=15)
    ap.add_argument("--dataset", default="oos2m")
    ap.add_argument("--n-show", type=int, default=200)
    ap.add_argument("--start", type=int, default=None, help="first closed-row index of the window (default: last n_show)")
    ap.add_argument("--series", default="cci_ma8")
    ap.add_argument("--const", type=float, default=None, help="train-median spacing (default: from apex_spacing_p05.csv)")
    ap.add_argument("--shift", type=float, default=0.0, help="candles to move the predicted apex EARLIER (moving-average delay)")
    ap.add_argument("--window", type=int, default=3, help="half-width of the window around the predicted apex (candles)")
    ap.add_argument("--confirm", type=int, default=None, help="draw a line N candles after each real apex (apex confirmed)")
    ap.add_argument("--zones", action="store_true", help="shade buy (green) / sell (red) zones: confirmed apex → predicted next apex + window")
    args = ap.parse_args()
    sdir = config.sidecar_dir(args.dataset)
    frame = pd.read_pickle(os.path.join(sdir, f"series_tf{args.tf}.pkl"))
    ohlc = load_ohlc(args.dataset, args.tf)
    assert len(ohlc) == len(frame), (len(ohlc), len(frame))
    y = frame[args.series].to_numpy(dtype=float)
    cci = frame["cci_14"].to_numpy(dtype=float)
    std = float(np.nanstd(y))
    idx, sgn = find_apexes(y, prominence=0.5 * std, distance=2)
    if args.const is None:
        summ = pd.read_csv(os.path.join(config.sidecar_dir("2y"), "apex_spacing_p05.csv"))
        errs = pd.read_pickle(os.path.join(config.sidecar_dir("2y"), "apex_spacing_errors_p05.pkl"))
        e = errs[(errs.series == args.series) & (errs.tf == args.tf) & (errs.method == "median")]
        const = float(e["pred_spacing"].iloc[0])
    else:
        const = args.const
    n = len(y)
    end = n - 1 if args.start is None else min(args.start + args.n_show, n - 1)
    start = max(end - args.n_show, 0)
    x = np.arange(start, end + 1)

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(15, 12.5), sharex=True,
                                                  gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5, 1.0], "hspace": 0.4})
    AXES = (ax0, ax1, ax2, ax3)
    fig.patch.set_facecolor(SURF)
    # --- candles
    o, h, l, c = (ohlc[k].to_numpy(dtype=float)[start:end + 1] for k in ("open", "high", "low", "close"))
    up = c >= o
    ax0.vlines(x, l, h, color=np.where(up, UP, DOWN), lw=0.9)
    ax0.bar(x, np.abs(c - o), bottom=np.minimum(o, c), width=0.7, color=np.where(up, UP, DOWN), lw=0)
    ax1.plot(x, cci[start:end + 1], color=INK, lw=1.0)
    ax1.axhline(0, color=INK2, lw=0.6)
    ax2.plot(x, y[start:end + 1], color=INK, lw=1.3)
    ax2.axhline(0, color=INK2, lw=0.6)
    rsi, rsi_ma = frame["rsi_14"].to_numpy(dtype=float), frame["rsi_ma8"].to_numpy(dtype=float)
    ax3.plot(x, rsi[start:end + 1], color=INK2, lw=0.9, label="rsi_14")
    ax3.plot(x, rsi_ma[start:end + 1], color=INK, lw=1.4, label="rsi_ma8")
    for lvl in (30, 50, 70):
        ax3.axhline(lvl, color=INK2, lw=0.6, ls=":" if lvl != 50 else "-")
    # RSI move zones: frozen sym0 7-class of rsi_ma8_diff (−3 … +3) as a background band on the RSI panel
    if "move_class_sym0" in ohlc.columns:
        mc = ohlc["move_class_sym0"].to_numpy(dtype=float)
        MC_COL = {-3: "#b2182b", -2: "#ef8a62", -1: "#fddbc7", 0: None, 1: "#d1e5f0", 2: "#67a9cf", 3: "#2166ac"}
        i = start
        while i <= end:
            k = mc[i]
            j = i
            while j + 1 <= end and mc[j + 1] == k:
                j += 1
            if not np.isnan(k) and MC_COL.get(int(k)):
                ax3.axvspan(i - 0.5, j + 0.5, ymin=0.0, ymax=0.14, color=MC_COL[int(k)], alpha=0.95, lw=0)
            i = j + 1
        for k, c in MC_COL.items():
            if c:
                ax3.fill_between([], [], [], color=c, alpha=0.6, label=f"move {k:+d}")
    ax3.legend(loc="upper left", fontsize=7, frameon=False, ncol=8)
    # --- real apexes
    a = idx[(idx >= start) & (idx <= end)]
    ax2.scatter(a, y[a], s=40, facecolor=SURF, edgecolor=APEX, lw=1.6, zorder=4, label="real apex (cci_ma8)")
    for t in a:
        for ax in AXES:
            ax.axvline(t, color=APEX, lw=0.7, alpha=0.45)
    zone_txt = ""
    if args.zones:
        conf = args.confirm if args.confirm is not None else 3
        z = build_zones(idx, sgn, const, conf, args.window, n)
        for ax in AXES:
            i = start
            while i <= end:
                if z[i] == 0:
                    i += 1
                    continue
                j = i
                while j + 1 <= end and z[j + 1] == z[i]:
                    j += 1
                ax.axvspan(i - 0.5, j + 0.5, color=UP if z[i] > 0 else DOWN, alpha=0.13, lw=0)
                i = j + 1
        ax2.fill_between([], [], [], color=UP, alpha=0.3, label="buy zone (bottom apex + confirm → prediction + window)")
        ax2.fill_between([], [], [], color=DOWN, alpha=0.3, label="sell zone (top apex + confirm → prediction + window)")
        # combined = cci zone (+1 buy / −1 sell / 0) + 0_rsi (move class with ±3 → 0)
        if "move_class_sym0" in ohlc.columns:
            mc_all = ohlc["move_class_sym0"].to_numpy(dtype=float)
            rsi0 = np.where(np.abs(mc_all) == 3, 0, np.nan_to_num(mc_all, nan=0.0)).astype(int)
            comb = z + rsi0
            CMB = {-3: "#b2182b", -2: "#ef8a62", -1: "#fddbc7", 0: None, 1: "#d1e5f0", 2: "#67a9cf", 3: "#2166ac"}
            i = start
            while i <= end:
                k = int(comb[i]); j = i
                while j + 1 <= end and comb[j + 1] == k:
                    j += 1
                if CMB.get(k):
                    ax4.axvspan(i - 0.5, j + 0.5, color=CMB[k], alpha=0.9, lw=0)
                i = j + 1
            ax4.step(x, comb[start:end + 1], where="mid", color=INK, lw=1.2)
            ax4.set_ylim(-3.5, 3.5); ax4.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            cl = ohlc["close"].to_numpy(float)
            rows = []
            for k in range(-3, 4):
                m = comb == k
                rec = {"combined": k, "n": int(m.sum()), "share": float(m.mean())}
                for h in (1, 3, 5):
                    r = np.full(n, np.nan); r[: n - h] = cl[h:] / cl[: n - h] - 1
                    rec[f"ret{h}_bp"] = float(1e4 * np.nanmean(r[m])) if m.sum() else np.nan
                    rec[f"up{h}"] = float(np.nanmean(r[m] > 0)) if m.sum() else np.nan
                rows.append(rec)
            ct = pd.DataFrame(rows)
            print(f"combined = cci_zone + 0_rsi, tf{args.tf} {args.dataset}, forward close return by value:\n{ct.round(3).to_string(index=False)}")
            ct.to_csv(os.path.join(config.sidecar_dir("2y"), f"combined_zone_rsi0_tf{args.tf}_{args.dataset}.csv"), index=False)
        st = zone_stats(z, ohlc["close"].to_numpy(float), ohlc["high"].to_numpy(float), ohlc["low"].to_numpy(float), args.tf)
        if len(st):
            g = st.groupby("zone").agg(n=("ret_bp", "size"), hit=("ret_bp", lambda r: (r > 0).mean()), mean_bp=("ret_bp", "mean"),
                                      median_bp=("ret_bp", "median"), mfe_bp=("mfe_bp", "median"), mae_bp=("mae_bp", "median"), len_med=("len", "median"))
            print(f"zone stats tf{args.tf} {args.dataset} (whole dataset, close start→end signed by zone):\n{g.round(2).to_string()}")
            zone_txt = "\nzones over the whole dataset (close start→end, signed): " + ", ".join(
                f"{k}: n {int(r.n)}, hit {r.hit:.2f}, mean {r.mean_bp:+.0f} bp" for k, r in g.iterrows())
            st.to_csv(os.path.join(config.sidecar_dir("2y"), f"zones_{args.series}_tf{args.tf}_{args.dataset}_c{conf}_w{args.window}.csv"), index=False)
    if args.confirm is not None:
        for t in a:
            for ax in AXES:
                ax.axvline(t + args.confirm, color=CONF, lw=1.1, ls=":", alpha=0.9)
        ax2.plot([], [], color=CONF, ls=":", lw=1.1, label=f"apex confirmed (+{args.confirm} candles)")
    # --- predicted next apex from each real apex: t + const, window
    first = True
    hi_all, lo_all = ohlc["high"].to_numpy(dtype=float), ohlc["low"].to_numpy(dtype=float)
    px_hits, px_n = 0, 0
    for t in idx[(idx + const - args.shift >= start) & (idx <= end)]:
        p = t + const - args.shift
        if p > end:
            continue
        # does the ±3 window contain a local price extreme (highest high / lowest low of ±8 candles)?
        lo_w, hi_w = int(round(p - args.window)), int(round(p + args.window))
        if lo_w - 5 >= 0 and hi_w + 5 < n:
            px_n += 1
            wide_h, wide_l = hi_all[lo_w - 5: hi_w + 6], lo_all[lo_w - 5: hi_w + 6]
            win_h, win_l = hi_all[lo_w: hi_w + 1], lo_all[lo_w: hi_w + 1]
            if win_h.max() >= wide_h.max() or win_l.min() <= wide_l.min():
                px_hits += 1
        for ax in AXES:
            if not args.zones:
                ax.axvspan(p - args.window, p + args.window, color=WIN, alpha=0.12, lw=0)
            ax.axvline(p, color=PRED, lw=1.2, ls="--", alpha=0.9)
        ax2.annotate("", xy=(p, y[t]), xytext=(t, y[t]), arrowprops=dict(arrowstyle="->", color=PRED, lw=1.0, alpha=0.8))
        if first:
            lab = f"predicted next apex = apex + {const:.0f} candles (train median)" + (f" − {args.shift:.0f} (MA delay)" if args.shift else "") + f", window ±{args.window}"
            ax2.plot([], [], color=PRED, ls="--", lw=1.2, label=lab)
            first = False
    hits = [t for t in idx[(idx >= start) & (idx <= end)]]
    # share of predictions (issued inside the window) whose real next apex falls within ±3
    issued = idx[(idx >= start - int(const) - 3) & (idx + const <= end)]
    ok = 0
    for i in np.where(np.isin(idx, issued))[0]:
        if i + 1 < len(idx) and abs(idx[i + 1] - (idx[i] + const - args.shift)) <= args.window:
            ok += 1
    ts = frame.index
    shift_txt = f", shifted {args.shift:.0f} candles earlier" if args.shift else ""
    _style(ax0, f"tf{args.tf} {args.dataset}: price (OHLC)  {ts[start]:%Y-%m-%d %H:%M} → {ts[end]:%Y-%m-%d %H:%M}\n"
                f"real cci_ma8 apexes (blue), predicted next apex (yellow dashed{shift_txt}) ±{args.window} candles — "
                f"{px_hits}/{px_n} windows contain a local price extreme (±8-candle high/low), {ok}/{len(issued)} within ±{args.window} of the next cci_ma8 apex{zone_txt}")
    print(f"tf{args.tf} start {start} shift {args.shift:g} window {args.window}: price-extreme windows {px_hits}/{px_n}, cci_ma8 apex within ±{args.window}: {ok}/{len(issued)}")
    _style(ax1, "cci_14")
    _style(ax2, f"{args.series}")
    _style(ax3, "rsi_14 + rsi_ma8 — bottom strip = RSI move class (frozen sym0 cuts on rsi_ma8_diff, −3 … +3; 0 = neutral, blank)")
    _style(ax4, "combined = cci zone (+1 buy / −1 sell / 0) + 0_rsi (RSI move class with ±3 set to 0)")
    ax3.set_ylim(0, 100)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.08, 0.98), ncol=3, fontsize=7.5, frameon=False)
    ax4.set_xlabel("closed-candle index", fontsize=8, color=INK2)
    out_dir = os.path.join(config.sidecar_dir("2y"), "plots")
    sfx = (f"_shift{args.shift:g}" if args.shift else "") + (f"_w{args.window}" if args.window != 3 else "") + (f"_c{args.confirm}" if args.confirm is not None else "") + ("_zones" if args.zones else "")
    out = os.path.join(out_dir, f"price_apex_tf{args.tf}_{args.dataset}_{start}{sfx}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
