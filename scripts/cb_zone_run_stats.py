#!/usr/bin/env python3
"""One-off: run-length statistics of CB zone markers (phase 20, spec
2026-10-19-cb-zone-run-stats-design.md). Writes a json + md report, no params.

Per dataset (2y training frame + closed-bounds sidecar; oos2m materialized
frame), per filter set, per (tf, side): block run lengths, blocks per touched
candle, minutes per candle, first-block length, remaining-given-streak. The
freeze table is set C (``cb_inzone ∧ cb_f_overshoot``), 2y, median block
length — the executor's ``CB_STOP_EXPECTED_POINTS``.

    docker compose -p cbrun run --rm ohlc_gen python3 scripts/cb_zone_run_stats.py \\
        [--train-ind /trader_data_long/train/2y_az_link_usdt/df_with_indicators.pkl] \\
        [--train-cb  /trader_data_long/train/2y_link_usdt/df_with_candle_bounds.pkl] \\
        [--oos-dir   /trader_data_long/train/oos2m_link_usdt] [--out DIR] \\
        [--skip-train] [--recompute-cb] [--make-subset PATH]

``--make-subset PATH`` writes the 2y frame trimmed to the columns this script
needs (the retry path when the full frame does not fit beside the sidecar) and
exits. The default ``--out`` is on the data volume, not under /code: the
container runs as root and a root-owned file in the worktree blocks merges.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebooks", "zone_profitability"))

from indicators.candle_bounds import kernel  # noqa: E402
from indicators.candle_bounds.params import ATR_COL, DIRECTIONS, EXTDONE_LADDER, TFS  # noqa: E402
import zone_runs  # noqa: E402

VOL = "/trader_data_long/train"
TRAIN_IND = f"{VOL}/2y_az_link_usdt/df_with_indicators.pkl"
TRAIN_CB = f"{VOL}/2y_link_usdt/df_with_candle_bounds.pkl"
OOS_DIR = f"{VOL}/oos2m_link_usdt"
OUT = f"{VOL}/cb_zone_run_stats"

SETS: tuple[str, ...] = ("raw", "extdone", "overshoot", "entry")
FREEZE_SET = "overshoot"
S_MAX = 12


# ----------------------------------------------------------------------------- frame helpers

def candle_id(df: pd.DataFrame, tf: int) -> np.ndarray:
    """``evlib.candle_id``: rows after a close through the next closing row share an id."""
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    return np.concatenate(([0], np.cumsum(closed)[:-1]))


def minutes_of(index: pd.Index) -> np.ndarray:
    """Epoch minutes of a DatetimeIndex (int64); a gap in the frame shows as a jump.
    Unit-independent: the stored frames carry ``datetime64[ms, UTC]``, where
    ``asi8`` is milliseconds, not nanoseconds."""
    idx = pd.DatetimeIndex(index)
    epoch = pd.Timestamp("1970-01-01", tz=idx.tz)
    return ((idx - epoch) // pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)


def need_columns() -> list[str]:
    cols = ["1_high", "1_low", "1_close"]
    for tf in TFS:
        cols += [f"{tf}_{c}" for c in ("open", "high", "low", "close", "ema_25", "is_closed")]
    return cols


def optional_columns() -> list[str]:
    return [ATR_COL] + [f"{tf}_{c}" for tf in TFS for c in ("ema_25_slope_class", "move_class")]


def atr1_ma5(df: pd.DataFrame) -> np.ndarray:
    """``run_zone_ext.atr1_ma5``: Wilder ATR14 on 1-min OHLC, then MA5 — the series the
    2y extdone ladder was fit against (rebuilt here, not imported: that module
    creates its output directory at import time)."""
    h, l, c = df["1_high"], df["1_low"], df["1_close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return atr.rolling(5).mean().to_numpy()


# ----------------------------------------------------------------------------- masks

def _flag(df: pd.DataFrame, name: str) -> np.ndarray:
    """A pass flag: NaN / absent passes (the viewer's rule)."""
    if name not in df.columns:
        return np.ones(len(df), dtype=bool)
    v = df[name]
    return (v.isna() | (v.astype(float) != 0.0)).to_numpy(dtype=bool)


def _sets(inzone: np.ndarray, f_ext: np.ndarray, f_over: np.ndarray, f_trend: np.ndarray) -> dict[str, np.ndarray]:
    over = inzone & f_over
    return {"raw": inzone, "extdone": inzone & f_ext, "overshoot": over, "entry": over & f_trend}


def masks_from_columns(df: pd.DataFrame, tf: int, side: str) -> dict[str, np.ndarray]:
    """oos2m path: the stored main-flow columns."""
    inzone = df[f"{tf}_cb_inzone_{side}"].fillna(False).astype(bool).to_numpy()
    return _sets(inzone, _flag(df, f"{tf}_cb_f_extdone_{side}"), _flag(df, f"{tf}_cb_f_overshoot_{side}"),
                 _flag(df, f"{tf}_cb_f_trend_{side}"))


def masks_from_kernel(df: pd.DataFrame, cb: pd.DataFrame, tf: int, side: str) -> dict[str, np.ndarray]:
    """2y path: inzone from the bounds sidecar, flags from the main-flow kernel."""
    cb = cb.reindex(df.index)
    inzone = cb[f"{tf}_cb_inzone_{side}"].fillna(False).astype(bool).to_numpy()
    ext = kernel.extdone(df, tf, EXTDONE_LADDER[tf])
    over = kernel.overshoot(cb, df, tf)
    flt = kernel.filters(df, ext, over, tf)
    return _sets(inzone, _flag(flt, f"{tf}_cb_f_extdone_{side}"), _flag(flt, f"{tf}_cb_f_overshoot_{side}"),
                 _flag(flt, f"{tf}_cb_f_trend_{side}"))


def edge_mask(df: pd.DataFrame, cb: pd.DataFrame | None, tf: int, side: str) -> np.ndarray:
    """Spec §4 edge rule: drop the first and last candle of the tf and every
    candle with a NaN closed bound (the side the zone hangs on) on any row."""
    cid = candle_id(df, tf)
    ok = (cid != cid[0]) & (cid != cid[-1])
    bound = f"{tf}_cb_low" if side == "long" else f"{tf}_cb_high"
    src = cb if cb is not None and bound in cb.columns else df
    if bound in src.columns:
        finite = np.isfinite(src[bound].reindex(df.index).to_numpy(dtype=float))
        # A candle with any NaN bound row is dropped whole, not row by row.
        ok &= ~np.isin(cid, np.unique(cid[~finite]))
    return ok


def kernel_parity(df: pd.DataFrame) -> dict[str, int]:
    """Mismatch count per ``{tf}_{side}_{set}`` between the 2y path
    (``masks_from_kernel`` on the frame's stored bounds) and the oos2m path
    (``masks_from_columns`` on its stored flags), on a frame that has both.
    The record that the kernel path reproduces main/'s own flags."""
    bounds = [c for c in df.columns if "_cb_" in c and "_cb_f_" not in c and "overshoot" not in c]
    cb = df[bounds]
    out: dict[str, int] = {}
    for tf in TFS:
        for side in DIRECTIONS:
            a = masks_from_columns(df, tf, side)
            b = masks_from_kernel(df, cb, tf, side)
            for name in SETS:
                out[f"{tf}_{side}_{name}"] = int((a[name] != b[name]).sum())
    return out


# ----------------------------------------------------------------------------- loaders

def _backfill(df: pd.DataFrame) -> pd.DataFrame:
    """Absent ``{tf}_ema_25_slope_class`` backfilled (materialize's rule), absent
    ``{tf}_move_class`` NaN (``cb_f_mcx`` is not used here), absent ATR rebuilt."""
    cols: dict[str, object] = {}
    missing_slope = [tf for tf in TFS if f"{tf}_ema_25_slope_class" not in df.columns]
    if missing_slope:
        from scripts.cb_markers_materialize import _ema_slope_columns
        new = _ema_slope_columns(df)
        for c in new.columns:
            if c.endswith("_ema_25_slope_class"):
                cols[c] = new[c]
        print(f"[run_stats] ema_25_slope_class backfilled for tf {missing_slope}", flush=True)
    for tf in TFS:
        if f"{tf}_move_class" not in df.columns:
            cols[f"{tf}_move_class"] = np.nan
    if ATR_COL in df.columns:
        print(f"[run_stats] ATR: stored {ATR_COL} used", flush=True)
    else:
        cols[ATR_COL] = atr1_ma5(df)
        print(f"[run_stats] ATR: {ATR_COL} absent, rebuilt (Wilder ATR14 → MA5, the ladder's series)", flush=True)
    if not cols:
        return df
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    keep = need_columns() + [c for c in optional_columns() if c in df.columns]
    missing = [c for c in need_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"cb_zone_run_stats: training frame lacks {missing}")
    return df[keep].copy()


def load_train(ind_path: str, cb_path: str, recompute_cb: bool, pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    df = _trim(pd.read_pickle(ind_path))
    print(f"[run_stats] train {df.shape} {df.index[0]} -> {df.index[-1]} loaded in {time.time() - t0:.0f}s", flush=True)
    df = _backfill(df)
    if recompute_cb:
        from indicators.candle_bounds.stage import compute_cb_columns, required_columns
        cb = compute_cb_columns(df[required_columns()], pair)
        if cb is None:
            raise RuntimeError(f"no candle-bounds model for pair {pair!r}")
    else:
        cb = pd.read_pickle(cb_path).reindex(df.index)
    return df, cb


def load_oos(oos_dir: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(oos_dir, "df_with_indicators.pkl"))


def write_subset(ind_path: str, out_path: str) -> None:
    df = _trim(pd.read_pickle(ind_path))
    df.to_pickle(out_path)
    print(f"[run_stats] subset {df.shape} -> {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


# ----------------------------------------------------------------------------- stats

def _cell(mask: np.ndarray, cid: np.ndarray, minute: np.ndarray, s_max: int) -> dict:
    lengths = zone_runs.blocks(mask, cid, minute)
    cell = zone_runs.summarize(lengths, zone_runs.per_candle(mask, cid, minute))
    cell["remaining_given_streak"] = zone_runs.remaining_given_streak(lengths, s_max)
    return cell


def dataset_stats(df: pd.DataFrame, cb: pd.DataFrame | None, from_kernel: bool, s_max: int = S_MAX) -> dict:
    out: dict[str, dict[str, dict]] = {name: {} for name in SETS}
    minute = minutes_of(df.index)
    for tf in TFS:
        cid = candle_id(df, tf)
        for side in DIRECTIONS:
            masks = masks_from_kernel(df, cb, tf, side) if from_kernel else masks_from_columns(df, tf, side)
            edge = edge_mask(df, cb, tf, side)
            for name in SETS:
                out[name][f"{tf}_{side}"] = _cell(masks[name] & edge, cid, minute, s_max)
            print(f"[run_stats] {tf} {side}: " + " ".join(
                f"{n}={out[n][f'{tf}_{side}']['n_blocks']}blk/p50={out[n][f'{tf}_{side}']['run_len'].get('p50', 'n/a')}"
                for n in SETS), flush=True)
    return out


# ----------------------------------------------------------------------------- report

def _fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def _get(report: dict, ds: str, name: str, key: str, *path) -> object:
    try:
        v = report[ds][name][key]
        for p in path:
            v = v[p]
        return v
    except KeyError:
        return None


def render_md(report: dict) -> str:
    o = ["# CB zone run-length statistics", ""]
    for ds in ("2y", "oos2m"):
        if ds not in report:
            o += [f"## {ds}", "", "not run", ""]
            continue
        o += [f"## {ds}", ""]
        for name in SETS:
            o += [f"### set {name}", "",
                  "| tf | side | blocks | candles | run mean | p50 | p75 | p90 | max | blocks/candle mean | p50 | min/candle mean | p50 | first blk mean | p50 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for tf in TFS:
                for side in DIRECTIONS:
                    k = f"{tf}_{side}"
                    c = report[ds][name][k]
                    rl, bpc, mpc, fb = c["run_len"], c["blocks_per_candle"], c["minutes_per_candle"], c["first_block_len"]
                    o.append(f"| {tf} | {side} | {c['n_blocks']} | {c['n_candles']} | {_fmt(rl.get('mean'))} | {_fmt(rl.get('p50'))} | "
                             f"{_fmt(rl.get('p75'))} | {_fmt(rl.get('p90'))} | {_fmt(rl.get('max'))} | {_fmt(bpc.get('mean'))} | "
                             f"{_fmt(bpc.get('p50'))} | {_fmt(mpc.get('mean'))} | {_fmt(mpc.get('p50'))} | {_fmt(fb.get('mean'))} | {_fmt(fb.get('p50'))} |")
            o.append("")
        o += [f"### remaining given streak — set {FREEZE_SET} ({ds}; p50 / n)", "",
              "| tf | side | " + " | ".join(f"s={s}" for s in range(1, S_MAX + 1)) + " |",
              "|---|---|" + "---|" * S_MAX]
        for tf in TFS:
            for side in DIRECTIONS:
                r = report[ds][FREEZE_SET][f"{tf}_{side}"]["remaining_given_streak"]
                cells = []
                for s in range(1, S_MAX + 1):
                    e = r.get(str(s), r.get(s, {"n": 0}))
                    cells.append(f"{_fmt(e.get('p50'))} / {e['n']}")
                o.append(f"| {tf} | {side} | " + " | ".join(cells) + " |")
        o.append("")
    o += ["## Freeze table", "",
          f"Set `{FREEZE_SET}` = `cb_inzone ∧ cb_f_overshoot` (the entry check's overshoot veto only; not its firing set — "
          "the check also vetoes on `cb_f_trend`, needs its own trade-price MA class, and rule 2 fires off-zone).", "",
          "| tf | side | 2y C p50 | oos2m C p50 | 2y A p50 | 2y C mean | 2y D p50 |", "|---|---|---|---|---|---|---|"]
    parts = []
    for tf in TFS:
        vals = []
        for side in DIRECTIONS:
            k = f"{tf}_{side}"
            c2 = _get(report, "2y", FREEZE_SET, k, "run_len", "p50")
            o.append(f"| {tf} | {side} | {_fmt(c2)} | {_fmt(_get(report, 'oos2m', FREEZE_SET, k, 'run_len', 'p50'))} | "
                     f"{_fmt(_get(report, '2y', 'raw', k, 'run_len', 'p50'))} | {_fmt(_get(report, '2y', FREEZE_SET, k, 'run_len', 'mean'))} | "
                     f"{_fmt(_get(report, '2y', 'entry', k, 'run_len', 'p50'))} |")
            vals.append(_fmt(c2))
        parts.append(f"{tf}:{vals[0]}/{vals[1]}")
    o += ["", "```", "CB_STOP_EXPECTED_POINTS=" + ";".join(parts), "```", ""]
    return "\n".join(o)


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ind", default=TRAIN_IND)
    ap.add_argument("--train-cb", default=TRAIN_CB)
    ap.add_argument("--oos-dir", default=OOS_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--recompute-cb", action="store_true")
    ap.add_argument("--pair", default=os.environ.get("PAIR", "link_usdt"))
    ap.add_argument("--make-subset", default=None, metavar="PATH")
    ap.add_argument("--s-max", type=int, default=S_MAX)
    ap.add_argument("--parity-kernel", action="store_true",
                    help="run the kernel mask path against the stored flags of --oos-dir and report mismatches")
    a = ap.parse_args(argv)

    if a.make_subset:
        write_subset(a.train_ind, a.make_subset)
        return 0
    if a.parity_kernel:
        report = kernel_parity(load_oos(a.oos_dir))
        bad = {k: v for k, v in report.items() if v}
        print(f"[run_stats] kernel parity on {a.oos_dir}: {len(report)} masks, mismatches {bad or 0}", flush=True)
        return 1 if bad else 0

    t0 = time.time()
    report: dict = {"freeze_set": FREEZE_SET, "s_max": a.s_max}
    if not a.skip_train:
        df, cb = load_train(a.train_ind, a.train_cb, a.recompute_cb, a.pair)
        report["2y"] = dataset_stats(df, cb, from_kernel=True, s_max=a.s_max)
        del df, cb
    oos = load_oos(a.oos_dir)
    print(f"[run_stats] oos2m {oos.shape}", flush=True)
    report["oos2m"] = dataset_stats(oos, None, from_kernel=False, s_max=a.s_max)
    del oos

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["run"] = {"peak_rss_mb": round(rss_mb), "wall_s": round(time.time() - t0)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cb_zone_run_stats.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    md = render_md(report)
    with open(os.path.join(a.out, "cb_zone_run_stats.md"), "w") as fh:
        fh.write(md)
    print(md[md.index("## Freeze table"):], flush=True)
    print(f"[run_stats] peak RSS {rss_mb:.0f} MB, wall {time.time() - t0:.0f}s, wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
