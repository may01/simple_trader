#!/usr/bin/env python3
"""Gate G8: live CB stage replayed against the prepared dataset (phase 18).

For the last ``--minutes`` minutes of a prepared dataset with CB columns, one
tick at the end of each minute:

1. build the per-tf frames the way live does from the dataset's raw 1-min
   OHLCV up to that minute — resample to each tf (``_resample_to_tf`` rule),
   ``tail(120)``, rename to ``{tf}_*``, ``is_closed = True``, run
   ``Indicators.compute(LiveDataPoint({tf: df}), tf)``;
2. ``LiveCbStage.update(ohlc)`` (one extra tick before the window so the held
   forming bound is set on the first measured minute);
3. compare each CB value with the prep row of that minute.

Prints, per column family and tf, the bool mismatch rate or the float max /
median relative difference (rows where both sides are finite) plus the rate of
rows finite on one side only; then the same for the stage's inputs (forming
row and last closed candle, live frame vs prep) to localise a difference; then
mean / p95 ``update`` time. Read-only on the dataset.

    TRAIN_ENV=configs/oos2m_dataset.env docker compose -p cbtest run --rm ohlc_gen \\
        python3 scripts/cb_live_parity.py --minutes 600

Not covered: live candles come from exchange klines at four base intervals,
gap-filled (``resample().ffill()``) before resampling (this replay resamples
stored 1-min data); fetch-timing skew between the tf frames of one tick (1m
fetched before 15m / 1h); and live's last tick of a minute can precede the
minute's final trade.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import LiveDataPoint  # noqa: E402
from helpers import wide_df_path  # noqa: E402
from indicators import Indicators  # noqa: E402
from indicators.candle_bounds.params import TFS  # noqa: E402
from indicators.candle_bounds.stage import (  # noqa: E402
    _TF_INPUTS, LiveCbStage, _is_bool_column, expected_columns,
)
from stocks.base_stock import StockInterface  # noqa: E402

# Agreed with the user after the first measurement; None = report only.
MAX_BOOL_MISMATCH_RATE: float | None = None
MAX_FLOAT_REL_DIFF: float | None = None

LIVE_TFS = (1, *TFS)
HISTORY_ROWS = 120
_RAW = {"1_open": "open", "1_high": "high", "1_low": "low", "1_close": "close",
        "1_volume": "volume", "1_buy_volume": "taker_base_vol"}
_CLASS_INPUTS = ("move_class", "ema_25_slope_class")


def live_frames(raw: pd.DataFrame, t: pd.Timestamp) -> dict[int, pd.DataFrame]:
    """Per-tf frames as ``LiveData.build_candles`` holds them at minute ``t``."""
    hist = raw.loc[t - pd.Timedelta(minutes=max(TFS) * (HISTORY_ROWS + 1)):t]
    ohlc = {}
    for tf in LIVE_TFS:
        df = hist if tf == 1 else StockInterface._resample_to_tf(None, hist, 1, tf)
        df = df.dropna(how="all").tail(HISTORY_ROWS).copy()
        df = df.rename(columns={c: f"{tf}_{c}" for c in df.columns})
        df[f"{tf}_is_closed"] = True
        df[f"{tf}_buy_volume"] = df[f"{tf}_taker_base_vol"]
        Indicators.compute(LiveDataPoint({tf: df}), tf)
        ohlc[tf] = df
    return ohlc


def _rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    """(max rel, median rel, one-side-finite rate, n both finite)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    one = float(np.mean(fa != fb)) if len(a) else float("nan")
    if not both.any():
        return float("nan"), float("nan"), one, 0
    denom = np.maximum(np.abs(b[both]), 1e-12)
    r = np.abs(a[both] - b[both]) / denom
    return float(r.max()), float(np.median(r)), one, int(both.sum())


def compare(live: pd.DataFrame, prep: pd.DataFrame) -> tuple[list[str], float, float]:
    """Report lines + worst bool mismatch rate + worst float max rel diff."""
    lines = [f"{'tf':>4} {'family':<22} {'kind':<5} {'n':>5} {'mismatch':>9} "
             f"{'max_rel':>10} {'med_rel':>10} {'one_side':>9}"]
    worst_bool, worst_float = 0.0, 0.0
    for tf in TFS:
        for col in [c for c in expected_columns() if c.startswith(f"{tf}_")]:
            fam = col.split("_", 1)[1]
            a = live[col].to_numpy(dtype=float)
            if _is_bool_column(col):
                b = prep[col].fillna(False).astype(bool).to_numpy(dtype=float)
                rate = float(np.mean(a != b))
                worst_bool = max(worst_bool, rate)
                lines.append(f"{tf:>4} {fam:<22} {'bool':<5} {len(a):>5} {rate:>9.4f}")
            else:
                b = pd.to_numeric(prep[col], errors="coerce").to_numpy(dtype=float)
                mx, md, one, n = _rel(a, b)
                if n:
                    worst_float = max(worst_float, mx)
                lines.append(f"{tf:>4} {fam:<22} {'float':<5} {n:>5} {'':>9} "
                             f"{mx:>10.3e} {md:>10.3e} {one:>9.4f}")
    return lines, worst_bool, worst_float


def compare_inputs(live_in: dict[str, list[float]], prep: pd.DataFrame,
                   ticks: pd.DatetimeIndex) -> list[str]:
    """Stage inputs, live vs prep: forming row at t, and last closed candle."""
    lines = [f"{'tf':>4} {'input':<22} {'row':<7} {'mismatch':>9} {'max_rel':>10} {'med_rel':>10}"]
    for tf in TFS:
        closed = prep.index[(prep[f"{tf}_is_closed"] == True).to_numpy()]  # noqa: E712
        for name in [n for n in _TF_INPUTS if n != "is_closed"]:
            col = f"{tf}_{name}"
            pv = pd.to_numeric(prep[col], errors="coerce")
            for row in ("forming", "closed"):
                if row == "forming":
                    b = pv.reindex(ticks).to_numpy(dtype=float)
                else:
                    # last candle closed strictly before t's candle opened
                    pos = closed.searchsorted(ticks.floor(f"{tf}min"), side="left") - 1
                    b = pv.reindex(closed[np.maximum(pos, 0)]).to_numpy(dtype=float, copy=True)
                    b[pos < 0] = np.nan                  # no closed candle yet (dataset start)
                a = np.asarray(live_in[f"{col}|{row}"], dtype=float)
                if name in _CLASS_INPUTS:
                    rate = float(np.mean((a != b) & ~(np.isnan(a) & np.isnan(b))))
                    lines.append(f"{tf:>4} {name:<22} {row:<7} {rate:>9.4f}")
                else:
                    mx, md, _, _ = _rel(a, b)
                    lines.append(f"{tf:>4} {name:<22} {row:<7} {'':>9} {mx:>10.3e} {md:>10.3e}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--minutes", type=int, default=600)
    args = ap.parse_args()

    path = wide_df_path()
    print(f"dataset: {path}")
    prep_full = pd.read_pickle(path)
    missing = [c for c in expected_columns() if c not in prep_full.columns]
    if missing:
        print(f"dataset has no CB columns ({len(missing)} missing); materialize first")
        return 2
    raw = prep_full[list(_RAW)].rename(columns=_RAW)
    in_cols = [f"{tf}_{n}" for tf in TFS for n in _TF_INPUTS]
    prep = prep_full[expected_columns() + in_cols].copy()
    del prep_full

    stage = LiveCbStage(os.environ["PAIR"])
    if not stage.enabled:
        print(f"no CB model for pair {os.environ['PAIR']!r}")
        return 2
    ticks = prep.index[-args.minutes:]
    warm = prep.index[-args.minutes - 1]
    stage.update(live_frames(raw, warm))

    rows: list[np.ndarray] = []
    live_in: dict[str, list[float]] = {}
    times_ms: list[float] = []
    for t in ticks:
        ohlc = live_frames(raw, t)
        for tf in TFS:
            last, prev = ohlc[tf].iloc[-1], ohlc[tf].iloc[-2]
            for n in _TF_INPUTS:
                if n == "is_closed":
                    continue
                live_in.setdefault(f"{tf}_{n}|forming", []).append(float(last[f"{tf}_{n}"]))
                live_in.setdefault(f"{tf}_{n}|closed", []).append(float(prev[f"{tf}_{n}"]))
        t0 = time.perf_counter()
        stage.update(ohlc)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        rows.append(np.concatenate([
            ohlc[tf][[c for c in expected_columns() if c.startswith(f"{tf}_")]].iloc[-1]
            .to_numpy(dtype=float) for tf in TFS
        ]))
    live = pd.DataFrame(np.vstack(rows), index=ticks, columns=expected_columns())
    prep_t = prep.loc[ticks]

    print(f"window: {ticks[0]} .. {ticks[-1]} ({len(ticks)} ticks)")
    lines, worst_bool, worst_float = compare(live, prep_t)
    print("\n== CB columns: live stage vs prep ==")
    print("\n".join(lines))
    print("\n== stage inputs: live frames vs prep ==")
    print("\n".join(compare_inputs(live_in, prep, ticks)))
    tm = np.asarray(times_ms)
    print(f"\nupdate time ms: mean={tm.mean():.1f} p95={np.percentile(tm, 95):.1f} "
          f"max={tm.max():.1f}")
    print(f"worst bool mismatch rate={worst_bool:.4f}  worst float max rel diff={worst_float:.3e}")

    fail = False
    if MAX_BOOL_MISMATCH_RATE is not None and worst_bool > MAX_BOOL_MISMATCH_RATE:
        print(f"FAIL bool mismatch {worst_bool:.4f} > {MAX_BOOL_MISMATCH_RATE}")
        fail = True
    if MAX_FLOAT_REL_DIFF is not None and worst_float > MAX_FLOAT_REL_DIFF:
        print(f"FAIL float rel diff {worst_float:.3e} > {MAX_FLOAT_REL_DIFF}")
        fail = True
    if MAX_BOOL_MISMATCH_RATE is None or MAX_FLOAT_REL_DIFF is None:
        print("thresholds not agreed yet: report only")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
