#!/usr/bin/env python3
"""export_executor_wide_df.py — main/'s wide 1-min frame from the executor's own
stored live data, so ``view_full.py`` can show exactly what trade_executor saw.

main/'s live loop keeps no frame on disk (``LiveCbStage`` writes each tick's
CB columns into the last row only), but the executor persists everything it
consumed: 1-min candles (``candle``) and every indicator reading main/
broadcast (``indicators`` — ``{tf}_cb_zone_long``, ``{tf}_cb_inzone_long``,
``{tf}_cb_f_*``, ``{tf}_ema_*``, ``1_sar_002_02`` …). This script reads both
through the executor visualiser's HTTP API, rebuilds the wide frame with the
same ``data._build_wide_df`` the prep pipeline uses, samples each indicator per
minute with the hold-until-next-row-or-expiry rule the executor chart applies,
and writes ``df_with_indicators.pkl`` into the live dataset folder
(``helpers.wide_df_path()`` under ``configs/live.env``). ``view_full.py`` then
renders it like any prepared dataset — CB zones, markers and filters included,
with the column names the viewer already knows.

Run inside the ``simple_trader`` image (same pandas as the viewer), attached to
the executor's compose network so ``http://visualizer:8090`` resolves::

    docker run --rm --network cbov-check_default \\
      -v "$PWD":/code -v simple_trader_vol:/trader_data \\
      --env-file configs/live.env -w /code simple_trader \\
      python3 scripts/export_executor_wide_df.py --hours 42

Then ``docker run -d --name view-full-live -p 8080:8080 -v "$PWD":/code
-v simple_trader_vol:/trader_data --env-file configs/live.env simple_trader
python3 view_full.py none``. Re-run the export and restart the viewer to
refresh: the dashboard loads the frame once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

MINUTE_MS = 60_000


def fetch(base: str, path: str, params: dict) -> object:
    url = f"{base}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=300) as resp:
        return json.load(resp)


def fetch_candles(base: str, pair: str, start_ms: int, end_ms: int, chunk_ms: int) -> pd.DataFrame:
    """1-min candles over ``[start_ms, end_ms]`` from ``/api/pair_history``,
    chunked because that route also returns every trade in the window."""
    frames = []
    for lo in range(start_ms, end_ms, chunk_ms):
        hi = min(lo + chunk_ms, end_ms)
        data = fetch(base, "/api/pair_history", {"pair": pair, "from": lo, "to": hi})
        rows = [c for c in data.get("candles", []) if c.get("interval") in ("1m", "1min", "1")]
        if rows:
            frames.append(pd.DataFrame(rows))
        print(f"  candles {lo}..{hi}: {len(rows)} rows", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates("open_time", keep="last")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]]


def fetch_indicators(base: str, pair: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = fetch(base, "/api/pair_indicators", {"pair": pair, "from": start_ms, "to": end_ms})
    print(f"  indicators: {len(rows)} folded rows", file=sys.stderr)
    if not rows:
        return pd.DataFrame(columns=["name", "value", "received_at", "expires_at"])
    df = pd.DataFrame(rows)[["name", "value", "received_at", "expires_at"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["received_at"] = df["received_at"].astype("int64")
    df["expires_at"] = df["expires_at"].astype("int64")
    return df


def sample_indicators(ind: pd.DataFrame, minutes: pd.DatetimeIndex, now_ms: int) -> pd.DataFrame:
    """One column per ``{tf}_name``: the value the executor held at each
    minute's last millisecond (or now, if sooner) — the last row received at
    or before that instant, if the instant is before its expiry."""
    # Not ``astype("int64")``: pandas 3 keeps ``to_datetime(unit="ms")`` at
    # millisecond resolution, so the raw integers would be ms, not ns.
    minute_ms = ((minutes - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).to_numpy().astype("int64")
    sample_ms = np.minimum(minute_ms + MINUTE_MS - 1, now_ms)
    out: dict[str, np.ndarray] = {}
    for name, rows in ind.groupby("name", sort=True):
        if not name[:1].isdigit():
            continue  # only main/'s ``{tf}_`` names belong in the wide frame
        rows = rows.sort_values("received_at")
        recv = rows["received_at"].to_numpy()
        pos = np.searchsorted(recv, sample_ms, side="right") - 1
        valid = pos >= 0
        safe = np.where(valid, pos, 0)
        exp = rows["expires_at"].to_numpy()[safe]
        val = rows["value"].to_numpy(dtype=float)[safe]
        live = valid & (sample_ms < exp)
        out[name] = np.where(live, val, np.nan)
    return pd.DataFrame(out, index=minutes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=os.environ.get("EXECUTOR_VISUALIZER_URL", "http://visualizer:8090"))
    ap.add_argument("--pair", default=os.environ.get("EXECUTOR_PAIR", "LINKUSDT"))
    ap.add_argument("--hours", type=float, default=48.0, help="look-back from now")
    ap.add_argument("--chunk-hours", type=float, default=4.0, help="pair_history window per request")
    ap.add_argument("--out", default=None, help="default: helpers.wide_df_path() (live dataset folder)")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from data import _build_wide_df  # noqa: E402
    from helpers import wide_df_path  # noqa: E402

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(args.hours * 3600 * 1000)
    print(f"export {args.pair} from {args.base}, last {args.hours:g}h", file=sys.stderr)

    candles = fetch_candles(args.base, args.pair, start_ms, now_ms, int(args.chunk_hours * 3600 * 1000))
    if candles.empty:
        print("no 1-min candles in the window — nothing to export", file=sys.stderr)
        return 1
    # A dense minute grid: gaps stay NaN (the viewer drops NaN-close rows).
    minutes = pd.date_range(candles.index.min(), candles.index.max(), freq="1min")
    candles = candles.reindex(minutes)
    # The executor's candles carry no taker volume; buy_volume stays NaN.
    candles["taker_base_vol"] = np.nan
    wide = _build_wide_df(candles)

    ind = fetch_indicators(args.base, args.pair, start_ms, now_ms)
    sampled = sample_indicators(ind, minutes, now_ms)
    # Indicator names main/ publishes are already ``{tf}_{name}``; a clash
    # with a wide column (none today) would keep the candle-derived one.
    sampled = sampled[[c for c in sampled.columns if c not in wide.columns]]
    wide = pd.concat([wide, sampled], axis=1)

    out = args.out or wide_df_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    wide.to_pickle(tmp)
    os.replace(tmp, out)

    cb_cols = [c for c in wide.columns if "_cb_" in c or "_cbnc_" in c or "_extdone_" in c]
    print(
        f"wrote {out}: {len(wide)} rows {wide.index.min()} .. {wide.index.max()}, "
        f"{wide.shape[1]} columns ({len(sampled.columns)} indicators, {len(cb_cols)} cb/cbnc/extdone)",
        file=sys.stderr,
    )
    for c in ("15_cb_zone_long", "15_cb_inzone_long", "60_cb_zone_long"):
        if c in wide.columns:
            s = wide[c]
            print(f"  {c}: {s.notna().sum()} minutes set, last={s.dropna().iloc[-1] if s.notna().any() else 'nan'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
