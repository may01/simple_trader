#!/usr/bin/env python3
"""One-shot: add the candle-bound (CB) columns to an already-prepared dataset.

Avoids a full re-prep. Steps on the loaded ``df_with_indicators.pkl``:

1. Backfill ``{tf}_ema_25_diff_prc`` / ``{tf}_ema_25_slope_class`` (tf in
   15/60/240) where absent — datasets baked before those fields existed.
   Forming ``{tf}_ema_25`` against the previous candle's EMA as the field's
   indicator window sees it (see ``_ema_prev``).
2. Drop any existing CB columns (``stage.expected_columns()``), recompute them
   with ``stage.compute_cb_columns`` (the same call prep makes) and append
   everything with one concat.

``{tf}_move_class`` must already be the 7-class one
(``scripts/move_class7_materialize.py``). Idempotent: a second run yields an
identical frame.

Run via the dataset env, inside the container (the volume is root-owned):

    TRAIN_ENV=configs/oos2m_dataset.env docker compose -p cbtest run --rm ohlc_gen \\
        python3 scripts/cb_markers_materialize.py

First run writes a one-time ``.precb_bak`` hard-link backup.
"""
import os
import sys

import pandas as pd
import talib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import INDICATOR_WINDOW_ROWS  # noqa: E402
from helpers import wide_df_path  # noqa: E402
from indicators.candle_bounds import kernel  # noqa: E402
from indicators.candle_bounds.params import TFS  # noqa: E402
from indicators.candle_bounds.stage import compute_cb_columns, expected_columns, required_columns  # noqa: E402
from indicators.library.classification import ema_slope_class  # noqa: E402

EMA_LENGTH = 25


def _ema_prev(df: pd.DataFrame, tf: int) -> pd.Series:
    """The previous-candle EMA-25 exactly as ``EmaDiffPrcField`` sees it.

    The field runs on a ``build_indicator_input`` slice: INDICATOR_WINDOW_ROWS
    rows ending at the row, so its ``ema[k-1]`` is a talib EMA over the
    INDICATOR_WINDOW_ROWS - 1 closed candles ending at the previous close —
    one candle fewer than the window behind the stored closed ``{tf}_ema_25``
    (talib seeds on the window's first 25 values, so the two differ by
    ~1e-4 % once a tf has more than INDICATOR_WINDOW_ROWS closed candles).
    Recomputed per closed candle where the frame holds that window; before
    that (the frame's first candles, whose window reached into the trimmed
    warm-up) the stored closed ``{tf}_ema_25`` is the nearest available value
    (exact when the dataset starts with no history). Broadcast with
    ``kernel.to_rows`` (last candle closed strictly before the row)."""
    closed = (df[f"{tf}_is_closed"] == True).to_numpy()  # noqa: E712
    c = df.loc[closed]
    prev = c[f"{tf}_ema_25"].to_numpy(dtype=float).copy()
    closes = c[f"{tf}_close"].to_numpy(dtype=float)
    w = INDICATOR_WINDOW_ROWS - 1
    for j in range(w - 1, len(closes)):
        prev[j] = talib.EMA(closes[j - w + 1:j + 1], timeperiod=EMA_LENGTH)[-1]
    return kernel.to_rows(pd.Series(prev, index=c.index), df.index)


def _ema_slope_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The absent ``{tf}_ema_25_diff_prc`` / ``{tf}_ema_25_slope_class`` columns."""
    cols: dict[str, object] = {}
    for tf in TFS:
        diff_col, cls_col = f"{tf}_ema_25_diff_prc", f"{tf}_ema_25_slope_class"
        if diff_col in df.columns and cls_col in df.columns:
            continue
        if diff_col in df.columns:
            diff = df[diff_col].astype(float)
        else:
            prev = _ema_prev(df, tf)
            diff = (df[f"{tf}_ema_25"].astype(float) - prev) / prev * 100.0
            cols[diff_col] = diff
        if cls_col not in df.columns:
            cols[cls_col] = pd.Series(ema_slope_class(diff, tf), index=df.index)
    return pd.DataFrame(cols, index=df.index)


def backfill_ema_slope(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus ``{tf}_ema_25_diff_prc`` / ``{tf}_ema_25_slope_class`` for tf
    in TFS where absent; existing columns are never overwritten."""
    new = _ema_slope_columns(df)
    if new.shape[1] == 0:
        return df
    return pd.concat([df, new], axis=1)


def materialize(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """``df`` with the EMA backfill and freshly computed CB columns appended."""
    missing = [f"{tf}_move_class" for tf in TFS if f"{tf}_move_class" not in df.columns]
    if missing:
        raise ValueError(
            f"cb_markers_materialize: {missing} absent — run "
            "scripts/move_class7_materialize.py on this dataset first")
    new = _ema_slope_columns(df)
    base = df.drop(columns=[c for c in expected_columns() if c in df.columns])
    inputs = pd.concat(
        [base[[c for c in required_columns() if c in base.columns]], new], axis=1)
    cb = compute_cb_columns(inputs, pair)
    if cb is None:
        raise RuntimeError(f"cb_markers_materialize: no candle-bounds model for pair {pair!r}")
    return pd.concat([base, new, cb], axis=1)


def main() -> None:
    path = wide_df_path()
    pair = os.environ["PAIR"]
    df = pd.read_pickle(path)
    print(f"[cb] input  {df.shape} <- {path}", flush=True)
    out = materialize(df, pair)
    lost = [c for c in df.columns if c not in out.columns]
    if lost or not out.index.equals(df.index):
        raise RuntimeError(f"[cb] output lost columns {lost[:10]} or changed the index; not writing")
    del df
    print(f"[cb] output {out.shape}", flush=True)

    bak = path + ".precb_bak"
    if not os.path.exists(bak):           # one-time; never overwritten
        os.link(path, bak)
        if not os.path.samefile(bak, path) or os.path.getsize(bak) != os.path.getsize(path):
            raise RuntimeError(f"[cb] backup {bak} does not match {path}; not writing")
        print(f"[cb] backup {bak} created ({os.path.getsize(bak)} bytes)", flush=True)
    elif os.path.getsize(bak) == 0:
        raise RuntimeError(f"[cb] backup {bak} is empty; not writing")
    else:
        print(f"[cb] backup {bak} kept ({os.path.getsize(bak)} bytes)", flush=True)

    tmp = path + ".tmp"
    try:
        out.to_pickle(tmp)
        os.rename(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"[cb] wrote {path} ({os.path.getsize(path)} bytes)", flush=True)


if __name__ == "__main__":
    main()
