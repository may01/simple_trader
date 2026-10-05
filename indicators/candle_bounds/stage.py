"""Candle-bounds (CB) stage entry point.

``compute_cb`` turns a 1-minute wide frame into every CB column; prep and the
materialize script call it unchanged.
``compute_cb_columns`` is the pair-aware wrapper that loads the frozen models
(and skips the stage, with one warning per pair, when none exist).
``LiveCbStage`` computes the current minute on every live tick without running
``compute_cb``: it caches the closed bounds / forming history per tf and
evaluates only the current row through ``tf_columns``. ``compute_cb`` (with
``nc_pred_prev=``) on ``LiveCbStage.build_wide`` / ``_wide`` is the prep-path
equivalent the tests and the parity script use as the oracle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.candle_bounds import kernel
from indicators.candle_bounds.models import ModelSet, load_models
from indicators.candle_bounds.params import ATR_COL, DIRECTIONS, EXTDONE_LADDER, SIDES, TFS
from logs import log_warning

NcPred = dict[tuple[int, str], tuple[float, float]]

_TF_INPUTS = (
    "open", "high", "low", "close", "is_closed", "rsi_14", "rsi_ma8",
    "macd_12_26_9", "macd_hist_12_26_9", "ema_25", "move_class", "ema_25_slope_class",
)
_WARNED_PAIRS: set[str] = set()


@dataclass(frozen=True)
class CbResult:
    columns: pd.DataFrame
    nc_pred_last: NcPred   # (tf, side) -> (bound, band) of the last row, unshifted


def required_columns() -> list[str]:
    cols = ["1_high", "1_low", "1_close", ATR_COL]
    for tf in TFS:
        cols += [f"{tf}_{name}" for name in _TF_INPUTS]
    return cols


def _tf_names(tf: int) -> list[str]:
    bands = ("", "_std", "_up", "_dn")
    names = [f"cb_{s}{b}" for s in SIDES for b in bands]
    names += [f"cb_zone_{d}" for d in DIRECTIONS] + [f"cb_inzone_{d}" for d in DIRECTIONS]
    names += [f"cbnc_{s}{b}" for s in SIDES for b in bands]
    for kind in ("cbnc_tgt", "cbnc_zone", "cbnc_inzone", "cbx_inzone"):
        names += [f"{kind}_{d}" for d in DIRECTIONS]
    names += [f"extdone_{s}" for s in SIDES] + [f"extdone_{s}_lvl" for s in SIDES]
    for kind in ("cb_f_extdone", "cb_f_trend", "cb_f_mcx"):
        names += [f"{kind}_{d}" for d in DIRECTIONS]
    return [f"{tf}_{n}" for n in names]


def expected_columns() -> list[str]:
    return [c for tf in TFS for c in _tf_names(tf)]


def _is_bool_column(name: str) -> bool:
    return ("inzone" in name or "_cb_f_" in name
            or (name.split("_", 1)[1].startswith("extdone_") and not name.endswith("_lvl")))


def _empty_columns(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype=bool if _is_bool_column(c) else float) for c in expected_columns()},
        index=index,
    )


def _held(nc_pred_prev: NcPred, tf: int, side: str) -> tuple[float, float]:
    bound, band = nc_pred_prev.get((tf, side), (np.nan, np.nan))
    return float(bound), float(band)


def closed_cb(df: pd.DataFrame, models: ModelSet, tf: int) -> pd.DataFrame:
    """Closed bounds of both sides for ``tf`` (``{tf}_cb_{side}[,_std,_up,_dn]``)."""
    return pd.concat([kernel.closed_bounds(df, models[(tf, s)]) for s in SIDES], axis=1)


def tf_columns(df: pd.DataFrame, models: ModelSet, tf: int, cb: pd.DataFrame, *,
               hist: dict[str, pd.DataFrame] | None = None,
               nc_pred_prev: NcPred | None = None) -> tuple[pd.DataFrame, NcPred]:
    """Every CB column of ``tf`` for ``df``'s rows, given its closed bounds
    ``cb`` (``closed_cb``, same index) and, optionally, precomputed
    ``kernel.forming_history`` rows per side (``hist``; default: from ``df``).
    Returns the columns (``_tf_names(tf)`` order) and the unshifted forming
    (bound, band) of the last row per side. See ``compute_cb`` for
    ``nc_pred_prev``."""
    cb = pd.concat([cb, kernel.closed_zones(cb, df, tf)], axis=1)

    nc_parts = []
    nc_pred_last: NcPred = {}
    for side in SIDES:
        pred = kernel.forming_prediction(df, models[(tf, side)],
                                         hist=None if hist is None else hist[side])
        nc_pred_last[(tf, side)] = (float(pred["bound"].iloc[-1]),
                                    float(pred["band"].iloc[-1]))
        nc = kernel.forming_bounds(pred, tf, side)
        if nc_pred_prev is not None:
            bound, band = _held(nc_pred_prev, tf, side)
            # columns of forming_bounds: cbnc, _std, _up, _dn
            nc.iloc[-1] = [bound, band, bound + band, bound - band]
        nc_parts.append(nc)
    cbnc = pd.concat(nc_parts, axis=1)
    zones = kernel.forming_zones(cbnc, cb, df, tf)

    ext = kernel.extdone(df, tf, EXTDONE_LADDER[tf])
    flt = kernel.filters(df, ext, tf)
    out = pd.concat([cb, cbnc, zones, ext, flt], axis=1)[_tf_names(tf)]
    return out, nc_pred_last


def compute_cb(wide_df: pd.DataFrame, models: ModelSet, *,
               nc_pred_prev: NcPred | None = None) -> CbResult:
    """All CB columns for ``wide_df`` (exactly ``expected_columns()``, same index).

    ``nc_pred_prev`` (live only): when given, the last row's ``cbnc_*`` come
    from it — ``bound``, ``band``, ``bound + band``, ``bound - band`` — instead
    of the one-row shift; a missing key or NaN value gives NaN. Hybrid zones and
    ``cbx`` of the last row follow from those values. ``nc_pred_last`` holds the
    unshifted (bound, band) of the last row for every (tf, side)."""
    missing = [c for c in required_columns() if c not in wide_df.columns]
    if missing:
        raise ValueError(f"candle_bounds: missing input columns: {missing}")
    df = wide_df[required_columns()].copy()
    if len(df) == 0:
        return CbResult(columns=_empty_columns(df.index), nc_pred_last={})

    parts: list[pd.DataFrame] = []
    nc_pred_last: NcPred = {}
    for tf in TFS:
        cols, last = tf_columns(df, models, tf, closed_cb(df, models, tf),
                                nc_pred_prev=nc_pred_prev)
        parts.append(cols)
        nc_pred_last.update(last)

    out = pd.concat(parts, axis=1)[expected_columns()]
    return CbResult(columns=out, nc_pred_last=nc_pred_last)


def _warn_no_model(pair: str) -> None:
    if pair not in _WARNED_PAIRS:
        _WARNED_PAIRS.add(pair)
        log_warning(f"candle_bounds: no model for pair {pair!r}; CB stage skipped")


def compute_cb_columns(wide_df: pd.DataFrame, pair: str | None = None) -> pd.DataFrame | None:
    """CB columns for ``pair`` (default ``os.environ["PAIR"]``), or None — after
    one warning per pair per process — when the pair has no model file."""
    if pair is None:
        pair = os.environ["PAIR"]
    models = load_models(pair)
    if models is None:
        _warn_no_model(pair)
        return None
    return compute_cb(wide_df, models).columns


# --------------------------------------------------------------------------- live

# Tied to kernel.closed_frame's widest window: V_std12 = rolling(12) over
# extensions, i.e. 13 closed candles. Fewer -> the tf is not warm (all NaN).
MIN_CLOSED_CANDLES = 13
_ONE_MIN = pd.Timedelta(minutes=1)
_ONE_MIN_INPUTS = ("1_high", "1_low", "1_close", ATR_COL)


@dataclass(frozen=True)
class _ClosedCache:
    key: tuple                          # (last closed candle open, closed-block bytes)
    cb: pd.DataFrame                    # closed_cb row at the current candle
    hist: dict[str, pd.DataFrame]       # forming_history row per side


class LiveCbStage:
    """CB columns for the current minute on every live tick.

    Same maths as ``compute_cb`` on the sparse wide frame of ``build_wide``
    (closed candles at their closing minute + one row for the current minute),
    split by what changes when: the closed bounds and the forming history of a
    tf depend only on its closed candles and are cached per (tf, closed
    block); every tick evaluates only the current row (``tf_columns`` on one
    row). No indicator computation.

    ``cbnc_*`` of the current minute are the forming prediction of the latest
    tick of the previous minute whose bound was finite (held per (tf, side));
    NaN after start, after a gap > 1 minute / clock going back, or when no
    tick of the previous minute had a usable frame for that tf.
    Writes ``{tf}_<name>`` into the last row of ``ohlc[tf]`` (bools as 1.0 /
    0.0, earlier rows NaN). A tf whose frame is missing, empty, lacks an input
    column, is stale or not yet warm gets NaN in all its columns (bools
    included: 0.0 would be a value) with one warning per (tf, cause); other
    tfs still compute. All tfs are computed before any is written."""

    def __init__(self, pair: str) -> None:
        self._models = load_models(pair)
        if self._models is None:
            _warn_no_model(pair)
        self._minute: pd.Timestamp | None = None
        self._fresh: NcPred = {}
        self._held: NcPred = {}
        self._cache: dict[int, _ClosedCache] = {}
        self._warned: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._models is not None

    def _warn_once(self, cause: str, message: str) -> None:
        if cause not in self._warned:
            self._warned.add(cause)
            log_warning(message)

    @staticmethod
    def _tf_cols(tf: int) -> list[str]:
        return [f"{tf}_{n}" for n in _TF_INPUTS if n != "is_closed"]

    @staticmethod
    def _usable_tfs(ohlc: dict[int, pd.DataFrame], cur: pd.Timestamp,
                    warn=None) -> list[int]:
        """tfs whose frame can be evaluated at ``cur``; ``warn(cause, message)``
        (optional) is called for every unusable one."""
        def skip(tf: int, cause: str, why: str) -> None:
            if warn is not None:
                warn(f"tf_{cause}:{tf}", f"candle_bounds live: tf={tf} frame {why}; "
                     "its CB columns are NaN")

        usable = []
        for tf in TFS:
            frame = ohlc.get(tf)
            if frame is None or len(frame) == 0:
                skip(tf, "missing", "missing or empty")
                continue
            absent = [c for c in LiveCbStage._tf_cols(tf) if c not in frame.columns]
            if absent:
                skip(tf, "column", f"lacks input column(s) {absent}")
                continue
            if frame.index[-1] != cur.floor(f"{tf}min"):
                skip(tf, "stale", f"is stale (last candle {frame.index[-1]}, "
                     f"current {cur.floor(f'{tf}min')})")
                continue
            if len(frame) - 1 < MIN_CLOSED_CANDLES:
                skip(tf, "short", f"is short ({len(frame) - 1} closed candles "
                     f"< {MIN_CLOSED_CANDLES})")
                continue
            usable.append(tf)
        return usable

    @staticmethod
    def _tf_part(frame: pd.DataFrame, tf: int, cur: pd.Timestamp) -> pd.DataFrame:
        """``tf``'s inputs: closed candles at their closing minute + the forming
        candle at ``cur``, with the stage's own ``is_closed`` flags."""
        part = frame[LiveCbStage._tf_cols(tf)].copy()
        close_ts = frame.index + pd.Timedelta(minutes=tf - 1)
        flags = np.ones(len(part), dtype=bool)
        flags[-1] = cur == close_ts[-1]
        part.index = close_ts[:-1].append(pd.DatetimeIndex([cur]))
        part[f"{tf}_is_closed"] = flags
        return part

    @staticmethod
    def _one_min_row(ohlc: dict[int, pd.DataFrame], cur: pd.Timestamp) -> pd.DataFrame:
        one = ohlc[1]
        last = {c: [float(one[c].iloc[-1]) if c in one.columns else np.nan] for c in _ONE_MIN_INPUTS}
        return pd.DataFrame(last, index=pd.DatetimeIndex([cur]))

    @staticmethod
    def _wide(ohlc: dict[int, pd.DataFrame], cur: pd.Timestamp, usable: list[int]) -> pd.DataFrame:
        cur_idx = pd.DatetimeIndex([cur])
        parts = []
        for tf in TFS:
            if tf in usable:
                parts.append(LiveCbStage._tf_part(ohlc[tf], tf, cur))
            else:
                part = pd.DataFrame(np.nan, index=cur_idx, columns=LiveCbStage._tf_cols(tf))
                part[f"{tf}_is_closed"] = False
                parts.append(part)
        parts.append(LiveCbStage._one_min_row(ohlc, cur))
        wide = pd.concat(parts, axis=1, sort=True)
        return wide[required_columns()]

    @staticmethod
    def build_wide(ohlc: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Sparse wide frame (``compute_cb`` input) whose last row is the current
        minute; ``update`` computes exactly ``compute_cb`` of it on that row."""
        cur = ohlc[1].index[-1]
        return LiveCbStage._wide(ohlc, cur, LiveCbStage._usable_tfs(ohlc, cur))

    def _closed(self, frame: pd.DataFrame, tf: int, cur: pd.Timestamp) -> _ClosedCache:
        """``frame``: the tf's input columns only (live frames are wide and
        fragmented; every selection on them is costly)."""
        closed_block = frame.iloc[:-1].to_numpy(dtype=float)
        key = (frame.index[-2], closed_block.tobytes())
        hit = self._cache.get(tf)
        if hit is not None and hit.key == key:
            return hit
        part = self._tf_part(frame, tf, cur)
        hist = {s: kernel.forming_history(part, tf, s).iloc[[-1]] for s in SIDES}
        entry = _ClosedCache(key=key, cb=closed_cb(part, self._models, tf).iloc[[-1]], hist=hist)
        self._cache[tf] = entry
        return entry

    def _check_one_min(self, row: pd.DataFrame, one: pd.DataFrame) -> None:
        for c in _ONE_MIN_INPUTS:
            if c not in one.columns:
                self._warn_once(f"missing:{c}", f"candle_bounds live: input column {c!r} "
                                "missing from the 1-min frame; dependent CB flags False")
            elif not np.isfinite(row[c].iloc[0]):
                self._warn_once(f"nan:{c}", f"candle_bounds live: input {c!r} is NaN on the "
                                "current 1-min row; dependent CB flags False")

    def update(self, ohlc: dict[int, pd.DataFrame]) -> None:
        if not self.enabled:
            return
        cur = ohlc[1].index[-1]
        if cur != self._minute:
            rolled = self._minute is not None and cur - self._minute == _ONE_MIN
            self._held = self._fresh if rolled else {}
            self._fresh = {}
            self._minute = cur
        usable = self._usable_tfs(ohlc, cur, warn=self._warn_once)
        cur_idx = pd.DatetimeIndex([cur])
        one_row = self._one_min_row(ohlc, cur)
        self._check_one_min(one_row, ohlc[1])

        # Compute every tf first, write after: an exception mid-loop leaves
        # no partially written tick.
        computed: dict[int, np.ndarray] = {}
        fresh: NcPred = {}
        for tf in TFS:
            if tf not in ohlc:
                continue
            if tf in usable:
                inputs = ohlc[tf][self._tf_cols(tf)]
                closed = self._closed(inputs, tf, cur)
                row = pd.concat([self._tf_part(inputs.iloc[[-1]], tf, cur), one_row], axis=1)
                cols, last = tf_columns(
                    row, self._models, tf, closed.cb.set_axis(cur_idx),
                    hist={s: h.set_axis(cur_idx) for s, h in closed.hist.items()},
                    nc_pred_prev=self._held,
                )
                for key, (bound, band) in last.items():
                    if np.isfinite(bound):
                        fresh[key] = (bound, band)
                computed[tf] = cols.iloc[0].to_numpy(dtype=float)
            else:
                computed[tf] = np.full(len(_tf_names(tf)), np.nan)
        self._fresh.update(fresh)

        for tf, values in computed.items():
            names = _tf_names(tf)
            frame = ohlc[tf]
            n_rows = len(frame)
            block = np.full((n_rows, len(names)), np.nan)
            if n_rows:
                block[-1] = values
            present = frame.columns.intersection(names)
            ohlc[tf] = pd.concat(
                [frame.drop(columns=present) if len(present) else frame,
                 pd.DataFrame(block, index=frame.index, columns=names)],
                axis=1,
            )
