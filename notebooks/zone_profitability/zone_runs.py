"""Run-length statistics of CB zone markers (phase 20, spec
2026-10-19-cb-zone-run-stats-design.md §2).

Pure functions over three aligned arrays: ``mask`` (bool, marker on the row),
``cid`` (int, candle id per row — ``evlib.candle_id``) and ``minute`` (int64
epoch minutes of the row). A *block* is a maximal run of True rows that share
one ``cid`` and have consecutive minutes; a gap in the frame ends a block the
way a silent minute ends the executor's streak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _block_bounds(mask: np.ndarray, cid: np.ndarray, minute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) row indices of every block."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    c = np.asarray(cid)
    t = np.asarray(minute, dtype=np.int64)
    # A row continues the previous block when both are True, same candle,
    # and the minute is the previous one + 1.
    cont = np.zeros(m.size, dtype=bool)
    cont[1:] = m[1:] & m[:-1] & (c[1:] == c[:-1]) & (t[1:] == t[:-1] + 1)
    starts = np.flatnonzero(m & ~cont)
    ends_mask = m.copy()
    ends_mask[:-1] &= ~cont[1:]
    ends = np.flatnonzero(ends_mask) + 1
    return starts.astype(np.int64), ends.astype(np.int64)


def blocks(mask: np.ndarray, cid: np.ndarray, minute: np.ndarray) -> np.ndarray:
    """Length (rows) of every block, in row order. ``int64``; empty when no marker."""
    starts, ends = _block_bounds(mask, cid, minute)
    return (ends - starts).astype(np.int64)


def per_candle(mask: np.ndarray, cid: np.ndarray, minute: np.ndarray) -> pd.DataFrame:
    """Per touched candle: ``blocks`` (count), ``minutes`` (True rows),
    ``first_block_len`` (length of the block holding the candle's first True
    row). Index = cid of touched candles, ascending; untouched candles absent."""
    starts, ends = _block_bounds(mask, cid, minute)
    cols = ["blocks", "minutes", "first_block_len"]
    if starts.size == 0:
        return pd.DataFrame(columns=cols, dtype=np.int64)
    c = np.asarray(cid)[starts]
    lengths = ends - starts
    frame = pd.DataFrame({"cid": c, "len": lengths})
    g = frame.groupby("cid", sort=True)["len"]
    out = pd.DataFrame({"blocks": g.size(), "minutes": g.sum(), "first_block_len": g.first()})
    out.index.name = None
    return out[cols].astype(np.int64)


def remaining_given_streak(lengths: np.ndarray, s_max: int = 12) -> dict[int, dict[str, float | int]]:
    """For s = 1..s_max: distribution of ``L - s`` over blocks with ``L >= s``.
    ``{s: {"p50", "p75", "n"}}``; an empty cell is ``{"n": 0}``."""
    L = np.asarray(lengths, dtype=np.int64)
    out: dict[int, dict[str, float | int]] = {}
    for s in range(1, int(s_max) + 1):
        rem = L[L >= s] - s
        if rem.size == 0:
            out[s] = {"n": 0}
            continue
        out[s] = {
            "p50": float(np.percentile(rem, 50)),
            "p75": float(np.percentile(rem, 75)),
            "n": int(rem.size),
        }
    return out


def _dist(x: np.ndarray, keys: tuple[str, ...]) -> dict[str, float]:
    if x.size == 0:
        return {}
    f = {
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "max": float(np.max(x)),
    }
    return {k: round(f[k], 3) for k in keys}


def summarize(lengths: np.ndarray, pc: pd.DataFrame) -> dict[str, object]:
    """The spec §3 json cell for one (set, tf, side)."""
    L = np.asarray(lengths, dtype=np.int64)
    return {
        "n_blocks": int(L.size),
        "n_candles": int(len(pc)),
        "minutes_total": int(L.sum()),
        "run_len": _dist(L, ("mean", "p50", "p75", "p90", "max")),
        "blocks_per_candle": _dist(pc["blocks"].to_numpy(dtype=np.int64), ("mean", "p50", "p90")),
        "minutes_per_candle": _dist(pc["minutes"].to_numpy(dtype=np.int64), ("mean", "p50")),
        "first_block_len": _dist(pc["first_block_len"].to_numpy(dtype=np.int64), ("mean", "p50", "p75")),
    }
