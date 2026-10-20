"""Phase 20 task 01: pure run-length statistics over (mask, cid, minute)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "notebooks", "zone_profitability")
_p = os.path.abspath(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)

import zone_runs  # noqa: E402


def _arr(mask, cid=None, minute=None):
    m = np.asarray(mask, dtype=bool)
    c = np.zeros(len(m), dtype=np.int64) if cid is None else np.asarray(cid, dtype=np.int64)
    t = np.arange(len(m), dtype=np.int64) if minute is None else np.asarray(minute, dtype=np.int64)
    return m, c, t


def test_blocks_counts_runs_within_one_candle():
    m, c, t = _arr([1, 1, 0, 1])
    assert zone_runs.blocks(m, c, t).tolist() == [2, 1]


def test_blocks_split_by_candle_boundary():
    m, c, t = _arr([1, 1, 1, 1], cid=[0, 0, 1, 1])
    assert zone_runs.blocks(m, c, t).tolist() == [2, 2]


def test_blocks_split_by_minute_gap():
    m, c, t = _arr([1, 1, 1], minute=[0, 1, 5])
    assert zone_runs.blocks(m, c, t).tolist() == [2, 1]


def test_blocks_empty_mask_gives_no_blocks():
    m, c, t = _arr([0, 0, 0])
    out = zone_runs.blocks(m, c, t)
    assert len(out) == 0
    assert out.dtype == np.int64


def _three_candles():
    # candle 0: 1,0,1,1 ; candle 1: 0,0,0 ; candle 2: 1
    return _arr([1, 0, 1, 1, 0, 0, 0, 1], cid=[0, 0, 0, 0, 1, 1, 1, 2])


def test_per_candle_three_candles():
    pc = zone_runs.per_candle(*_three_candles())
    assert pc.index.tolist() == [0, 2]
    assert pc["blocks"].tolist() == [2, 1]
    assert pc["minutes"].tolist() == [3, 1]
    assert pc["first_block_len"].tolist() == [1, 1]


def test_per_candle_untouched_candle_absent():
    pc = zone_runs.per_candle(*_three_candles())
    assert 1 not in pc.index


def test_remaining_given_streak_small():
    r = zone_runs.remaining_given_streak(np.array([3, 5]), s_max=4)
    assert r[1] == {"p50": 3.0, "p75": 3.5, "n": 2}
    assert r[3]["n"] == 2 and r[3]["p50"] == 1.0
    assert r[4] == {"p50": 1.0, "p75": 1.0, "n": 1}


def test_remaining_given_streak_beyond_every_block():
    r = zone_runs.remaining_given_streak(np.array([3]), s_max=5)
    assert r[4] == {"n": 0}
    assert r[5] == {"n": 0}
    assert "p50" not in r[4]


def test_summarize_matches_hand_numbers():
    m, c, t = _three_candles()
    lengths = zone_runs.blocks(m, c, t)
    pc = zone_runs.per_candle(m, c, t)
    s = zone_runs.summarize(lengths, pc)
    assert lengths.tolist() == [1, 2, 1]
    assert s["n_blocks"] == 3 and s["n_candles"] == 2
    assert s["run_len"]["p50"] == 1.0
    assert s["run_len"]["mean"] == pytest.approx(1.333, abs=1e-3)
    assert s["run_len"]["max"] == 2
    assert s["blocks_per_candle"]["mean"] == 1.5
    assert s["minutes_per_candle"]["mean"] == 2.0
    assert s["first_block_len"]["p50"] == 1.0
    assert s["minutes_total"] == 4


def test_summarize_empty():
    s = zone_runs.summarize(np.array([], dtype=np.int64), pd.DataFrame(columns=["blocks", "minutes", "first_block_len"]))
    assert s["n_blocks"] == 0 and s["n_candles"] == 0
    assert s["run_len"] == {}
    assert s["minutes_total"] == 0
