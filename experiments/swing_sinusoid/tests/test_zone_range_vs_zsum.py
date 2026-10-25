import numpy as np
import pandas as pd
import pytest

from experiments.swing_sinusoid.zone_range_vs_zsum import episodes, partial_spearman, spearman_perm


def test_episodes_two_blocks_one_candle_and_one_block_next():
    # 10 rows, tf candle of 5 rows: cid 0 = rows 0-4, cid 1 = rows 5-9
    cid = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    inz = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0], bool)
    hi = np.array([10, 12, 13, 20, 11, 9, 9, 15, 14, 30], float)
    lo = np.array([9, 11, 12, 19, 8, 8, 8, 14, 13, 29], float)
    ep = episodes(inz, cid, hi, lo).set_index("cid")
    assert list(ep.index) == [0, 1]
    e0 = ep.loc[0]
    assert e0["first"] == 1 and e0["last"] == 4 and e0["n_min"] == 3
    assert e0["hi"] == 13 and e0["lo"] == 8            # in-zone rows 1, 2, 4 only (row 3 excluded)
    assert e0["fb_hi"] == 13 and e0["fb_lo"] == 11 and e0["fb_n"] == 2
    assert e0["cdl_hi"] == 20 and e0["cdl_lo"] == 8 and e0["cdl_start"] == 0
    assert e0["post_hi"] == 20 and e0["post_lo"] == 8          # rows 1..4 of candle 0
    e1 = ep.loc[1]
    assert e1["first"] == 7 and e1["n_min"] == 2 and e1["hi"] == 15 and e1["lo"] == 13
    assert e1["fb_n"] == 2 and e1["cdl_hi"] == 30 and e1["cdl_start"] == 5
    assert e1["post_hi"] == 30 and e1["post_lo"] == 13         # rows 7..9 (row 5-6 lo 8 excluded)


def test_episodes_block_split_by_candle_boundary():
    cid = np.array([0, 0, 1, 1])
    inz = np.ones(4, bool)
    hi = np.array([1, 2, 3, 4], float); lo = hi - 0.5
    ep = episodes(inz, cid, hi, lo).set_index("cid")
    assert ep.loc[0, "fb_n"] == 2 and ep.loc[1, "fb_n"] == 2       # contiguous rows, but two candles → two blocks


def test_episodes_empty():
    assert len(episodes(np.zeros(5, bool), np.zeros(5, int), np.ones(5), np.ones(5))) == 0


def test_spearman_perm_detects_monotone_and_null_is_centred():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500); y = x + rng.normal(size=500)
    rho, null = spearman_perm(x, y, 200, rng)
    assert rho > 0.6 and abs(null.mean()) < 0.05 and np.percentile(null, 95) < 0.15


def test_partial_spearman_removes_shared_driver():
    rng = np.random.default_rng(1)
    z = rng.normal(size=2000)
    x = z + 0.3 * rng.normal(size=2000); y = z + 0.3 * rng.normal(size=2000)
    assert abs(partial_spearman(x, y, z)) < 0.1
