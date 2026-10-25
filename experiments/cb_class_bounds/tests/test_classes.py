import numpy as np
import pandas as pd
import pytest

from experiments.cb_class_bounds import classes as C
from experiments.cb_class_bounds.swing_causal_zones import CONST, STD, causal_apexes, cci_zone


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(0, 30, n))
    y[:7] = np.nan
    return y


def test_move_class_matches_digitize_convention():
    rsi = np.array([50.0, 50.2, 53.0, 49.0, 49.0])
    mc = C.move_class(rsi, 15)
    assert mc[0] == 0                 # NaN diff → 0
    assert mc[1] == 0                 # |0.2| < 0.409
    assert mc[2] == 3                 # 2.8 > 2.728
    assert mc[3] == -3                # -4 < -2.728
    assert mc[4] == 0


def test_zone_class_tiers():
    z = C.zone_class(np.array([30.0, 46.0, 50.0, 54.0, 70.0, np.nan]), 50.0, 5.0)
    assert list(z) == [0, 1, 2, 3, 4, 2]


def test_ema_slope_class_closed_sign():
    e = np.array([100.0, 100.0, 101.0, 100.0])
    cl = C.ema_slope_class_closed(e, 15)
    assert cl[0] == 0 and cl[2] == 1 and cl[3] == -1


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_zigzag_incremental_equals_batch(seed):
    y = _series(seed=seed)
    prom = 0.5 * STD[15]
    zz = C.ZigZag(prom)
    got = [a for a in (zz.feed(k, y[k]) for k in range(len(y))) if a is not None]
    assert got == causal_apexes(y, prom)


def test_zone_from_apexes_equals_cci_zone():
    y = _series(seed=5)
    prom, const = 0.5 * STD[60], CONST[60]
    z_ref, apexes = cci_zone(y, prom, const, 3, 3)
    assert np.array_equal(C.zone_from_apexes(apexes, len(y), const), z_ref)


@pytest.mark.parametrize("seed", [0, 7])
def test_cci_zone_forming_matches_closed_when_forming_is_the_next_close(seed):
    """Forming value == the next closed value → forming zone == closed zone at k+1."""
    y = _series(seed=seed)
    tf = 15
    z_closed = C.cci_zone_closed(y, tf)
    last_closed = np.arange(len(y) - 1)           # rows observing candle k+1 forming, k = 0..n-2
    y_forming = y[1:]
    zf = C.cci_zone_forming(y, last_closed, y_forming, tf)
    assert np.array_equal(zf, z_closed[1:])


def test_cci_zone_forming_peek_does_not_mutate():
    y = _series(seed=9)
    tf = 240
    last_closed = np.array([50, 50, 120, -1])
    yf = np.array([y[50] + 1.0, y[50] - 300.0, np.nan, 10.0])
    out = C.cci_zone_forming(y, last_closed, yf, tf)
    assert out[2] == 0 and out[3] == 0
    assert set(out.tolist()) <= {-1, 0, 1}
