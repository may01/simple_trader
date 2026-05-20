import pytest
import pandas as pd
import numpy as np
from simple_trader.data.data_point import LiveDataPoint


def _make_live_point(values: dict) -> LiveDataPoint:
    ohlc = {}
    for tf, cols in values.items():
        data = {f"{tf}_{col}": vals for col, vals in cols.items()}
        ohlc[tf] = pd.DataFrame(data)
    return LiveDataPoint(ohlc)


@pytest.fixture
def point_rsi_50():
    return _make_live_point({1: {"rsi_14": [30.0, 50.0]}, 15: {"rsi_14": [45.0, 55.0]}})


@pytest.fixture
def point_rsi_series():
    """Point with rsi rising: [20, 40, 60, 80]."""
    return _make_live_point({1: {"rsi_14": [20.0, 40.0, 60.0, 80.0]}})
