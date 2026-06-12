"""Integration test: warmup margin → indicator pass → no NaN at start date.

Proves INDICATOR_WINDOW_ROWS closed candles of leading history are enough for
the slowest base indicator (ema_100) to be non-NaN at the simulation start.
Uses tf=5 so the fixture stays small (105 × 5 minutes of warmup).
"""

import numpy as np
import pandas as pd

from constants import INDICATOR_WINDOW_ROWS

TF = 5


def test_base_indicators_non_nan_at_start_after_warmup():
    data_start = pd.Timestamp("2024-01-08 00:00", tz="UTC")
    warmup_min = INDICATOR_WINDOW_ROWS * TF

    idx = pd.date_range(
        data_start - pd.Timedelta(minutes=warmup_min),
        periods=warmup_min + TF,
        freq="1min",
        tz="UTC",
    )
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    raw = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(idx), 100.0),
            "taker_base_vol": np.full(len(idx), 50.0),
        },
        index=idx,
    )

    from data import _build_wide_df
    df = _build_wide_df(raw)

    from training.data_preparer import DataPreparer
    preparer = DataPreparer("configs/indicators_config.yaml", "unused.pkl", "unused.pkl")
    preparer._run_indicator_pass(df, ["momentum", "trend"], [TF], start_ts=data_start)

    from indicators import Indicators
    fields = Indicators._sorted_fields(TF, groups=["momentum", "trend"])
    cols = [f"{TF}_{f.name}" for f in fields]
    assert cols, "Expected momentum/trend fields for tf=5"
    assert any(c.endswith("ema_100") for c in cols), "ema_100 must be in the checked set"

    row = df.loc[data_start]
    nan_cols = [c for c in cols if pd.isna(row[c])]
    assert not nan_cols, f"NaN at simulation start for: {nan_cols}"

    # Warmup rows are input only — no indicators computed before data_start
    before = df.loc[df.index < data_start, cols]
    assert before.isna().all().all(), "Indicators must not be computed before DATA_START"
