"""Unit tests for indicator warmup margin — constants.INDICATOR_WINDOW_ROWS and
config_loader.warmup_minutes / warmup_start_ms."""

import pandas as pd

from constants import INDICATOR_WINDOW_ROWS
from config_loader import CANDLES, warmup_minutes, warmup_start_ms


# ---------------------------------------------------------------------------
# INDICATOR_WINDOW_ROWS — single source for the build_indicator_input window
# ---------------------------------------------------------------------------

def test_indicator_window_rows_value():
    """Constant mirrors the build_indicator_input 105-row window."""
    assert INDICATOR_WINDOW_ROWS == 105


def test_build_indicator_input_window_matches_constant():
    """build_indicator_input returns at most INDICATOR_WINDOW_ROWS rows."""
    from indicators import build_indicator_input

    n = INDICATOR_WINDOW_ROWS * 3
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({"1_close": range(n), "1_is_closed": [True] * n}, index=idx)

    out = build_indicator_input(df, idx[-1], tf=1)
    assert len(out) == INDICATOR_WINDOW_ROWS


# ---------------------------------------------------------------------------
# warmup_minutes
# ---------------------------------------------------------------------------

def test_warmup_minutes_is_window_times_max_tf():
    """warmup_minutes() == INDICATOR_WINDOW_ROWS * max(CANDLES)."""
    assert warmup_minutes() == INDICATOR_WINDOW_ROWS * max(CANDLES)


# ---------------------------------------------------------------------------
# warmup_start_ms
# ---------------------------------------------------------------------------

def test_warmup_start_ms_subtracts_warmup():
    """Start is moved back by warmup_minutes() in milliseconds."""
    data_start_ms = 1_700_000_000_000
    expected = data_start_ms - warmup_minutes() * 60_000
    assert warmup_start_ms(data_start_ms) == expected


def test_warmup_start_ms_clamps_to_zero():
    """Result never goes below 0 (epoch start)."""
    assert warmup_start_ms(1_000) == 0


def test_warmup_start_ms_zero_input():
    """DATA_START=0 (live.env placeholder) stays 0."""
    assert warmup_start_ms(0) == 0
