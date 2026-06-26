# tests/unit/data_layer/test_chunk_pct_progress.py
# Tests for _PctProgress — intra-chunk 1% progress logging (phase 16).

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

import training.data_preparer as dp_mod  # noqa: E402
from training.data_preparer import _PctProgress  # noqa: E402


def test_logs_once_per_percent_crossing():
    """tick() emits exactly one line for each integer percent reached (1..100)."""
    lines: list[str] = []
    p = _PctProgress(total=200, label="x", log=lines.append)
    for done in range(1, 201):
        p.tick(done)
    assert len(lines) == 100


def test_log_line_format():
    """Line is '<label> <pct>% (<done>/<total>)'."""
    lines: list[str] = []
    p = _PctProgress(total=100, label="pass1 tf=15", log=lines.append)
    p.tick(1)
    assert lines == ["pass1 tf=15 1% (1/100)"]


def test_total_zero_is_noop():
    lines: list[str] = []
    _PctProgress(total=0, label="x", log=lines.append).tick(5)
    assert lines == []


def test_done_zero_no_log():
    lines: list[str] = []
    _PctProgress(total=100, label="x", log=lines.append).tick(0)
    assert lines == []


def test_small_total_logs_each_new_percent():
    """total<100: each row is >1%, log once per percent reached, in order."""
    lines: list[str] = []
    p = _PctProgress(total=4, label="x", log=lines.append)
    for done in range(1, 5):
        p.tick(done)
    assert lines == [
        "x 25% (1/4)", "x 50% (2/4)", "x 75% (3/4)", "x 100% (4/4)",
    ]


def test_no_duplicate_percent_logs():
    lines: list[str] = []
    p = _PctProgress(total=1000, label="x", log=lines.append)
    p.tick(10); p.tick(11); p.tick(15); p.tick(20)  # 1%,1%,1%,2%
    assert lines == ["x 1% (10/1000)", "x 2% (20/1000)"]


def test_compute_tf_rows_emits_pct_progress(monkeypatch):
    """The per-row indicator loop logs ~one line per 1% of its rows, tagged tf."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({"1_close": range(n)}, index=idx)

    # Mock the indicator framework (mirrors existing _run_indicator_pass tests).
    field = MagicMock(); field.name = "f1"
    indicators = MagicMock()
    indicators.Indicators._sorted_fields.return_value = [field]
    indicators.build_indicator_input.return_value = pd.DataFrame({"1_f1": [1.0]})
    monkeypatch.setitem(sys.modules, "indicators", indicators)

    captured: list[str] = []
    monkeypatch.setattr("logs.log", captured.append)

    dp_mod._compute_tf_rows(df, tf=1, rows=df.index, groups=["volume"])

    pct_lines = [l for l in captured if "tf=1" in l and "%" in l]
    assert len(pct_lines) == 100  # 200 rows → one line per 1%
