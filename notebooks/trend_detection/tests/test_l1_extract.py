"""Layer 1 tests -- tdlib.config (SLIM whitelist + constants) and
tdlib.extract (extract_slim: wide df -> SLIM pickle). See task-1-brief.md.

Filename carries "l1" so `pytest -k l1` selects every test in this module
(mirrors test_l0_conftest.py's own "l0" naming and action_zones's
test_layer1_loader.py "layer1" naming -- same convention, this experiment's
own module-numbering prefix).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tdlib.config import (
    ANALYSIS_TFS,
    CONTEXT_TFS,
    LABEL_COLS,
    MOVE_CUTS,
    artifacts_dir,
    slim_columns,
    slim_out_path,
)
from tdlib.extract import extract_slim

from .conftest import make_slim

# --- independently-typed expectations (NOT imported from tdlib.config) -----
#
# Copied verbatim from the task-1 brief, exactly like test_l0_conftest.py's
# own _EXPECTED_LABEL_COLUMNS -- so this file independently verifies
# tdlib.config's transcription of the brief's exact values rather than only
# checking internal self-consistency against tdlib.config's own constants.

_EXPECTED_ANALYSIS_TFS = [15, 60, 240]
_EXPECTED_CONTEXT_TFS = [5, 15, 60, 240, 1440]

# Verified against the actual source
# (worktrees/rsi-quantile-sym0/stats/train/link_usdt/rsi_classification.json,
# keys "15"/"60"/"240" -> move_cuts) rather than the brief's own inline
# literals -- the two did NOT match byte-for-byte (see task-1-report.md).
# These are the FILE's values.
_EXPECTED_MOVE_CUTS = {
    15: [-1.3642919195372998, -0.4092875758611899, 0.4092875758611899, 1.3642919195372998],
    60: [-1.4096495558610285, -0.4228948667583085, 0.4228948667583085, 1.4096495558610285],
    240: [-1.4597374604647457, -0.4379212381394237, 0.4379212381394237, 1.4597374604647457],
}

_EXPECTED_LABEL_COLUMNS = [
    # tf=15
    "15_pslong_n1_m1_x0p3_l15_y0p2",
    "15_psshort_n1_m1_x0p3_l15_y0p2",
    "15_pslong_n2_m1_x0p3_l15_y0p2",
    "15_psshort_n2_m1_x0p3_l15_y0p2",
    "15_plong_n1_m1_x0p3",
    "15_pshort_n1_m1_x0p3",
    # tf=60
    "60_pslong_n1_m1_x0p2_l15_y0p1",
    "60_psshort_n1_m1_x0p2_l15_y0p1",
    "60_pslong_n2_m1_x0p2_l15_y0p1",
    "60_psshort_n2_m1_x0p2_l15_y0p1",
    "60_plong_n1_m1_x0p2",
    "60_pshort_n1_m1_x0p2",
    # tf=240
    "240_pslong_n1_m1_x0p1_l15_y0p1",
    "240_psshort_n1_m1_x0p1_l15_y0p1",
    "240_pslong_n2_m1_x0p1_l15_y0p1",
    "240_psshort_n2_m1_x0p1_l15_y0p1",
    "240_plong_n1_m1_x0p1",
    "240_pshort_n1_m1_x0p1",
]


def _minimal_required_columns() -> list[str]:
    """Independently re-derive the REQUIRED column set from the task-1
    brief's own spec (NOT by calling any tdlib.config private helper) --
    used to build minimal `available` lists for slim_columns tests without
    coupling the test to config.py's internal structure.

    Brief: "{tf}_{open_index,open,high,low,close,is_closed}` for tf in
    CONTEXT_TFS, `{tf}_rsi_ma8_diff` + `{tf}_atr_14` + bb_20_2 triple for tf
    in [15,60,240,1440], all LABEL_COLS values."
    """
    cols: list[str] = []
    for tf in _EXPECTED_CONTEXT_TFS:
        for suffix in ("open_index", "open", "high", "low", "close", "is_closed"):
            cols.append(f"{tf}_{suffix}")
    for tf in (15, 60, 240, 1440):
        for suffix in ("rsi_ma8_diff", "atr_14", "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"):
            cols.append(f"{tf}_{suffix}")
    cols.extend(_EXPECTED_LABEL_COLUMNS)
    return cols


# --- tdlib.config constants -------------------------------------------------


def test_analysis_tfs_matches_brief():
    assert ANALYSIS_TFS == _EXPECTED_ANALYSIS_TFS


def test_context_tfs_matches_brief():
    assert CONTEXT_TFS == _EXPECTED_CONTEXT_TFS


def test_move_cuts_matches_verified_source_values():
    assert MOVE_CUTS == _EXPECTED_MOVE_CUTS


def test_label_cols_flattened_matches_brief_verbatim_list():
    flattened = [col for tf in LABEL_COLS for col in LABEL_COLS[tf].values()]
    assert flattened == _EXPECTED_LABEL_COLUMNS


def test_slim_out_path_and_artifacts_dir_built_on_dataset_folder(monkeypatch):
    monkeypatch.setenv("ROOT_FOLDER", "long")
    monkeypatch.setenv("DATA_ROOT", "train")
    monkeypatch.setenv("DATA_SET_NAME", "link")
    monkeypatch.setenv("PAIR", "usdt")

    assert artifacts_dir() == "/trader_data_long/train/link_usdt/trend_detection/"
    assert slim_out_path() == "/trader_data_long/train/link_usdt/trend_detection/slim.pkl"


# --- tdlib.config.slim_columns ----------------------------------------------


def test_slim_columns_keeps_required_and_drops_columns_not_in_whitelist(synthetic_slim_df):
    available = list(synthetic_slim_df.columns) + ["foo_bar", "15_totally_made_up_col"]
    result = slim_columns(available)

    assert "foo_bar" not in result
    assert "15_totally_made_up_col" not in result
    # every required column that IS available must survive into the result
    for col in _minimal_required_columns():
        if col in synthetic_slim_df.columns:
            assert col in result


def test_slim_columns_raises_keyerror_naming_all_missing_required():
    full_required = _minimal_required_columns()
    to_remove = {"1440_is_closed", "60_atr_14", "240_pshort_n1_m1_x0p1"}
    available = [c for c in full_required if c not in to_remove]

    with pytest.raises(KeyError) as exc_info:
        slim_columns(available)

    # Read the raw message tdlib.config built (not str(exc), which KeyError
    # wraps in its own extra repr layer) so quoted-substring checks below
    # are exact, not dependent on Python's outer-repr quoting choice.
    message = exc_info.value.args[0]
    for col in to_remove:
        assert f"'{col}'" in message, f"{col!r} not named in KeyError message: {message}"
    # columns that WERE available must not be reported as missing.
    for col in available:
        assert f"'{col}'" not in message


def test_slim_columns_order_is_stable_regardless_of_available_order():
    full_required = _minimal_required_columns()
    extra_optional = ["15_close_diff_prc", "60_natr_14", "240_adx_14"]

    result_forward = slim_columns(full_required + extra_optional)
    result_reversed = slim_columns(list(reversed(full_required)) + list(reversed(extra_optional)))

    assert result_forward == result_reversed
    assert len(result_forward) == len(set(result_forward))


def test_slim_columns_dedupes_when_available_has_duplicate_entries():
    full_required = _minimal_required_columns()
    dupe_col = full_required[0]
    result = slim_columns(full_required + [dupe_col, dupe_col])

    assert len(result) == len(set(result))
    assert result.count(dupe_col) == 1


# --- tdlib.extract.extract_slim ---------------------------------------------


@pytest.fixture
def slim_paths(tmp_path, monkeypatch):
    """Patch tdlib.extract's three path functions to tmp_path locations;
    return the resolved paths. Callers write whatever source df they want to
    `.wide_path` themselves before calling extract_slim() -- shared here
    since every extract_slim test in this module needs the same wiring
    (mirrors test_layer1_loader.py's per-test `monkeypatch.setattr` on the
    *importing* module's own bound name, zone-selection worktree, applied
    to all three of tdlib.extract's path functions instead of just one).
    """
    wide_path = tmp_path / "df_with_indicators.pkl"
    out_dir = tmp_path / "dataset_folder" / "trend_detection"
    out_path = out_dir / "slim.pkl"

    monkeypatch.setattr("tdlib.extract.wide_df_path", lambda: str(wide_path))
    monkeypatch.setattr("tdlib.extract.slim_out_path", lambda: str(out_path))
    monkeypatch.setattr("tdlib.extract.artifacts_dir", lambda: str(out_dir))

    return SimpleNamespace(wide_path=wide_path, out_path=out_path, out_dir=out_dir)


def test_extract_slim_returns_rows_cols_matching_saved_pkl(slim_paths, synthetic_slim_df):
    # synthetic_slim_df is already fully 15_is_closed==True (Task 0 fixture)
    # -- every row survives the filter, so this test isolates the
    # return-value / saved-shape correspondence from filtering behavior
    # (covered separately by test_slim_feeds_point_selection below).
    synthetic_slim_df.to_pickle(slim_paths.wide_path)

    rows, cols = extract_slim()

    saved = pd.read_pickle(slim_paths.out_path)
    assert (rows, cols) == saved.shape


def test_extract_slim_does_not_modify_source_file(slim_paths, synthetic_slim_df):
    synthetic_slim_df.to_pickle(slim_paths.wide_path)
    before_bytes = slim_paths.wide_path.read_bytes()
    before_mtime_ns = slim_paths.wide_path.stat().st_mtime_ns

    extract_slim()

    assert slim_paths.wide_path.read_bytes() == before_bytes
    assert slim_paths.wide_path.stat().st_mtime_ns == before_mtime_ns


def test_extract_slim_overwrite_is_idempotent(slim_paths, synthetic_slim_df):
    synthetic_slim_df.to_pickle(slim_paths.wide_path)

    result_1 = extract_slim()
    saved_1 = pd.read_pickle(slim_paths.out_path)
    result_2 = extract_slim()
    saved_2 = pd.read_pickle(slim_paths.out_path)

    assert result_1 == result_2
    pd.testing.assert_frame_equal(saved_1, saved_2)


def test_extract_slim_missing_source_raises_filenotfounderror_with_path(slim_paths):
    with pytest.raises(FileNotFoundError, match=re.escape(str(slim_paths.wide_path))):
        extract_slim()


# --- integration -> Layer 2 (point selection, next task) -------------------


def test_slim_feeds_point_selection(slim_paths):
    """L1 -> L2 integration boundary (L2 lands in a later task): build a
    WIDE synthetic frame -- make_slim()'s rows (real completed 15m closes)
    interleaved with extra 15_is_closed=False 1-minute rows, plus a junk
    column outside the whitelist entirely -- run extract_slim() against it,
    and verify the saved SLIM output is exactly what the next layer will
    need: only the real completed-15m rows, junk column gone, label columns
    (and their NaN pattern) intact. Covers the L1 side of the boundary only
    -- no L2 module exists yet to import.
    """
    slim = make_slim(seed=1, days=2)  # small: only filtering/columns matter here, not volume
    wide = slim.copy()
    wide["foo_bar"] = np.arange(len(wide))  # junk column: outside the whitelist entirely

    # Extra non-closing 1-minute rows, one minute before each real slim row.
    # Every slim row's minute is in {14, 29, 44, 59} (mod 15), so shifting
    # back 1 minute (-> {13, 28, 43, 58}) never crosses a 15/60/240/1440-min
    # floor boundary -- the copied {tf}_open_index columns stay numerically
    # correct for the shifted timestamp. 15_is_closed is force-set False
    # (the only column this test's filter actually depends on); other tfs'
    # *_is_closed columns are stale copies, which is harmless here since
    # extract_slim() only ever filters on 15_is_closed.
    extra = wide.copy()
    extra.index = wide.index - pd.Timedelta(minutes=1)
    extra["15_is_closed"] = False

    wide_full = pd.concat([wide, extra]).sort_index()
    wide_full.index.name = "timestamp"
    assert len(wide_full) == 2 * len(slim)  # sanity: interleave doubled the row count
    assert (~wide_full["15_is_closed"]).sum() == len(slim)  # sanity: exactly `extra`'s rows are non-closed

    wide_full.to_pickle(slim_paths.wide_path)

    rows, cols = extract_slim()
    output = pd.read_pickle(slim_paths.out_path)

    assert (rows, cols) == output.shape
    assert rows == len(slim)  # every non-closed row dropped, none of the real ones lost
    assert output["15_is_closed"].all()
    assert "foo_bar" not in output.columns

    expected_cols = slim_columns(wide_full.columns.tolist())
    assert list(output.columns) == expected_cols

    label_col = "15_pslong_n1_m1_x0p3_l15_y0p2"
    assert label_col in output.columns
    pd.testing.assert_series_equal(output[label_col].sort_index(), slim[label_col].sort_index())
