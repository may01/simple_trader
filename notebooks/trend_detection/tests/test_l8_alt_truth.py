"""Layer 8 tests -- tdlib.alt_truth (truth decomposition: strict vs plain vs
fwd, on the same points/combos/models). See task-8-brief.md.

Filename carries "l8" so `pytest -k l8` selects every test in this module
(mirrors test_l0_conftest.py .. test_l7_diag.py's own "l0".."l7" naming).

Reuses test_l5_loop.py's own ``_engineered_slim`` fixture-construction
helper (same "reuse L5 scaffolding" instruction task-7-brief.md already
established, applied again here) -- it forces tf=15 strong points on both
sides while leaving tf=60 naturally sparse. Verified against a throwaway
probe against the real fixture (not committed) that this SAME fixture
clears MIN_COMBO_POINTS (60) for tf=15 under BOTH "plain" and "fwd" truth
(not just the "strict" truth it was originally engineered around) while
tf=60 stays under the floor for every truth_kind -- its raw strong-point
count alone (the upper bound on any truth_kind's usable-point count, since
truth-marking can only ever DROP rows, never add them) is already below 60,
so no truth-kind-specific re-engineering was needed for the skip path.

Most unit tests below build small hand-controlled DataFrames directly --
faster and easier to hand-verify than routing everything through
make_slim + real column data, matching test_l4_screen_models.py's/
test_l7_diag.py's own stated convention. Only the smoke test needs the real
fixture and a real (non-monkeypatched) run_alt call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdlib.alt_truth import (
    feature_matrix_alt,
    mark_truth_fwd,
    mark_truth_plain,
    run_alt,
    write_alt_report,
)
from tdlib.config import LABEL_COLS, MOVE_CUTS
from tdlib.truth import mark_truth

from .conftest import make_slim
from .test_l5_loop import _engineered_slim

# --- shared helpers ------------------------------------------------------------


def _force_strong_points(df: pd.DataFrame, tf: int, positions: list) -> None:
    """Single-sided version of test_l5_loop.py's own
    ``_force_strong_points_both_sides`` -- verbatim copy of test_l3/l4/l5's
    shared helper of this name (neutralize the whole column to class 0,
    force class +2 at exactly ``positions``)."""
    diff_col = df.columns.get_loc(f"{tf}_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0
    df.iloc[positions, diff_col] = MOVE_CUTS[tf][3] + 1.0


def _set_plain_labels(df: pd.DataFrame, tf: int, positions: list, long_val: float, short_val: float) -> None:
    """Writes the PLAIN long/short pair directly by column name -- unlike
    the profit_strict pair, ``conftest.set_labels`` never touches this pair
    (see its own docstring: "Only the n1 profit_strict pair is touched"),
    so a test that wants CONTROLLED plain labels must write those two
    columns itself."""
    long_col = LABEL_COLS[tf]["plong_n1"]
    short_col = LABEL_COLS[tf]["pshort_n1"]
    df.iloc[positions, df.columns.get_loc(long_col)] = long_val
    df.iloc[positions, df.columns.get_loc(short_col)] = short_val


def _plant_fake_metrics(path: Path, gbc_test_auc: float, n: int = 100) -> None:
    """Minimal but real-shaped metrics.json, matching what a real
    tdlib.loop.run_iteration / tdlib.diag.run_diag would have persisted,
    trimmed to just the values write_alt_report's baseline read actually
    uses (metrics["gbc"]["test"]["roc_auc"]/["n"])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "logistic": {"train": {"roc_auc": 0.5}, "test": {"roc_auc": 0.5, "n": n}},
        "gbc": {"train": {"roc_auc": 0.99}, "test": {"roc_auc": gbc_test_auc, "n": n}},
    }
    with open(path, "w") as f:
        json.dump(metrics, f)


# --- (a) mark_truth_plain -------------------------------------------------------


def test_mark_truth_plain_reads_plain_cols_and_differs_from_strict():
    """Plants DIFFERENT strict vs plain label patterns on the SAME 4 rows --
    if mark_truth_plain accidentally read the strict (profit_strict)
    columns instead, every row would come back "neither" (the strict
    columns are all 0.0 here), not the real plain-derived pattern.
    """
    idx = pd.date_range("2023-01-01", periods=4, freq="15min", tz="UTC")
    strict_long_col = LABEL_COLS[15]["pslong_n1"]
    strict_short_col = LABEL_COLS[15]["psshort_n1"]
    plain_long_col = LABEL_COLS[15]["plong_n1"]
    plain_short_col = LABEL_COLS[15]["pshort_n1"]

    pts = pd.DataFrame(
        {
            strict_long_col: [0.0, 0.0, 0.0, 0.0],   # strict: always "neither" if this were read
            strict_short_col: [0.0, 0.0, 0.0, 0.0],
            plain_long_col: [1.0, 0.0, 1.0, 0.0],
            plain_short_col: [0.0, 1.0, 1.0, 0.0],
        },
        index=idx,
    )

    plain_marked = mark_truth_plain(pts, 15)
    strict_marked = mark_truth(pts, 15, "n1")

    assert plain_marked.tolist() == ["long", "short", "both", "neither"]
    assert strict_marked.tolist() == ["neither", "neither", "neither", "neither"]
    assert plain_marked.tolist() != strict_marked.tolist()
    assert list(plain_marked.index) == list(pts.index)


def test_mark_truth_plain_nan_overrides_like_strict():
    idx = pd.date_range("2023-01-01", periods=1, freq="15min", tz="UTC")
    pts = pd.DataFrame(
        {LABEL_COLS[15]["plong_n1"]: [1.0], LABEL_COLS[15]["pshort_n1"]: [float("nan")]},
        index=idx,
    )
    assert mark_truth_plain(pts, 15).tolist() == ["nan"]


# --- (b) mark_truth_fwd ----------------------------------------------------------


def test_mark_truth_fwd_sign_mapping_nan_drop_no_both_neither():
    idx = pd.date_range("2023-01-01", periods=6, freq="15min", tz="UTC")
    close = pd.Series([100.0, 110.0, 100.0, 100.0, 100.0, np.nan], index=idx)
    slim = pd.DataFrame({"15_is_closed": pd.Series(True, index=idx), "15_close": close}, index=idx)

    marked = mark_truth_fwd(slim, 15)

    # bar0->1: up (+) -> long; bar1->2: down (-) -> short; bar2->3 and
    # bar3->4: flat (== 0 exactly) -> nan; bar4->5: close[5] NaN -> nan;
    # bar5: no bar ahead -> nan.
    assert marked.tolist() == ["long", "short", "nan", "nan", "nan", "nan"]
    assert set(marked.unique()) <= {"long", "short", "nan"}  # never "both"/"neither"
    assert list(marked.index) == list(slim.index)  # full slim index, not a subset


def test_mark_truth_fwd_restricts_cleanly_to_a_points_subset():
    idx = pd.date_range("2023-01-01", periods=4, freq="15min", tz="UTC")
    close = pd.Series([100.0, 90.0, 90.0, 120.0], index=idx)
    slim = pd.DataFrame({"15_is_closed": pd.Series(True, index=idx), "15_close": close}, index=idx)

    full = mark_truth_fwd(slim, 15)
    subset_index = idx[[0, 2]]
    restricted = full.loc[subset_index]

    assert restricted.tolist() == ["short", "long"]  # bar0->1 down, bar2->3 up


# --- (c) feature_matrix_alt: y-encoding, leak blocklist, no fwd col --------------


def test_feature_matrix_alt_plain_y_encoding_and_both_neither_dropped():
    df = make_slim(seed=210, days=2)
    closed = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed[:4].tolist()
    _force_strong_points(df, 60, positions)

    _set_plain_labels(df, 60, [positions[0]], 1.0, 0.0)  # long
    _set_plain_labels(df, 60, [positions[1]], 0.0, 1.0)  # short
    _set_plain_labels(df, 60, [positions[2]], 1.0, 1.0)  # both -> dropped
    _set_plain_labels(df, 60, [positions[3]], 0.0, 0.0)  # neither -> dropped

    X, y = feature_matrix_alt(df, 60, 2, "plain")

    expected_index = [df.index[positions[0]], df.index[positions[1]]]
    assert list(y.index) == expected_index
    assert y.tolist() == [1, 0]
    assert y.dtype == np.int8
    assert list(X.index) == list(y.index)


def test_feature_matrix_alt_fwd_y_encoding_and_no_fwd_col_in_x():
    df = make_slim(seed=211, days=2)
    closed = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed[:20].tolist()
    _force_strong_points(df, 60, positions)

    # Real (unforced) 60_close random walk already gives a mix of up/down
    # forward moves across 20 points -- no extra engineering needed here,
    # this test only needs SOME of each class to exist.
    X, y = feature_matrix_alt(df, 60, 2, "fwd")

    assert set(y.unique().tolist()) <= {0, 1}
    assert y.dtype == np.int8
    assert list(X.index) == list(y.index)
    assert len(X) > 0

    # fwd truth never enters X, for either truth_kind -- no column in a
    # real feature_matrix_alt call ever matches the leak blocklist's own
    # ".*fwd_.*" pattern (features.LEAK_BLOCKLIST_RE).
    for c in X.columns:
        assert not re.search(r"fwd_", c), f"fwd-derived column {c!r} leaked into X"


def test_feature_matrix_alt_leak_blocklist_planted_col_raises_value_error():
    df = make_slim(seed=212, days=2)
    closed = np.flatnonzero(df["60_is_closed"].to_numpy())
    positions = closed[:2].tolist()
    _force_strong_points(df, 60, positions)
    _set_plain_labels(df, 60, positions, long_val=1.0, short_val=0.0)

    leak_col = LABEL_COLS[60]["pslong_n1"]  # a real, blocklisted column -- not exempted just because it's "strict"

    with pytest.raises(ValueError):
        feature_matrix_alt(df, 60, 2, "plain", feature_cols=[leak_col, "60_rsi_ma8_diff"])


def test_feature_matrix_alt_unknown_truth_kind_raises_value_error():
    df = make_slim(seed=213, days=2)
    with pytest.raises(ValueError):
        feature_matrix_alt(df, 60, 2, "not_a_real_truth_kind")


# --- (d) run_alt / write_alt_report: smoke + skip floor -------------------------


def test_run_alt_smoke_report_and_skip_floor_honored(tmp_path):
    df = _engineered_slim()  # tf=15 both sides clear 60 for BOTH plain and fwd; tf=60 clears for neither
    base_dir = tmp_path / "trend_detection"
    base_dir.mkdir(parents=True)

    table = run_alt(df, str(base_dir))

    assert list(table.columns) == [
        "combo", "truth_kind", "feature_set", "model", "split",
        "roc_auc", "acc", "base_rate", "lift_long", "lift_short", "n",
    ]
    assert set(table["combo"].unique()) == {"15_up", "15_dn"}  # tf=60 skipped for every truth_kind
    assert set(table["truth_kind"].unique()) == {"plain", "fwd"}
    assert set(table["feature_set"].unique()) == {"full", "noshape"}
    # 2 combos x 2 truth_kinds x 2 feature_sets x 2 models x 2 splits
    assert len(table) == 2 * 2 * 2 * 2 * 2

    for combo in ("15_up", "15_dn"):
        for truth_kind in ("plain", "fwd"):
            for feature_set in ("full", "noshape"):
                combo_dir = base_dir / "alt_truth" / combo / f"{truth_kind}_{feature_set}"
                for fname in ("metrics.json", "screen.csv", "importance.csv"):
                    assert (combo_dir / fname).exists()

    for combo in ("60_up", "60_dn"):
        for sub in ("plain_full", "plain_noshape", "fwd_full", "fwd_noshape"):
            assert not (base_dir / "alt_truth" / combo / sub).exists()  # no files at all for a skipped cell

    # plant a real prior "strict" baseline for ONE combo so the headline/
    # interpretation sections have something concrete to decompose, leaving
    # the other combo's (and 60_up/60_dn's) baseline absent to also exercise
    # the "n/a" (missing source) path.
    _plant_fake_metrics(base_dir / "iter_01" / "15_up" / "metrics.json", gbc_test_auc=0.96)
    _plant_fake_metrics(base_dir / "diag" / "15_up" / "metrics.json", gbc_test_auc=0.70)

    report_path = write_alt_report(table, str(base_dir))

    assert report_path == str(base_dir / "alt_truth" / "report.md")
    assert Path(report_path).exists()
    text = Path(report_path).read_text()

    assert "15_up" in text
    assert "15_dn" in text
    assert "60_up" in text  # headline row still shown even though skipped
    assert "0.9600" in text  # planted strict_full baseline
    assert "0.7000" in text  # planted strict_noshape baseline
    assert "n/a" in text  # 15_dn's / 60_up's missing strict baseline
