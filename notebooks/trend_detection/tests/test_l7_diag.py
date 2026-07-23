"""Layer 7 tests -- tdlib.diag (the geometry-excluded diagnostic layer:
what separation remains once candle-k's own per-candle shape/geometry
feature family is excluded entirely). See task-7-brief.md.

Filename carries "l7" so `pytest -k l7` selects every test in this module
(mirrors test_l0_conftest.py .. test_l6_oos.py's own "l0".."l6" naming).

Reuses test_l5_loop.py's own ``_engineered_slim`` fixture-construction
helper (task-7-brief.md's explicit instruction: "reuse L5 scaffolding")
rather than duplicating it -- it already engineers tf=15 to have >=60 usable
long/short points on BOTH sides while leaving tf=60 naturally far too sparse
(<60) at seed=0/days=12, which conveniently exercises BOTH the smoke-test
path (tf=15 combos) AND the skip path (tf=60 combos) from the SAME fixture,
with no extra engineering needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tdlib.diag import (
    SHAPE_SUFFIXES,
    is_shape_excluded,
    run_diag,
    write_diag_report,
)

from .test_l5_loop import _engineered_slim

# --- is_shape_excluded: exact-match regex over the 8 named suffixes ----------


@pytest.mark.parametrize("suffix", SHAPE_SUFFIXES)
@pytest.mark.parametrize("tf", [5, 15, 60, 240, 1440])
def test_is_shape_excluded_matches_every_suffix_across_every_tf_prefix(tf, suffix):
    assert is_shape_excluded(f"{tf}_{suffix}") is True


@pytest.mark.parametrize(
    "col",
    [
        # _rm_6* rolling-smoothed family: KEPT IN (module docstring's judgment call) --
        # extra trailing tokens after the bare suffix mean this is NOT a full-string match.
        "60_close_diff_prc_rm_6",
        "60_close_diff_prc_rm_6_std_above",
        "60_close_diff_prc_rm_6_std_below",
        "15_high_diff_prc_rm_6",
        "240_low_diff_prc_rm_6_std_above",
        "240_low_diff_prc_rm_6_std_below",
        # engineered-feature names (tdlib.features.engineered_features's own
        # output) -- never start with a bare digit-string tf prefix, so
        # trivially outside the shape regex, but worth pinning explicitly.
        "bb_pos_15",
        "time_left_60",
        "swing_dist_hi_60_20",
        "htf_move_agree_240",
        # a real, unrelated per-tf raw column -- shares nothing with SHAPE_SUFFIXES.
        "60_rsi_ma8_diff",
        "60_close",
    ],
)
def test_is_shape_excluded_keeps_rm6_family_engineered_and_unrelated_names(col):
    assert is_shape_excluded(col) is False


# --- run_diag / write_diag_report: smoke + skip path --------------------------


def _plant_fake_loop_baseline(base_dir: Path, combo: str, gbc_test_auc: float) -> None:
    """Plant a minimal but real-shaped ``iter_01/{combo}/metrics.json`` under
    ``base_dir`` -- what a real ``tdlib.loop.run_iteration`` baseline
    iteration would have persisted, trimmed to just the one value
    ``write_diag_report``'s baseline comparison actually reads
    (``metrics["gbc"]["test"]["roc_auc"]``)."""
    combo_dir = base_dir / "iter_01" / combo
    combo_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "logistic": {"train": {"roc_auc": 0.5}, "test": {"roc_auc": 0.5}},
        "gbc": {"train": {"roc_auc": 0.99}, "test": {"roc_auc": gbc_test_auc}},
    }
    with open(combo_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)


def test_run_diag_smoke_report_and_no_shape_cols_persisted(tmp_path):
    """report.md + metrics exist for the non-skipped combos, the returned
    table has exactly the spec'd columns, and no shape-suffix column ever
    appears in a persisted screen.csv -- the diagnostic's own core promise.
    """
    df = _engineered_slim()  # tf=15 both sides >=60 pts; tf=60 naturally <60
    base_dir = tmp_path / "trend_detection"
    base_dir.mkdir(parents=True)

    table = run_diag(df, str(base_dir))

    assert list(table.columns) == [
        "combo", "model", "split", "roc_auc", "acc", "base_rate", "lift_long", "lift_short", "n",
    ]
    # tf=15 both sides non-skipped; tf=60 both sides skipped (see module docstring).
    assert set(table["combo"].unique()) == {"15_up", "15_dn"}
    assert set(table["model"].unique()) == {"logistic", "gbc"}
    assert set(table["split"].unique()) == {"train", "test"}
    assert len(table) == 2 * 2 * 2  # 2 combos x 2 models x 2 splits

    for combo in ("15_up", "15_dn"):
        combo_dir = base_dir / "diag" / combo
        for fname in ("metrics.json", "screen.csv", "importance.csv"):
            assert (combo_dir / fname).exists()
            assert (combo_dir / fname).stat().st_size > 0

        scr = pd.read_csv(combo_dir / "screen.csv")
        assert "feature" in scr.columns
        assert len(scr) > 0
        for feat in scr["feature"]:
            assert not is_shape_excluded(feat), f"shape-excluded column {feat!r} leaked into screen.csv"

        imp = pd.read_csv(combo_dir / "importance.csv")
        assert "feature" in imp.columns
        for feat in imp["feature"]:
            assert not is_shape_excluded(feat), f"shape-excluded column {feat!r} leaked into importance.csv"

    # plant a real-shaped baseline for ONE combo so the report's headline
    # comparison has something concrete to show; leave the other combo's
    # baseline absent to also exercise the "n/a" (missing baseline) path.
    _plant_fake_loop_baseline(base_dir, "15_up", gbc_test_auc=0.96)
    loop_summary_path = str(base_dir / "summary.md")
    (base_dir / "summary.md").write_text("# fake loop summary\n")

    report_path = write_diag_report(table, str(base_dir), loop_summary_path)

    assert report_path == str(base_dir / "diag" / "report.md")
    assert Path(report_path).exists()
    text = Path(report_path).read_text()

    assert "15_up" in text
    assert "15_dn" in text
    assert "0.9600" in text  # the planted baseline auc for 15_up
    assert "n/a" in text  # 15_dn's own missing baseline
    for suffix in SHAPE_SUFFIXES:
        # sanity: the report text itself never surfaces a bare shape column
        # name (it only ever lists screened/important features, none of
        # which can be shape-excluded per the assertions above).
        assert f"15_{suffix} |" not in text
        assert f"60_{suffix} |" not in text


def test_run_diag_skip_path_n_less_than_60_honored(tmp_path):
    """tf=60 (both sides) has fewer than MIN_COMBO_POINTS usable points at
    this fixture -- no files, no table rows, and the report shows an
    explicit 'skipped' headline row for both instead of silently omitting
    them or crashing.
    """
    df = _engineered_slim()
    base_dir = tmp_path / "trend_detection"
    base_dir.mkdir(parents=True)

    table = run_diag(df, str(base_dir))

    assert "60_up" not in set(table["combo"].unique())
    assert "60_dn" not in set(table["combo"].unique())

    for combo in ("60_up", "60_dn"):
        combo_dir = base_dir / "diag" / combo
        assert not combo_dir.exists()  # no files at all for a skipped combo

    loop_summary_path = str(base_dir / "summary.md")
    (base_dir / "summary.md").write_text("# fake loop summary\n")
    report_path = write_diag_report(table, str(base_dir), loop_summary_path)
    text = Path(report_path).read_text()

    assert "60_up" in text
    assert "60_dn" in text
    assert "skipped" in text
