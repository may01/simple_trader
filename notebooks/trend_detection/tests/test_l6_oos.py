"""Layer 6 tests -- tdlib.oos (frozen OOS validation: scores the BEST 2y
iteration's frozen artifacts on an oos slim with ZERO refit). See
task-6-brief.md.

Filename carries "l6" so `pytest -k l6` selects every test in this module
(mirrors test_l0_conftest.py .. test_l5_loop.py's own "l0".."l5" naming).

Two families of fixture-construction helper live here, both duplicated from
test_l5_loop.py's own module-local helpers (per that file's own stated
convention: "duplicated here rather than imported cross-module since each
test_l*.py file in this suite is self-contained, importing only from
.conftest and tdlib") -- task-6-brief.md's own "Context you cannot infer"
explicitly points at test_l5_loop.py's ``loop_artifacts_dir`` fixture
pattern for planting genuine frozen artifacts via
``loop.improvement_loop(slim, max_iters=1)``:

- ``_force_strong_points_both_sides`` / ``_strong_point_positions`` /
  ``_assign_mixed_labels`` / ``_engineered_slim`` -- verbatim copies, used to
  build BOTH the "train" slim (planted into a tmp train_base via
  ``improvement_loop``) and a second, different-seed "oos" slim per
  task-6-brief.md's fixture note.
- ``_oos_slim_with_skip`` -- a deliberately ASYMMETRIC variant (plenty of
  tf=15 "up" points, deliberately few "dn" points) used only by the
  skip-path test: the train_base still has BOTH combos genuinely trained
  (via the unmodified ``_engineered_slim`` train slim), but the OOS slim
  itself is too sparse for "dn" to clear run_oos's own 30-point floor --
  exercising OOS-side data sparsity, not a combo Layer 5 itself skipped.
- ``_oos_slim_zero_dn`` -- an even more extreme variant: tf=15 "dn" gets
  ZERO forced positions at all (not just few), so ``strong_points`` selects
  truly nothing for that combo. Coordinator-review CRITICAL-fix regression
  test: an earlier version of ``_score_combo`` built X/transform/bundle
  machinery BEFORE checking the point count, and ``feature_matrix``'s own
  ``dropna(axis=1, how="all")`` drops EVERY column on a 0-row selection
  (vacuous "all NaN" truth on an empty array) -- producing a misleading
  "could not be reproduced" error that aborted scoring for every OTHER
  combo in the same ``run_oos`` call too.

A third family covers the frozen-baked-column crosscheck gate
(``_inject_baked_move_class`` / ``_flip_baked_move_class``) and a fourth
covers hand-planting a single combo's artifacts directly
(``_plant_manual_combo`` / ``_write_best_json`` / ``_write_fake_train_metrics``)
for the transform-reproduction and verdict-symmetry tests -- steering a REAL
``improvement_loop`` run to land on a specific transform (e.g.
"interact_time_left") and KEEP it, or to produce a specific test-vs-oos lift
SIGN mismatch, is not reliably controllable from a test (depends on
synthetic-data AUC dynamics), so those tests hand-build a minimal train_base
instead, matching task-6-brief.md's own "plant best.json with
transform=..." phrasing literally.

Fast/pure unit tests (``_reproduce_transform``, ``_parse_combo_name``,
``crosscheck_gates`` directly) come first, matching test_l5_loop.py's own
"cheap hand-controlled objects before real pipeline integration tests"
convention; only 5 tests in this file pay for a REAL
``improvement_loop(max_iters=1)`` run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdlib.config import MOVE_CUTS
from tdlib.features import feature_matrix
from tdlib.loop import improvement_loop
from tdlib.models import fit_classifiers, save_bundle
from tdlib.oos import (
    _parse_combo_name,
    _reproduce_transform,
    crosscheck_gates,
    run_oos,
    write_oos_report,
)
from tdlib.points import sym0_class

from .conftest import make_slim

# --- shared fixture-construction helpers (verbatim from test_l5_loop.py) ----


def _force_strong_points_both_sides(df: pd.DataFrame, tf: int, up_positions: list, dn_positions: list) -> None:
    diff_col = df.columns.get_loc(f"{tf}_rsi_ma8_diff")
    df.iloc[:, diff_col] = 0.0
    df.iloc[up_positions, diff_col] = MOVE_CUTS[tf][3] + 1.0
    df.iloc[dn_positions, diff_col] = MOVE_CUTS[tf][0] - 1.0


def _assign_mixed_labels(df: pd.DataFrame, tf: int, positions: list) -> None:
    from .conftest import set_labels

    pattern = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]  # long, short, both, neither
    for i, (long_val, short_val) in enumerate(pattern):
        subset = positions[i::4]
        if subset:
            set_labels(df, tf, subset, long_val=long_val, short_val=short_val)


def _strong_point_positions(n_rows: int) -> tuple:
    up_positions = list(range(0, n_rows, 3))[:200]
    dn_positions = list(range(1, n_rows, 3))[:200]
    return up_positions, dn_positions


def _engineered_slim(seed: int = 0, days: int = 12) -> pd.DataFrame:
    """tf=15 engineered to have >=60 usable long/short points on BOTH sides
    (50 long + 50 short per side) -- both combos train successfully and end
    up in best.json's combos. tf=60/240 left untouched (naturally too sparse
    to ever be trained/kept, so never appear in best.json at all)."""
    df = make_slim(seed=seed, days=days)
    up_positions, dn_positions = _strong_point_positions(len(df))
    _force_strong_points_both_sides(df, 15, up_positions, dn_positions)
    _assign_mixed_labels(df, 15, up_positions)
    _assign_mixed_labels(df, 15, dn_positions)
    return df


def _oos_slim_with_skip(seed: int = 1, days: int = 12) -> pd.DataFrame:
    """Deliberately ASYMMETRIC oos slim: tf=15 "up" gets 120 forced
    positions (-> 30 long + 30 short usable, comfortably >= run_oos's own
    30-point floor); "dn" gets only 8 (-> 2 long + 2 short usable,
    comfortably below it). Used only by the skip-path test -- the train_base
    it is scored against (built from the UNMODIFIED ``_engineered_slim``) has
    BOTH combos genuinely trained, so this exercises OOS-side sparsity for
    one combo while the other combo, from the SAME best.json, still scores.

    "dn"'s 8 positions are spread from row 400 to the end of the frame
    (NOT clustered at the very start, unlike a naive small-stride slice) --
    the swing-level engineered features (tdlib.features.engineered_features)
    need a full rolling window of PRIOR closed candles before they produce a
    non-NaN value (e.g. tf=240's w=20 window needs ~320 rows of history), so
    a few-points selection clustered near row 0 would make those columns
    entirely NaN over the (tiny) kept subset and get silently dropped by
    feature_matrix's own dropna(how="all") -- indistinguishable from a
    genuine reproducibility bug and NOT what this test means to exercise
    (verified empirically: an earlier draft using rows 1..22 tripped run_oos's
    "feature(s) ... could not be reproduced" error instead of a clean skip).
    """
    df = make_slim(seed=seed, days=days)
    n_rows = len(df)
    up_positions = list(range(0, n_rows, 3))[:120]
    dn_positions = list(range(400, n_rows, 90))[:8]
    _force_strong_points_both_sides(df, 15, up_positions, dn_positions)
    _assign_mixed_labels(df, 15, up_positions)
    _assign_mixed_labels(df, 15, dn_positions)
    return df


def _oos_slim_zero_dn(seed: int = 1, days: int = 12) -> pd.DataFrame:
    """Even more extreme than ``_oos_slim_with_skip``: tf=15 "dn" gets NO
    forced positions at all (an empty list, not just a small one) -- since
    ``_force_strong_points_both_sides`` first zeroes the WHOLE
    ``{tf}_rsi_ma8_diff`` column (class 0) before forcing any override, an
    empty ``dn_positions`` leaves genuinely ZERO rows at class -2, so
    ``strong_points(df, 15, -2)`` returns a truly EMPTY (0-row) frame (plus
    its own "no rows selected" UserWarning) -- the exact structurally-
    degenerate case the coordinator-review CRITICAL fix targets. "up" is
    forced normally (plenty of points, scores cleanly) so the same run_oos
    call proves BOTH halves of the fix: the zero-point combo is cleanly
    skipped AND it does not take down the other combo's scoring.
    """
    df = make_slim(seed=seed, days=days)
    n_rows = len(df)
    up_positions = list(range(0, n_rows, 3))[:120]
    dn_positions: list = []
    _force_strong_points_both_sides(df, 15, up_positions, dn_positions)
    _assign_mixed_labels(df, 15, up_positions)
    return df


# --- crosscheck-gate fixture helpers -----------------------------------------


def _inject_baked_move_class(df: pd.DataFrame, tf: int) -> None:
    """Add ``{tf}_move_class_sym0`` equal to ``points.sym0_class``'s own
    computed classes -- the production-baked column's HAPPY-PATH stand-in
    (task-6-brief.md's fixture note: the real oos2m slim carries this column
    for at least one tf; a synthetic oos slim needs it injected by hand)."""
    computed = sym0_class(df[f"{tf}_rsi_ma8_diff"], MOVE_CUTS[tf])
    df[f"{tf}_move_class_sym0"] = computed.astype(np.int8)


def _flip_baked_move_class(df: pd.DataFrame, tf: int, frac: float) -> None:
    """In place: shift the (already-injected) baked column by +1 modulo the
    5-class [-2..2] range at the first ``frac`` fraction of tf's CLOSED
    rows -- a +1 shift out of 5 possible classes is NEVER a no-op, so every
    flipped row is guaranteed to genuinely disagree with
    ``sym0_class``'s own computation, deterministically driving the
    crosscheck fraction below run_oos's 0.999 gate."""
    col = f"{tf}_move_class_sym0"
    closed_idx = df.index[df[f"{tf}_is_closed"] == True]  # noqa: E712
    n_flip = max(1, int(len(closed_idx) * frac))
    flip_idx = closed_idx[:n_flip]
    current = df.loc[flip_idx, col].to_numpy().astype(int)
    df.loc[flip_idx, col] = (((current + 2 + 1) % 5) - 2).astype(np.int8)


# --- manual-plant helpers (transform-reproduction integration tests) --------


def _plant_manual_combo(train_base: Path, combo_name: str, X: pd.DataFrame, y: pd.Series) -> None:
    """Hand-plant one combo's frozen artifacts (bundle + selected_features.json)
    directly under train_base/iter_01/{combo_name}/, from caller-supplied X/y
    -- bypassing run_iteration's full screening pipeline so a test can control
    EXACTLY which columns land in feature_order.json/selected_features.json
    (e.g. a pre-built "a__x__b" interaction column, or a deliberately
    unreproducible bogus one). Steering a REAL improvement_loop run to land
    on a specific transform AND keep it is not reliably controllable (depends
    on synthetic-data AUC dynamics), so these two tests hand-build instead --
    matching task-6-brief.md's own "plant best.json with transform=..."
    phrasing literally.
    """
    combo_dir = train_base / "iter_01" / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)
    bundle = fit_classifiers(X, y)
    save_bundle(bundle, combo_dir / "bundle")
    with open(combo_dir / "selected_features.json", "w") as f:
        json.dump(list(X.columns), f)


def _write_best_json(train_base: Path, transform: str, horizon: str, combos: dict) -> None:
    best = {
        "best_iter": 1, "transform": transform, "horizon": horizon,
        "mean_test_auc": 0.6, "combos": combos,
    }
    train_base.mkdir(parents=True, exist_ok=True)
    with open(train_base / "best.json", "w") as f:
        json.dump(best, f)


def _write_fake_train_metrics(train_base: Path, iter_no: int, combo_name: str,
                               test_lift_long: float, test_lift_short: float) -> None:
    """Hand-plant just enough of ``{train_base}/iter_NN/{combo}/metrics.json``
    (Layer 5's own persisted shape -- see tdlib.loop.run_iteration) for
    write_oos_report's test-vs-oos comparison to read back real gbc
    test-split ``lift_long``/``lift_short`` values -- lets a test precisely
    control the sign-agreement/disagreement scenario the symmetric verdict
    rule needs to be proven against, independent of what a real model fit
    happens to produce.
    """
    combo_dir = train_base / f"iter_{iter_no:02d}" / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "logistic": {"train": {}, "test": {}},
        "gbc": {"train": {}, "test": {
            "roc_auc": 0.6, "acc": 0.55, "base_rate": 0.5, "n": 100,
            "prec_top_decile": 0.6, "prec_bottom_decile": 0.6,
            "lift_long": test_lift_long, "lift_short": test_lift_short,
        }},
    }
    with open(combo_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)


@pytest.fixture
def train_base_dir(tmp_path, monkeypatch):
    """Patch tdlib.loop's own ``artifacts_dir`` binding to a tmp_path
    location, so a real ``improvement_loop`` call plants genuine frozen
    artifacts there. Mirrors test_l5_loop.py's ``loop_artifacts_dir``
    fixture exactly (same monkeypatch target) -- task-6-brief.md's own
    "Context you cannot infer" names this fixture pattern directly."""
    base = tmp_path / "train_2y"
    monkeypatch.setattr("tdlib.loop.artifacts_dir", lambda: str(base))
    return base


# --- _parse_combo_name --------------------------------------------------------


def test_parse_combo_name_up_and_dn():
    assert _parse_combo_name("15_up") == (15, 2)
    assert _parse_combo_name("240_dn") == (240, -2)


def test_parse_combo_name_unrecognized_raises_value_error():
    with pytest.raises(ValueError):
        _parse_combo_name("not_a_combo")


# --- _reproduce_transform ------------------------------------------------------


def test_reproduce_transform_baseline_and_horizon_n2_are_identity():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    out_baseline = _reproduce_transform(X, "baseline", ["a", "b"])
    out_horizon = _reproduce_transform(X, "horizon_n2", ["a", "b"])

    pd.testing.assert_frame_equal(out_baseline, X)
    pd.testing.assert_frame_equal(out_horizon, X)


def test_reproduce_transform_prune_top40_selects_only_named_columns_present():
    X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    selected_features = ["c", "a"]  # deliberately not X's own column order

    out = _reproduce_transform(X, "prune_top40", selected_features)

    assert list(out.columns) == ["c", "a"]


def test_reproduce_transform_interact_time_left_builds_product_and_leaves_bogus_unresolved():
    X = pd.DataFrame({
        "bb_pos_15": [2.0, 4.0, 6.0],
        "time_left_60": [0.5, 0.25, 0.1],
        "other": [1.0, 1.0, 1.0],
    })
    selected_features = [
        "bb_pos_15", "time_left_60", "other",
        "bb_pos_15__x__time_left_60",  # rebuildable: both bases present
        "totally_unknown_feature_xyz",  # NOT rebuildable: not a__x__b, not present
    ]

    out = _reproduce_transform(X, "interact_time_left", selected_features)

    assert "bb_pos_15__x__time_left_60" in out.columns
    expected = X["bb_pos_15"] * X["time_left_60"]
    pd.testing.assert_series_equal(out["bb_pos_15__x__time_left_60"], expected, check_names=False)
    # unresolved name is simply absent -- run_oos's own feature_order check
    # (integration tests below) is what turns this into a named error.
    assert "totally_unknown_feature_xyz" not in out.columns
    for c in X.columns:
        assert c in out.columns


def test_reproduce_transform_interact_time_left_leaves_unresolved_when_base_column_absent():
    """MINOR-2 (coordinator review): distinct code path from the
    "totally_unknown_feature_xyz" case above -- THAT name never matches
    ``a__x__b`` at all (``_INTERACT_RE`` doesn't match, ``m`` is None).
    THIS name DOES match the pattern (a real "__x__" separator, so ``m`` is
    truthy) but its base column "missing_base" is absent from X -- must
    still land in the SAME unresolved/absent outcome (never attempt
    ``X["missing_base"]`` and raise a raw pandas KeyError here); run_oos's
    own feature_order check downstream is what turns this into a named,
    actionable error, not this function.
    """
    X = pd.DataFrame({"time_left_60": [0.5, 0.25, 0.1], "other": [1.0, 1.0, 1.0]})
    selected_features = ["time_left_60", "other", "missing_base__x__time_left_60"]

    out = _reproduce_transform(X, "interact_time_left", selected_features)

    assert "missing_base__x__time_left_60" not in out.columns
    for c in X.columns:
        assert c in out.columns


def test_reproduce_transform_unknown_transform_raises_value_error():
    X = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError):
        _reproduce_transform(X, "not_a_real_transform", ["a"])


# --- crosscheck_gates (pure, no train_base needed) ---------------------------


def test_crosscheck_gates_direct_pass_warn_and_raise():
    # (a) matching baked column -> passes
    df_ok = _engineered_slim(seed=1)
    _inject_baked_move_class(df_ok, 15)
    gates_ok = crosscheck_gates(df_ok)
    assert gates_ok[15]["status"] == "passed"
    assert gates_ok[15]["fraction"] == pytest.approx(1.0)

    # (b) no baked column at all -> warns, records "skipped"
    df_missing = _engineered_slim(seed=2)
    with pytest.warns(UserWarning):
        gates_missing = crosscheck_gates(df_missing)
    assert gates_missing[15]["status"] == "skipped"

    # (c) drifted baked column -> RuntimeError naming tf=15 + a fraction
    df_bad = _engineered_slim(seed=1)
    _inject_baked_move_class(df_bad, 15)
    _flip_baked_move_class(df_bad, 15, frac=0.01)
    with pytest.raises(RuntimeError) as exc_info:
        crosscheck_gates(df_bad)
    assert "15" in str(exc_info.value)


# --- run_oos: missing best.json ------------------------------------------------


def test_missing_best_json_raises_file_not_found_naming_path(tmp_path):
    missing_base = tmp_path / "nope"
    oos_slim = _engineered_slim(seed=1)

    with pytest.raises(FileNotFoundError) as exc_info:
        run_oos(oos_slim, str(missing_base))

    assert str(missing_base) in str(exc_info.value)


# --- run_oos: missing bundle file ----------------------------------------------


def test_missing_bundle_file_raises_actionable_error(train_base_dir):
    train_slim = _engineered_slim(seed=0)
    improvement_loop(train_slim, max_iters=1)

    bundle_file = train_base_dir / "iter_01" / "15_up" / "bundle" / "gbc.joblib"
    assert bundle_file.exists()
    bundle_file.unlink()

    oos_slim = _engineered_slim(seed=1)
    with pytest.raises(FileNotFoundError) as exc_info:
        run_oos(oos_slim, str(train_base_dir))

    assert "gbc.joblib" in str(exc_info.value)


# --- run_oos: skip path ---------------------------------------------------------


def test_skip_path_low_oos_points_marks_skipped_others_scored(train_base_dir):
    train_slim = _engineered_slim(seed=0)  # both 15_up/15_dn well-trained
    improvement_loop(train_slim, max_iters=1)

    oos_slim = _oos_slim_with_skip(seed=1)  # 15_up plenty, 15_dn < 30
    table = run_oos(oos_slim, str(train_base_dir))

    dn_rows = table[table["combo"] == "15_dn"]
    assert len(dn_rows) > 0
    assert bool(dn_rows["skipped"].all())
    assert (dn_rows["n_long"] + dn_rows["n_short"] < 30).all()

    up_rows = table[table["combo"] == "15_up"]
    assert len(up_rows) > 0
    assert not up_rows["skipped"].any()
    for _, row in up_rows.iterrows():
        assert math.isfinite(row["roc_auc"])


def test_zero_oos_points_combo_skipped_others_scored_and_reported(train_base_dir, tmp_path):
    """CRITICAL fix regression test (coordinator review): a combo with
    EXACTLY ZERO oos-side strong points must never reach the missing-
    feature_order check. Before the fix, ``_score_combo`` built X (via
    feature_matrix)/transform/bundle machinery BEFORE checking the point
    count -- feature_matrix's own ``dropna(axis=1, how="all")`` drops EVERY
    column on a 0-row selection (``.all()`` over an empty array is vacuously
    True), so the missing-feature_order check then reported the ENTIRE
    frozen feature set as "could not be reproduced", raising and aborting
    scoring for every OTHER combo in the same run_oos call too. Realistic in
    production: tf=240 has only ~360 closed candles in a 2-month OOS window,
    and the +-2 strong-move tails can legitimately be empty on one side.

    Also covers IMPORTANT-2 (coordinator review): write_oos_report's
    "skipped (n=...)" verdict line was previously never exercised by any
    test (every prior skip-path table had n>0, just <30).
    """
    train_slim = _engineered_slim(seed=0)  # both 15_up/15_dn well-trained
    improvement_loop(train_slim, max_iters=1)

    oos_slim = _oos_slim_zero_dn(seed=1)  # 15_up plenty, 15_dn EXACTLY 0 points
    with pytest.warns(UserWarning):  # strong_points' own "no rows selected" warning
        table = run_oos(oos_slim, str(train_base_dir))  # must NOT raise/abort

    dn_rows = table[table["combo"] == "15_dn"]
    assert len(dn_rows) == 2  # still 2 rows (one per model) -- uniform table shape
    assert bool(dn_rows["skipped"].all())
    assert (dn_rows["n_long"] == 0).all()
    assert (dn_rows["n_short"] == 0).all()
    assert (dn_rows["n"] == 0).all()

    up_rows = table[table["combo"] == "15_up"]
    assert len(up_rows) == 2
    assert not up_rows["skipped"].any()
    for _, row in up_rows.iterrows():
        assert math.isfinite(row["roc_auc"])

    # IMPORTANT-2: write_oos_report's skipped-verdict branch, exercised for real.
    gates = crosscheck_gates(oos_slim)
    out_dir = tmp_path / "zero_dn_report"
    md_path = write_oos_report(table, gates, str(train_base_dir), str(out_dir))
    text = Path(md_path).read_text()
    assert "15_dn: skipped (n=0" in text


# --- run_oos: no-refit purity ----------------------------------------------------


def test_no_refit_purity_freezestats_fit_never_called(train_base_dir, monkeypatch):
    train_slim = _engineered_slim(seed=0)
    improvement_loop(train_slim, max_iters=1)

    def _boom(self, X):
        raise AssertionError("FreezeStats.fit must never be called during run_oos")

    monkeypatch.setattr("tdlib.features.FreezeStats.fit", _boom)

    oos_slim = _engineered_slim(seed=1)
    table = run_oos(oos_slim, str(train_base_dir))  # must NOT raise

    assert not table.empty
    assert not table["skipped"].all()


# --- run_oos + write_oos_report: full integration -------------------------------


def test_oos_no_refit_end_to_end_and_report(train_base_dir, tmp_path):
    train_slim = _engineered_slim(seed=0)
    improvement_loop(train_slim, max_iters=1)

    oos_slim = _engineered_slim(seed=1)  # different seed than train
    _inject_baked_move_class(oos_slim, 15)  # exercise the gate's happy path

    freeze_up = train_base_dir / "iter_01" / "15_up" / "bundle" / "freeze.json"
    freeze_dn = train_base_dir / "iter_01" / "15_dn" / "bundle" / "freeze.json"
    before_up = freeze_up.read_bytes()
    before_dn = freeze_dn.read_bytes()

    table = run_oos(oos_slim, str(train_base_dir))

    # no-refit: freeze.json bytes on disk unchanged after scoring.
    assert freeze_up.read_bytes() == before_up
    assert freeze_dn.read_bytes() == before_dn

    # rows for the non-skipped combos x 2 models.
    assert set(table["combo"].unique()) == {"15_up", "15_dn"}
    non_skipped = table[~table["skipped"]]
    assert len(non_skipped) == 4  # 2 combos x 2 models
    assert set(non_skipped["model"]) == {"logistic", "gbc"}
    for _, row in non_skipped.iterrows():
        assert math.isfinite(row["roc_auc"])
        assert math.isfinite(row["acc"])
        assert math.isfinite(row["base_rate"])

    # crosscheck gate propagates a RuntimeError through run_oos itself when
    # the SAME train_base is scored against a drifted oos slim.
    oos_bad = _engineered_slim(seed=1)
    _inject_baked_move_class(oos_bad, 15)
    _flip_baked_move_class(oos_bad, 15, frac=0.01)
    with pytest.raises(RuntimeError):
        run_oos(oos_bad, str(train_base_dir))

    # write_oos_report: md + csv.
    gates = crosscheck_gates(oos_slim)
    out_dir = tmp_path / "oos_out"
    md_path = write_oos_report(table, gates, str(train_base_dir), str(out_dir))

    assert md_path == str(out_dir / "oos_report.md")
    text = Path(md_path).read_text()
    assert "tf=15" in text  # gate line
    assert "passed" in text
    assert "15_up" in text and "15_dn" in text
    assert ("holds" in text) or ("fails" in text)  # per-combo verdict line
    # IMPORTANT-1: verdict line shows BOTH lift signs, not lift_long alone.
    assert "oos_lift_long=" in text and "test_lift_long=" in text
    assert "oos_lift_short=" in text and "test_lift_short=" in text

    csv_path = out_dir / "oos_table.csv"
    assert csv_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == len(table)


# --- write_oos_report: verdict symmetry (both lift signs) ------------------------


def _make_oos_row(combo: str, model: str, roc_auc: float, lift_long: float, lift_short: float) -> dict:
    return {
        "combo": combo, "tf": 15, "side": 2, "model": model,
        "n": 100, "n_long": 50, "n_short": 50, "base_rate": 0.5,
        "roc_auc": roc_auc, "acc": 0.58, "prec_top_decile": 0.6, "prec_bottom_decile": 0.6,
        "lift_long": lift_long, "lift_short": lift_short,
        "long_fwd1": 0.5, "long_fwd4": 0.5, "short_fwd1": 0.5, "short_fwd4": 0.5,
        "skipped": False,
    }


def test_write_oos_report_verdict_is_symmetric_across_both_lift_signs(tmp_path):
    """IMPORTANT-1 fix (coordinator review): the per-combo verdict must
    check BOTH lift_long and lift_short sign agreement, not lift_long
    alone -- a combo whose oos lift_long sign matches test but whose
    lift_short sign does NOT must verdict "fails", never "holds". Hand-built
    table + hand-planted metrics.json (rather than a real run_oos call) so
    the exact sign-agreement/disagreement scenario is fully controlled,
    independent of what a real model fit happens to produce.
    """
    train_base = tmp_path / "train_base"
    _write_best_json(train_base, transform="baseline", horizon="n1", combos={
        "15_up": {
            "bundle_dir": "iter_01/15_up/bundle",
            "features": "iter_01/15_up/selected_features.json",
            "n_test": 100,
        },
    })
    # test-side: lift_long positive, lift_short positive.
    _write_fake_train_metrics(train_base, 1, "15_up", test_lift_long=0.05, test_lift_short=0.05)

    table = pd.DataFrame([
        # oos: lift_long POSITIVE (matches test's +) but lift_short NEGATIVE
        # (disagrees with test's +) -- must fail despite roc_auc > 0.5 and
        # lift_long agreeing.
        _make_oos_row("15_up", "gbc", roc_auc=0.62, lift_long=0.04, lift_short=-0.02),
        _make_oos_row("15_up", "logistic", roc_auc=0.55, lift_long=0.03, lift_short=-0.01),
    ])
    gates = {15: {"status": "skipped", "fraction": None}}

    out_dir = tmp_path / "report_mismatch"
    md_path = write_oos_report(table, gates, str(train_base), str(out_dir))
    text = Path(md_path).read_text()

    assert "15_up: fails" in text  # NOT holds -- lift_short sign disagrees
    assert "15_up: holds" not in text
    assert "oos_lift_short=-0.0200" in text
    assert "test_lift_short=0.0500" in text

    # Sanity: flipping test_lift_short to ALSO be negative (matching oos)
    # flips the verdict to "holds" -- proves the rule is genuinely
    # sign-based (both signs independently gating), not "always fails
    # whenever both lift_long and lift_short are present".
    _write_fake_train_metrics(train_base, 1, "15_up", test_lift_long=0.05, test_lift_short=-0.03)
    out_dir2 = tmp_path / "report_match"
    md_path2 = write_oos_report(table, gates, str(train_base), str(out_dir2))
    text2 = Path(md_path2).read_text()
    assert "15_up: holds" in text2


# --- run_oos: transform reproduction (hand-planted) ------------------------------


def test_run_oos_interact_time_left_rebuilds_product_column_end_to_end(tmp_path):
    train_base = tmp_path / "train_base"
    oos_slim = _engineered_slim(seed=1)

    X, y = feature_matrix(oos_slim, 15, 2, horizon="n1")
    assert "bb_pos_15" in X.columns and "time_left_60" in X.columns

    interact_col = "bb_pos_15__x__time_left_60"
    X_planted = X.copy()
    X_planted[interact_col] = X_planted["bb_pos_15"] * X_planted["time_left_60"]

    _plant_manual_combo(train_base, "15_up", X_planted, y)
    _write_best_json(train_base, transform="interact_time_left", horizon="n1", combos={
        "15_up": {
            "bundle_dir": "iter_01/15_up/bundle",
            "features": "iter_01/15_up/selected_features.json",
            "n_test": int(len(y)),
        },
    })

    table = run_oos(oos_slim, str(train_base))

    row = table[(table["combo"] == "15_up") & (table["model"] == "gbc")].iloc[0]
    assert row["skipped"] == False  # noqa: E712
    assert math.isfinite(row["roc_auc"])


def test_run_oos_unrebuildable_feature_raises_naming_it_end_to_end(tmp_path):
    train_base = tmp_path / "train_base"
    oos_slim = _engineered_slim(seed=1)

    X, y = feature_matrix(oos_slim, 15, 2, horizon="n1")
    bogus_col = "totally_unknown_feature_xyz"
    X_planted = X.copy()
    X_planted[bogus_col] = 0.0  # only ever present at "train" time

    _plant_manual_combo(train_base, "15_up", X_planted, y)
    _write_best_json(train_base, transform="interact_time_left", horizon="n1", combos={
        "15_up": {
            "bundle_dir": "iter_01/15_up/bundle",
            "features": "iter_01/15_up/selected_features.json",
            "n_test": int(len(y)),
        },
    })

    with pytest.raises(ValueError) as exc_info:
        run_oos(oos_slim, str(train_base))

    assert bogus_col in str(exc_info.value)
