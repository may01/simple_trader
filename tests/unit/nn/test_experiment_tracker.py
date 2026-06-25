"""Tests for ExperimentTracker — trial store with holdout promotion gate.

TDD: write tests first (RED), then implement nn/experiment_tracker.py (GREEN).
"""

import dataclasses
import json
import os
import sqlite3
import tempfile
import uuid

import pytest

from nn.experiment_tracker import ExperimentTracker
from nn.nn_model_spec import NNModelSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spec():
    return NNModelSpec.default()


@pytest.fixture
def tracker_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def tracker(tracker_dir):
    return ExperimentTracker(tracker_dir, "study_a", metric="holdout_score", margin=0.02, mode="max")


@pytest.fixture
def min_tracker(tracker_dir):
    """mode='min' for regression metric (lower is better)."""
    return ExperimentTracker(tracker_dir, "study_min", metric="holdout_score", margin=0.01, mode="min")


def holdout(score):
    return {"holdout_score": score, "score": score, "per_target": {}, "n_rows": 100}


# ---------------------------------------------------------------------------
# is_improvement
# ---------------------------------------------------------------------------


class TestIsImprovement:
    def test_no_incumbent_returns_true(self, tracker):
        """No incumbent → is_improvement always returns True."""
        assert tracker.is_improvement("all", holdout(0.70)) is True

    def test_within_margin_returns_false(self, tracker, spec):
        """Gain of 0.015 with margin=0.02 is NOT an improvement."""
        h1 = holdout(0.70)
        id1 = tracker.record(spec, {"accuracy": 0.72, "val_accuracy": 0.69}, h1, round=1)
        tracker.promote("all", id1)

        h2 = holdout(0.715)
        assert tracker.is_improvement("all", h2) is False

    def test_beyond_margin_returns_true(self, tracker, spec):
        """Gain of 0.05 with margin=0.02 IS an improvement."""
        h1 = holdout(0.70)
        id1 = tracker.record(spec, {"accuracy": 0.72, "val_accuracy": 0.69}, h1, round=1)
        tracker.promote("all", id1)

        h3 = holdout(0.75)
        assert tracker.is_improvement("all", h3) is True

    def test_mode_min_no_incumbent_returns_true(self, min_tracker):
        """mode=min, no incumbent → True."""
        assert min_tracker.is_improvement("all", holdout(0.30)) is True

    def test_mode_min_improvement(self, min_tracker, spec):
        """mode=min: lower score is improvement when beyond margin."""
        h1 = holdout(0.30)
        id1 = min_tracker.record(spec, {}, h1, round=1)
        min_tracker.promote("all", id1)

        # 0.30 → 0.28 = delta 0.02 > margin 0.01 → improvement
        assert min_tracker.is_improvement("all", holdout(0.28)) is True

    def test_mode_min_within_margin(self, min_tracker, spec):
        """mode=min: gain within margin → not improvement."""
        h1 = holdout(0.30)
        id1 = min_tracker.record(spec, {}, h1, round=1)
        min_tracker.promote("all", id1)

        # 0.30 → 0.295 = delta 0.005 < margin 0.01 → NOT improvement
        assert min_tracker.is_improvement("all", holdout(0.295)) is False


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_returns_trial_id_string(self, tracker, spec):
        tid = tracker.record(spec, {"accuracy": 0.7}, holdout(0.65), round=1)
        assert isinstance(tid, str) and len(tid) > 0

    def test_record_writes_trial_json(self, tracker, spec, tracker_dir):
        tid = tracker.record(spec, {"accuracy": 0.7, "val_accuracy": 0.65}, holdout(0.65), round=1)
        json_path = os.path.join(tracker_dir, "study_a", "trials", f"{tid}.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)

        # Required JSON keys from spec
        assert data["trial_id"] == tid
        assert data["round"] == 1
        assert "group_key" in data
        assert "spec_hash" in data
        assert "spec" in data
        assert "metrics" in data
        assert "holdout" in data
        assert "promoted" in data
        assert "status" in data

    def test_record_inserts_sqlite_row(self, tracker, spec, tracker_dir):
        tid = tracker.record(spec, {"accuracy": 0.7}, holdout(0.65), round=1)
        db_path = os.path.join(tracker_dir, "study_a", "index.sqlite")
        con = sqlite3.connect(db_path)
        row = con.execute("SELECT trial_id FROM trials WHERE trial_id = ?", (tid,)).fetchone()
        con.close()
        assert row is not None
        assert row[0] == tid

    def test_record_sqlite_columns(self, tracker, spec, tracker_dir):
        """SQLite row has all required columns from spec."""
        tid = tracker.record(
            spec,
            {"accuracy": 0.72, "val_accuracy": 0.69},
            holdout(0.70),
            round=3,
        )
        db_path = os.path.join(tracker_dir, "study_a", "index.sqlite")
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM trials WHERE trial_id = ?", (tid,)).fetchone()
        con.close()

        assert row["trial_id"] == tid
        assert row["round"] == 3
        assert "group_key" in row.keys()
        assert "spec_hash" in row.keys()
        assert row["holdout_score"] == pytest.approx(0.70)
        assert row["promoted"] == 0  # not promoted yet
        assert row["status"] == "ok"
        assert "created_at" in row.keys()
        assert "train_acc" in row.keys()
        assert "val_acc" in row.keys()

    def test_record_status_failed(self, tracker, spec):
        tid = tracker.record(spec, {}, {}, round=1, status="failed")
        assert tid is not None

        # Verify in SQLite
        import sqlite3 as _sqlite3
        db_path = os.path.join(tracker._tracking_dir, tracker._study_name, "index.sqlite")
        con = _sqlite3.connect(db_path)
        row = con.execute("SELECT status FROM trials WHERE trial_id = ?", (tid,)).fetchone()
        con.close()
        assert row[0] == "failed"

    def test_record_status_pruned(self, tracker, spec):
        tid = tracker.record(spec, {}, holdout(0.5), round=2, status="pruned")
        db_path = os.path.join(tracker._tracking_dir, tracker._study_name, "index.sqlite")
        con = sqlite3.connect(db_path)
        row = con.execute("SELECT status FROM trials WHERE trial_id = ?", (tid,)).fetchone()
        con.close()
        assert row[0] == "pruned"

    def test_record_spec_stored_via_dataclasses_asdict(self, tracker, spec, tracker_dir):
        """spec in trial JSON matches dataclasses.asdict(spec)."""
        tid = tracker.record(spec, {"accuracy": 0.7}, holdout(0.65), round=1)
        json_path = os.path.join(tracker_dir, "study_a", "trials", f"{tid}.json")
        with open(json_path) as f:
            data = json.load(f)
        assert data["spec"] == dataclasses.asdict(spec)

    def test_record_spec_hash_matches(self, tracker, spec, tracker_dir):
        """spec_hash in trial JSON matches spec.spec_hash."""
        tid = tracker.record(spec, {}, holdout(0.5), round=1)
        json_path = os.path.join(tracker_dir, "study_a", "trials", f"{tid}.json")
        with open(json_path) as f:
            data = json.load(f)
        assert data["spec_hash"] == spec.spec_hash

    def test_record_with_rationale(self, tracker, spec, tracker_dir):
        tid = tracker.record(
            spec, {}, holdout(0.5), round=1, rationale="test rationale"
        )
        json_path = os.path.join(tracker_dir, "study_a", "trials", f"{tid}.json")
        with open(json_path) as f:
            data = json.load(f)
        assert data["strategist_rationale"] == "test rationale"


# ---------------------------------------------------------------------------
# promote()
# ---------------------------------------------------------------------------


class TestPromote:
    def test_promote_updates_best_json(self, tracker, spec, tracker_dir):
        """promote() writes best.json for the group_key."""
        h1 = holdout(0.70)
        id1 = tracker.record(spec, {"accuracy": 0.72, "val_accuracy": 0.69}, h1, round=1)
        tracker.promote("all", id1)

        best_path = os.path.join(tracker_dir, "study_a", "best.json")
        assert os.path.exists(best_path)

        with open(best_path) as f:
            best = json.load(f)

        assert "all" in best
        assert best["all"]["trial_id"] == id1

    def test_promote_marks_trial_promoted_true(self, tracker, spec, tracker_dir):
        """promote() sets promoted=true in the trial JSON file."""
        h1 = holdout(0.70)
        id1 = tracker.record(spec, {"accuracy": 0.72}, h1, round=1)
        tracker.promote("all", id1)

        json_path = os.path.join(tracker_dir, "study_a", "trials", f"{id1}.json")
        with open(json_path) as f:
            data = json.load(f)
        assert data["promoted"] is True

    def test_promote_updates_sqlite_promoted_flag(self, tracker, spec, tracker_dir):
        """promote() sets promoted=1 in SQLite."""
        h1 = holdout(0.70)
        id1 = tracker.record(spec, {"accuracy": 0.72}, h1, round=1)
        tracker.promote("all", id1)

        db_path = os.path.join(tracker_dir, "study_a", "index.sqlite")
        con = sqlite3.connect(db_path)
        row = con.execute("SELECT promoted FROM trials WHERE trial_id = ?", (id1,)).fetchone()
        con.close()
        assert row[0] == 1

    def test_promote_replaces_previous_best(self, tracker, spec, tracker_dir):
        """Second promote for same group replaces the first in best.json."""
        id1 = tracker.record(spec, {}, holdout(0.70), round=1)
        tracker.promote("all", id1)

        id2 = tracker.record(spec, {}, holdout(0.75), round=2)
        tracker.promote("all", id2)

        with open(os.path.join(tracker_dir, "study_a", "best.json")) as f:
            best = json.load(f)

        assert best["all"]["trial_id"] == id2

    def test_promote_multiple_group_keys(self, tracker, spec, tracker_dir):
        """Different group_keys have independent entries in best.json."""
        id_a = tracker.record(spec, {}, holdout(0.70), round=1, status="ok")
        tracker.promote("group_a", id_a)

        id_b = tracker.record(spec, {}, holdout(0.80), round=1, status="ok")
        tracker.promote("group_b", id_b)

        with open(os.path.join(tracker_dir, "study_a", "best.json")) as f:
            best = json.load(f)

        assert best["group_a"]["trial_id"] == id_a
        assert best["group_b"]["trial_id"] == id_b


# ---------------------------------------------------------------------------
# Resume (reuse = resume)
# ---------------------------------------------------------------------------


class TestResume:
    def test_new_tracker_reads_existing_best(self, tracker_dir, spec):
        """New tracker on same dir resumes existing best + index."""
        t1 = ExperimentTracker(tracker_dir, "study_r", metric="holdout_score", margin=0.01, mode="max")
        h1 = holdout(0.70)
        id1 = t1.record(spec, {}, h1, round=1)
        t1.promote("all", id1)

        # Create a fresh tracker on the same directory
        t2 = ExperimentTracker(tracker_dir, "study_r", metric="holdout_score", margin=0.01, mode="max")
        # It should see the existing incumbent
        assert t2.is_improvement("all", holdout(0.705)) is False  # within margin 0.01
        assert t2.is_improvement("all", holdout(0.72)) is True   # beyond margin

    def test_resume_sees_existing_trials_count(self, tracker_dir, spec):
        """Resumed tracker reflects trials already recorded."""
        t1 = ExperimentTracker(tracker_dir, "study_r2", metric="holdout_score", margin=0.0, mode="max")
        for i in range(3):
            t1.record(spec, {}, holdout(0.5 + i * 0.01), round=1)

        t2 = ExperimentTracker(tracker_dir, "study_r2", metric="holdout_score", margin=0.0, mode="max")
        s = t2.summary()
        assert s["total_trials"] >= 3


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_returns_dict(self, tracker):
        s = tracker.summary()
        assert isinstance(s, dict)

    def test_summary_is_aggregated_not_raw(self, tracker, spec):
        """summary() must be compact/aggregated, not a raw list of trial dumps."""
        for i in range(5):
            tracker.record(spec, {"accuracy": 0.5 + i * 0.01}, holdout(0.5 + i * 0.01), round=1)

        s = tracker.summary()
        # Must NOT be a list of raw trials
        assert not isinstance(s, list)
        # Must have summary-level keys
        assert "total_trials" in s or "trial_count" in s or "rounds" in s or "incumbents" in s

    def test_summary_includes_incumbent_metrics(self, tracker, spec):
        """After promote, summary includes incumbent info."""
        id1 = tracker.record(spec, {"accuracy": 0.72}, holdout(0.70), round=1)
        tracker.promote("all", id1)

        s = tracker.summary()
        # Should mention incumbents somewhere
        assert "incumbents" in s or "best" in s

    def test_summary_empty_study(self, tracker):
        """summary() works with no trials recorded yet."""
        s = tracker.summary()
        assert isinstance(s, dict)


# ---------------------------------------------------------------------------
# round_summary()
# ---------------------------------------------------------------------------


class TestRoundSummary:
    def test_round_summary_returns_dict(self, tracker, spec):
        tracker.record(spec, {}, holdout(0.5), round=1)
        s = tracker.round_summary(1)
        assert isinstance(s, dict)

    def test_round_summary_counts(self, tracker, spec):
        """round_summary includes best/median/failed counts."""
        tracker.record(spec, {"val_accuracy": 0.7}, holdout(0.70), round=2)
        tracker.record(spec, {"val_accuracy": 0.65}, holdout(0.65), round=2)
        tracker.record(spec, {}, {}, round=2, status="failed")
        tracker.record(spec, {}, holdout(0.5), round=2, status="pruned")

        s = tracker.round_summary(2)
        assert isinstance(s, dict)
        assert "failed" in s or "failed_count" in s or "total" in s

    def test_round_summary_empty_round(self, tracker):
        """round_summary on non-existent round returns dict (not exception)."""
        s = tracker.round_summary(99)
        assert isinstance(s, dict)


# ---------------------------------------------------------------------------
# best()
# ---------------------------------------------------------------------------


class TestBest:
    def test_best_returns_incumbent_record(self, tracker, spec):
        id1 = tracker.record(spec, {}, holdout(0.70), round=1)
        tracker.promote("all", id1)

        b = tracker.best("all")
        assert b is not None
        assert b["trial_id"] == id1

    def test_best_no_incumbent_returns_none(self, tracker):
        b = tracker.best("nonexistent_group")
        assert b is None

    def test_best_none_group_key_returns_all_incumbents(self, tracker, spec):
        """best(None) or best() returns all incumbent records as a dict."""
        id_a = tracker.record(spec, {}, holdout(0.70), round=1)
        tracker.promote("group_a", id_a)
        id_b = tracker.record(spec, {}, holdout(0.80), round=1)
        tracker.promote("group_b", id_b)

        b = tracker.best()
        # Should return all groups
        assert isinstance(b, dict)
        assert "group_a" in b
        assert "group_b" in b


# ---------------------------------------------------------------------------
# export()
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json_returns_string(self, tracker, spec):
        tracker.record(spec, {}, holdout(0.5), round=1)
        result = tracker.export(format="json")
        assert isinstance(result, str)

    def test_export_json_is_valid(self, tracker, spec):
        tracker.record(spec, {}, holdout(0.5), round=1)
        result = tracker.export(format="json")
        parsed = json.loads(result)
        assert isinstance(parsed, (list, dict))

    def test_export_csv_returns_string(self, tracker, spec):
        tracker.record(spec, {}, holdout(0.5), round=1)
        result = tracker.export(format="csv")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Resilience: corrupt SQLite → rebuild from trials/*.json
# ---------------------------------------------------------------------------


class TestResilience:
    def test_corrupt_sqlite_rebuilds_from_json(self, tracker_dir, spec):
        """After corrupting index.sqlite, a new tracker rebuilds from JSON files."""
        t1 = ExperimentTracker(tracker_dir, "study_res", metric="holdout_score", margin=0.0, mode="max")
        ids = []
        for i in range(3):
            tid = t1.record(spec, {"accuracy": 0.5 + i * 0.05}, holdout(0.5 + i * 0.05), round=1)
            ids.append(tid)
        t1.promote("all", ids[-1])

        # Corrupt the SQLite file
        db_path = os.path.join(tracker_dir, "study_res", "index.sqlite")
        with open(db_path, "wb") as f:
            f.write(b"CORRUPTED_GARBAGE_DATA_NOT_A_VALID_SQLITE")

        # Create a new tracker — should detect corruption and rebuild
        t2 = ExperimentTracker(tracker_dir, "study_res", metric="holdout_score", margin=0.0, mode="max")

        # The rebuilt tracker should know about the 3 trials
        s = t2.summary()
        assert s.get("total_trials", 0) >= 3 or s.get("trial_count", 0) >= 3

    def test_corrupt_sqlite_best_restored_from_json(self, tracker_dir, spec):
        """After rebuild, promote state is restored from JSON promoted flag."""
        t1 = ExperimentTracker(tracker_dir, "study_res2", metric="holdout_score", margin=0.0, mode="max")
        id1 = t1.record(spec, {}, holdout(0.70), round=1)
        t1.promote("all", id1)

        # Corrupt SQLite
        db_path = os.path.join(tracker_dir, "study_res2", "index.sqlite")
        with open(db_path, "wb") as f:
            f.write(b"CORRUPTED")

        # Rebuild — is_improvement should use the restored best (0.70 with margin 0)
        t2 = ExperimentTracker(tracker_dir, "study_res2", metric="holdout_score", margin=0.01, mode="max")
        # 0.70 → 0.705 = delta 0.005 < margin 0.01 → not improvement
        assert t2.is_improvement("all", holdout(0.705)) is False


# ---------------------------------------------------------------------------
# Verification scenario from brief (adapted to nn.nn_model_spec)
# ---------------------------------------------------------------------------


class TestBriefVerification:
    def test_full_scenario(self, tracker_dir):
        """Mirror the verification snippet in the task brief."""
        t = ExperimentTracker(tracker_dir, "study_a", metric="holdout_score", margin=0.02, mode="max")
        spec = NNModelSpec.default()

        # First trial: no incumbent → improvement, record + promote
        h1 = {"holdout_score": 0.70, "score": 0.70, "per_target": {}, "n_rows": 100}
        assert t.is_improvement("all", h1) is True
        id1 = t.record(spec, {"accuracy": 0.72, "val_accuracy": 0.69}, h1, round=1)
        t.promote("all", id1)

        # Within-margin gain (0.70 → 0.715, margin 0.02) → NOT an improvement
        h2 = {"holdout_score": 0.715, "score": 0.715, "per_target": {}, "n_rows": 100}
        assert t.is_improvement("all", h2) is False

        # Beyond-margin gain (0.70 → 0.75) → improvement
        h3 = {"holdout_score": 0.75, "score": 0.75, "per_target": {}, "n_rows": 100}
        assert t.is_improvement("all", h3) is True
        id3 = t.record(spec, {"accuracy": 0.77, "val_accuracy": 0.71}, h3, round=2)
        t.promote("all", id3)

        # promote updated best.json
        with open(os.path.join(tracker_dir, "study_a", "best.json")) as f:
            best = json.load(f)
        assert best["all"]["trial_id"] == id3, best

        # Storage artifacts exist
        assert os.path.exists(os.path.join(tracker_dir, "study_a", "index.sqlite"))
        assert os.path.exists(os.path.join(tracker_dir, "study_a", "trials", id3 + ".json"))

        # summary() returns an LLM-sized dict
        s = t.summary()
        assert isinstance(s, dict)
