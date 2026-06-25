"""nn/experiment_tracker.py — Trial store with holdout promotion gate.

Replaces the legacy ResultAggregator.  Records every training trial (spec,
metrics, holdout, rationale) durably and queryably — the history the
NNStrategist reads.  Gates promotion: a candidate becomes the incumbent only
when it beats the current best on a time-ordered holdout by at least `margin`.

Storage layout:
    tracking/{study_name}/
    ├── index.sqlite           # one row per trial (queryable)
    ├── trials/{trial_id}.json # full record: spec, metrics, rationale
    └── best.json              # current incumbent per group_key

Resilience:
    - SQLite WAL mode for concurrent writers
    - Corrupt index.sqlite → rebuild from trials/*.json (JSON is source of truth)
    - best.json updates serialised via write-then-rename (atomic on POSIX)
    - Missing/empty holdout → trial recorded but not promotable
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nn.nn_model_spec import NNModelSpec


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trials (
    trial_id        TEXT PRIMARY KEY,
    round           INTEGER,
    group_key       TEXT,
    spec_hash       TEXT,
    train_acc       REAL,
    val_acc         REAL,
    holdout_score   REAL,
    promoted        INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'ok',
    created_at      TEXT
);
"""

_INDEX_COLS = [
    "trial_id", "round", "group_key", "spec_hash",
    "train_acc", "val_acc", "holdout_score", "promoted", "status", "created_at",
]


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Trial store with holdout-based, margin-gated promotion.

    Parameters
    ----------
    tracking_dir:
        Root directory for all studies.  The tracker creates
        ``{tracking_dir}/{study_name}/`` and initialises storage.
    study_name:
        Keys the tracking subdirectory.  Reuse = resume.
    metric:
        Field inside the holdout dict used for promotion comparisons.
        Defaults to ``"holdout_score"``.
    margin:
        Minimum improvement over the incumbent required for promotion.
        Gains within margin are treated as noise.
    mode:
        ``"max"`` (default) or ``"min"``.  Determines whether a higher or
        lower metric value is better.
    """

    def __init__(
        self,
        tracking_dir: str,
        study_name: str,
        metric: str = "holdout_score",
        margin: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self._tracking_dir = tracking_dir
        self._study_name = study_name
        self._metric = metric
        self._margin = margin
        self._mode = mode
        self._best_lock = threading.Lock()

        # Build directory structure
        self._study_dir = Path(tracking_dir) / study_name
        self._trials_dir = self._study_dir / "trials"
        self._db_path = self._study_dir / "index.sqlite"
        self._best_path = self._study_dir / "best.json"

        self._study_dir.mkdir(parents=True, exist_ok=True)
        self._trials_dir.mkdir(parents=True, exist_ok=True)

        # Initialise or recover SQLite
        self._init_sqlite()

        # Load existing best.json into memory cache
        self._best_cache: dict = self._load_best_json()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record(
        self,
        spec: "NNModelSpec",
        metrics: dict,
        holdout: dict,
        round: int,
        status: str = "ok",
        rationale: str = None,
        group_key: str = "all",
    ) -> str:
        """Write trial JSON and insert SQLite row; returns trial_id."""
        trial_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Determine scores
        holdout_score = None
        if holdout:
            holdout_score = holdout.get(self._metric, holdout.get("score"))

        train_acc = metrics.get("accuracy") if metrics else None
        val_acc = metrics.get("val_accuracy") if metrics else None

        # Build trial JSON record (spec verbatim from brief)
        trial_record = {
            "trial_id": trial_id,
            "round": round,
            "group_key": group_key,
            "spec_hash": spec.spec_hash,
            "spec": dataclasses.asdict(spec),
            "metrics": {
                "loss": metrics.get("loss") if metrics else None,
                "accuracy": metrics.get("accuracy") if metrics else None,
                "val_loss": metrics.get("val_loss") if metrics else None,
                "val_accuracy": metrics.get("val_accuracy") if metrics else None,
                "per_target": metrics.get("per_target", {}) if metrics else {},
            },
            "holdout": holdout if holdout else {},
            "promoted": False,
            "strategist_rationale": rationale,
            "status": status,
        }

        # Write trial JSON
        json_path = self._trials_dir / f"{trial_id}.json"
        _atomic_json_write(json_path, trial_record)

        # Insert SQLite row
        self._db_insert(
            trial_id=trial_id,
            round=round,
            group_key=group_key,
            spec_hash=spec.spec_hash,
            train_acc=train_acc,
            val_acc=val_acc,
            holdout_score=holdout_score,
            promoted=0,
            status=status,
            created_at=now,
        )

        return trial_id

    def is_improvement(self, group_key: str, holdout: dict) -> bool:
        """Return True if holdout beats the incumbent by at least margin.

        No incumbent → always True.
        Missing/empty holdout → False (not promotable).
        A gain exactly equal to margin IS an improvement (>= comparison).
        """
        if not holdout:
            return False

        candidate_score = holdout.get(self._metric, holdout.get("score"))
        if candidate_score is None:
            return False

        incumbent = self._best_cache.get(group_key)
        if incumbent is None:
            return True

        inc_holdout = incumbent.get("holdout", {})
        inc_score = inc_holdout.get(self._metric, inc_holdout.get("score"))
        if inc_score is None:
            return True

        if self._mode == "max":
            return (candidate_score - inc_score) >= self._margin
        else:  # mode == "min"
            return (inc_score - candidate_score) >= self._margin

    def promote(self, group_key: str, trial_id: str) -> None:
        """Update best.json for group_key; mark trial promoted=true."""
        # Load trial JSON
        json_path = self._trials_dir / f"{trial_id}.json"
        with open(json_path) as f:
            trial_record = json.load(f)

        # Mark promoted in the JSON
        trial_record["promoted"] = True
        _atomic_json_write(json_path, trial_record)

        # Mark promoted in SQLite
        self._db_set_promoted(trial_id)

        # Update best.json atomically
        with self._best_lock:
            self._best_cache[group_key] = trial_record
            _atomic_json_write(self._best_path, self._best_cache)

    def summary(self) -> dict:
        """Compact history for the NNStrategist.

        Aggregated (not raw): incumbent metrics, per-indicator and per-target
        performance deltas, recent trial outcomes, trial/round counts.
        Sized to fit an LLM prompt.
        """
        trials = self._load_all_trials_from_sqlite()

        total = len(trials)
        rounds = sorted({t["round"] for t in trials}) if trials else []
        statuses: dict = {}
        for t in trials:
            statuses[t["status"]] = statuses.get(t["status"], 0) + 1

        # Recent outcomes (last 10)
        recent = sorted(trials, key=lambda t: t.get("created_at") or "")[-10:]
        recent_outcomes = [
            {
                "trial_id": t["trial_id"][:8],
                "round": t["round"],
                "group_key": t["group_key"],
                "holdout_score": t["holdout_score"],
                "status": t["status"],
                "promoted": bool(t["promoted"]),
            }
            for t in recent
        ]

        # Incumbent metrics
        incumbents_summary: dict = {}
        for group_key, inc in self._best_cache.items():
            inc_holdout = inc.get("holdout", {})
            inc_metrics = inc.get("metrics", {})
            incumbents_summary[group_key] = {
                "trial_id": inc.get("trial_id", "")[:8],
                "holdout_score": inc_holdout.get(self._metric, inc_holdout.get("score")),
                "val_accuracy": inc_metrics.get("val_accuracy"),
                "round": inc.get("round"),
                "spec_hash": inc.get("spec_hash", "")[:8],
            }

        # Per-target deltas if available
        per_target_deltas: dict = {}
        for group_key, inc in self._best_cache.items():
            per_target = inc.get("holdout", {}).get("per_target", {})
            if per_target:
                per_target_deltas[group_key] = per_target

        return {
            "study_name": self._study_name,
            "metric": self._metric,
            "mode": self._mode,
            "margin": self._margin,
            "total_trials": total,
            "rounds": rounds,
            "status_counts": statuses,
            "incumbents": incumbents_summary,
            "per_target_deltas": per_target_deltas,
            "recent_outcomes": recent_outcomes,
        }

    def round_summary(self, round: int) -> dict:
        """Metrics for one round: best/median/failed counts."""
        trials = self._load_all_trials_from_sqlite()
        round_trials = [t for t in trials if t["round"] == round]

        if not round_trials:
            return {"round": round, "total": 0, "ok": 0, "failed_count": 0, "pruned_count": 0}

        ok_trials = [t for t in round_trials if t["status"] == "ok"]
        failed_count = sum(1 for t in round_trials if t["status"] == "failed")
        pruned_count = sum(1 for t in round_trials if t["status"] == "pruned")

        scores = [t["holdout_score"] for t in ok_trials if t["holdout_score"] is not None]
        best_score = None
        median_score = None
        if scores:
            best_score = max(scores) if self._mode == "max" else min(scores)
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            if len(sorted_scores) % 2 == 0:
                median_score = (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
            else:
                median_score = sorted_scores[mid]

        return {
            "round": round,
            "total": len(round_trials),
            "ok": len(ok_trials),
            "failed_count": failed_count,
            "pruned_count": pruned_count,
            "best_score": best_score,
            "median_score": median_score,
        }

    def best(self, group_key: str = None) -> dict | None:
        """Return incumbent trial record(s).

        If group_key is given, return that group's incumbent dict or None.
        If group_key is None, return all incumbents as {group_key: record}.
        """
        if group_key is None:
            return dict(self._best_cache)
        return self._best_cache.get(group_key)

    def export(self, format: str = "json") -> str:
        """Export full history as JSON string or CSV string.

        JSON: loads each trial's full JSON record (spec, metrics, holdout,
        strategist_rationale) for complete offline analysis.
        CSV: exports the SQLite index columns (flat format).
        """
        index_rows = self._load_all_trials_from_sqlite()

        if format == "json":
            full_records = []
            for row in index_rows:
                trial_id = row.get("trial_id", "")
                json_path = self._trials_dir / f"{trial_id}.json"
                try:
                    with open(json_path) as f:
                        full_records.append(json.load(f))
                except Exception:
                    # Fall back to index row if JSON file is missing
                    full_records.append(row)
            return json.dumps(full_records, indent=2, default=str)

        # CSV — index columns (flat format)
        buf = io.StringIO()
        if index_rows:
            writer = csv.DictWriter(buf, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
        else:
            writer = csv.DictWriter(buf, fieldnames=_INDEX_COLS)
            writer.writeheader()
        return buf.getvalue()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        """Create the SQLite schema; recover from corruption by rebuilding."""
        if self._db_path.exists():
            if not self._sqlite_healthy():
                self._db_path.unlink()
                self._rebuild_sqlite_from_json()
                return

        con = sqlite3.connect(str(self._db_path))
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(_SCHEMA_SQL)
            con.commit()
        finally:
            con.close()

    def _sqlite_healthy(self) -> bool:
        """Return True if the SQLite file is readable and has the right schema."""
        con = None
        try:
            con = sqlite3.connect(str(self._db_path))
            ok = con.execute("PRAGMA integrity_check").fetchone()[0]
            con.execute("SELECT 1 FROM trials LIMIT 1")
            return ok == "ok"
        except Exception:
            return False
        finally:
            if con:
                con.close()

    def _rebuild_sqlite_from_json(self) -> None:
        """Rebuild index.sqlite from trials/*.json (JSON is source of truth)."""
        con = sqlite3.connect(str(self._db_path))
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(_SCHEMA_SQL)
            con.commit()

            for json_file in sorted(self._trials_dir.glob("*.json")):
                try:
                    with open(json_file) as f:
                        rec = json.load(f)
                except Exception:
                    continue

                holdout = rec.get("holdout", {})
                holdout_score = (
                    holdout.get(self._metric, holdout.get("score"))
                    if holdout else None
                )
                metrics = rec.get("metrics", {})
                con.execute(
                    """INSERT OR REPLACE INTO trials
                       (trial_id, round, group_key, spec_hash, train_acc,
                        val_acc, holdout_score, promoted, status, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        rec.get("trial_id", ""),
                        rec.get("round"),
                        rec.get("group_key", "all"),
                        rec.get("spec_hash", ""),
                        metrics.get("accuracy") if metrics else None,
                        metrics.get("val_accuracy") if metrics else None,
                        holdout_score,
                        1 if rec.get("promoted") else 0,
                        rec.get("status", "ok"),
                        None,  # created_at not stored in JSON
                    ),
                )
            con.commit()
        finally:
            con.close()

    def _db_insert(
        self,
        trial_id: str,
        round: int,
        group_key: str,
        spec_hash: str,
        train_acc,
        val_acc,
        holdout_score,
        promoted: int,
        status: str,
        created_at: str,
    ) -> None:
        con = sqlite3.connect(str(self._db_path))
        try:
            con.execute(
                """INSERT INTO trials
                   (trial_id, round, group_key, spec_hash, train_acc,
                    val_acc, holdout_score, promoted, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (trial_id, round, group_key, spec_hash, train_acc,
                 val_acc, holdout_score, promoted, status, created_at),
            )
            con.commit()
        finally:
            con.close()

    def _db_set_promoted(self, trial_id: str) -> None:
        con = sqlite3.connect(str(self._db_path))
        try:
            con.execute(
                "UPDATE trials SET promoted = 1 WHERE trial_id = ?",
                (trial_id,),
            )
            con.commit()
        finally:
            con.close()

    def _load_all_trials_from_sqlite(self) -> list[dict]:
        con = None
        try:
            con = sqlite3.connect(str(self._db_path))
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM trials ORDER BY created_at ASC"
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            if con:
                con.close()

    # ------------------------------------------------------------------
    # best.json helpers
    # ------------------------------------------------------------------

    def _load_best_json(self) -> dict:
        if not self._best_path.exists():
            return {}
        try:
            with open(self._best_path) as f:
                return json.load(f)
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Utility: atomic JSON write (write-then-rename)
# ---------------------------------------------------------------------------


def _atomic_json_write(path: Path, data: dict) -> None:
    """Write JSON atomically using a temp file + rename (POSIX atomic)."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, path)
