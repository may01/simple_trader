import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from nn.device import nn_artefact_root


@dataclass
class VersionResult:
    study: str
    holdout_score: float
    best: dict


def _read_best_json(tracking_dir: Path, study: str) -> dict | None:
    """Parse {tracking_dir}/{study}/best.json, or return None if absent."""
    path = Path(tracking_dir) / study / "best.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_holdout_score(best: dict) -> float:
    """Max holdout_score across all group_keys in best.json; 0.0 if none.

    Each value is a trial record that either nests the score under
    "holdout" -> "holdout_score", or carries a flat "holdout_score".
    """
    if not best:
        return 0.0
    scores: list[float] = []
    for record in best.values():
        if not isinstance(record, dict):
            continue
        holdout = record.get("holdout")
        if isinstance(holdout, dict) and "holdout_score" in holdout:
            scores.append(float(holdout["holdout_score"]))
        else:
            scores.append(float(record.get("holdout_score", 0.0)))
    return max(scores) if scores else 0.0


def run_version_training(spec_path: str, study: str, *, pair: str,
                         timeout_s: float, extra_env: dict | None = None) -> VersionResult:
    """Launch the nn-train service for one version, poll best.json, return score.

    Runs `docker compose run --rm` in the foreground to completion with
    NN_SPEC_PATH / NN_STUDY / NN_TRAIN_MODE=search set, then reads the study's
    best.json off the shared artefact volume.

    Raises:
        TimeoutError: the container did not finish within timeout_s.
        RuntimeError: the container finished but no best.json was produced.
    """
    cmd = [
        "docker", "compose", "run", "--rm",
        "-e", f"NN_SPEC_PATH={spec_path}",
        "-e", f"NN_STUDY={study}",
        "-e", "NN_TRAIN_MODE=search",
    ]
    for key, value in (extra_env or {}).items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append("nn-train")

    try:
        subprocess.run(cmd, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"nn-train timed out after {timeout_s}s for study {study!r}"
        ) from exc

    tracking_dir = nn_artefact_root(pair) / "tracking"

    best = None
    # absorb the flush/visibility window between container exit and file landing
    for _ in range(10):
        best = _read_best_json(tracking_dir, study)
        if best is not None:
            break
        time.sleep(0.5)

    if best is None:
        raise RuntimeError(
            f"nn-train produced no best.json for study {study!r} "
            f"under {tracking_dir / study}"
        )

    return VersionResult(
        study=study,
        holdout_score=_extract_holdout_score(best),
        best=best,
    )
