import json
import os
import subprocess

import pytest

from nn.orchestration import runner
from nn.orchestration.runner import (
    VersionResult,
    _extract_holdout_score,
    _read_best_json,
    run_version_training,
)


def test_read_best_json_returns_none_when_absent(tmp_path):
    # tracking dir exists but the {study}/best.json was never written
    assert _read_best_json(tmp_path, "phase17_v0") is None


def test_read_best_json_parses_written_file(tmp_path):
    study = "phase17_v0"
    study_dir = tmp_path / study
    study_dir.mkdir()
    payload = {"all": {"holdout": {"holdout_score": 0.41}}}
    (study_dir / "best.json").write_text(json.dumps(payload))

    assert _read_best_json(tmp_path, study) == payload


def test_extract_holdout_score_max_across_groups():
    best = {
        "all": {"holdout": {"holdout_score": 0.41}},
        "g2": {"holdout": {"holdout_score": 0.45}},
    }
    assert _extract_holdout_score(best) == 0.45


def test_extract_holdout_score_empty_is_zero():
    assert _extract_holdout_score({}) == 0.0
    # record present but no holdout info anywhere -> 0.0 fallback
    assert _extract_holdout_score({"all": {"metrics": {"loss": 1.0}}}) == 0.0


def test_run_version_training_launches_and_returns_score(tmp_path, monkeypatch):
    study = "phase17_v0"
    spec_path = "/trader_data_long/link_usdt/nn/specs/v0.yaml"
    pair = "link_usdt"

    # tracking dir -> tmp_path/tracking ; pre-write best.json the "container" left
    tracking_dir = tmp_path / "tracking"
    study_dir = tracking_dir / study
    study_dir.mkdir(parents=True)
    payload = {"all": {"holdout": {"holdout_score": 0.41}}}
    (study_dir / "best.json").write_text(json.dumps(payload))

    monkeypatch.setattr(runner, "nn_artefact_root", lambda p: tmp_path)

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        captured["check"] = kwargs.get("check")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    result = run_version_training(
        spec_path, study, pair=pair, timeout_s=123.0,
        extra_env={"NN_EXTRA": "x"},
    )

    cmd = captured["cmd"]
    assert "nn-train" in cmd
    assert "-e" in cmd and f"NN_SPEC_PATH={spec_path}" in cmd
    assert f"NN_STUDY={study}" in cmd
    assert "NN_TRAIN_MODE=search" in cmd
    assert "NN_EXTRA=x" in cmd
    assert cmd[-1] == "nn-train"  # service name is the trailing positional
    assert captured["timeout"] == 123.0
    assert captured["check"] is False

    assert isinstance(result, VersionResult)
    assert result.study == study
    assert result.holdout_score == 0.41
    assert result.best == payload


def test_run_version_training_raises_when_no_best_json(tmp_path, monkeypatch):
    study = "phase17_v0"
    monkeypatch.setattr(runner, "nn_artefact_root", lambda p: tmp_path)

    # subprocess is a no-op: container "ran" but wrote nothing
    monkeypatch.setattr(
        runner.subprocess, "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0),
    )
    # keep the poll loop instant — no real waiting for a file that never comes
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="no best.json"):
        run_version_training(
            "/specs/v0.yaml", study, pair="link_usdt", timeout_s=5.0,
        )


@pytest.mark.docker_e2e
@pytest.mark.skipif(
    not os.environ.get("NN_DATA_ROOT")
    or not os.path.isdir(os.environ.get("NN_DATA_ROOT", "")),
    reason="NN_DATA_ROOT data absent; real nn-train e2e lives in Task 11",
)
def test_run_version_training_real_docker():
    # Genuine end-to-end (build a spec via spec_store, launch nn-train, assert a
    # real holdout_score lands) is implemented in Task 11. This placeholder only
    # documents the seam and stays skipped unless NN_DATA_ROOT data is present.
    pytest.skip("real nn-train e2e covered by Task 11")
