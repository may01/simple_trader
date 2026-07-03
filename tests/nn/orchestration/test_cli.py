import json

import pytest

from nn.orchestration import cli
from nn.orchestration.cli import main


def test_decide_subcommand_calls_helper_and_prints_json(capsys, monkeypatch):
    captured = {}

    class FakeDecision:
        action = "promote"
        new_best = True
        strike_count = 0
        stop = False
        reason = "promote: +0.0200 ≥ margin 0.0100"

    def fake_decide(**kwargs):
        captured.update(kwargs)
        return FakeDecision()

    monkeypatch.setattr(cli, "decide", fake_decide)

    rc = main([
        "decide",
        "--best", "0.40", "--candidate", "0.42", "--margin", "0.01",
        "--strike", "0", "--k", "2", "--version-n", "2",
        "--max-versions", "6", "--elapsed", "100.0", "--budget", "28800.0",
    ])

    assert rc == 0
    # the literal float "0.40" parses to best_score=0.40 (NOT the "none" sentinel)
    assert captured == dict(
        best_score=0.40, candidate_score=0.42, margin=0.01,
        strike_count=0, K=2, version_n=2, max_versions=6,
        elapsed_s=100.0, budget_s=28800.0,
    )
    out = json.loads(capsys.readouterr().out)
    assert out == {
        "action": "promote", "new_best": True, "strike_count": 0,
        "stop": False, "reason": "promote: +0.0200 ≥ margin 0.0100",
    }


def test_decide_subcommand_best_none_sentinel(capsys, monkeypatch):
    captured = {}

    class FakeDecision:
        action = "promote"; new_best = True; strike_count = 0
        stop = False; reason = "promote: first version"

    monkeypatch.setattr(cli, "decide",
                        lambda **kw: captured.update(kw) or FakeDecision())

    rc = main([
        "decide",
        "--best", "none", "--candidate", "0.33", "--margin", "0.01",
        "--strike", "0", "--k", "2", "--version-n", "1",
        "--max-versions", "6", "--elapsed", "0.0", "--budget", "28800.0",
    ])

    assert rc == 0
    assert captured["best_score"] is None
    out = json.loads(capsys.readouterr().out)
    assert out["action"] == "promote"


def test_run_version_subcommand_calls_runner_and_prints_json(capsys, monkeypatch):
    captured = {}

    class FakeResult:
        study = "link_attn_v1"
        holdout_score = 0.41
        best = {"all": {"holdout": {"holdout_score": 0.41}}}

    def fake_run(spec_path, study, *, pair, timeout_s):
        captured.update(dict(spec_path=spec_path, study=study,
                             pair=pair, timeout_s=timeout_s))
        return FakeResult()

    monkeypatch.setattr(cli, "run_version_training", fake_run)

    rc = main([
        "run-version",
        "--spec-path", "/trader_data_long/train/link_usdt/nn/specs/abc/spec.yaml",
        "--study", "link_attn_v1", "--pair", "link_usdt", "--timeout", "28800",
    ])

    assert rc == 0
    assert captured == dict(
        spec_path="/trader_data_long/train/link_usdt/nn/specs/abc/spec.yaml",
        study="link_attn_v1", pair="link_usdt", timeout_s=28800.0,
    )
    out = json.loads(capsys.readouterr().out)
    assert out == {"study": "link_attn_v1", "holdout_score": 0.41}


def test_materialize_spec_subcommand_loads_and_prints_path(capsys, monkeypatch):
    calls = {}
    sentinel_spec = object()
    sentinel_root = object()

    monkeypatch.setattr(
        cli.NNModelSpec, "from_yaml",
        classmethod(lambda cls, path: [calls.setdefault("from_yaml", path), sentinel_spec][-1]),
    )
    monkeypatch.setattr(cli, "nn_artefact_root",
                        lambda pair: [calls.setdefault("root_pair", pair), sentinel_root][-1])

    def fake_materialize(spec, artefact_root):
        calls["materialize"] = (spec, artefact_root)
        return "/trader_data_long/train/link_usdt/nn/specs/abc/spec.yaml"

    monkeypatch.setattr(cli, "materialize_spec", fake_materialize)

    rc = main(["materialize-spec",
               "--spec-yaml", "/in/spec.yaml", "--pair", "link_usdt"])

    assert rc == 0
    assert calls["from_yaml"] == "/in/spec.yaml"
    assert calls["root_pair"] == "link_usdt"
    assert calls["materialize"] == (sentinel_spec, sentinel_root)
    assert capsys.readouterr().out.strip() == \
        "/trader_data_long/train/link_usdt/nn/specs/abc/spec.yaml"


def test_no_command_returns_usage_code():
    assert main([]) == 2
