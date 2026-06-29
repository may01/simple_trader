"""End-to-end smoke test for one tiny orchestration lineage.

Marker: @pytest.mark.docker_e2e
Skipped when NN_DATA_ROOT / link_usdt data is absent (common in dev / CI).

To run manually with the data volume mounted:
    docker compose run --rm \\
      -e NN_DATA_ROOT=/trader_data_long/train \\
      nn-train python -m pytest tests/nn/orchestration/test_lineage_e2e.py \\
               -q -m docker_e2e -p no:cacheprovider
"""

import json
import os
from pathlib import Path

import pytest

from nn.nn_model_spec import NNModelSpec
from nn.device import nn_artefact_root
from nn.orchestration.spec_store import materialize_spec
from nn.orchestration.runner import run_version_training
from nn.orchestration.lineage import decide
from nn.orchestration.report import parse_report


pytestmark = pytest.mark.docker_e2e


def _data_present() -> bool:
    root = os.environ.get("NN_DATA_ROOT")
    if not root:
        return False
    # Require the actual pair data directory to exist (not just the root mount).
    return os.path.isdir(os.path.join(root, "link_usdt"))


@pytest.mark.skipif(
    not _data_present(),
    reason="NN_DATA_ROOT / link_usdt data absent; e2e needs the populated volume",
)
def test_one_tiny_lineage_produces_artifacts(tmp_path):
    pair = "link_usdt"
    archetype = "e2e_smoke"

    # 1. tiny spec: NNModelSpec.default() shrunk to epochs=5
    spec = NNModelSpec.default()
    spec.epochs = 5
    in_yaml = tmp_path / "v1.yaml"
    spec.to_yaml(str(in_yaml))

    # 2. materialize onto the artefact volume
    spec_path = materialize_spec(spec, nn_artefact_root(pair))
    assert Path(spec_path).exists()

    # 3. run ONE version via the real nn-train container (multi-minute, tiny)
    study = f"{archetype}_v1"
    result = run_version_training(
        str(spec_path), study, pair=pair, timeout_s=3600,
    )
    assert result.study == study

    # tracking/{study}/best.json exists on the volume
    best_path = nn_artefact_root(pair) / "tracking" / study / "best.json"
    assert best_path.exists(), f"missing {best_path}"

    # 4. drive ONE evolve cycle: write a v1 report.md, then decide
    version_dir = tmp_path / archetype / "v1"
    version_dir.mkdir(parents=True)
    report_md = version_dir / "report.md"
    report_md.write_text(_render_v1_report(archetype, spec, result))
    assert report_md.exists()

    # v1 report parses via parse_report (contract with Task 06)
    meta = parse_report(report_md.read_text())
    assert meta.archetype == archetype
    assert meta.version == 1

    # decide: first version always promotes -> lineage produced >= 1 version
    d = decide(
        best_score=None, candidate_score=result.holdout_score, margin=0.01,
        strike_count=0, K=2, version_n=1, max_versions=6,
        elapsed_s=0.0, budget_s=28800.0,
    )
    assert d.action == "promote"          # >= 1 version in the lineage
    assert d.new_best is True

    # 5. if the loop continues, a v2 spec would be emitted next; assert the
    #    lineage is in a continuable state (not an immediate forced stop)
    assert d.stop is False


def _render_v1_report(archetype: str, spec, result) -> str:
    # minimal frontmatter that parse_report (Task 06) accepts
    meta = {
        "archetype": archetype,
        "version": 1,
        "parent": None,
        "spec_hash": spec.spec_hash,
        "hypothesis": "baseline default architecture, epochs=5",
        "holdout_acc": float(result.holdout_score),
        "holdout_loss": 0.0,
        "per_class": {"up": 0.0, "neutral": 0.0, "down": 0.0},
        "decision": "pending",
        "strike": 0,
        "next_hypothesis": None,
    }
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {json.dumps(v)}")
    lines.append("---")
    lines.append("")
    lines.append("# e2e smoke v1")
    return "\n".join(lines)
