from dataclasses import asdict

import pytest

from nn.orchestration.report import ReportMeta, parse_report, render_report


def _meta(**over):
    base = dict(
        archetype="cnn_lstm",
        version=3,
        parent="v2",
        spec_hash="9f1c4ad2e7b0",
        hypothesis="add a second conv1d block before the lstm",
        holdout_acc=0.6312,
        holdout_loss=0.8741,
        per_class={"up": 0.71, "neutral": 0.49, "down": 0.66},
        decision="promote",
        strike=0,
        next_hypothesis="try a residual skip around the lstm",
    )
    base.update(over)
    return ReportMeta(**base)


def test_render_report_shape_and_roundtrip():
    m = _meta()
    out = render_report(m, "## body\n\nsome prose")

    assert out.startswith("---\n")
    assert "archetype:" in out
    assert out.endswith("## body\n\nsome prose")

    reparsed = parse_report(render_report(m, "hi"))
    assert asdict(reparsed) == asdict(m)
    assert reparsed == m


HANDWRITTEN_V3 = """---
archetype: cnn_lstm
version: 3
parent: v2
spec_hash: 9f1c4ad2e7b0
hypothesis: add a second conv1d block before the lstm
holdout_acc: 0.6312
holdout_loss: 0.8741
per_class:
  up: 0.71
  neutral: 0.49
  down: 0.66
decision: promote
strike: 0
next_hypothesis: try a residual skip around the lstm
---

## v3 — second conv block

Result: holdout accuracy rose, so this version is promoted.
"""

HANDWRITTEN_V1 = """---
archetype: cnn_lstm
version: 1
parent: null
spec_hash: 1a2b3c4d5e6f
hypothesis: baseline declared cnn_lstm archetype
holdout_acc: 0.5503
holdout_loss: 1.0212
per_class:
  up: 0.58
  neutral: 0.41
  down: 0.61
decision: pending
strike: 0
next_hypothesis: null
---

## v1 — baseline

First declared architecture; nothing to compare against yet.
"""


def test_parse_handwritten_types():
    m = parse_report(HANDWRITTEN_V3)
    assert m.archetype == "cnn_lstm"
    assert m.version == 3 and isinstance(m.version, int)
    assert m.parent == "v2"
    assert m.spec_hash == "9f1c4ad2e7b0"
    assert m.holdout_acc == 0.6312 and isinstance(m.holdout_acc, float)
    assert m.holdout_loss == 0.8741 and isinstance(m.holdout_loss, float)
    assert isinstance(m.per_class, dict)
    assert m.decision == "promote"
    assert m.strike == 0 and isinstance(m.strike, int)
    assert m.next_hypothesis == "try a residual skip around the lstm"


def test_parse_handwritten_nullable_parent():
    m = parse_report(HANDWRITTEN_V1)
    assert m.parent is None
    assert m.next_hypothesis is None
    assert m.version == 1
    assert m.decision == "pending"


MISSING_SPEC_HASH = """---
archetype: cnn_lstm
version: 2
parent: v1
hypothesis: drop dropout to 0.1
holdout_acc: 0.60
holdout_loss: 0.95
per_class:
  up: 0.62
  neutral: 0.45
  down: 0.63
decision: revert
strike: 1
next_hypothesis: null
---

## v2

no spec_hash in the frontmatter above.
"""


def test_parse_missing_key_raises():
    with pytest.raises(ValueError) as exc:
        parse_report(MISSING_SPEC_HASH)
    assert "missing report keys" in str(exc.value)
    assert "spec_hash" in str(exc.value)


def test_parse_no_fences_raises():
    plain = "# just a heading\n\nsome prose with no frontmatter at all.\n"
    with pytest.raises(ValueError) as exc:
        parse_report(plain)
    assert "frontmatter" in str(exc.value)


def test_per_class_roundtrips_as_float_dict():
    m = _meta(per_class={"up": 0.71, "neutral": 0.49, "down": 0.66})
    reparsed = parse_report(render_report(m, "body"))

    assert set(reparsed.per_class) == {"up", "neutral", "down"}
    assert reparsed.per_class == {"up": 0.71, "neutral": 0.49, "down": 0.66}
    for v in reparsed.per_class.values():
        assert isinstance(v, float)
