"""Contract: an nn-evolve report.md round-trips through the Layer B codec and the
decision threaded through lineage.decide matches the documented verdict.

This pins the nn-evolve SKILL's promise: it renders a ReportMeta via render_report,
it parses back via parse_report, and feeding the parsed holdout_acc into decide
yields the recorded decision. A below-margin second strike must revert + strike +
stop the lineage.
"""
from dataclasses import asdict

from nn.orchestration.report import ReportMeta, parse_report, render_report
from nn.orchestration.lineage import decide


PROSE = """## v3 — second strike, below margin

**Hypothesis.** Swap the trailing dense block for a gru to capture temporal
structure the dense-only v2 missed (up/down were confused with each other).

**Result.** holdout accuracy 0.4040 (+0.0040 vs incumbent 0.4000) — below the
0.01 margin; up/down recall barely moved.

**What went bad.** decide: revert: +0.0040 < margin (strike 2/2).

**What to improve.** up/down still collapse into neutral; the NEXT change must
target up/down recall, not layer width.

**Decision.** revert — second consecutive strike; lineage stops.
"""


def _meta(**over):
    base = dict(
        archetype="cnn_lstm",
        version=3,
        parent="v2",
        spec_hash="abc123def456",
        hypothesis="swap trailing dense for gru to capture temporal structure",
        holdout_acc=0.4040,          # incumbent 0.40 + 0.004 < margin 0.01
        holdout_loss=0.9300,
        per_class={"up": 0.31, "neutral": 0.78, "down": 0.29},
        decision="revert",
        strike=2,
        next_hypothesis=None,        # lineage stops, so no next hypothesis
    )
    base.update(over)
    return ReportMeta(**base)


def test_report_roundtrips_through_codec():
    meta = _meta()
    text = render_report(meta, PROSE)

    assert text.startswith("---\n")
    assert text.endswith(PROSE)

    back = parse_report(text)
    assert asdict(back) == asdict(meta)
    assert back == meta


def test_below_margin_version_reverts_strikes_and_stops():
    # The skill records the candidate's holdout in the report; the orchestrator
    # threads it through decide with the locked values + the prior strike (1).
    meta = _meta()
    back = parse_report(render_report(meta, PROSE))

    d = decide(
        best_score=0.4000,                 # the incumbent (prior promoted version)
        candidate_score=back.holdout_acc,  # 0.4040 -> +0.0040 < margin 0.01
        margin=0.01,
        strike_count=1,                    # one strike already on the board
        K=2,
        version_n=3,
        max_versions=6,
        elapsed_s=1000.0,
        budget_s=28800,
    )

    assert d.action == "revert"
    assert d.new_best is False
    assert d.strike_count == 2            # 1 -> 2 == K
    assert d.stop is True                 # second consecutive strike stops
    assert "strike" in d.reason and "2/2" in d.reason

    # the recorded report decision matches the verdict the skill must write
    assert back.decision == d.action      # "revert"
    assert back.strike == d.strike_count  # 2
    assert back.next_hypothesis is None   # stopped lineage carries no next step
