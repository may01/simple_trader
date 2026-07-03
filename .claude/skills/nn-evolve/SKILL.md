---
name: nn-evolve
description: >-
  Use after a single NN version has been trained (Tier 1 / Optuna) to evolve the
  lineage: record the version's report.md (frontmatter + prose), call lineage.decide
  to promote/revert/strike/stop, and — if continuing — reason ONE structural change
  (never random — ADR-0001) and emit the next version's spec.yaml to the volume.
---

# nn-evolve — evolve one NN version (Tier 2)

You are handed: a trained version's `VersionResult` (study, holdout_score, best),
the prior version's `report.md` text (or None for v1), and the lineage counters
(version_n, current best_score, strike_count, elapsed_s). Locked values:
margin=0.01, K=2, max_versions=6, budget_s=28800.

## 1. Read the inputs

- If a prior report exists: `prior = parse_report(prior_md)`. Its `holdout_acc` is
  the running best; its `strike` is the strike count; its `next_hypothesis` is the
  hypothesis THIS version was built to test (becomes this report's `hypothesis`).
- For v1: no prior; `best_score=None`, `strike_count=0`, parent=None, and the
  hypothesis is the archetype declaration from `nn-investigate` (Task 09).
- From `result.best`, pull this version's holdout accuracy, holdout loss, the
  per-class accuracy {up, neutral, down}, and the confusion matrix (3×3 over
  up/neutral/down). These live in each trial record's `metrics`.

## 2. Write the version report

- Decide first (step 3), then render — so the frontmatter `decision`/`strike`/
  `next_hypothesis` reflect the verdict. (You may draft prose before deciding, but
  the rendered frontmatter is post-decision.)
- Build `meta = ReportMeta(archetype=..., version=version_n, parent=prior_label,
  spec_hash=this_spec_hash, hypothesis=this_hypothesis,
  holdout_acc=..., holdout_loss=..., per_class={...}, decision=..., strike=...,
  next_hypothesis=...)`.
- `body` = the PROSE TEMPLATE below, filled in.
- `text = render_report(meta, body)`; write to
  `external/docs/superpowers/specs/supporting-systems/nn-training-orchestration/{archetype}/v{version_n}/report.md`.
- The report MUST parse back: `parse_report(text)` returns an equal `ReportMeta`.

## 3. Decide the lineage move

Call exactly:

```python
from nn.orchestration.report import ReportMeta, parse_report, render_report
from nn.orchestration.lineage import decide, LineageDecision
from nn.orchestration.spec_store import materialize_spec

d = decide(
    best_score=prior.holdout_acc if prior else None,
    candidate_score=result.holdout_score,
    margin=0.01, strike_count=(prior.strike if prior else 0),
    K=2, version_n=version_n, max_versions=6,
    elapsed_s=elapsed_s, budget_s=28800,
)
```

Branch on `d`:

- `d.action == "promote"`: this version is the new incumbent. Set the report's
  `decision="promote"`, `strike=0` (decide reset it). The next version's parent is
  THIS version; the next best_score is THIS holdout_acc.
- `d.action == "revert"`: keep the prior incumbent. Set `decision="revert"`,
  `strike=d.strike_count` (incremented). The next version's parent is still the
  incumbent (the prior promoted version), and best_score stays the incumbent's.

Record `d.reason` verbatim in the report's "what bad / what to improve" prose.

## 4. Stop or continue

- If `d.stop` is True: FINALIZE. Set `next_hypothesis=None`. Write the report.
  Return to the orchestrator: lineage done, winner = the current incumbent, with
  `d.reason` as the stop cause (strikes / max_versions / budget). Do NOT build a
  new spec.
- If `d.stop` is False: CONTINUE. Reason ONE structural change (step 5), set it as
  `next_hypothesis`, build + materialise the next spec (step 6), then return it to
  the orchestrator to train as the next version.

## 5. Reason ONE structural change (NOT random — ADR-0001)

Read the failure in the per-class/confusion table you recorded:

- Diagnose the dominant failure mode (e.g. always-neutral; up/down confused with
  each other; underfit-all; overfit gap; one regime drowning a minority class).
- Pick a SINGLE edit from the structural menu below that DIRECTLY targets that
  failure. State the causal link in one sentence:
  `"<failure> → <structural edit> because <why it addresses recall/separation/etc>"`.
- FORBIDDEN as a structural change: changing layer width/units, learning_rate, or
  dropout — those are Tier-1 (Optuna) and happen inside the version, not here.
- Exactly ONE structural edit per version, so the next report can attribute the
  holdout delta to that one change (clean lineage hill-climb).

### Structural change menu (reason over this — NOT a sampler)

| Structural edit | What it changes | When to reach for it |
|---|---|---|
| add / remove a layer | layer **count** → new version | underfitting (all classes weak) / overfitting (train ≫ holdout) |
| swap a layer **kind** (e.g. `dense`→`lstm`/`gru`/`conv1d`) | layer kind | no temporal structure being captured; flat per-class |
| widen `history_points` | input window length | model can't see far enough back to separate up/down |
| change / add a `target` (TargetSpec) | output head(s) | the head's labelling is the bottleneck (e.g. neutral band too wide) |
| add / drop an indicator group | input features | a class lacks the signal that distinguishes it |
| change `grouping` | row partitioning (e.g. by vol regime) | one regime dominates and drowns minority-class recall |

**NOT structural (Tier-1 / Optuna only — DO NOT pick these as `next_hypothesis`):**
layer **width** (`units`), `learning_rate`, `dropout`. If the only thing wrong is
"needs a wider layer / lower lr", that is already inside the version's own Optuna
search — Tier 2 changes the SHAPE, not the tunable values.

**Example diagnosis:** an "always-neutral" model (`per_class` shows high neutral,
near-zero up/down recall; confusion rows for up/down collapse into the neutral
column) is gaming the accuracy metric. The correct structural move targets recall
on up/down — e.g. narrow the direction target's neutral band (`label_m`/`label_x`
on the TargetSpec), add an indicator group that separates direction, or switch to
`by_indicator` grouping — and must NOT be "widen the dense layer."

## 6. Build + materialise the next spec

- Start from the INCUMBENT spec (the promoted version's spec, or v1's for a revert),
  apply the one structural edit with `dataclasses.replace` / list edits on `layers`,
  `targets`, `history_points`, `indicators`, or `grouping`.
- `path = materialize_spec(next_spec, nn_artefact_root(pair))`. The new spec_hash
  gives it its own `specs/{hash}/` dir; the prior version is untouched (preserved
  automatically). Hand `path` (the next NN_SPEC_PATH) and the next study label
  (`{archetype}_v{version_n+1}`) back to the orchestrator.

## Report.md path

Reports live in the EXTERNAL docs repo (never in the code repo):

```
external/docs/superpowers/specs/supporting-systems/nn-training-orchestration/{archetype}/v{version_n}/report.md
```

## Report.md prose body template (fill the `{...}` slots verbatim)

```markdown
## v{version_n} — {short title of this version's change}

**Hypothesis.** {what structural change this version tested, and the failure in the
prior version it was meant to fix.}

**Result.**

| metric | value | vs incumbent |
|---|---|---|
| holdout accuracy | {holdout_acc:.4f} | {delta vs best, e.g. +0.0261} |
| holdout loss | {holdout_loss:.4f} | {delta} |
| up recall | {per_class.up:.2f} |  |
| neutral recall | {per_class.neutral:.2f} |  |
| down recall | {per_class.down:.2f} |  |

Confusion (rows = actual, cols = predicted):

|        | pred up | pred neutral | pred down |
|--------|---------|--------------|-----------|
| up     | {c_uu}  | {c_un}       | {c_ud}    |
| neutral| {c_nu}  | {c_nn}       | {c_nd}    |
| down   | {c_du}  | {c_dn}       | {c_dd}    |

**What went good.** {what improved — e.g. the change lifted the targeted class's
recall; loss fell; the lineage promoted.}

**What went bad.** {what regressed or stayed broken — e.g. still gaming neutral;
up/down still confused; overfit gap widened. Quote decide(...).reason here.}

**What to improve.** {the diagnosed dominant failure mode that the NEXT structural
change must target — names the class/confusion cell, not "make it bigger".}

**Decision.** {promote | revert} — {decide(...).reason}. {if continuing:}
Next hypothesis: {next_hypothesis} (one structural change). {if stopped:}
Lineage stops: {stop cause}; winner = {incumbent label}.
```

## Invariants

- Any emitted spec MUST load via `NNModelSpec.from_yaml`.
- Any emitted report MUST parse via `parse_report`.
- `decide(...)` is ALWAYS called with the locked values above — never override them.
- Exactly ONE structural change per continuing version. Never sample architecture.
- Layer kinds for any new spec must be in `{dense, lstm, gru, conv1d}` (buildable
  today); any other kind requires the engine extension recipe first (ADR-0001).
- Reports go to the EXTERNAL docs repo; specs go to the artefact volume. Neither
  belongs in the code repo (`main/`).
