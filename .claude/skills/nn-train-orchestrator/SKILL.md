---
name: nn-train-orchestrator
description: Use when running the Phase 17 NN training orchestration — walk the candidate archetypes one at a time, drive each archetype's autonomous version lineage to its stop, re-validate the winner on 4y, then STOP for a human gate before the next archetype. Re-invoke under ralph-loop or /loop for the multi-hour trainings.
---

# NN Train Orchestrator (Tier 2 conductor)

## Overview

You conduct the per-archetype training lineages for one pair (iterate on **2y
link_usdt**, confirm on **4y**). You do NOT design architectures (that is
`nn-investigate`, Tier 0) or tune numerics (that is the in-container Optuna
search, Tier 1). Your job is the **walk**: for each candidate archetype, run its
lineage to a decided stop, re-validate the winner on 4y, then **hand back to a
human** before the next archetype.

**You never compute path math, the Docker launch, or promote/revert arithmetic
by hand.** Those are deterministic and live in the CLI — call it:

- `python -m nn.orchestration.cli materialize-spec --spec-yaml <in> --pair <pair>`
  → prints the on-volume spec path.
- `python -m nn.orchestration.cli run-version --spec-path <vol> --study <name> --pair <pair> --timeout 28800`
  → prints JSON `{study, holdout_score}`.
- `python -m nn.orchestration.cli decide --best <f|none> --candidate <f> --margin 0.01 --strike <i> --k 2 --version-n <i> --max-versions 6 --elapsed <f> --budget 28800`
  → prints JSON of the `LineageDecision` (`action`, `new_best`, `strike_count`,
  `stop`, `reason`).

## Locked values (do not change)

`trials_per_round 8` · `max_rounds 1` · `K=2` · `max_versions 6` ·
`margin 0.01` · `budget 28800s` (8 h per archetype) · iterate **2y link_usdt** ·
confirm **4y**.

## The walk (archetypes)

Read the archetype list from the Tier-0a meta plan (`nn-investigate` output).
Process **one archetype per run**:

1. Run that archetype's lineage loop (below) to `decide.stop`.
2. Re-validate the promoted best on **4y** (the final lineage step — see below).
3. **STOP.** Emit a short summary and end the run for a **human gate**: the
   operator reviews the lineage + 4y result, then resumes you for the NEXT
   archetype. The gate is BETWEEN archetypes — never pause inside a lineage.

## The per-archetype lineage loop (autonomous)

State lives in files, not in your memory — on every (re-)invocation, read the
lineage's own artifacts to resume.

1. **Investigate (Tier 0b)** — get/confirm this archetype's v1 spec via
   `nn-investigate`.
2. **Materialize v1** — `cli materialize-spec` → on-volume spec path.
3. **Run version** — `cli run-version --timeout 28800` → `{study, holdout_score}`.
4. **Evolve (Tier 2)** — invoke `nn-evolve`: it writes the version `report.md`
   (under the external `{archetype}/v{n}/`, parseable by `parse_report`) and then
   calls `cli decide` (current best vs this candidate, margin 0.01, K 2,
   max_versions 6, elapsed vs budget 28800). Thread the returned `strike_count`
   into the next `decide`.
5. **Branch on `decide.stop`:**
   - `stop == false` → take the next spec `nn-evolve` proposed, go to step 2 with
     `version_n += 1`.
   - `stop == true` → the 2y lineage is done. Go to the 4y confirm step.

## 4y confirm step (final step before the gate)

Once the 2y lineage stops, take the **promoted best** spec and run it ONCE on
**4y** (`--pair` for the 4y dataset, fresh `--study` suffix `_4y`). Record the 4y
holdout in the winner's report. This re-validation that the 2y winner generalises
to 4y is the LAST action of the archetype — then STOP for the human gate.

## Long trainings — ralph-loop / /loop

A single `run-version` is a multi-hour `docker compose run`. Do NOT block one
turn for hours. This skill is built to be **re-invoked between version steps**:

- **ralph-loop** (`/ralph-loop "<this skill's task>" --completion-promise
  "ARCHETYPE_DONE"`): the Stop hook feeds the same prompt back; each iteration
  reads the lineage's files (last `report.md`, `tracking/{study}/best.json`) and
  advances one step. Emit the completion promise only after the 4y confirm step,
  so the loop stops at the human gate.
- **/loop** (interval re-invoke) works the same way for unattended polling.

The contract that makes this safe: **the prompt never changes between iterations
and all state is in files** — you reconstruct "where am I" from the artifacts, not
from conversation history.

## Stop conditions you must honour

Stop the lineage exactly when `cli decide` says `stop == true` — that is the OR of
`strike_count >= 2`, `version_n >= 6`, or `elapsed_s >= 28800`. Never override it.
After stop + 4y confirm: STOP the run for the human gate.

## Done means

- One archetype's lineage ran to a decided stop (`decide.stop`).
- The promoted best was re-validated on 4y and recorded.
- A v{n} `report.md` exists per version under `{archetype}/v{n}/`, each parseable
  by `parse_report`.
- You ended the run for a human gate (did not auto-start the next archetype).
