---
name: nn-investigate
description: Use when starting a new NN archetype investigation for a trading pair, or when asked to propose candidate NN designs or emit a v1 model spec for stock-price prediction — Tier 0 of the NN training orchestration loop.
---

# nn-investigate (Tier 0a + 0b)

## Overview

Tier-0 reasoning for the NN training orchestration loop. Two stages:
**0a** proposes the archetype list; **0b** turns one archetype into a written
`description.md` and a buildable v1 `NNModelSpec` materialized on the artefact
volume. You reason which indicators / targets / layers to use and WHY each layer
is a real-data dependency — you do NOT search numerics (Optuna does that) and you
do NOT launch training (the orchestrator does that).

**Inputs:** `pair` (e.g. `link_usdt`); optional `archetype` (present ⇒ run 0b for
it; absent ⇒ run 0a).

## Allowed vocabulary — read these FIRST, every run

You may use ONLY these sets. This is the same vocabulary the in-loop NNStrategist
uses. Read them fresh each run; do not invent names from memory.

| Set | Source | Use |
|---|---|---|
| Indicators (features) | `main/configs/indicators_config.yaml` field names (140+, e.g. `rsi_14`, `logret`, `macd_12_26_9_slope`) | feature columns → `indicators:` in the spec |
| Timeframes | `main/configs/candles_config.yaml` `candles: [1,5,15,60,240,1440]` | candle resolutions → `timeframes:` in the spec |
| Targets | `indicators_config.yaml` `labels:` entries + TargetSpec kinds `direction`/`label`/`regression` | the prediction target |
| Buildable layer kinds | `dense`, `lstm`, `gru`, `conv1d` (+ mixed stacks) | layer `kind` |

A name not in its set is not allowed. A layer kind not in the buildable set is
**not-buildable-now**: do not declare it until the engine extension recipe in
`phase-17.../DECISIONS-LOG.md` has added it via TDD — flag the archetype and link
the recipe instead.

**Buildable kinds today:** `dense`, `lstm`, `gru`, `conv1d`.
**New kinds (e.g. `attention`) require the engine extension recipe** (added on demand
via TDD) BEFORE an archetype may declare them. Flag such archetypes as
not-buildable-now and link the recipe.

## Tier 0a — meta-investigation (when no `archetype` given)

1. Read the four vocabulary sources above.
2. Brainstorm 4–8 distinct archetypes — each a single real-data hypothesis with a
   distinct architecture shape. Examples of the *shape* (not a fixed list):
   - snapshot dense (cross-feature interaction at one instant),
   - conv1d→dense (local candle pattern over a short window),
   - lstm/gru sequence (temporal dependency across bar history),
   - cnn_lstm (local pattern then temporal),
   - multi-timeframe (parallel branches over several `candles`).
3. For each, decide its declared layer kinds and set **buildable-now = yes** iff
   every kind is in `{dense,lstm,gru,conv1d}`; otherwise **no** + link the recipe.
4. Rank by expected signal-to-effort.
5. Write `external/docs/superpowers/specs/supporting-systems/nn-training-orchestration/meta-investigation.md`:
   a short intro + a table with columns `archetype | thesis (1 line) | layer kinds | buildable-now`.
   STOP. (Choosing which archetype to pursue is the orchestrator's call.)

## Tier 0b — per-archetype deep investigation (when `archetype` given)

1. Re-read the vocabulary sources. Confirm the archetype is **buildable-now**;
   if not, STOP and report the recipe link — do not emit a spec.
2. Reason the design and write
   `external/docs/superpowers/specs/supporting-systems/nn-training-orchestration/{archetype}/description.md`
   using the template below VERBATIM (Design thesis; Indicators rationale; Layers
   real-data-dependency rationale; Target rationale; Initial hyperparameters +
   why; VRAM note). Every indicator name must exist in `indicators_config.yaml`;
   every layer must map to a stated real-data dependency; every layer kind must be
   buildable-now.
3. Build the v1 `NNModelSpec` from that reasoning (in a short python snippet run
   under `main/.venv`): construct `LayerSpec(...)` per the description's layer
   stack, `TargetSpec(...)` per the target section, and set the **locked starting
   values** — `batch_size=128` (or `64` if the stack contains `lstm`/`gru`),
   `epochs=50`, `early_stopping_patience=8`, `learning_rate=1e-3`, `dropout=0.2`.
   Optuna will tune only `units`/`learning_rate`/`dropout` later (ADR-0001) —
   kinds and count are fixed here.
4. Materialize to the volume, NOT the repo:
   ```python
   from nn.device import nn_artefact_root
   from nn.orchestration.spec_store import materialize_spec
   path = materialize_spec(spec, nn_artefact_root(pair))  # → …/specs/{spec_hash}/spec.yaml
   ```
   Record the returned path (and `spec.spec_hash`) in the description's footer.
5. Self-check before handing off: re-load it — `NNModelSpec.from_yaml(str(path))`
   must succeed; `layers` non-empty with kinds ⊆ buildable set; `targets`
   non-empty with kinds ⊆ `{direction,label,regression}`; starting values set.

## The archetype `description.md` template (use verbatim)

Tier 0b MUST produce a `description.md` with exactly these sections, in this order:

```markdown
# Archetype: {archetype}

## Design thesis
One paragraph: the single real-data hypothesis this architecture tests
(e.g. "local candlestick shapes over the last N bars predict the next-bar
direction better than a flat snapshot"). What would make it win; what would
make it lose.

## Indicators (features) — rationale
For EACH selected indicator (each name MUST exist in indicators_config.yaml):
- `{indicator_name}` — what real-data signal it carries and why this archetype
  needs it (not "it's commonly used" — the concrete dependency).
Timeframes used (subset of [1,5,15,60,240,1440]) and why these resolutions.

## Layers — real-data dependency rationale
For EACH layer (in stack order), state the real-data dependency it captures:
- `conv1d` → local candle/pattern structure over a short window (shape, not value).
- `lstm` / `gru` → temporal dependency across the bar history (order matters).
- `dense` → snapshot feature interaction at a single instant (cross-feature, no time).
Every layer must map to a stated dependency; a layer with no dependency rationale
is removed. If any kind is NOT in {dense,lstm,gru,conv1d}, this archetype is
not-buildable-now — STOP and link the engine extension recipe in DECISIONS-LOG.md.

## Target — rationale
The TargetSpec kind ({direction|label|regression}) and, for a label target, which
labels_config entry; why this target matches the thesis (horizon, class balance).

## Initial hyperparameters — and why
batch_size, epochs, patience, learning_rate, dropout, units per layer — each with
one clause of justification. Use the locked starting values (batch_size 128, or 64
for recurrent archetypes; epochs 50; patience 8; lr 1e-3; dropout 0.2). State the
Optuna-tuned dims (units/lr/dropout) vs the fixed ones (kinds/count — ADR-0001).

## VRAM note (4 GB GPU)
Estimated footprint at the chosen width/sequence length; why it fits the 4 GB
ceiling; the fallback (OOM→CPU retry is the safety net). Recurrent/attention
widths kept small.
```

## Red flags — STOP

- An indicator name you "remember" but did not find in `indicators_config.yaml`.
- A layer kind outside `{dense,lstm,gru,conv1d}` without the recipe applied first.
- Writing `spec.yaml` anywhere under `main/` (it goes on the volume, via
  `materialize_spec`).
- Picking `units`/`lr`/`dropout` as "the answer" — those are Optuna's to tune; you
  fix only kinds, count, target, and the locked starting values.
- A layer with no real-data-dependency sentence — remove it.
- Using `features:` as a YAML key — the spec field is `indicators:`.
- Using `patience:` as a YAML key — the spec field is `early_stopping_patience:`.
- Using `params: {horizon: N}` in a target — the spec field is `horizons: [N]`.
