"""nn/nn_strategist.py — LLM-driven search steering for the NN training loop.

The NNStrategist reasons over experiment history and decides:
  - Which indicators / timeframes / targets to explore next
  - What bounds Optuna should sample within (search_space)
  - When and why to stop

It never trains or touches weights.  Optuna owns numeric sampling;
NNStrategist owns *scope* decisions.

Design principles
-----------------
- Pluggable LLM boundary: constructor accepts an optional ``client`` object
  with a ``call(prompt: str) -> str`` method.  When ``client is None`` a
  default AnthropicClient is built lazily (imports anthropic only on first
  use) so importing this module works even without the anthropic package.
- Model ID: read from env ``NN_STRATEGIST_MODEL``, default
  ``claude-sonnet-4-6``.  One constant, easily changed.
- Schema validation: proposals are JSON-validated against the Proposal
  dataclass.  Any out-of-vocab indicator, timeframe, or target triggers
  rejection → fallback to previous proposal + rejection log line.
- search_space clamped to search_config limits after every LLM call.
- review() verb constrained to allowed vocab; out-of-vocab → ``continue``.
- All LLM calls logged (prompt + full proposal + rationale) for
  reproducibility, distinct from the one-line investigation log.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from logs import log, log_error, log_warning
from nn.nn_model_spec import TargetSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-6"
REVIEW_VERBS = frozenset({"continue", "narrow", "broaden", "change_targets", "stop"})

# ---------------------------------------------------------------------------
# Proposal dataclass
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    """Structured output from NNStrategist.propose().

    indicators   — which indicator names to include in the next NNModelSpec
    timeframes   — which candle timeframes (minutes) to include
    targets      — list of TargetSpec output heads
    search_space — Optuna sampling bounds: lr, depth, units, dropout
    rationale    — why this scope (logged + stored with the round)
    """

    indicators: list[str]
    timeframes: list[int]
    targets: list[TargetSpec]
    search_space: dict
    rationale: str


# ---------------------------------------------------------------------------
# Default Anthropic client (lazy import)
# ---------------------------------------------------------------------------


class _AnthropicClient:
    """Thin wrapper around the Anthropic Messages API.

    anthropic is imported lazily so this module can be imported without the
    package installed; the error surfaces only when propose/review is called.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("NN_STRATEGIST_MODEL", DEFAULT_MODEL)

    def call(self, prompt: str) -> str:  # pragma: no cover — requires real API
        try:
            import anthropic  # lazy import
        except ImportError as exc:
            raise ImportError(
                "anthropic package is not installed.  Install it in the "
                "nn-train image or inject a fake client for tests."
            ) from exc

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# ---------------------------------------------------------------------------
# NNStrategist
# ---------------------------------------------------------------------------


class NNStrategist:
    """LLM agent that steers Optuna search scope across training rounds.

    Parameters
    ----------
    search_config:
        Hard bounds for Optuna: ``{"lr": (lo, hi), "depth": (lo, hi),
        "units": (lo, hi), "dropout": (lo, hi)}``.  LLM proposals cannot
        exceed these limits.
    client:
        Optional LLM client with a ``call(prompt: str) -> str`` method.
        If None, uses _AnthropicClient (lazy anthropic import).
    allowed_indicators:
        Vocabulary of valid indicator names.  Proposals with unknown
        indicators are rejected.  If None, any name passes.
    allowed_timeframes:
        Vocabulary of valid timeframe integers.  Proposals with unknown
        timeframes are rejected.  If None, any value passes.
    """

    def __init__(
        self,
        search_config: dict,
        client: Any = None,
        allowed_indicators: list[str] | None = None,
        allowed_timeframes: list[int] | None = None,
    ) -> None:
        self._search_config = search_config
        self._client = client if client is not None else _AnthropicClient()
        self._allowed_indicators: frozenset[str] | None = (
            frozenset(allowed_indicators) if allowed_indicators is not None else None
        )
        self._allowed_timeframes: frozenset[int] | None = (
            frozenset(allowed_timeframes) if allowed_timeframes is not None else None
        )
        self._previous_proposal: Proposal | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def propose(self, history: dict) -> Proposal:
        """Call the LLM with tracker history; return a validated, clamped Proposal.

        Falls back to previous proposal (+ rejection log) if the LLM response
        fails vocab validation.  Raises RuntimeError if no previous proposal
        exists and the first call is rejected.
        """
        round_n = history.get("n_rounds", history.get("rounds", [None]))
        if isinstance(round_n, list):
            round_n = round_n[-1] if round_n else 0
        round_n = round_n or 0

        prompt = self._build_propose_prompt(history)

        # Log full prompt for reproducibility
        log(f"[NNStrategist] LLM CALL propose round={round_n}: prompt_len={len(prompt)}")

        raw = self._client.call(prompt)

        # Log full response for reproducibility
        log(f"[NNStrategist] LLM RESPONSE round={round_n}: {raw[:500]}")

        # Parse + validate
        try:
            proposal = self._parse_and_validate_proposal(raw)
        except (ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
            reason = str(exc)
            log(f"[NNStrategist] round={round_n}: proposal rejected ({reason}); reusing previous scope")
            if self._previous_proposal is None:
                raise RuntimeError(
                    f"NNStrategist: proposal rejected ({reason}) and no previous "
                    "proposal to fall back to"
                ) from exc
            return self._previous_proposal

        # Vocab validation
        try:
            self._validate_vocab(proposal)
        except ValueError as exc:
            reason = str(exc)
            log(f"[NNStrategist] round={round_n}: proposal rejected ({reason}); reusing previous scope")
            if self._previous_proposal is None:
                raise RuntimeError(
                    f"NNStrategist: proposal rejected ({reason}) and no previous "
                    "proposal to fall back to"
                ) from exc
            return self._previous_proposal

        # Clamp search_space
        proposal = self._clamp_search_space(proposal)

        # Emit one-line investigation log
        self._emit_investigation_log(round_n, history, proposal)

        self._previous_proposal = proposal
        return proposal

    def review(self, round_summary: dict) -> str:
        """Call the LLM to decide what to do after a round.

        Returns one of: continue | narrow | broaden | change_targets | stop.
        Out-of-vocab response → returns 'continue'.
        """
        prompt = self._build_review_prompt(round_summary)

        log(f"[NNStrategist] LLM CALL review round={round_summary.get('round')}: prompt_len={len(prompt)}")

        raw = self._client.call(prompt)

        log(f"[NNStrategist] LLM RESPONSE review: {raw[:200]}")

        try:
            data = json.loads(raw)
            verb = str(data.get("verb", "continue")).strip()
        except (json.JSONDecodeError, AttributeError):
            verb = "continue"

        if verb not in REVIEW_VERBS:
            log_warning(
                f"[NNStrategist] review() returned unknown verb {verb!r}; falling back to 'continue'"
            )
            return "continue"

        return verb

    # ------------------------------------------------------------------
    # Private: prompt builders
    # ------------------------------------------------------------------

    def _build_propose_prompt(self, history: dict) -> str:
        allowed_ind = (
            sorted(self._allowed_indicators)
            if self._allowed_indicators is not None
            else "any"
        )
        allowed_tf = (
            sorted(self._allowed_timeframes)
            if self._allowed_timeframes is not None
            else "any"
        )
        return (
            "You are an expert ML experiment strategist steering a neural-network training loop.\n\n"
            f"## Experiment History\n{json.dumps(history, indent=2, default=str)}\n\n"
            f"## Constraints\n"
            f"- Allowed indicators: {allowed_ind}\n"
            f"- Allowed timeframes: {allowed_tf}\n"
            f"- search_space bounds (hard limits — do NOT exceed):\n"
            f"  lr:      [{self._search_config.get('lr', (1e-5, 1e-2))[0]}, "
            f"{self._search_config.get('lr', (1e-5, 1e-2))[1]}]\n"
            f"  depth:   [{self._search_config.get('depth', (1, 6))[0]}, "
            f"{self._search_config.get('depth', (1, 6))[1]}]\n"
            f"  units:   [{self._search_config.get('units', (16, 256))[0]}, "
            f"{self._search_config.get('units', (16, 256))[1]}]\n"
            f"  dropout: [{self._search_config.get('dropout', (0.0, 0.5))[0]}, "
            f"{self._search_config.get('dropout', (0.0, 0.5))[1]}]\n\n"
            "## Task\n"
            "Return a single JSON object (no markdown) with exactly these keys:\n"
            '  "indicators": list of indicator names (from allowed list)\n'
            '  "timeframes": list of timeframe integers (from allowed list)\n'
            '  "targets": list of TargetSpec dicts with keys: name, kind, horizons,\n'
            '             label_tf (optional), label_m (optional), label_x (optional),\n'
            '             strict (optional, bool), transform (optional, str)\n'
            '  "search_space": {"lr": [lo, hi], "depth": [lo, hi], "units": [lo, hi],\n'
            '                   "dropout": [lo, hi]}\n'
            '  "rationale": brief explanation of your choices\n\n'
            "Respond ONLY with valid JSON, no other text."
        )

    def _build_review_prompt(self, round_summary: dict) -> str:
        return (
            "You are an expert ML experiment strategist reviewing a completed training round.\n\n"
            f"## round_summary\n{json.dumps(round_summary, indent=2, default=str)}\n\n"
            "## Task\n"
            "Return a single JSON object with exactly:\n"
            '  "verb": one of continue | narrow | broaden | change_targets | stop\n'
            '  "rationale": brief explanation\n\n'
            "Respond ONLY with valid JSON, no other text."
        )

    # ------------------------------------------------------------------
    # Private: parsing & validation
    # ------------------------------------------------------------------

    def _parse_and_validate_proposal(self, raw: str) -> Proposal:
        """Parse LLM JSON response into a Proposal; raises on malformed input."""
        data = json.loads(raw)

        indicators = list(data["indicators"])
        timeframes = [int(tf) for tf in data["timeframes"]]

        targets_raw = data["targets"]
        targets = []
        for tr in targets_raw:
            tr = dict(tr)
            horizons = [int(h) for h in tr.pop("horizons", [1])]
            # Remove keys not in TargetSpec or with None values for optional fields
            ts = TargetSpec(
                name=tr["name"],
                kind=tr["kind"],
                horizons=horizons,
                label_tf=tr.get("label_tf"),
                label_m=tr.get("label_m"),
                label_x=tr.get("label_x"),
                strict=bool(tr.get("strict", False)),
                transform=tr.get("transform", "logret"),
            )
            targets.append(ts)

        raw_ss = data["search_space"]
        search_space = {
            k: (float(v[0]), float(v[1])) for k, v in raw_ss.items()
        }

        rationale = str(data["rationale"])

        return Proposal(
            indicators=indicators,
            timeframes=timeframes,
            targets=targets,
            search_space=search_space,
            rationale=rationale,
        )

    def _validate_vocab(self, proposal: Proposal) -> None:
        """Raise ValueError if any indicator/timeframe is outside the allowed vocab."""
        if self._allowed_indicators is not None:
            unknown_ind = [i for i in proposal.indicators if i not in self._allowed_indicators]
            if unknown_ind:
                raise ValueError(f"unknown indicators: {unknown_ind}")

        if self._allowed_timeframes is not None:
            unknown_tf = [tf for tf in proposal.timeframes if tf not in self._allowed_timeframes]
            if unknown_tf:
                raise ValueError(f"unknown timeframes: {unknown_tf}")

    def _clamp_search_space(self, proposal: Proposal) -> Proposal:
        """Return a new Proposal with search_space clamped to search_config limits."""
        clamped = {}
        for key in ("lr", "depth", "units", "dropout"):
            if key in self._search_config:
                cfg_lo, cfg_hi = self._search_config[key]
                if key in proposal.search_space:
                    raw_lo, raw_hi = proposal.search_space[key]
                    new_lo = max(cfg_lo, min(raw_lo, cfg_hi))
                    new_hi = max(cfg_lo, min(raw_hi, cfg_hi))
                    if new_lo > new_hi:
                        new_lo, new_hi = new_hi, new_lo
                    clamped[key] = (new_lo, new_hi)
                else:
                    clamped[key] = (cfg_lo, cfg_hi)
            else:
                if key in proposal.search_space:
                    clamped[key] = proposal.search_space[key]

        # Carry over any extra keys the LLM included
        for key, val in proposal.search_space.items():
            if key not in clamped:
                clamped[key] = val

        return Proposal(
            indicators=proposal.indicators,
            timeframes=proposal.timeframes,
            targets=proposal.targets,
            search_space=clamped,
            rationale=proposal.rationale,
        )

    # ------------------------------------------------------------------
    # Private: investigation log
    # ------------------------------------------------------------------

    def _emit_investigation_log(
        self, round_n: int, history: dict, proposal: Proposal
    ) -> None:
        """Emit the one-line investigation log via logs.log().

        Shape (verbatim from brief):
          [NNStrategist] round={r}: best={metric}={value:.4f} | signals: <driver/weakest> →
          propose {n} ind, tf={timeframes}, targets=[…]; space lr={lo}-{hi}, depth={lo}-{hi}
          | why: {rationale[:120]}
        """
        # Extract best metric from history
        incumbent = history.get("incumbent", {})
        if incumbent:
            metric_key, metric_val = next(iter(incumbent.items()))
            metric_str = f"{metric_key}={metric_val:.4f}" if isinstance(metric_val, (int, float)) else f"{metric_key}={metric_val}"
        else:
            metric_str = "score=N/A"

        # Top driver / weakest indicator from history
        per_ind = history.get("per_indicator", {})
        if per_ind:
            sorted_ind = sorted(per_ind.items(), key=lambda x: x[1], reverse=True)
            top = f"{sorted_ind[0][0]}(+{sorted_ind[0][1]:.3f})"
            weak = f"{sorted_ind[-1][0]}({sorted_ind[-1][1]:.3f})" if len(sorted_ind) > 1 else ""
            signals_str = f"{top} weakest:{weak}" if weak else top
        else:
            signals_str = "no signal deltas"

        # search_space summary (lr + depth)
        ss = proposal.search_space
        lr_bounds = ss.get("lr", ("?", "?"))
        depth_bounds = ss.get("depth", ("?", "?"))
        space_str = f"lr={lr_bounds[0]:.1e}-{lr_bounds[1]:.1e}, depth={int(depth_bounds[0])}-{int(depth_bounds[1])}"

        targets_names = [t.name for t in proposal.targets]
        rationale_short = proposal.rationale[:120]

        log(
            f"[NNStrategist] round={round_n}: best={metric_str} | "
            f"signals: {signals_str} → propose {len(proposal.indicators)} ind, "
            f"tf={proposal.timeframes}, targets={targets_names}; "
            f"space {space_str} | why: {rationale_short}"
        )
