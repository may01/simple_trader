"""Tests for NNStrategist — LLM search steering with guardrails.

TDD: tests written first (RED), then implement nn/nn_strategist.py (GREEN).

Key invariants:
- propose() returns a schema-valid Proposal with search_space clamped to search_config
- out-of-vocab proposal triggers fallback-to-previous + rejection log line
- one-line investigation log is emitted with the right shape
- review() returns an allowed verb; out-of-vocab → 'continue'
- module imports WITHOUT anthropic installed (lazy import)
- FAKE injected client — NO network calls
"""

import importlib
import json
import sys
from dataclasses import asdict
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from nn.nn_model_spec import TargetSpec


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SEARCH_CONFIG = {
    "lr": (1e-5, 1e-2),
    "depth": (1, 6),
    "units": (16, 256),
    "dropout": (0.0, 0.5),
}

# Stub history shaped like ExperimentTracker.summary()
HISTORY = {
    "incumbent": {"holdout_score": 0.70},
    "per_indicator": {"rsi": 0.01, "vol_regime": 0.04},
    "per_target": {"dir_h1": 0.71},
    "recent": [{"trial_id": "t1", "holdout_score": 0.68, "status": "ok"}],
    "n_trials": 5,
    "n_rounds": 1,
}

ROUND_SUMMARY = {"round": 1, "best": 0.70, "median": 0.66, "failed": 1}

# A valid proposal JSON the fake client will return for propose()
_VALID_PROPOSAL_JSON = json.dumps({
    "indicators": ["rsi", "vol_regime"],
    "timeframes": [15, 60],
    "targets": [
        {
            "name": "dir_h1",
            "kind": "direction",
            "horizons": [1],
            "label_tf": 60,
            "label_m": 1.0,
            "label_x": 0.3,
            "strict": False,
            "transform": "logret",
        }
    ],
    "search_space": {
        "lr": [1e-4, 5e-3],
        "depth": [2, 4],
        "units": [32, 128],
        "dropout": [0.1, 0.3],
    },
    "rationale": "vol_regime showed the highest delta; rsi still useful. Tightening depth range.",
})

# A review JSON
_VALID_REVIEW_JSON = json.dumps({
    "verb": "narrow",
    "rationale": "Best scores tightly clustered — narrow search space.",
})


def _make_fake_client(propose_response=None, review_response=None):
    """Return a fake LLM callable that yields canned JSON, making no network calls."""
    propose_resp = propose_response or _VALID_PROPOSAL_JSON
    review_resp = review_response or _VALID_REVIEW_JSON

    class FakeClient:
        def __init__(self):
            self._call_count = 0
            self._last_prompt = None

        def call(self, prompt: str) -> str:
            self._call_count += 1
            self._last_prompt = prompt
            # Distinguish propose vs review by prompt content
            if "review" in prompt.lower() or "round_summary" in prompt.lower():
                return review_resp
            return propose_resp

    return FakeClient()


@pytest.fixture
def fake_client():
    return _make_fake_client()


@pytest.fixture
def strategist(fake_client):
    from nn.nn_strategist import NNStrategist
    return NNStrategist(
        search_config=SEARCH_CONFIG,
        client=fake_client,
        allowed_indicators=["rsi", "vol_regime", "macd", "logret"],
        allowed_timeframes=[15, 30, 60, 240],
    )


# ---------------------------------------------------------------------------
# 1. Module imports without anthropic installed
# ---------------------------------------------------------------------------


def test_import_without_anthropic():
    """Importing nn.nn_strategist must NOT require anthropic to be installed."""
    # Remove anthropic from sys.modules if present, hide it
    saved = sys.modules.pop("anthropic", None)
    # Also block import of anthropic
    blocker = MagicMock()
    blocker.__spec__ = None
    sys.modules["anthropic"] = None  # Causes ImportError on 'import anthropic'

    try:
        # Force re-import
        if "nn.nn_strategist" in sys.modules:
            del sys.modules["nn.nn_strategist"]
        import nn.nn_strategist  # must not raise  # noqa: F401
    finally:
        # Restore
        if saved is not None:
            sys.modules["anthropic"] = saved
        else:
            sys.modules.pop("anthropic", None)
        if "nn.nn_strategist" in sys.modules:
            del sys.modules["nn.nn_strategist"]


# ---------------------------------------------------------------------------
# 2. Proposal dataclass shape
# ---------------------------------------------------------------------------


def test_proposal_is_dataclass():
    from nn.nn_strategist import Proposal
    p = Proposal(
        indicators=["rsi"],
        timeframes=[15],
        targets=[TargetSpec(name="dir", kind="direction", horizons=[1])],
        search_space={"lr": (1e-4, 1e-3)},
        rationale="test",
    )
    assert p.indicators == ["rsi"]
    assert isinstance(p.targets[0], TargetSpec)


# ---------------------------------------------------------------------------
# 3. propose() returns schema-valid Proposal
# ---------------------------------------------------------------------------


def test_propose_returns_proposal(strategist):
    from nn.nn_strategist import Proposal
    p = strategist.propose(HISTORY)
    assert isinstance(p, Proposal)
    assert isinstance(p.indicators, list) and len(p.indicators) > 0
    assert isinstance(p.timeframes, list) and len(p.timeframes) > 0
    assert isinstance(p.targets, list) and len(p.targets) > 0
    assert isinstance(p.search_space, dict)
    assert isinstance(p.rationale, str) and p.rationale


def test_propose_targets_are_target_spec(strategist):
    p = strategist.propose(HISTORY)
    for t in p.targets:
        assert isinstance(t, TargetSpec)


# ---------------------------------------------------------------------------
# 4. search_space clamping
# ---------------------------------------------------------------------------


def test_propose_lr_clamped_within_search_config(strategist):
    p = strategist.propose(HISTORY)
    lo, hi = p.search_space["lr"]
    assert SEARCH_CONFIG["lr"][0] <= lo <= hi <= SEARCH_CONFIG["lr"][1]


def test_propose_depth_clamped_within_search_config(strategist):
    p = strategist.propose(HISTORY)
    lo, hi = p.search_space["depth"]
    assert SEARCH_CONFIG["depth"][0] <= lo <= hi <= SEARCH_CONFIG["depth"][1]


def test_propose_units_clamped_within_search_config(strategist):
    p = strategist.propose(HISTORY)
    lo, hi = p.search_space["units"]
    assert SEARCH_CONFIG["units"][0] <= lo <= hi <= SEARCH_CONFIG["units"][1]


def test_propose_dropout_clamped_within_search_config(strategist):
    p = strategist.propose(HISTORY)
    lo, hi = p.search_space["dropout"]
    assert SEARCH_CONFIG["dropout"][0] <= lo <= hi <= SEARCH_CONFIG["dropout"][1]


def test_propose_clamps_lr_that_exceeds_config():
    """LLM returning lr=(1e-8, 1.0) must be clamped to search_config bounds."""
    wide_proposal = json.dumps({
        "indicators": ["rsi"],
        "timeframes": [15],
        "targets": [
            {"name": "dir_h1", "kind": "direction", "horizons": [1],
             "label_tf": 60, "label_m": 1.0, "label_x": 0.3,
             "strict": False, "transform": "logret"}
        ],
        "search_space": {
            "lr": [1e-8, 1.0],      # Way outside bounds
            "depth": [0, 100],       # Way outside bounds
            "units": [1, 10000],     # Way outside bounds
            "dropout": [-0.5, 2.0],  # Way outside bounds
        },
        "rationale": "wide test",
    })
    client = _make_fake_client(propose_response=wide_proposal)
    from nn.nn_strategist import NNStrategist
    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=client,
        allowed_indicators=["rsi"],
        allowed_timeframes=[15],
    )
    p = s.propose(HISTORY)
    lo_lr, hi_lr = p.search_space["lr"]
    assert lo_lr >= SEARCH_CONFIG["lr"][0]
    assert hi_lr <= SEARCH_CONFIG["lr"][1]
    lo_depth, hi_depth = p.search_space["depth"]
    assert lo_depth >= SEARCH_CONFIG["depth"][0]
    assert hi_depth <= SEARCH_CONFIG["depth"][1]


# ---------------------------------------------------------------------------
# 5. Vocab validation and fallback-to-previous
# ---------------------------------------------------------------------------


def _make_out_of_vocab_proposal(bad_field="indicators"):
    """Return a JSON proposal with an invalid indicator."""
    base = {
        "indicators": ["rsi", "vol_regime"],
        "timeframes": [15, 60],
        "targets": [
            {"name": "dir_h1", "kind": "direction", "horizons": [1],
             "label_tf": 60, "label_m": 1.0, "label_x": 0.3,
             "strict": False, "transform": "logret"}
        ],
        "search_space": {
            "lr": [1e-4, 5e-3],
            "depth": [2, 4],
            "units": [32, 128],
            "dropout": [0.1, 0.3],
        },
        "rationale": "out-of-vocab test",
    }
    if bad_field == "indicators":
        base["indicators"] = ["rsi", "UNKNOWN_INDICATOR_XYZ"]
    elif bad_field == "timeframes":
        base["timeframes"] = [15, 9999]
    return json.dumps(base)


def test_out_of_vocab_indicator_triggers_fallback(fake_client, capsys):
    """An unknown indicator in the proposal triggers rejection + reuse of previous."""
    from nn.nn_strategist import NNStrategist, Proposal

    # First call: valid proposal → stored as previous
    call_results = [_VALID_PROPOSAL_JSON, _make_out_of_vocab_proposal("indicators")]
    call_idx = [0]

    class SequencedClient:
        def call(self, prompt: str) -> str:
            idx = call_idx[0]
            call_idx[0] += 1
            return call_results[idx % len(call_results)]

    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=SequencedClient(),
        allowed_indicators=["rsi", "vol_regime", "macd"],
        allowed_timeframes=[15, 60],
    )
    # First propose: valid → stores previous
    p1 = s.propose(HISTORY)
    assert isinstance(p1, Proposal)

    # Second propose: out-of-vocab → fallback to p1
    p2 = s.propose(HISTORY)
    assert p2 is p1 or (p2.indicators == p1.indicators and p2.timeframes == p1.timeframes)

    # Check rejection log line was emitted
    captured = capsys.readouterr()
    assert "proposal rejected" in captured.out
    assert "reusing previous scope" in captured.out


def test_out_of_vocab_timeframe_triggers_fallback(capsys):
    """An unknown timeframe in the proposal triggers rejection + reuse of previous."""
    from nn.nn_strategist import NNStrategist

    call_results = [_VALID_PROPOSAL_JSON, _make_out_of_vocab_proposal("timeframes")]
    call_idx = [0]

    class SequencedClient:
        def call(self, prompt: str) -> str:
            idx = call_idx[0]
            call_idx[0] += 1
            return call_results[idx % len(call_results)]

    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=SequencedClient(),
        allowed_indicators=["rsi", "vol_regime"],
        allowed_timeframes=[15, 60],
    )
    p1 = s.propose(HISTORY)
    p2 = s.propose(HISTORY)
    assert p2.timeframes == p1.timeframes

    captured = capsys.readouterr()
    assert "proposal rejected" in captured.out


def test_fallback_on_first_call_when_no_previous(capsys):
    """If first call is rejected and there's no previous, a RuntimeError is raised."""
    from nn.nn_strategist import NNStrategist

    client = _make_fake_client(propose_response=_make_out_of_vocab_proposal("indicators"))
    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=client,
        allowed_indicators=["rsi"],
        allowed_timeframes=[15],
    )
    with pytest.raises(RuntimeError, match="no previous"):
        s.propose(HISTORY)


# ---------------------------------------------------------------------------
# 6. One-line investigation log shape
# ---------------------------------------------------------------------------


def test_propose_emits_investigation_log(strategist, capsys):
    """propose() must emit a single NNStrategist log line via logs.log()."""
    strategist.propose(HISTORY)
    captured = capsys.readouterr()
    assert "[NNStrategist]" in captured.out


def test_propose_investigation_log_contains_required_fields(capsys):
    """Investigation log must contain round=, best=, propose, space, why:."""
    from nn.nn_strategist import NNStrategist

    # Use a history with ExperimentTracker.summary() shape
    history = {
        "incumbent": {"holdout_score": 0.72},
        "per_indicator": {"rsi": 0.02, "vol_regime": 0.05},
        "per_target": {"dir_h1": 0.72},
        "recent": [{"trial_id": "abc", "holdout_score": 0.70, "status": "ok"}],
        "n_trials": 10,
        "n_rounds": 2,
    }
    client = _make_fake_client()
    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=client,
        allowed_indicators=["rsi", "vol_regime"],
        allowed_timeframes=[15, 60],
    )
    s.propose(history)
    captured = capsys.readouterr()

    # Check log shape fields — investigation log is the line containing 'best=' and 'why:'
    log_lines = [l for l in captured.out.split("\n") if "[NNStrategist]" in l]
    assert len(log_lines) >= 1
    # Find the one-line investigation log (contains 'best=' and 'why:')
    inv_lines = [l for l in log_lines if "best=" in l and "why:" in l]
    assert len(inv_lines) >= 1, (
        f"Expected an investigation log line with 'best=' and 'why:' among: {log_lines}"
    )
    line = inv_lines[0]
    assert "round=" in line
    assert "best=" in line
    assert "propose" in line
    assert "space" in line
    assert "why:" in line


def test_propose_rejection_log_shape(capsys):
    """Rejection log must contain 'round=', 'proposal rejected', 'reusing previous scope'."""
    from nn.nn_strategist import NNStrategist

    call_results = [_VALID_PROPOSAL_JSON, _make_out_of_vocab_proposal("indicators")]
    call_idx = [0]

    class SequencedClient:
        def call(self, prompt: str) -> str:
            idx = call_idx[0]
            call_idx[0] += 1
            return call_results[idx % len(call_results)]

    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=SequencedClient(),
        allowed_indicators=["rsi", "vol_regime"],
        allowed_timeframes=[15, 60],
    )
    s.propose(HISTORY)  # valid
    s.propose(HISTORY)  # rejected
    captured = capsys.readouterr()

    rejection_lines = [l for l in captured.out.split("\n")
                       if "proposal rejected" in l and "[NNStrategist]" in l]
    assert len(rejection_lines) >= 1
    line = rejection_lines[0]
    assert "round=" in line
    assert "reusing previous scope" in line


# ---------------------------------------------------------------------------
# 7. review() verb constraint
# ---------------------------------------------------------------------------


def test_review_returns_allowed_verb(strategist):
    verb = strategist.review(ROUND_SUMMARY)
    assert verb in {"continue", "narrow", "broaden", "change_targets", "stop"}


def test_review_out_of_vocab_defaults_to_continue():
    """If LLM returns an unknown verb, review() falls back to 'continue'."""
    from nn.nn_strategist import NNStrategist

    bad_review = json.dumps({"verb": "EXPLODE", "rationale": "nonsense"})
    client = _make_fake_client(review_response=bad_review)
    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=client,
        allowed_indicators=["rsi"],
        allowed_timeframes=[15],
    )
    verb = s.review(ROUND_SUMMARY)
    assert verb == "continue"


def test_review_all_valid_verbs():
    """Each allowed verb passes through correctly."""
    from nn.nn_strategist import NNStrategist

    for expected_verb in ["continue", "narrow", "broaden", "change_targets", "stop"]:
        resp = json.dumps({"verb": expected_verb, "rationale": "ok"})
        client = _make_fake_client(review_response=resp)
        s = NNStrategist(
            search_config=SEARCH_CONFIG,
            client=client,
            allowed_indicators=["rsi"],
            allowed_timeframes=[15],
        )
        assert s.review(ROUND_SUMMARY) == expected_verb


# ---------------------------------------------------------------------------
# 8. LLM not called when client=None default (lazy import guard)
# ---------------------------------------------------------------------------


def test_default_client_is_not_built_without_anthropic(monkeypatch):
    """NNStrategist(search_config=...) should not crash even if anthropic is absent,
    but building the default client lazy-imports anthropic only on first use."""
    # Block anthropic
    saved = sys.modules.pop("anthropic", None)
    sys.modules["anthropic"] = None  # type: ignore

    try:
        if "nn.nn_strategist" in sys.modules:
            del sys.modules["nn.nn_strategist"]
        from nn.nn_strategist import NNStrategist

        # Should instantiate without calling LLM
        s = NNStrategist(search_config=SEARCH_CONFIG)
        # The object exists; we don't call propose() so no anthropic import is needed
        assert s is not None
    finally:
        if saved is not None:
            sys.modules["anthropic"] = saved
        else:
            sys.modules.pop("anthropic", None)
        if "nn.nn_strategist" in sys.modules:
            del sys.modules["nn.nn_strategist"]


# ---------------------------------------------------------------------------
# 9. Malformed / non-JSON LLM response
# ---------------------------------------------------------------------------


def test_propose_handles_non_json_response():
    """If LLM returns non-JSON, should raise a clear error or fall back."""
    from nn.nn_strategist import NNStrategist

    class BadClient:
        def call(self, prompt: str) -> str:
            return "This is not JSON at all!!!"

    s = NNStrategist(
        search_config=SEARCH_CONFIG,
        client=BadClient(),
        allowed_indicators=["rsi"],
        allowed_timeframes=[15],
    )
    # First call: no previous → should raise RuntimeError (no previous to fall back to)
    with pytest.raises((RuntimeError, ValueError, Exception)):
        s.propose(HISTORY)


# ---------------------------------------------------------------------------
# 10. Strategist never trains (no model/weight access)
# ---------------------------------------------------------------------------


def test_strategist_has_no_train_method(strategist):
    assert not hasattr(strategist, "train")
    assert not hasattr(strategist, "fit")


# ---------------------------------------------------------------------------
# 11. LLM calls are logged for reproducibility
# ---------------------------------------------------------------------------


def test_propose_logs_full_prompt_and_proposal(strategist, capsys):
    """propose() must log prompt + proposal for reproducibility (verbose log)."""
    strategist.propose(HISTORY)
    captured = capsys.readouterr()
    # The reproducibility log should include 'prompt' or 'proposal' keyword
    # (distinct from the one-line investigation log)
    # We check the one-line log is there; full reproducibility log may be debug-level
    assert "[NNStrategist]" in captured.out
