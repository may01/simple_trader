"""Tests for chunked holdout inference (fix for GPU OOM on large holdouts).

We refactor chunked inference into a pure helper `_chunked_predict(model,
X_flat, chunk)` and test THAT directly — no TrainingLoop wiring required.

Rationale for helper approach: `evaluate_on_holdout` requires a full
NNDataset, a real spec, and a live orchestrator.  Spinning all of that up in a
unit test is heavyweight and couples these tests to Dataset + orchestrator
internals.  A tiny pure function `_chunked_predict(model, X_flat, chunk_size)`
is side-effect-free, deterministic, and perfectly testable in isolation.
`evaluate_on_holdout` delegates to it in one line, so any breakage in the
delegation path would surface in an integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from nn.training_loop import HOLDOUT_EVAL_CHUNK, _chunked_predict


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class _RecordingModel:
    """Fake model that records call sizes and returns a fixed function of input."""

    def __init__(self, out_width: int = 3):
        self.call_sizes: list[int] = []
        self.out_width = out_width

    def run_batch(self, X_flat: np.ndarray) -> np.ndarray:
        n = X_flat.shape[0]
        self.call_sizes.append(n)
        # Deterministic: each output row = mean of input row, broadcast to out_width.
        row_means = X_flat.mean(axis=1, keepdims=True)
        return np.broadcast_to(row_means, (n, self.out_width)).copy()


# ---------------------------------------------------------------------------
# Test 1 — chunking behaviour: multiple calls, each <= HOLDOUT_EVAL_CHUNK
# ---------------------------------------------------------------------------

def test_chunked_predict_calls_run_batch_in_chunks():
    """_chunked_predict never calls run_batch with more than HOLDOUT_EVAL_CHUNK rows."""
    n_rows = HOLDOUT_EVAL_CHUNK * 3 + 17  # clearly larger than one chunk
    n_features = 10
    out_width = 3
    X_flat = np.random.default_rng(42).random((n_rows, n_features), dtype=np.float32)

    model = _RecordingModel(out_width=out_width)
    preds = _chunked_predict(model, X_flat, HOLDOUT_EVAL_CHUNK)

    # Must have been called more than once (chunked).
    assert len(model.call_sizes) > 1, (
        "run_batch should be called multiple times for n_rows > HOLDOUT_EVAL_CHUNK"
    )
    # No single call may exceed the chunk size.
    assert all(s <= HOLDOUT_EVAL_CHUNK for s in model.call_sizes), (
        f"Some chunk exceeded HOLDOUT_EVAL_CHUNK={HOLDOUT_EVAL_CHUNK}: {model.call_sizes}"
    )
    # Output covers all rows.
    assert preds.shape == (n_rows, out_width), (
        f"Expected ({n_rows}, {out_width}), got {preds.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 — equivalence: chunked == single-pass for deterministic model
# ---------------------------------------------------------------------------

def test_chunked_predict_equals_single_pass():
    """Chunked predictions are identical row-for-row to a single full-batch call."""
    n_rows = HOLDOUT_EVAL_CHUNK + 500
    n_features = 8
    out_width = 4
    rng = np.random.default_rng(7)
    X_flat = rng.random((n_rows, n_features)).astype(np.float32)

    # Chunked path.
    model_chunked = _RecordingModel(out_width=out_width)
    preds_chunked = _chunked_predict(model_chunked, X_flat, HOLDOUT_EVAL_CHUNK)

    # Single-pass reference (directly call run_batch on the whole array).
    model_single = _RecordingModel(out_width=out_width)
    preds_single = model_single.run_batch(X_flat)

    np.testing.assert_array_equal(
        preds_chunked,
        preds_single,
        err_msg="Chunked predictions must be identical to single-pass predictions",
    )


# ---------------------------------------------------------------------------
# Test 3 — edge case: exactly one chunk (no splitting needed)
# ---------------------------------------------------------------------------

def test_chunked_predict_single_chunk_one_call():
    """When n_rows <= chunk_size, run_batch is called exactly once."""
    n_rows = HOLDOUT_EVAL_CHUNK
    n_features = 5
    out_width = 2
    X_flat = np.ones((n_rows, n_features), dtype=np.float32)

    model = _RecordingModel(out_width=out_width)
    preds = _chunked_predict(model, X_flat, HOLDOUT_EVAL_CHUNK)

    assert model.call_sizes == [n_rows], (
        f"Expected exactly one call of size {n_rows}, got {model.call_sizes}"
    )
    assert preds.shape == (n_rows, out_width)


# ---------------------------------------------------------------------------
# Test 4 — edge case: n_rows smaller than chunk_size
# ---------------------------------------------------------------------------

def test_chunked_predict_smaller_than_chunk_one_call():
    """When n_rows < chunk_size, run_batch is still called once."""
    n_rows = 100
    chunk_size = HOLDOUT_EVAL_CHUNK  # n_rows << chunk_size
    n_features = 6
    out_width = 3
    X_flat = np.zeros((n_rows, n_features), dtype=np.float32)

    model = _RecordingModel(out_width=out_width)
    preds = _chunked_predict(model, X_flat, chunk_size)

    assert model.call_sizes == [100]
    assert preds.shape == (100, out_width)
