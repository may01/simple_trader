"""tests/nn/test_target_spec_strict.py — TDD tests for label_l/label_y on TargetSpec.

Tests:
1. Strict TargetSpec resolves to the correct profit-label column names.
2. Strict TargetSpec without label_l/label_y raises ValueError.
3. Round-trip NNModelSpec.to_yaml -> from_yaml preserves label_l/label_y and spec_hash.
"""
import os
import tempfile

import pytest

from nn.nn_dataset import _profit_long_col, _profit_short_col
from nn.nn_model_spec import NNModelSpec, TargetSpec


# ---------------------------------------------------------------------------
# Test 1: strict spec resolves to correct column names
# ---------------------------------------------------------------------------


def test_strict_target_resolves_correct_long_col():
    """A strict TargetSpec with label_l=15, label_y=0.2 resolves to the
    expected strict profit-long column name."""
    spec = TargetSpec(
        name="dir15",
        kind="direction",
        label_tf=15,
        label_m=1.0,
        label_x=0.3,
        strict=True,
        label_l=15,
        label_y=0.2,
        horizons=[1],
    )
    result = _profit_long_col(spec, 1)
    assert result == "15_pslong_n1_m1_x0p3_l15_y0p2"


def test_strict_target_resolves_correct_short_col():
    """A strict TargetSpec with label_l=15, label_y=0.2 resolves to the
    expected strict profit-short column name."""
    spec = TargetSpec(
        name="dir15",
        kind="direction",
        label_tf=15,
        label_m=1.0,
        label_x=0.3,
        strict=True,
        label_l=15,
        label_y=0.2,
        horizons=[1],
    )
    result = _profit_short_col(spec, 1)
    assert result == "15_psshort_n1_m1_x0p3_l15_y0p2"


# ---------------------------------------------------------------------------
# Test 2: strict spec WITHOUT label_l/label_y raises ValueError
# ---------------------------------------------------------------------------


def test_strict_target_missing_label_l_y_raises():
    """A strict TargetSpec with label_l=None/label_y=None raises ValueError
    mentioning 'label_l/label_y' when resolving the column name."""
    spec = TargetSpec(
        name="dir15",
        kind="direction",
        label_tf=15,
        label_m=1.0,
        label_x=0.3,
        strict=True,
        # label_l and label_y omitted → None
        horizons=[1],
    )
    with pytest.raises(ValueError, match="label_l/label_y"):
        _profit_long_col(spec, 1)


# ---------------------------------------------------------------------------
# Test 3: round-trip through to_yaml / from_yaml preserves spec_hash and fields
# ---------------------------------------------------------------------------


def test_strict_target_yaml_roundtrip_preserves_hash_and_fields():
    """NNModelSpec containing a strict TargetSpec survives to_yaml/from_yaml
    with identical spec_hash and label_l/label_y preserved."""
    original = NNModelSpec(
        name="strict_test",
        timeframes=[15],
        indicators=["rsi_14"],
        history_points=16,
        targets=[
            TargetSpec(
                name="dir15strict",
                kind="direction",
                label_tf=15,
                label_m=1.0,
                label_x=0.3,
                strict=True,
                label_l=15,
                label_y=0.2,
                horizons=[1],
            )
        ],
    )

    original_hash = original.spec_hash

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        tmp_path = f.name

    try:
        original.to_yaml(tmp_path)
        restored = NNModelSpec.from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert restored.spec_hash == original_hash, (
        f"spec_hash changed after round-trip: {original_hash!r} → {restored.spec_hash!r}"
    )

    t = restored.targets[0]
    assert t.label_l == 15, f"label_l not preserved: got {t.label_l!r}"
    assert t.label_y == 0.2, f"label_y not preserved: got {t.label_y!r}"
    assert t.strict is True
