"""Tests for the conv1d_seq layer kind — sequence-preserving 1-D conv.

Unlike conv1d (which mean-pools the time axis to (B, C), collapsing to 2-D),
conv1d_seq keeps the time axis: (B, T, F) -> (B, T, C). This lets a
conv1d_seq -> lstm stack work; otherwise the following lstm errors with
"too many indices for tensor of dimension 2" because conv1d already
collapsed the sequence dim.

Runs in the nn-train image (real torch):
    docker compose run --rm --no-deps -e PYTHONDONTWRITEBYTECODE=1 nn-train \
        python -m pytest tests/nn/test_conv1d_seq.py -q -p no:cacheprovider
"""

import dataclasses

import torch

from nn.nn_model import _SpecNet
from nn.nn_model_spec import LayerSpec, NNModelSpec, TargetSpec


def _base_spec(layers, history_points=8, n_features=4):
    return dataclasses.replace(
        NNModelSpec.default(),
        layers=layers,
        history_points=history_points,
        targets=[
            TargetSpec(
                name="dir15",
                kind="label",
                horizons=[1],
                label_tf=15,
                label_m=1.0,
                label_x=0.3,
            )
        ],
    )


def test_conv1d_seq_then_lstm_builds_and_forwards():
    """conv1d_seq -> lstm: time axis must survive the conv so lstm sees a sequence."""
    T, F, B = 8, 4, 4
    spec = _base_spec(
        layers=[
            LayerSpec(kind="conv1d_seq", units=16, params={"kernel_size": 3}),
            LayerSpec(kind="lstm", units=32, params={}),
        ],
        history_points=T,
        n_features=F,
    )
    net = _SpecNet(spec, history_points=T, n_features=F)
    out = net(torch.randn(B, T, F))
    assert out.shape == (B, 1)  # label head width == 1


def test_conv1d_seq_preserves_time_axis_alone():
    """conv1d_seq with no following recurrent: post-loop flatten guard kicks in."""
    T, F, B = 8, 4, 4
    spec = _base_spec(
        layers=[LayerSpec(kind="conv1d_seq", units=8, params={})],
        history_points=T,
        n_features=F,
    )
    net = _SpecNet(spec, history_points=T, n_features=F)
    out = net(torch.randn(B, T, F))
    assert out.shape == (B, 1)


def test_conv1d_pooling_regression_unchanged():
    """Existing conv1d (pooling) kind still builds and forwards unchanged."""
    T, F, B = 8, 4, 4
    spec = _base_spec(
        layers=[LayerSpec(kind="conv1d", units=12, params={"kernel_size": 3})],
        history_points=T,
        n_features=F,
    )
    net = _SpecNet(spec, history_points=T, n_features=F)
    out = net(torch.randn(B, T, F))
    assert out.shape == (B, 1)
