"""nn/device.py — device resolver and artefact-root path helpers.

Single source of device policy for the entire NN subsystem.
All os.environ reads happen at CALL TIME, never at import time.

OOM policy note: a CUDA out-of-memory error on a trial triggers ONE CPU retry
before the trial is marked failed (handled in TrainingCoordinator). This module
exposes the resolver; the retry logic lives in the training coordinator.
"""

import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> torch.device:
    """Return the torch.device to use for training/inference.

    Args:
        pref: device preference — "auto" | "cpu" | "cuda"

    Returns:
        torch.device("cpu") or torch.device("cuda")

    Behaviour:
        - "cpu"            → torch.device("cpu"), CUDA never probed.
        - "auto"           → cuda if available, else cpu (silent).
        - "cuda"           → cuda if available; if not, logs a WARNING and
                             returns cpu ("cuda requested but unavailable;
                             falling back to cpu").
    """
    if pref == "cpu":
        return torch.device("cpu")

    cuda_ok = torch.cuda.is_available()

    if pref == "cuda" and not cuda_ok:
        log.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")

    if cuda_ok:
        return torch.device("cuda")

    # pref == "auto" with no CUDA
    return torch.device("cpu")


def nn_artefact_root(pair: str) -> Path:
    """Return the per-pair artefact root: {NN_ARTEFACT_ROOT}/{DATA_ROOT}/{pair}/nn.

    The three sibling namespaces live under this root:
        datasets/   — content-addressed tensor cache
        checkpoints/ — one dir per model spec
        tracking/   — one dir per Optuna study

    Reads NN_ARTEFACT_ROOT and DATA_ROOT from the environment at call time.
    Raises KeyError if either env var is absent (same idiom as helpers.py).
    """
    artefact_root = os.environ["NN_ARTEFACT_ROOT"]
    data_root = os.environ["DATA_ROOT"]
    return Path(artefact_root) / data_root / pair / "nn"


def nn_spec_path() -> str:
    """Return the path to the NN model spec YAML file.

    Reads NN_SPEC_PATH from the environment at call time.
    Falls back to "configs/nn_spec.yaml" when the variable is unset.
    """
    return os.environ.get("NN_SPEC_PATH", "configs/nn_spec.yaml")
