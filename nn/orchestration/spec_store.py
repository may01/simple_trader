"""spec_store — persist NNModelSpec to the artefact volume (content-addressed).

Phase 17 Task 05. The loop driver materialises each version's spec under
specs/{spec_hash}/spec.yaml and hands the path to the trainer via NN_SPEC_PATH.
"""

from pathlib import Path

from nn.nn_model_spec import NNModelSpec


def materialize_spec(spec: NNModelSpec, artefact_root: Path) -> Path:
    """Persist spec to {artefact_root}/specs/{spec.spec_hash}/spec.yaml.

    Content-addressed by spec_hash: the same spec always maps to the same path,
    so re-materialising is idempotent (overwrites in place, never raises) and
    distinct specs are isolated under distinct specs/{hash}/ directories.

    NNModelSpec.to_yaml creates the parent directory, so no makedirs here.

    Returns the written path.
    """
    path = artefact_root / "specs" / spec.spec_hash / "spec.yaml"
    spec.to_yaml(str(path))
    return path
