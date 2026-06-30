"""nn/nn_model_spec.py — Declarative model definition for the NN subsystem.

Single source of truth for model architecture, data selection, training
hyper-parameters, and target heads.  A NNModelSpec uniquely identifies a
model: two models differ only by their specs.

Content-addressing: spec_hash is a sha256 over the canonically-normalised
spec fields (device and seed excluded — runtime-only, do not change model
identity).  Checkpoints and tensor caches are keyed by this hash.

Usage
-----
# Load from YAML
spec = NNModelSpec.from_yaml("configs/nn_spec.yaml")
print(spec.spec_hash[:8], spec.timeframes)

# Minimal valid spec for tests / smoke-checks
spec = NNModelSpec.default()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# LayerSpec
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    """One hidden layer in the network.

    kind: "dense" | "lstm" | "gru" | "conv1d" | "conv1d_seq"
        conv1d_seq — conv over time, preserves the sequence for a following
        recurrent layer (unlike conv1d, which mean-pools the time axis away).
    units: layer width / hidden-state size
    params: kind-specific kwargs (e.g. kernel_size, bidirectional)
    """

    kind: str
    units: int
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GroupingSpec
# ---------------------------------------------------------------------------


@dataclass
class GroupingSpec:
    """How training rows are partitioned into classes/regimes.

    mode="single"        — one model trained on all rows (default)
    mode="by_indicator"  — orchestrator trains one model per class, where
                           classes come from `column` split by `bins` or
                           `classes`.
    """

    mode: str = "single"                   # "single" | "by_indicator"
    column: str | None = None              # e.g. "60_vol_regime"
    bins: list[float] | None = None        # numeric edges → classes
    classes: list | None = None            # explicit categorical values


# ---------------------------------------------------------------------------
# TargetSpec
# ---------------------------------------------------------------------------


@dataclass
class TargetSpec:
    """One output head produced by the model.

    name      — used in output column names: nn_res_{name}_*
    kind      — "direction" | "label" | "regression"
    horizons  — list of look-ahead candle counts; len>1 → multi-horizon heads

    direction / label fields (profit-labels pipeline, Phase 03 Task 07):
        label_tf, label_m, label_x, strict; label_l, label_y for strict

    regression fields:
        transform — "logret"
    """

    name: str
    kind: str                                   # "direction"|"label"|"regression"
    horizons: list[int] = field(default_factory=lambda: [1])

    # direction / label
    label_tf: int | None = None
    label_m: float | None = None
    label_x: float | None = None
    strict: bool = False
    label_l: int | None = None      # strict-only: clean-entry lookback window
    label_y: float | None = None    # strict-only: clean-entry threshold (ATR mult)

    # regression
    transform: str = "logret"

    def out_columns(self) -> list[str]:
        """Return timeframe-agnostic output column names for this target.

        Naming rules (single-horizon [1] has NO _h suffix):
          direction  → nn_res_{name}_prob_up / _prob_neutral / _prob_down
          label      → nn_res_{name}_prob
          regression → nn_res_{name}

          multi-horizon (len(horizons)>1) inserts _h{hk} after {name}:
          direction  → nn_res_{name}_h{hk}_prob_up / _prob_neutral / _prob_down
          label      → nn_res_{name}_h{hk}_prob
          regression → nn_res_{name}_h{hk}
        """
        multi = len(self.horizons) > 1
        cols: list[str] = []

        for hk in self.horizons:
            if multi:
                base = f"nn_res_{self.name}_h{hk}"
            else:
                base = f"nn_res_{self.name}"

            if self.kind == "direction":
                cols += [
                    f"{base}_prob_up",
                    f"{base}_prob_neutral",
                    f"{base}_prob_down",
                ]
            elif self.kind == "label":
                cols.append(f"{base}_prob")
            elif self.kind == "regression":
                cols.append(base)
            else:
                raise ValueError(f"Unknown TargetSpec kind: {self.kind!r}")

        return cols


# ---------------------------------------------------------------------------
# NNModelSpec
# ---------------------------------------------------------------------------


@dataclass
class NNModelSpec:
    """Full declarative specification of a neural-network model.

    Fields are grouped:
      Identity     — name
      Data         — grouping, timeframes, indicators, history_points
      Architecture — layers, activation, dropout
      Targets      — targets (output heads)
      Learning     — loss_fn, optimizer, learning_rate, weight_decay,
                     batch_size, epochs, validation_split, val_strategy,
                     early_stopping_patience, class_weight, shuffle_train
      Runtime      — device, seed  (excluded from spec_hash)

    spec_hash (property) — sha256 content-address; callers use [:8] for labels.
    """

    # --- Identity ---
    name: str

    # --- Data grouping ---
    grouping: GroupingSpec = field(default_factory=GroupingSpec)
    timeframes: list[int] = field(default_factory=list)
    indicators: list[str] = field(default_factory=list)
    history_points: int = 32

    # --- Architecture ---
    layers: list[LayerSpec] = field(default_factory=list)
    activation: str = "relu"
    dropout: float = 0.0

    # --- Targets ---
    targets: list[TargetSpec] = field(default_factory=list)

    # --- Learning parameters ---
    loss_fn: str = "auto"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    val_strategy: str = "time_holdout"
    early_stopping_patience: int | None = 10
    class_weight: str = "balanced"
    shuffle_train: bool = True

    # --- Runtime (excluded from spec_hash) ---
    device: str = "auto"
    seed: int = 0

    # ------------------------------------------------------------------
    # Content-addressed hash
    # ------------------------------------------------------------------

    @property
    def spec_hash(self) -> str:
        """sha256 hex digest over canonically-normalised spec fields.

        Excluded fields (runtime-only): device, seed.
        Normalisation guarantees:
          - nested dataclasses → plain dicts
          - dict keys are sorted recursively (so params key order is stable)
          - ordered sequences (layers, targets, timeframes, indicators) stay
            ordered (order is semantically meaningful)
        """
        canonical = _canonical(self)
        serialised = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()

    # ------------------------------------------------------------------
    # YAML loader
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str = "configs/nn_spec.yaml") -> "NNModelSpec":
        """Parse a YAML spec file and return a fully-populated NNModelSpec.

        Builds nested GroupingSpec / LayerSpec / TargetSpec from their
        sub-mappings; coerces types (timeframes→list[int],
        horizons→list[int]); applies dataclass defaults for absent keys.
        """
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        raw = dict(raw)  # shallow copy so we can pop/replace

        # grouping
        grouping_raw = raw.pop("grouping", {})
        grouping = GroupingSpec(**grouping_raw) if grouping_raw else GroupingSpec()

        # timeframes → list[int]
        timeframes = [int(tf) for tf in raw.pop("timeframes", [])]

        # layers
        layers_raw = raw.pop("layers", [])
        layers = [
            LayerSpec(
                kind=lr["kind"],
                units=int(lr["units"]),
                params=dict(lr.get("params") or {}),
            )
            for lr in layers_raw
        ]

        # targets
        targets_raw = raw.pop("targets", [])
        targets = []
        for tr in targets_raw:
            tr = dict(tr)
            horizons = [int(h) for h in tr.pop("horizons", [1])]
            targets.append(
                TargetSpec(
                    horizons=horizons,
                    **tr,
                )
            )

        return cls(
            grouping=grouping,
            timeframes=timeframes,
            layers=layers,
            targets=targets,
            **raw,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain nested dict representation of this spec.

        Calls dataclasses.asdict(self), so all nested dataclasses (LayerSpec,
        GroupingSpec, TargetSpec) become plain dicts.  device and seed are
        included (unlike spec_hash, which excludes them).
        """
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write this spec to a YAML file at *path*.

        Creates parent directories if they do not exist.  Dumps with
        sort_keys=False to preserve dataclass field order for readability;
        from_yaml reads by key so order does not affect the round-trip.
        """
        import os

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # ------------------------------------------------------------------
    # Default factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "NNModelSpec":
        """Return a minimal but valid NNModelSpec.

        Suitable for smoke-tests and downstream task scaffolding.
        Uses a small history window, two indicators, one dense layer,
        and one direction target on the 15-minute timeframe.
        """
        return cls(
            name="default",
            timeframes=[15],
            indicators=["rsi_14", "logret"],
            history_points=16,
            layers=[LayerSpec(kind="dense", units=64)],
            targets=[
                TargetSpec(
                    name="dir15",
                    kind="direction",
                    horizons=[1],
                    label_tf=15,
                    label_m=1.0,
                    label_x=0.3,
                )
            ],
            epochs=5,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canonical(spec: NNModelSpec) -> dict:
    """Return a plain-dict representation of spec, excluding device and seed,
    with all nested dicts sorted by key recursively.

    Ordered sequences (layers, targets, timeframes, indicators) preserve order.
    """
    full = asdict(spec)

    # Remove runtime-only fields before hashing
    full.pop("device", None)
    full.pop("seed", None)

    return _sort_dicts(full)


def _sort_dicts(obj: Any) -> Any:
    """Recursively sort dict keys; leave lists in their original order."""
    if isinstance(obj, dict):
        return {k: _sort_dicts(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_sort_dicts(item) for item in obj]
    return obj
