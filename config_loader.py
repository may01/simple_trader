# config_loader.py — loads candles and indicators configuration from YAML files.
# No field names are hardcoded here; all come from the YAML files.

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class IndicatorFieldConfig:
    name: str
    group: str
    applies_to: list  # concrete list of TF ints (expanded at load time)
    depends_on: list
    library: Optional[str]
    params: dict


def load_candles_config(path: str = "configs/candles_config.yaml") -> list:
    """Returns the candles list, e.g. [1, 5, 15, 60, 240, 1440]."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return [int(c) for c in data["candles"]]


def _topological_sort(fields: list) -> list:
    """Kahn's algorithm: returns fields ordered so dependencies appear before dependants."""
    name_to_field = {f.name: f for f in fields}
    # Build in-degree and adjacency lists
    in_degree = {f.name: 0 for f in fields}
    dependants = {f.name: [] for f in fields}  # dep -> list of fields that depend on it

    for f in fields:
        for dep in f.depends_on:
            if dep in name_to_field:
                in_degree[f.name] += 1
                dependants[dep].append(f.name)

    # Start with nodes that have no unsatisfied dependencies
    queue = [f.name for f in fields if in_degree[f.name] == 0]
    # Preserve relative insertion order within same in-degree bucket
    queue.sort(key=lambda n: [f.name for f in fields].index(n))

    result = []
    while queue:
        current = queue.pop(0)
        result.append(name_to_field[current])
        # Reduce in-degree for all fields that depend on current
        next_ready = []
        for dependant in dependants[current]:
            in_degree[dependant] -= 1
            if in_degree[dependant] == 0:
                next_ready.append(dependant)
        # Sort next_ready to preserve original YAML ordering
        yaml_order = [f.name for f in fields]
        next_ready.sort(key=lambda n: yaml_order.index(n))
        queue.extend(next_ready)

    if len(result) != len(fields):
        cyclic = [f.name for f in fields if f.name not in {r.name for r in result}]
        raise ValueError(f"Cycle detected in indicator dependencies involving: {cyclic}")

    return result


def load_indicators_config(path: str = "configs/indicators_config.yaml") -> list:
    """Returns IndicatorFieldConfig list in dependency order (topologically sorted)."""
    candles = load_candles_config()

    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    raw_fields = data.get("fields", [])
    configs = []
    for entry in raw_fields:
        applies_raw = entry.get("applies_to", "all")
        if applies_raw == "all":
            applies_to = list(candles)
        else:
            applies_to = [int(x) for x in applies_raw]

        configs.append(
            IndicatorFieldConfig(
                name=entry["name"],
                group=entry["group"],
                applies_to=applies_to,
                depends_on=list(entry.get("depends_on", [])),
                library=entry.get("library", None),
                params=dict(entry.get("params", {})),
            )
        )

    return _topological_sort(configs)


def load_nn_config(path: str = "configs/indicators_config.yaml") -> dict:
    """Returns the 'nn' section as a dict with feature_cols and checkpoint_dir."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data["nn"]


@dataclass
class LabelSpecConfig:
    """One profit-label family from the 'labels' config section."""

    type: str            # "profit" | "profit_strict"
    tfs: list
    n: int
    m: float
    x: float
    atr_period: int = 14
    ma_length: int = 5         # SMA window of the atr_{atr_period}_ma_ column
    l: Optional[int] = None    # strict only
    y: Optional[float] = None  # strict only


def load_labels_config(path: str = "configs/indicators_config.yaml") -> list:
    """Parse the optional 'labels' section into LabelSpecConfig entries.

    Missing section → []. Raises ValueError on unknown type or a
    profit_strict entry without l/y.
    """
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)

    specs = []
    for entry in data.get("labels") or []:
        spec = LabelSpecConfig(
            type=entry["type"],
            tfs=[int(tf) for tf in entry["tfs"]],
            n=int(entry["n"]),
            m=float(entry["m"]),
            x=float(entry["x"]),
            atr_period=int(entry.get("atr_period", 14)),
            ma_length=int(entry.get("ma_length", 5)),
            l=int(entry["l"]) if "l" in entry else None,
            y=float(entry["y"]) if "y" in entry else None,
        )
        if spec.type not in ("profit", "profit_strict"):
            raise ValueError(f"Unknown label type: {spec.type!r}")
        if spec.type == "profit_strict" and (spec.l is None or spec.y is None):
            raise ValueError(
                f"profit_strict label entry requires 'l' and 'y' (got l={spec.l}, y={spec.y})"
            )
        specs.append(spec)
    return specs


# Module-level constant — loaded at import from default path
CANDLES: list = load_candles_config()


def warmup_minutes() -> int:
    """Minutes of leading history needed for NaN-free indicators on all TFs:
    INDICATOR_WINDOW_ROWS closed candles of the largest timeframe."""
    from constants import INDICATOR_WINDOW_ROWS
    return INDICATOR_WINDOW_ROWS * max(CANDLES)


def warmup_start_ms(data_start_ms: int) -> int:
    """Grab-range start: data_start_ms moved back by the warmup margin.

    Clamped at 0 — live.env uses DATA_START=0 as a placeholder.
    The simulation window itself must keep using the raw DATA_START.
    """
    return max(data_start_ms - warmup_minutes() * 60_000, 0)
