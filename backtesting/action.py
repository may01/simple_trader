"""Action — simulation event record (Phase 14, Task 02).

A single record type capturing every meaningful event during a simulation:
a position opened/closed/stopped out, a stop-loss moved, or a strategy signal
fired. JSONL-serialisable; one file per simulation.

Attribution is *action-type only* — no strategy-class name, no signal-chain
name (Phase 14 design decision 2). This module performs NO file I/O; ActionLog
only produces a JSONL string. Persistence is the orchestrator's job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# Event categories (lifecycle of the position / signal)
EVENT_OPEN = "OPEN"
EVENT_CLOSE = "CLOSE"
EVENT_STOP_LOSS = "STOP_LOSS"
EVENT_MOVE_STOP_LOSS = "MOVE_STOP_LOSS"
EVENT_SIGNAL_FIRED = "SIGNAL_FIRED"

# JSON-safe sentinel for non-finite floats (strict parsers reject raw inf/nan)
_INF_SENTINEL = "inf"


@dataclass
class Action:
    """One simulation event.

    Fields are concrete scalars so the JSONL round-trip is lossless.

    Attributes:
        sim_id: Owning simulation id.
        timestamp: Unix seconds of the tick.
        tick_index: Ordinal of the tick within the worker segment.
        event: One of EVENT_* — lifecycle category (what happened).
        action_type: Originating STRATEGY_ACTION_* constant (strategy intent).
        position_type: POSITION_TYPE_* constant.
        was_stop_loss: True when a close was forced by a stop-loss trigger.
        target_price: Price the strategy provided to the position; 0.0 if N/A.
        executed_price: Simulated fill price; 0.0 if no fill.
        stop_loss_price: Stop-loss price in effect; 0.0 if none.
        revenue_pct: Set on CLOSE/STOP_LOSS, else 0.0.
        revenue_abs: Set on CLOSE/STOP_LOSS, else 0.0.
    """

    sim_id: int
    timestamp: float
    tick_index: int
    event: str
    action_type: str
    position_type: str
    was_stop_loss: bool
    target_price: float
    executed_price: float
    stop_loss_price: float
    revenue_pct: float
    revenue_abs: float

    def to_dict(self) -> dict:
        """Return a flat, JSON-safe dict of all fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        """Rebuild an Action from a dict produced by to_dict()."""
        return cls(
            sim_id=int(data["sim_id"]),
            timestamp=float(data["timestamp"]),
            tick_index=int(data["tick_index"]),
            event=str(data["event"]),
            action_type=str(data["action_type"]),
            position_type=str(data["position_type"]),
            was_stop_loss=bool(data["was_stop_loss"]),
            target_price=float(data["target_price"]),
            executed_price=float(data["executed_price"]),
            stop_loss_price=float(data["stop_loss_price"]),
            revenue_pct=float(data["revenue_pct"]),
            revenue_abs=float(data["revenue_abs"]),
        )


class ActionLog:
    """In-memory accumulator of Action records for one simulation.

    Args:
        sim_id: The simulation id every recorded action must belong to.
    """

    def __init__(self, sim_id: int) -> None:
        self.sim_id: int = sim_id
        self.actions: list[Action] = []

    def record(self, action: Action) -> None:
        """Append an action, enforcing the sim_id invariant."""
        if action.sim_id != self.sim_id:
            raise ValueError(
                f"Action sim_id {action.sim_id} != log sim_id {self.sim_id}"
            )
        self.actions.append(action)

    def extend(self, other: "ActionLog") -> None:
        """Merge another log's actions (e.g. combining worker logs)."""
        if other.sim_id != self.sim_id:
            raise ValueError(
                f"Cannot extend: sim_id {other.sim_id} != {self.sim_id}"
            )
        self.actions.extend(other.actions)

    def to_jsonl(self) -> str:
        """Serialise to newline-separated JSON, one Action per line."""
        return "\n".join(json.dumps(a.to_dict()) for a in self.actions)

    @classmethod
    def from_jsonl(cls, sim_id: int, jsonl: str) -> "ActionLog":
        """Rebuild a log from a JSONL string (skips blank lines)."""
        log = cls(sim_id)
        for line in jsonl.splitlines():
            line = line.strip()
            if line:
                log.record(Action.from_dict(json.loads(line)))
        return log

    def __len__(self) -> int:
        return len(self.actions)
