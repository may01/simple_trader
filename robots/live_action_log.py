"""LiveActionLog — append-only JSONL store of live Action records.

The live Robot writes one line per lifecycle Action (OPEN / CLOSE / STOP_LOSS)
so a separate process (scripts/tail_live_actions.py) can observe what the bot
did without stopping it. Unlike the in-memory backtesting ActionLog, this
persists incrementally to disk.

Live records use a fixed sentinel sim_id (LIVE_SIM_ID); the simulation
sim_id space is positive, so 0 marks "not a simulation".
"""

import json
import os

from backtesting.action import Action

LIVE_SIM_ID = 0


class LiveActionLog:
    """Append-only, tailable JSONL log of live Action records.

    Args:
        path: File path the records are appended to (one JSON object per line).
    """

    def __init__(self, path: str) -> None:
        self.path: str = path

    def record(self, action: Action) -> None:
        """Append one Action as a JSON line, creating the parent dir if needed."""
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(action.to_dict()) + "\n")

    def tail(self, n: int = 20) -> list[Action]:
        """Return the most recent ``n`` Action records (oldest-first), or []."""
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            lines = [line.strip() for line in f if line.strip()]
        return [Action.from_dict(json.loads(line)) for line in lines[-n:]]
