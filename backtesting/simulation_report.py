"""SimulationReport — per-simulation report.json writer (Phase 14, Task 06).

Assembles PerformanceAnalyzer metrics with run metadata and action tallies into
a JSON report that lives beside actions.jsonl, so one folder fully describes one
simulation.
"""

import json
import os


def _json_safe(value):
    """Convert non-finite floats to JSON-safe sentinels (no raw inf/nan)."""
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:  # NaN
            return 0.0
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


class SimulationReport:
    """Self-describing summary of one simulation run.

    Args:
        sim_id: The run's simulation id.
        metrics: Output of PerformanceAnalyzer.analyze().
        action_counts: Count of actions per event (see action_counts_from_jsonl).
        context: Run metadata (strategy_set, pair, window, fee, num_workers, ...).
    """

    def __init__(self, sim_id: int, metrics: dict, action_counts: dict, context: dict) -> None:
        self.sim_id = sim_id
        self.metrics = metrics
        self.action_counts = action_counts
        self.context = context

    def to_dict(self) -> dict:
        """Return the JSON-safe report dict, asserting trade/close consistency."""
        closes = self.action_counts.get("CLOSE", 0) + self.action_counts.get("STOP_LOSS", 0)
        total_trades = self.metrics.get("total_trades", 0)
        if total_trades != closes:
            raise ValueError(
                f"Report inconsistency: total_trades={total_trades} != "
                f"CLOSE+STOP_LOSS={closes}"
            )
        return _json_safe({
            "sim_id": self.sim_id,
            "metrics": self.metrics,
            "action_counts": self.action_counts,
            "context": self.context,
        })

    def write(self, folder: str) -> str:
        """Write report.json into *folder*; return its path."""
        os.makedirs(folder, exist_ok=True)
        path = folder + "report.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def action_counts_from_jsonl(jsonl: str) -> dict:
        """Tally actions per event from a merged actions JSONL string."""
        counts: dict = {}
        for line in jsonl.splitlines():
            line = line.strip()
            if not line:
                continue
            event = json.loads(line).get("event", "UNKNOWN")
            counts[event] = counts.get(event, 0) + 1
        return counts
