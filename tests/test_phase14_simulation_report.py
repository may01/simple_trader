"""Phase 14 Task 06 — SimulationReport."""

import json

import pytest

from backtesting.simulation_report import SimulationReport


def _counts(*events):
    jsonl = "\n".join(json.dumps({"event": e}) for e in events)
    return SimulationReport.action_counts_from_jsonl(jsonl)


def test_action_counts_from_jsonl():
    c = _counts("OPEN", "CLOSE", "SIGNAL_FIRED", "CLOSE")
    assert c == {"OPEN": 1, "CLOSE": 2, "SIGNAL_FIRED": 1}


def test_write_report(tmp_path):
    counts = _counts("OPEN", "CLOSE")
    rep = SimulationReport(5, {"total_trades": 1, "win_rate": 1.0}, counts, {"strategy_set": "test"})
    folder = str(tmp_path) + "/"
    path = rep.write(folder)
    data = json.load(open(path))
    assert data["sim_id"] == 5
    assert data["action_counts"]["CLOSE"] == 1


def test_inf_profit_factor_serialised_as_string(tmp_path):
    counts = _counts("CLOSE")
    rep = SimulationReport(
        1, {"total_trades": 1, "profit_factor": float("inf")}, counts, {}
    )
    data = json.loads(json.dumps(rep.to_dict()))  # must be strict-JSON-safe
    assert data["metrics"]["profit_factor"] == "inf"


def test_consistency_assertion():
    counts = _counts("OPEN", "OPEN")  # zero closes
    rep = SimulationReport(1, {"total_trades": 2}, counts, {})
    with pytest.raises(ValueError):
        rep.to_dict()


def test_stop_loss_counts_toward_trades():
    counts = _counts("CLOSE", "STOP_LOSS")
    rep = SimulationReport(1, {"total_trades": 2}, counts, {})
    assert rep.to_dict()["metrics"]["total_trades"] == 2
