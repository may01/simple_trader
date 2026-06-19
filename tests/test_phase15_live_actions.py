"""Phase 15 Task 03 — live Action persistence.

The live Robot writes lifecycle Action records to an append-only JSONL store
so a separate process can tail them. Mock-only, no creds, runs in Docker.
"""

from backtesting.action import Action
from robots.live_action_log import LIVE_SIM_ID, LiveActionLog
from robots.robot import Robot
from strategies.strategy_manager import StrategyManager


def _action(event: str = "OPEN") -> Action:
    return Action(
        sim_id=LIVE_SIM_ID,
        timestamp=1.0,
        tick_index=0,
        event=event,
        action_type=event,
        position_type="LONG",
        was_stop_loss=False,
        target_price=10.0,
        executed_price=10.0,
        stop_loss_price=0.0,
        revenue_pct=0.0,
        revenue_abs=0.0,
    )


def test_live_action_log_round_trip(tmp_path):
    log = LiveActionLog(str(tmp_path / "live_actions.jsonl"))
    log.record(_action("OPEN"))
    log.record(_action("CLOSE"))
    out = log.tail(10)
    assert [a.event for a in out] == ["OPEN", "CLOSE"]
    assert all(a.sim_id == LIVE_SIM_ID for a in out)


def test_tail_empty_when_no_file(tmp_path):
    assert LiveActionLog(str(tmp_path / "missing.jsonl")).tail() == []


def test_tail_returns_last_n(tmp_path):
    log = LiveActionLog(str(tmp_path / "live_actions.jsonl"))
    for _ in range(5):
        log.record(_action("OPEN"))
    assert len(log.tail(2)) == 2


class _FakeStock:
    fee = 0.001


class _DataPoint:
    timestamp = 123.0


def _robot(tmp_path, action_log_path):
    return Robot(
        StrategyManager(_FakeStock.fee),
        live_data=None,
        stock=_FakeStock(),
        fee=_FakeStock.fee,
        persist_path=str(tmp_path / "tracker.json"),
        action_log_path=action_log_path,
    )


def test_robot_records_live_action_on_position_change(tmp_path):
    path = str(tmp_path / "live_actions.jsonl")
    robot = _robot(tmp_path, path)
    robot.position._changes.append({
        "kind": "OPEN",
        "position_type": "LONG",
        "was_stop_loss": False,
        "target_price": 10.0,
        "executed_price": 10.0,
        "stop_loss_price": 0.0,
        "revenue_pct": 0.0,
        "revenue_abs": 0.0,
    })
    robot._record_live_actions(_DataPoint())
    recs = LiveActionLog(path).tail()
    assert len(recs) == 1
    assert recs[0].event == "OPEN"
    assert recs[0].position_type == "LONG"


def test_robot_no_action_log_is_inert(tmp_path):
    # action_log_path=None → recording is a no-op, no crash, no file.
    robot = _robot(tmp_path, None)
    robot.position._changes.append({
        "kind": "OPEN", "position_type": "LONG", "was_stop_loss": False,
        "target_price": 10.0, "executed_price": 10.0, "stop_loss_price": 0.0,
        "revenue_pct": 0.0, "revenue_abs": 0.0,
    })
    robot._record_live_actions(_DataPoint())  # must not raise
