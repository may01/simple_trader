"""Phase 14 Task 02 — Action / ActionLog."""

import json

import pytest

from backtesting.action import Action, ActionLog, EVENT_OPEN
from constants import STRATEGY_ACTION_OPEN_LONG, POSITION_TYPE_LONG


def _make(sim_id=7):
    return Action(
        sim_id=sim_id, timestamp=1000.0, tick_index=3, event=EVENT_OPEN,
        action_type=STRATEGY_ACTION_OPEN_LONG, position_type=POSITION_TYPE_LONG,
        was_stop_loss=False, target_price=20.0, executed_price=20.0,
        stop_loss_price=19.0, revenue_pct=0.0, revenue_abs=0.0,
    )


def test_jsonl_roundtrip_lossless():
    a = _make()
    log = ActionLog(7)
    log.record(a)
    line = log.to_jsonl()
    assert Action.from_dict(json.loads(line)).to_dict() == a.to_dict()


def test_record_enforces_sim_id():
    log = ActionLog(1)
    with pytest.raises(ValueError):
        log.record(_make(sim_id=2))


def test_extend_merges_and_enforces_sim_id():
    a, b = ActionLog(5), ActionLog(5)
    a.record(_make(5))
    b.record(_make(5))
    a.extend(b)
    assert len(a) == 2
    with pytest.raises(ValueError):
        a.extend(ActionLog(6))


def test_from_jsonl_skips_blank_lines():
    log = ActionLog(7)
    log.record(_make())
    rebuilt = ActionLog.from_jsonl(7, log.to_jsonl() + "\n\n")
    assert len(rebuilt) == 1
