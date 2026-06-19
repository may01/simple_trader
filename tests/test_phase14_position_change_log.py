"""Phase 14 Task 03 — Position change log."""

from position.position import Position
from constants import (
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_DO_STOP_LOSS,
)


def _open_long():
    p = Position(fee=0.001)
    p.full_position = 1000.0
    p.open(STRATEGY_ACTION_OPEN_LONG, [20.0], [20.16], 19.5, 3600, None)
    return p


def test_open_records_change_and_drains():
    p = _open_long()
    ch = p.drain_changes()
    # OPEN: target_price = take-profit (price_close[0]); executed_price = entry.
    assert ch and ch[0]["kind"] == "OPEN"
    assert ch[0]["target_price"] == 20.16
    assert ch[0]["executed_price"] == 20.0
    assert p.drain_changes() == []  # cleared


def test_close_and_finalize_settle_revenue():
    p = _open_long()
    p.drain_changes()  # discard OPEN
    p.record_entry_fill(50.0, 20.0)
    p.close(STRATEGY_ACTION_CLOSE_LONG, [20.16], 19.5, 3600, None)
    p.record_exit_fill(50.0, 20.16)
    p.finalize()
    ch = p.drain_changes()
    closes = [c for c in ch if c["kind"] == "CLOSE"]
    assert closes and closes[-1]["revenue_pct"] != 0.0


def test_stop_loss_close_flagged():
    p = _open_long()
    p.drain_changes()
    p.record_entry_fill(50.0, 20.0)
    p.close(STRATEGY_ACTION_DO_STOP_LOSS, [19.5], 19.5, 3600, None)
    ch = p.drain_changes()
    sl = [c for c in ch if c["kind"] == "STOP_LOSS"]
    assert sl and sl[0]["was_stop_loss"] is True


def test_forced_stop_loss_set_does_not_log_move():
    # close() sets stop-loss with force=True — must NOT record a MOVE_STOP_LOSS.
    p = _open_long()
    p.drain_changes()
    p.record_entry_fill(50.0, 20.0)
    p.close(STRATEGY_ACTION_CLOSE_LONG, [20.16], 19.0, 3600, None)
    ch = p.drain_changes()
    assert not any(c["kind"] == "MOVE_STOP_LOSS" for c in ch)
