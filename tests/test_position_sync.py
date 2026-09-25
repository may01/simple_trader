"""`Position.sync_from_executor` — the aggregate, replaced not appended.

main/ tracks a position for two reasons (position-management design
§6.4): to record actions for level visualisation, and to hold
`avg_price_open` so risk and strategies compute against the price
actually paid. These tests hold both, and the no-drift property that
justifies a setter over a synthetic fill.
"""

import pytest

from constants import STRATEGY_ACTION_OPEN_LONG, STRATEGY_ACTION_OPEN_SHORT
from position.position import Position


def opened_long() -> Position:
    position = Position(thread_num=0, fee=0.001)
    position.open(
        STRATEGY_ACTION_OPEN_LONG,
        [15.00],
        [15.60],
        14.85,
        15,
        action_msg=None,
    )
    return position


# ---------------------------------------------------------------------
# Purpose 2: avg_price_open is the price actually paid
# ---------------------------------------------------------------------

def test_avg_price_open_returns_the_executors_number_after_a_sync():
    position = opened_long()
    assert position.sync_from_executor(0.8, 15.0150, 14.85, 15.60) is True
    assert position.avg_price_open() == pytest.approx(15.0150)


def test_repeated_syncs_with_the_same_value_do_not_drift():
    """The reason this is a replace-in-place setter and not
    `record_entry_fill(net_size, avg)`: that appends, so each repeated
    update would re-average against its own previous output."""
    position = opened_long()
    for _ in range(5):
        position.sync_from_executor(0.8, 15.0150, 14.85, 15.60)
    assert position.avg_price_open() == pytest.approx(15.0150)


def test_a_sync_records_no_fills():
    """main/ tracks no fills from this path (§6.4)."""
    position = opened_long()
    before_open = list(position.posImpl.executed_open)
    before_amount = list(position.posImpl.executed_open_amount)
    position.sync_from_executor(0.8, 15.0150, 14.85, 15.60)
    assert position.posImpl.executed_open == before_open
    assert position.posImpl.executed_open_amount == before_amount


def test_a_changing_average_is_followed_not_blended():
    position = opened_long()
    position.sync_from_executor(0.8, 15.0150, None, None)
    position.sync_from_executor(1.2, 15.0400, None, None)
    assert position.avg_price_open() == pytest.approx(15.0400)


def test_without_a_sync_the_fill_derived_average_still_wins():
    """Backtests and the armed legacy order path have no executor and
    must behave exactly as before."""
    position = opened_long()
    position.record_entry_fill(100.0, 15.00)
    assert position.avg_price_open() == pytest.approx(15.00)


# ---------------------------------------------------------------------
# The levels risk is computed against
# ---------------------------------------------------------------------

def test_a_synced_stop_drives_the_stop_loss_check():
    position = opened_long()
    position.sync_from_executor(0.8, 15.0150, 14.90, 15.60)
    assert position.is_stop_loss_triggered(14.89) is True
    assert position.is_stop_loss_triggered(14.95) is False


def test_omitted_levels_leave_the_existing_ones_alone():
    position = opened_long()
    position.sync_from_executor(0.8, 15.0150, 14.90, 15.60)
    position.sync_from_executor(0.8, 15.0150, None, None)
    assert position.is_stop_loss_triggered(14.89) is True


# ---------------------------------------------------------------------
# A sync that changes nothing records nothing
# ---------------------------------------------------------------------

def test_a_sync_that_changes_nothing_reports_no_change():
    position = opened_long()
    assert position.sync_from_executor(0.8, 15.0150, 14.85, 15.60) is True
    assert position.sync_from_executor(0.8, 15.0150, 14.85, 15.60) is False


def test_syncing_a_flat_position_is_a_no_op():
    """No local intent for the executor's truth to attach to. The
    divergence check in Robot is what notices and says so."""
    position = Position(thread_num=0, fee=0.001)
    assert position.sync_from_executor(0.8, 15.0150, 14.85, 15.60) is False


def test_a_short_position_syncs_too():
    position = Position(thread_num=0, fee=0.001)
    position.open(STRATEGY_ACTION_OPEN_SHORT, [15.00], [14.40], 15.15, 15, action_msg=None)
    assert position.sync_from_executor(0.8, 14.9850, 15.15, 14.40) is True
    assert position.avg_price_open() == pytest.approx(14.9850)
    assert position.is_stop_loss_triggered(15.20) is True
