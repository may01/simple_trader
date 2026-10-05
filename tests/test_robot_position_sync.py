"""Robot applies the executor's position and reports divergence.

main/'s `position.open()`/`close()` express intent; the executor's
events are what make them true (position-management design §6.4). A
Robot with no consumer must behave exactly as it did before any of this
existed.
"""

import logging

import pytest

from constants import STRATEGY_ACTION_OPEN_LONG
from mq.position_consumer import PositionEvent


def event(**overrides) -> PositionEvent:
    base = dict(
        schema=2,
        pair="LINKUSDT",
        event="opened",
        position_id="11111111-2222-4333-8444-555555555555",
        reason=None,
        status="open",
        side="long",
        market_kind="margin",
        target_size=1.2,
        net_size=0.8,
        intended_open_price=15.00,
        avg_entry_price=15.0150,
        avg_exit_price=None,
        stop_loss_price=14.85,
        take_profit_price=15.60,
        realized_pnl=None,
        unrealized_pnl=None,
        settlement_complete=False,
        leverage=None,
        liquidation_price=None,
        created_at=1_758_500_000_000,
        ts="2026-09-25T00:00:00Z",
    )
    base.update(overrides)
    return PositionEvent(**base)


def flat_event(**overrides) -> PositionEvent:
    return event(
        event="already_closed",
        status="flat",
        side=None,
        net_size=None,
        avg_entry_price=None,
        target_size=None,
        intended_open_price=None,
        stop_loss_price=None,
        take_profit_price=None,
        **overrides,
    )


class FakeConsumer:
    def __init__(self, events):
        self._events = list(events)
        self.polls = 0

    def poll(self, timeout_ms: int = 0):
        self.polls += 1
        drained, self._events = self._events, []
        return drained


@pytest.fixture
def robot():
    """A Robot with only what these tests touch.

    Built via `__new__` rather than the full constructor: `Robot.__init__`
    reaches for live data, a stock and a tracker, none of which this
    behaviour depends on.
    """
    from position.position import Position
    from robots.robot import Robot

    r = Robot.__new__(Robot)
    r.position = Position(thread_num=0, fee=0.001)
    r.position_consumer = None
    r._divergences = 0
    return r


def open_local(robot):
    robot.position.open(
        STRATEGY_ACTION_OPEN_LONG, [15.00], [15.60], 14.85, 15, action_msg=None
    )


# ---------------------------------------------------------------------
# Applying the executor's aggregate
# ---------------------------------------------------------------------

def test_an_executor_event_sets_the_average_entry_price(robot):
    open_local(robot)
    robot.position_consumer = FakeConsumer([event()])
    robot._apply_executor_position()
    assert robot.position.avg_price_open() == pytest.approx(15.0150)


def test_a_sync_that_changes_something_records_an_action(robot):
    """Purpose 1: the executor's actions belong in the action log, not
    only main/'s own intent."""
    open_local(robot)
    robot.position.drain_changes()  # discard the OPEN from strategy intent
    robot.position_consumer = FakeConsumer([event()])
    robot._apply_executor_position()
    kinds = [c["kind"] for c in robot.position.drain_changes()]
    assert "EXECUTOR_OPENED" in kinds


def test_a_sync_that_changes_nothing_records_nothing(robot):
    open_local(robot)
    robot.position_consumer = FakeConsumer([event()])
    robot._apply_executor_position()
    robot.position.drain_changes()
    robot.position_consumer = FakeConsumer([event()])
    robot._apply_executor_position()
    assert robot.position.drain_changes() == []


def test_every_queued_event_is_drained_in_one_tick(robot):
    open_local(robot)
    consumer = FakeConsumer([event(), event(avg_entry_price=15.05)])
    robot.position_consumer = consumer
    robot._apply_executor_position()
    assert robot.position.avg_price_open() == pytest.approx(15.05)


def test_no_consumer_is_a_complete_no_op(robot):
    open_local(robot)
    before = robot.position.avg_price_open()
    robot._apply_executor_position()
    assert robot.position.avg_price_open() == before
    assert robot._divergences == 0


# ---------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------

def test_executor_holds_a_position_main_thinks_is_flat(robot, caplog):
    robot.position_consumer = FakeConsumer([event()])
    with caplog.at_level(logging.WARNING):
        robot._apply_executor_position()
    assert robot._divergences == 1
    assert "executor holds" in " ".join(r.getMessage() for r in caplog.records)


def test_main_thinks_open_but_the_executor_reports_flat(robot, caplog):
    """The signal that a decision was silently refused: wire v2 carries
    `not_placed` with a reason, but a dropped PUSH message carries
    nothing."""
    open_local(robot)
    robot.position_consumer = FakeConsumer([flat_event()])
    with caplog.at_level(logging.WARNING):
        robot._apply_executor_position()
    assert robot._divergences == 1
    assert "reports flat" in " ".join(r.getMessage() for r in caplog.records)


def test_a_side_mismatch_is_reported(robot, caplog):
    open_local(robot)
    robot.position_consumer = FakeConsumer([event(side="short")])
    with caplog.at_level(logging.WARNING):
        robot._apply_executor_position()
    assert robot._divergences == 1
    assert "side divergence" in " ".join(r.getMessage() for r in caplog.records)


def test_agreement_produces_no_warning(robot, caplog):
    """A heartbeat matching main/'s view must not produce a warning
    storm -- one arrives every interval per live position."""
    open_local(robot)
    robot.position_consumer = FakeConsumer([event(), event(), event()])
    with caplog.at_level(logging.WARNING):
        robot._apply_executor_position()
    assert robot._divergences == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_both_flat_is_not_a_divergence(robot, caplog):
    robot.position_consumer = FakeConsumer([flat_event()])
    with caplog.at_level(logging.WARNING):
        robot._apply_executor_position()
    assert robot._divergences == 0
