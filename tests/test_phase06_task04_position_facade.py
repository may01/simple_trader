"""Tests for Position facade — Phase 06 Task 04."""

import pytest
from constants import (
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
    POSITION_STATE_WAIT,
    POSITION_STATE_WAIT_BUY,
    POSITION_STATE_WAIT_SELL,
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
)
from position.long_position import LongPosition
from position.short_position import ShortPosition
from position.position import Position


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

class TestPositionInit:
    def test_defaults(self):
        pos = Position()
        assert pos.posImpl is None
        assert pos.fee == 0.0
        assert pos.thread_num == 0
        assert pos.full_position == 10000.0

    def test_custom_params(self):
        pos = Position(thread_num=3, fee=0.001)
        assert pos.thread_num == 3
        assert pos.fee == 0.001

    def test_is_opened_false_initially(self):
        pos = Position()
        assert pos.is_opened() is False


# ---------------------------------------------------------------------------
# get_action / get_state / get_target with no impl
# ---------------------------------------------------------------------------

class TestNullDelegation:
    def test_get_action_no_impl(self):
        pos = Position()
        assert pos.get_action() == STRATEGY_ACTION_NOTHING

    def test_get_state_no_impl(self):
        pos = Position()
        assert pos.get_state() == POSITION_STATE_WAIT

    def test_get_target_no_impl(self):
        pos = Position()
        assert pos.get_target() == 0.0


# ---------------------------------------------------------------------------
# open() — Long
# ---------------------------------------------------------------------------

class TestOpenLong:
    def setup_method(self):
        self.pos = Position(fee=0.001)
        self.pos.full_position = 1000.0

    def test_open_long_creates_long_impl(self):
        result = self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0, 22.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert result is True
        assert isinstance(self.pos.posImpl, LongPosition)

    def test_open_long_state_becomes_wait_buy(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.get_state() == POSITION_STATE_WAIT_BUY

    def test_open_long_position_type(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.posImpl.position_type == POSITION_TYPE_LONG

    def test_open_long_is_opened(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.is_opened() is True

    def test_open_long_uses_fee_and_full_position(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.posImpl.fee == 0.001
        assert self.pos.posImpl.full_position == 1000.0

    def test_open_long_returns_false_when_already_open(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        result = self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert result is False


# ---------------------------------------------------------------------------
# open() — Short
# ---------------------------------------------------------------------------

class TestOpenShort:
    def setup_method(self):
        self.pos = Position(fee=0.001)
        self.pos.full_position = 1000.0

    def test_open_short_creates_short_impl(self):
        result = self.pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0, 18.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        assert result is True
        assert isinstance(self.pos.posImpl, ShortPosition)

    def test_open_short_state_becomes_wait_sell(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.get_state() == POSITION_STATE_WAIT_SELL

    def test_open_short_position_type(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.posImpl.position_type == POSITION_TYPE_SHORT

    def test_open_short_returns_false_when_already_open(self):
        self.pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        result = self.pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        assert result is False


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:
    def setup_method(self):
        self.pos = Position(fee=0.001)
        self.pos.full_position = 1000.0
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )

    def test_close_delegates_updates_price_close(self):
        self.pos.close(
            STRATEGY_ACTION_CLOSE_LONG,
            price_close=[21.5],
            price_stop_loss=19.5,
            time_period=15,
            action_msg=None,
        )
        assert self.pos.posImpl.price_close == [21.5]

    def test_close_no_impl_is_noop(self):
        pos = Position()
        # Should not raise
        pos.close(STRATEGY_ACTION_CLOSE_LONG, [21.0], 19.0, 15, None)


# ---------------------------------------------------------------------------
# set_stop_loss()
# ---------------------------------------------------------------------------

class TestSetStopLoss:
    def setup_method(self):
        self.pos = Position(fee=0.001)
        self.pos.full_position = 1000.0
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )

    def test_set_stop_loss_improves(self):
        result = self.pos.set_stop_loss(19.5, action_msg=None)
        assert result is True
        assert self.pos.posImpl.price_stop_loss == 19.5

    def test_set_stop_loss_rejected_when_worse(self):
        result = self.pos.set_stop_loss(18.0, action_msg=None)
        assert result is False

    def test_set_stop_loss_no_impl_is_noop(self):
        pos = Position()
        # Should not raise; return value doesn't matter (no impl)
        pos.set_stop_loss(100.0, action_msg=None)


# ---------------------------------------------------------------------------
# record_entry_fill / record_exit_fill
# ---------------------------------------------------------------------------

class TestFillDelegation:
    def setup_method(self):
        self.pos = Position(fee=0.001)
        self.pos.full_position = 1000.0
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )

    def test_record_entry_fill_records_price(self):
        self.pos.record_entry_fill(coin_amount=5.0, price=20.0)
        assert 20.0 in self.pos.posImpl.executed_open

    def test_record_entry_fill_state_changes_to_wait_sell(self):
        self.pos.record_entry_fill(coin_amount=5.0, price=20.0)
        assert self.pos.get_state() == POSITION_STATE_WAIT_SELL

    def test_record_exit_fill_increments_close_idx(self):
        self.pos.record_entry_fill(coin_amount=5.0, price=20.0)
        initial_idx = self.pos.posImpl.close_idx
        self.pos.record_exit_fill(coin_amount=2.0, price=21.0)
        assert self.pos.posImpl.close_idx == initial_idx + 1

    def test_record_entry_fill_no_impl_is_noop(self):
        pos = Position()
        pos.record_entry_fill(5.0, 20.0)  # Should not raise

    def test_record_exit_fill_no_impl_is_noop(self):
        pos = Position()
        pos.record_exit_fill(5.0, 21.0)  # Should not raise


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------

class TestFinalize:
    def test_finalize_clears_posImpl(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        pos.record_entry_fill(5.0, 20.0)
        pos.record_exit_fill(5.0, 21.0)
        result = pos.finalize()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert pos.posImpl is None

    def test_finalize_is_opened_false_after(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        pos.record_entry_fill(5.0, 20.0)
        pos.record_exit_fill(5.0, 21.0)
        pos.finalize()
        assert pos.is_opened() is False

    def test_get_state_after_finalize_is_wait(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        pos.record_entry_fill(5.0, 20.0)
        pos.record_exit_fill(5.0, 21.0)
        pos.finalize()
        assert pos.get_state() == POSITION_STATE_WAIT

    def test_can_reopen_after_finalize(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        pos.record_entry_fill(5.0, 20.0)
        pos.record_exit_fill(5.0, 21.0)
        pos.finalize()
        result = pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert result is True


# ---------------------------------------------------------------------------
# get_action()
# ---------------------------------------------------------------------------

class TestGetAction:
    def test_get_action_after_open_long(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert pos.get_action() == STRATEGY_ACTION_OPEN_LONG

    def test_get_action_after_open_short(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        assert pos.get_action() == STRATEGY_ACTION_OPEN_SHORT


# ---------------------------------------------------------------------------
# get_target()
# ---------------------------------------------------------------------------

class TestGetTarget:
    def test_get_target_returns_first_close_price(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0, 22.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert pos.get_target() == 21.0


# ---------------------------------------------------------------------------
# is_stop_loss_triggered()
# ---------------------------------------------------------------------------

class TestStopLossTrigger:
    def setup_method(self):
        self.pos = Position(fee=0.001)
        self.pos.full_position = 1000.0
        self.pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )

    def test_stop_loss_triggered_below(self):
        assert self.pos.is_stop_loss_triggered(18.5) is True

    def test_stop_loss_not_triggered_above(self):
        assert self.pos.is_stop_loss_triggered(20.0) is False


# ---------------------------------------------------------------------------
# check_stop_open()
# ---------------------------------------------------------------------------

class TestCheckStopOpen:
    def test_check_stop_open_long_far_below(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        # 4 * fee = 0.004; threshold = 20 * 0.996 = 19.92; 15.0 is far below
        assert pos.check_stop_open(15.0) is True

    def test_check_stop_open_long_at_price(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        assert pos.check_stop_open(20.0) is False


# ---------------------------------------------------------------------------
# close_by_time()
# ---------------------------------------------------------------------------

class TestCloseByTime:
    def test_close_by_time_before_timeout(self):
        import time
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        # Immediately after open, should not timeout
        assert pos.close_by_time(time.time()) is False

    def test_close_by_time_after_timeout(self):
        import time
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        # Simulate far-future time
        future = time.time() + 999999
        assert pos.close_by_time(future) is True


# ---------------------------------------------------------------------------
# to_dict() / from_dict()
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_contains_position_type_long(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        d = pos.to_dict()
        assert d["position_type"] == POSITION_TYPE_LONG

    def test_to_dict_contains_position_type_short(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        d = pos.to_dict()
        assert d["position_type"] == POSITION_TYPE_SHORT

    def test_from_dict_restores_long_position(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        d = pos.to_dict()

        pos2 = Position(fee=0.001)
        pos2.from_dict(d)
        assert isinstance(pos2.posImpl, LongPosition)
        assert pos2.posImpl.position_type == POSITION_TYPE_LONG

    def test_from_dict_restores_short_position(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[20.0],
            price_close=[19.0],
            price_stop_loss=21.0,
            time_period=15,
            action_msg=None,
        )
        d = pos.to_dict()

        pos2 = Position(fee=0.001)
        pos2.from_dict(d)
        assert isinstance(pos2.posImpl, ShortPosition)
        assert pos2.posImpl.position_type == POSITION_TYPE_SHORT

    def test_from_dict_restores_state(self):
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[20.0],
            price_close=[21.0, 22.0],
            price_stop_loss=19.0,
            time_period=15,
            action_msg=None,
        )
        d = pos.to_dict()

        pos2 = Position(fee=0.001)
        pos2.from_dict(d)
        assert pos2.get_state() == pos.get_state()
        assert pos2.posImpl.price_stop_loss == 19.0

    def test_roundtrip_fee_and_full_position(self):
        pos = Position(fee=0.002)
        pos.full_position = 5000.0
        pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[100.0],
            price_close=[110.0],
            price_stop_loss=90.0,
            time_period=60,
            action_msg=None,
        )
        d = pos.to_dict()
        pos2 = Position()
        pos2.from_dict(d)
        assert pos2.posImpl.fee == 0.002
        assert pos2.posImpl.full_position == 5000.0
