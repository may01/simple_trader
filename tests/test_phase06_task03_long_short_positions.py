"""Tests for LongPosition and ShortPosition (Phase 06, Task 03)."""

import time
import pytest
from constants import (
    POSITION_STATE_WAIT,
    POSITION_STATE_WAIT_BUY,
    POSITION_STATE_WAIT_SELL,
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_CLOSE_SHORT,
)
from position.long_position import LongPosition
from position.short_position import ShortPosition
from position.coin import Coin


# ---------------------------------------------------------------------------
# LongPosition — instantiation
# ---------------------------------------------------------------------------

class TestLongPositionInit:
    """Test LongPosition initialization."""

    def test_is_concrete_subclass(self):
        """LongPosition can be instantiated (not abstract)."""
        pos = LongPosition(fee=0.001)
        assert pos is not None

    def test_position_type_long(self):
        """position_type is POSITION_TYPE_LONG."""
        pos = LongPosition(fee=0.001)
        assert pos.position_type == POSITION_TYPE_LONG

    def test_coin_use_is_usdt(self):
        """coinUse is named 'usdt'."""
        pos = LongPosition(fee=0.001)
        assert pos.coinUse.name == "usdt"

    def test_coin_get_is_coin(self):
        """coinGet is named 'coin'."""
        pos = LongPosition(fee=0.001)
        assert pos.coinGet.name == "coin"

    def test_initial_state_is_wait(self):
        """state starts as POSITION_STATE_WAIT."""
        pos = LongPosition(fee=0.001)
        assert pos.state == POSITION_STATE_WAIT

    def test_default_full_position(self):
        """full_position defaults to 10000.0."""
        pos = LongPosition(fee=0.001)
        assert pos.full_position == 10000.0

    def test_custom_full_position(self):
        """full_position can be customized."""
        pos = LongPosition(fee=0.001, full_position=5000.0)
        assert pos.full_position == 5000.0

    def test_thread_num_default(self):
        """thread_num defaults to 0."""
        pos = LongPosition(fee=0.001)
        assert pos.thread_num == 0


# ---------------------------------------------------------------------------
# LongPosition — open()
# ---------------------------------------------------------------------------

class TestLongPositionOpen:
    """Test LongPosition.open() method."""

    def _make_pos(self):
        return LongPosition(fee=0.001, full_position=10000.0)

    def test_open_returns_true_on_success(self):
        """open() returns True on successful open."""
        pos = self._make_pos()
        result = pos.open(
            STRATEGY_ACTION_OPEN_LONG,
            price_open=[100.0],
            price_close=[105.0, 110.0],
            price_stop_loss=95.0,
            time_period=60,
            action_msg="test",
        )
        assert result is True

    def test_open_state_becomes_wait_buy(self):
        """After open(), state is POSITION_STATE_WAIT_BUY."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert pos.state == POSITION_STATE_WAIT_BUY

    def test_open_action_set(self):
        """After open(), action is STRATEGY_ACTION_OPEN_LONG."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert pos.action == STRATEGY_ACTION_OPEN_LONG

    def test_open_sets_price_targets(self):
        """open() stores price_open, price_close, price_stop_loss."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0, 98.0], [108.0, 112.0], 94.0, 60, "test")
        assert pos.price_open == [100.0, 98.0]
        assert pos.price_close == [108.0, 112.0]
        assert pos.price_stop_loss == 94.0

    def test_open_seeds_coin_use_size(self):
        """coinUse.size set to full_position."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert pos.coinUse.size == pos.full_position

    def test_open_sets_safety_close_time(self):
        """safety_close_time = time_period * 60 * 4."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert pos.safety_close_time == 60 * 60 * 4

    def test_open_records_open_time(self):
        """open_time is set to a recent timestamp."""
        pos = self._make_pos()
        before = time.time()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        after = time.time()
        assert before <= pos.open_time <= after

    def test_open_sets_want_to_use(self):
        """coinUse.want_to_use is set to computed size."""
        pos = self._make_pos()
        # With risk_per_trade=0.01, avg=100, stop=95 → risk/avg=0.05
        # size = min(10000, 10000 * (0.01/0.05)) = min(10000, 2000) = 2000
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert pos.coinUse.want_to_use == pytest.approx(2000.0)

    def test_open_sets_action_amount(self):
        """coinUse.action_amount is set to computed size."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert pos.coinUse.action_amount == pytest.approx(2000.0)

    def test_open_rejects_if_already_open(self):
        """open() returns False if state is not POSITION_STATE_WAIT."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        # Now state is WAIT_BUY — second open should fail
        result = pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        assert result is False

    def test_open_rejects_zero_risk(self):
        """open() returns False when stop_loss == avg_price (risk=0)."""
        pos = self._make_pos()
        result = pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 100.0, 60, "test")
        assert result is False

    def test_open_rejects_negative_risk(self):
        """open() returns False when stop_loss > avg_price (negative risk for long)."""
        pos = self._make_pos()
        result = pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 105.0, 60, "test")
        assert result is False

    def test_open_size_capped_at_full_position(self):
        """size is capped at full_position when computed size exceeds it."""
        pos = self._make_pos()
        # avg=100, stop=99 → risk/avg=0.01; size = min(10000, 10000 * (0.01/0.01)) = 10000
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 99.0, 60, "test")
        assert pos.coinUse.want_to_use == pytest.approx(10000.0)

    def test_open_avg_price_from_multiple_entries(self):
        """avg_price is computed as avg of price_open list."""
        pos = self._make_pos()
        # avg([100, 98]) = 99; risk = 99 - 95 = 4; risk/avg = 4/99
        # size = min(10000, 10000 * (0.01 / (4/99)))
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0, 98.0], [110.0], 95.0, 60, "test")
        avg = (100.0 + 98.0) / 2  # 99.0
        risk = avg - 95.0  # 4.0
        expected_size = min(10000.0, 10000.0 * (0.01 / (risk / avg)))
        assert pos.coinUse.want_to_use == pytest.approx(expected_size)


# ---------------------------------------------------------------------------
# LongPosition — record_entry_fill()
# ---------------------------------------------------------------------------

class TestLongPositionRecordEntryFill:
    """Test LongPosition.record_entry_fill()."""

    def _make_open_pos(self):
        pos = LongPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        return pos

    def test_state_transitions_to_wait_sell(self):
        """After entry fill, state becomes POSITION_STATE_WAIT_SELL."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.state == POSITION_STATE_WAIT_SELL

    def test_coin_get_size_credited(self):
        """coinGet.size increases by coin_amount."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.coinGet.size == pytest.approx(10.0)

    def test_coin_use_size_debited(self):
        """coinUse.size decreases by coin_amount * price * (1 + fee)."""
        pos = self._make_open_pos()
        initial = pos.coinUse.size
        coin_amount = 10.0
        price = 100.0
        fee = 0.001
        pos.record_entry_fill(coin_amount=coin_amount, price=price)
        usd_spent = coin_amount * price * (1 + fee)
        assert pos.coinUse.size == pytest.approx(initial - usd_spent)

    def test_coin_use_used_increases(self):
        """coinUse.used increases by coin_amount * price * (1 + fee)."""
        pos = self._make_open_pos()
        coin_amount = 10.0
        price = 100.0
        fee = 0.001
        pos.record_entry_fill(coin_amount=coin_amount, price=price)
        usd_spent = coin_amount * price * (1 + fee)
        assert pos.coinUse.used == pytest.approx(usd_spent)

    def test_executed_open_records_price(self):
        """executed_open records the fill price."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.executed_open == [100.0]

    def test_executed_open_amount_records_usd(self):
        """executed_open_amount records USD amount (coin_amount * price, pre-fee)."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.executed_open_amount == [pytest.approx(1000.0)]  # 10 * 100

    def test_multiple_entry_fills_accumulate(self):
        """Multiple fills accumulate in executed_open lists."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=5.0, price=100.0)
        pos.record_entry_fill(coin_amount=5.0, price=98.0)
        assert len(pos.executed_open) == 2
        assert len(pos.executed_open_amount) == 2
        assert pos.coinGet.size == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# LongPosition — record_exit_fill()
# ---------------------------------------------------------------------------

class TestLongPositionRecordExitFill:
    """Test LongPosition.record_exit_fill()."""

    def _make_pos_after_entry(self):
        pos = LongPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        return pos

    def test_coin_get_size_debited(self):
        """coinGet.size decreases by coin_amount sold."""
        pos = self._make_pos_after_entry()
        initial_coins = pos.coinGet.size  # 10.0
        pos.record_exit_fill(coin_amount=5.0, price=110.0)
        assert pos.coinGet.size == pytest.approx(initial_coins - 5.0)

    def test_coin_use_size_credited(self):
        """coinUse.size increases by coin_amount * price * (1 - fee)."""
        pos = self._make_pos_after_entry()
        initial_usd = pos.coinUse.size
        coin_amount = 5.0
        price = 110.0
        fee = 0.001
        pos.record_exit_fill(coin_amount=coin_amount, price=price)
        usd_received = coin_amount * price * (1 - fee)
        assert pos.coinUse.size == pytest.approx(initial_usd + usd_received)

    def test_coin_use_returned_increases(self):
        """coinUse.returned increases by coin_amount * price * (1 - fee)."""
        pos = self._make_pos_after_entry()
        coin_amount = 5.0
        price = 110.0
        fee = 0.001
        pos.record_exit_fill(coin_amount=coin_amount, price=price)
        usd_received = coin_amount * price * (1 - fee)
        assert pos.coinUse.returned == pytest.approx(usd_received)

    def test_executed_close_records_price(self):
        """executed_close records the fill price."""
        pos = self._make_pos_after_entry()
        pos.record_exit_fill(coin_amount=5.0, price=110.0)
        assert pos.executed_close == [110.0]

    def test_executed_close_amount_records_coin(self):
        """executed_close_amount records coin amount sold."""
        pos = self._make_pos_after_entry()
        pos.record_exit_fill(coin_amount=5.0, price=110.0)
        assert pos.executed_close_amount == [5.0]

    def test_close_idx_incremented(self):
        """close_idx increments after each exit fill."""
        pos = self._make_pos_after_entry()
        assert pos.close_idx == 0
        pos.record_exit_fill(coin_amount=5.0, price=110.0)
        assert pos.close_idx == 1

    def test_multiple_exit_fills(self):
        """Multiple exit fills accumulate correctly."""
        pos = self._make_pos_after_entry()
        pos.record_exit_fill(coin_amount=5.0, price=110.0)
        pos.record_exit_fill(coin_amount=5.0, price=112.0)
        assert len(pos.executed_close) == 2
        assert pos.close_idx == 2


# ---------------------------------------------------------------------------
# LongPosition — avg_price_open() and avg_price_close()
# ---------------------------------------------------------------------------

class TestLongPositionAvgPrices:
    """Test LongPosition average price calculations."""

    def test_avg_price_open_single_fill(self):
        """Single fill: avg_price_open returns that price."""
        pos = LongPosition(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [1000.0]  # 10 coins * 100
        # formula: sum(amounts) / sum(amounts[i] / prices[i]) = 1000 / (1000/100) = 1000/10 = 100
        assert pos.avg_price_open() == pytest.approx(100.0)

    def test_avg_price_open_two_fills(self):
        """Two fills at different prices: weighted average."""
        pos = LongPosition(fee=0.001)
        # Fill 1: 10 coins at 100 → amount=1000
        # Fill 2: 10 coins at 90  → amount=900
        pos.executed_open = [100.0, 90.0]
        pos.executed_open_amount = [1000.0, 900.0]
        # coins1 = 1000/100 = 10, coins2 = 900/90 = 10
        # avg = (1000+900) / (10+10) = 1900/20 = 95
        assert pos.avg_price_open() == pytest.approx(95.0)

    def test_avg_price_open_no_fills_returns_zero(self):
        """No fills: avg_price_open returns 0.0."""
        pos = LongPosition(fee=0.001)
        assert pos.avg_price_open() == 0.0

    def test_avg_price_close_single_fill(self):
        """Single exit fill: avg_price_close returns that price."""
        pos = LongPosition(fee=0.001)
        pos.executed_close = [110.0]
        pos.executed_close_amount = [10.0]  # 10 coins sold
        # formula: sum(price * amount) / sum(amount) = 1100/10 = 110
        assert pos.avg_price_close() == pytest.approx(110.0)

    def test_avg_price_close_two_fills(self):
        """Two exit fills: weighted average price."""
        pos = LongPosition(fee=0.001)
        pos.executed_close = [110.0, 115.0]
        pos.executed_close_amount = [5.0, 5.0]
        # (110*5 + 115*5) / (5+5) = (550+575)/10 = 1125/10 = 112.5
        assert pos.avg_price_close() == pytest.approx(112.5)

    def test_avg_price_close_no_fills_returns_zero(self):
        """No fills: avg_price_close returns 0.0."""
        pos = LongPosition(fee=0.001)
        assert pos.avg_price_close() == 0.0


# ---------------------------------------------------------------------------
# LongPosition — direction helpers
# ---------------------------------------------------------------------------

class TestLongPositionDirectionHelpers:
    """Test LongPosition direction helper methods."""

    def test_direction_profit_returns_val_unchanged(self):
        """direction_profit returns val unchanged (long: up is profit)."""
        pos = LongPosition(fee=0.001)
        assert pos.direction_profit(5.0) == 5.0
        assert pos.direction_profit(-3.0) == -3.0

    def test_direction_loss_returns_negated_val(self):
        """direction_loss returns -val (long: down is loss)."""
        pos = LongPosition(fee=0.001)
        assert pos.direction_loss(5.0) == -5.0
        assert pos.direction_loss(-3.0) == 3.0

    def test_first_in_profit_higher_is_better(self):
        """first_in_profit: long returns a > b."""
        pos = LongPosition(fee=0.001)
        assert pos.first_in_profit(105.0, 100.0) is True
        assert pos.first_in_profit(95.0, 100.0) is False
        assert pos.first_in_profit(100.0, 100.0) is False

    def test_is_stop_loss_triggered_at_or_below(self):
        """Long stop-loss triggers when cur_price <= price_stop_loss."""
        pos = LongPosition(fee=0.001)
        pos.price_stop_loss = 95.0
        assert pos.is_stop_loss_triggered(95.0) is True
        assert pos.is_stop_loss_triggered(90.0) is True
        assert pos.is_stop_loss_triggered(96.0) is False


# ---------------------------------------------------------------------------
# LongPosition — close()
# ---------------------------------------------------------------------------

class TestLongPositionClose:
    """Test LongPosition.close() method."""

    def test_close_updates_price_close(self):
        """close() updates price_close."""
        pos = LongPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.close(STRATEGY_ACTION_CLOSE_LONG, [108.0, 112.0], 97.0, 60, "test")
        assert pos.price_close == [108.0, 112.0]

    def test_close_sets_action(self):
        """close() sets action."""
        pos = LongPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.close(STRATEGY_ACTION_CLOSE_LONG, [108.0], 97.0, 60, "test")
        assert pos.action == STRATEGY_ACTION_CLOSE_LONG

    def test_close_force_updates_stop_loss(self):
        """close() force-sets stop loss regardless of direction."""
        pos = LongPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        # Set stop loss lower (would normally be rejected by first_in_profit)
        pos.close(STRATEGY_ACTION_CLOSE_LONG, [108.0], 90.0, 60, "test")
        assert pos.price_stop_loss == 90.0

    def test_close_does_not_change_state(self):
        """close() does not change state — state is managed by fill callbacks."""
        pos = LongPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        state_before = pos.state
        pos.close(STRATEGY_ACTION_CLOSE_LONG, [108.0], 97.0, 60, "test")
        assert pos.state == state_before


# ---------------------------------------------------------------------------
# ShortPosition — instantiation
# ---------------------------------------------------------------------------

class TestShortPositionInit:
    """Test ShortPosition initialization."""

    def test_is_concrete_subclass(self):
        """ShortPosition can be instantiated."""
        pos = ShortPosition(fee=0.001)
        assert pos is not None

    def test_position_type_short(self):
        """position_type is POSITION_TYPE_SHORT."""
        pos = ShortPosition(fee=0.001)
        assert pos.position_type == POSITION_TYPE_SHORT

    def test_coin_use_is_coin(self):
        """coinUse is named 'coin'."""
        pos = ShortPosition(fee=0.001)
        assert pos.coinUse.name == "coin"

    def test_coin_get_is_usdt(self):
        """coinGet is named 'usdt'."""
        pos = ShortPosition(fee=0.001)
        assert pos.coinGet.name == "usdt"

    def test_initial_state_is_wait(self):
        """state starts as POSITION_STATE_WAIT."""
        pos = ShortPosition(fee=0.001)
        assert pos.state == POSITION_STATE_WAIT


# ---------------------------------------------------------------------------
# ShortPosition — open()
# ---------------------------------------------------------------------------

class TestShortPositionOpen:
    """Test ShortPosition.open() method."""

    def _make_pos(self):
        return ShortPosition(fee=0.001, full_position=10000.0)

    def test_open_returns_true_on_success(self):
        """open() returns True on success."""
        pos = self._make_pos()
        result = pos.open(
            STRATEGY_ACTION_OPEN_SHORT,
            price_open=[100.0],
            price_close=[95.0],
            price_stop_loss=105.0,
            time_period=60,
            action_msg="test",
        )
        assert result is True

    def test_open_state_becomes_wait_sell(self):
        """Short: after open(), state is POSITION_STATE_WAIT_SELL."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert pos.state == POSITION_STATE_WAIT_SELL

    def test_open_action_set(self):
        """After open(), action is STRATEGY_ACTION_OPEN_SHORT."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert pos.action == STRATEGY_ACTION_OPEN_SHORT

    def test_open_seeds_coin_use_size(self):
        """coinUse.size set to full_position."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert pos.coinUse.size == pos.full_position

    def test_open_sets_safety_close_time(self):
        """safety_close_time = time_period * 60 * 4."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert pos.safety_close_time == 60 * 60 * 4

    def test_open_records_open_time(self):
        """open_time is set to a recent timestamp."""
        pos = self._make_pos()
        before = time.time()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        after = time.time()
        assert before <= pos.open_time <= after

    def test_open_rejects_if_already_open(self):
        """open() returns False if state is not POSITION_STATE_WAIT."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        result = pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert result is False

    def test_open_rejects_zero_risk(self):
        """open() returns False when stop_loss == avg_price (risk=0)."""
        pos = self._make_pos()
        result = pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 100.0, 60, "test")
        assert result is False

    def test_open_rejects_negative_risk(self):
        """open() returns False when stop_loss < avg_price (wrong direction for short)."""
        pos = self._make_pos()
        # For short, risk = stop_loss - avg_price. If stop < avg, risk <= 0.
        result = pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 95.0, 60, "test")
        assert result is False

    def test_open_sets_want_to_use(self):
        """coinUse.want_to_use is set to computed size."""
        pos = self._make_pos()
        # avg=100, stop=105 → risk/avg=0.05; size = min(10000, 10000*(0.01/0.05))=2000
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert pos.coinUse.want_to_use == pytest.approx(2000.0)

    def test_open_sets_action_amount(self):
        """coinUse.action_amount equals computed size."""
        pos = self._make_pos()
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        assert pos.coinUse.action_amount == pytest.approx(2000.0)

    def test_open_size_capped_at_full_position(self):
        """size capped at full_position when computed exceeds it."""
        pos = self._make_pos()
        # avg=100, stop=101 → risk/avg=0.01; size = min(10000, 10000*(0.01/0.01))=10000
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 101.0, 60, "test")
        assert pos.coinUse.want_to_use == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# ShortPosition — record_entry_fill()
# ---------------------------------------------------------------------------

class TestShortPositionRecordEntryFill:
    """Test ShortPosition.record_entry_fill() (short entry = SELL)."""

    def _make_open_pos(self):
        pos = ShortPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        return pos

    def test_state_transitions_to_wait_buy(self):
        """After short entry fill (sell), state becomes POSITION_STATE_WAIT_BUY."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.state == POSITION_STATE_WAIT_BUY

    def test_coin_use_size_debited(self):
        """coinUse.size (coin) decreases by coin_amount sold."""
        pos = self._make_open_pos()
        initial = pos.coinUse.size
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.coinUse.size == pytest.approx(initial - 10.0)

    def test_coin_get_size_credited(self):
        """coinGet.size (usdt) increases by coin_amount * price * (1 - fee)."""
        pos = self._make_open_pos()
        coin_amount = 10.0
        price = 100.0
        fee = 0.001
        pos.record_entry_fill(coin_amount=coin_amount, price=price)
        usd_received = coin_amount * price * (1 - fee)
        assert pos.coinGet.size == pytest.approx(usd_received)

    def test_executed_open_records_price(self):
        """executed_open records the fill price."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.executed_open == [100.0]

    def test_executed_open_amount_records_usd(self):
        """executed_open_amount records USD amount (coin_amount * price)."""
        pos = self._make_open_pos()
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        assert pos.executed_open_amount == [pytest.approx(1000.0)]


# ---------------------------------------------------------------------------
# ShortPosition — record_exit_fill()
# ---------------------------------------------------------------------------

class TestShortPositionRecordExitFill:
    """Test ShortPosition.record_exit_fill() (short exit = BUY back)."""

    def _make_pos_after_entry(self):
        pos = ShortPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        pos.record_entry_fill(coin_amount=10.0, price=100.0)
        return pos

    def test_coin_get_size_debited(self):
        """coinGet.size (usdt) decreases by coin_amount * price * (1 + fee)."""
        pos = self._make_pos_after_entry()
        initial_usd = pos.coinGet.size
        coin_amount = 5.0
        price = 95.0
        fee = 0.001
        pos.record_exit_fill(coin_amount=coin_amount, price=price)
        usd_spent = coin_amount * price * (1 + fee)
        assert pos.coinGet.size == pytest.approx(initial_usd - usd_spent)

    def test_coin_use_size_credited(self):
        """coinUse.size (coin) increases by coin_amount bought back."""
        pos = self._make_pos_after_entry()
        initial_coins = pos.coinUse.size
        pos.record_exit_fill(coin_amount=5.0, price=95.0)
        assert pos.coinUse.size == pytest.approx(initial_coins + 5.0)

    def test_executed_close_records_price(self):
        """executed_close records the fill price."""
        pos = self._make_pos_after_entry()
        pos.record_exit_fill(coin_amount=5.0, price=95.0)
        assert pos.executed_close == [95.0]

    def test_executed_close_amount_records_coin(self):
        """executed_close_amount records coin amount bought back."""
        pos = self._make_pos_after_entry()
        pos.record_exit_fill(coin_amount=5.0, price=95.0)
        assert pos.executed_close_amount == [5.0]

    def test_close_idx_incremented(self):
        """close_idx increments after each exit fill."""
        pos = self._make_pos_after_entry()
        assert pos.close_idx == 0
        pos.record_exit_fill(coin_amount=5.0, price=95.0)
        assert pos.close_idx == 1


# ---------------------------------------------------------------------------
# ShortPosition — avg_price_open() and avg_price_close()
# ---------------------------------------------------------------------------

class TestShortPositionAvgPrices:
    """Test ShortPosition average price calculations (same formula as long)."""

    def test_avg_price_open_single_fill(self):
        """Single fill: avg_price_open returns that price."""
        pos = ShortPosition(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [1000.0]
        assert pos.avg_price_open() == pytest.approx(100.0)

    def test_avg_price_open_two_fills(self):
        """Two fills at different prices: weighted average."""
        pos = ShortPosition(fee=0.001)
        pos.executed_open = [100.0, 90.0]
        pos.executed_open_amount = [1000.0, 900.0]
        assert pos.avg_price_open() == pytest.approx(95.0)

    def test_avg_price_open_no_fills_returns_zero(self):
        """No fills: avg_price_open returns 0.0."""
        pos = ShortPosition(fee=0.001)
        assert pos.avg_price_open() == 0.0

    def test_avg_price_close_single_fill(self):
        """Single exit fill: avg_price_close returns that price."""
        pos = ShortPosition(fee=0.001)
        pos.executed_close = [90.0]
        pos.executed_close_amount = [10.0]
        assert pos.avg_price_close() == pytest.approx(90.0)

    def test_avg_price_close_two_fills(self):
        """Two exit fills: weighted average price."""
        pos = ShortPosition(fee=0.001)
        pos.executed_close = [90.0, 85.0]
        pos.executed_close_amount = [5.0, 5.0]
        assert pos.avg_price_close() == pytest.approx(87.5)

    def test_avg_price_close_no_fills_returns_zero(self):
        """No fills: avg_price_close returns 0.0."""
        pos = ShortPosition(fee=0.001)
        assert pos.avg_price_close() == 0.0


# ---------------------------------------------------------------------------
# ShortPosition — direction helpers
# ---------------------------------------------------------------------------

class TestShortPositionDirectionHelpers:
    """Test ShortPosition direction helper methods."""

    def test_direction_profit_returns_negated_val(self):
        """direction_profit returns -val (short: down is profit)."""
        pos = ShortPosition(fee=0.001)
        assert pos.direction_profit(5.0) == -5.0
        assert pos.direction_profit(-3.0) == 3.0

    def test_direction_loss_returns_val_unchanged(self):
        """direction_loss returns val unchanged (short: up is loss)."""
        pos = ShortPosition(fee=0.001)
        assert pos.direction_loss(5.0) == 5.0
        assert pos.direction_loss(-3.0) == -3.0

    def test_first_in_profit_lower_is_better(self):
        """first_in_profit: short returns a < b."""
        pos = ShortPosition(fee=0.001)
        assert pos.first_in_profit(95.0, 100.0) is True
        assert pos.first_in_profit(105.0, 100.0) is False
        assert pos.first_in_profit(100.0, 100.0) is False

    def test_is_stop_loss_triggered_at_or_above(self):
        """Short stop-loss triggers when cur_price >= price_stop_loss."""
        pos = ShortPosition(fee=0.001)
        pos.price_stop_loss = 105.0
        assert pos.is_stop_loss_triggered(105.0) is True
        assert pos.is_stop_loss_triggered(110.0) is True
        assert pos.is_stop_loss_triggered(104.0) is False


# ---------------------------------------------------------------------------
# ShortPosition — close()
# ---------------------------------------------------------------------------

class TestShortPositionClose:
    """Test ShortPosition.close() method."""

    def test_close_updates_price_close(self):
        """close() updates price_close."""
        pos = ShortPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.close(STRATEGY_ACTION_CLOSE_SHORT, [93.0, 90.0], 103.0, 60, "test")
        assert pos.price_close == [93.0, 90.0]

    def test_close_sets_action(self):
        """close() sets action."""
        pos = ShortPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.close(STRATEGY_ACTION_CLOSE_SHORT, [93.0], 103.0, 60, "test")
        assert pos.action == STRATEGY_ACTION_CLOSE_SHORT

    def test_close_force_updates_stop_loss(self):
        """close() force-sets stop loss regardless of direction."""
        pos = ShortPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        # Set stop loss higher (would normally be rejected by first_in_profit for short)
        pos.close(STRATEGY_ACTION_CLOSE_SHORT, [93.0], 110.0, 60, "test")
        assert pos.price_stop_loss == 110.0

    def test_close_does_not_change_state(self):
        """close() does not change state."""
        pos = ShortPosition(fee=0.001)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [95.0], 105.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        state_before = pos.state
        pos.close(STRATEGY_ACTION_CLOSE_SHORT, [93.0], 103.0, 60, "test")
        assert pos.state == state_before


# ---------------------------------------------------------------------------
# Integration: full round-trip through finalize()
# ---------------------------------------------------------------------------

class TestFullRoundTrip:
    """Integration tests for full position lifecycle."""

    def test_long_full_lifecycle_pnl(self):
        """Long: open → entry fill → exit fill → finalize produces correct P&L."""
        pos = LongPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(coin_amount=10.0, price=100.0)  # buy 10 @ 100
        pos.record_exit_fill(coin_amount=10.0, price=110.0)   # sell 10 @ 110

        revenue_pct, revenue_abs = pos.finalize()

        # avg_open=100, avg_close=110; pct = (110/100 - 1) - 2*0.001 = 0.1 - 0.002 = 0.098
        assert revenue_pct == pytest.approx(0.098)
        assert revenue_abs == pytest.approx(980.0)

    def test_long_state_after_finalize(self):
        """After finalize, state resets to POSITION_STATE_WAIT."""
        pos = LongPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.record_exit_fill(10.0, 110.0)
        pos.finalize()
        assert pos.state == POSITION_STATE_WAIT

    def test_short_full_lifecycle_pnl(self):
        """Short: open → entry fill → exit fill → finalize produces correct P&L."""
        pos = ShortPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [90.0], 105.0, 60, "test")
        pos.record_entry_fill(coin_amount=10.0, price=100.0)  # sell 10 @ 100
        pos.record_exit_fill(coin_amount=10.0, price=90.0)    # buy back 10 @ 90

        revenue_pct, revenue_abs = pos.finalize()

        # avg_open=100, avg_close=90; pct = (1 - 90/100) - 2*0.001 = 0.1 - 0.002 = 0.098
        assert revenue_pct == pytest.approx(0.098)
        assert revenue_abs == pytest.approx(980.0)

    def test_short_state_after_finalize(self):
        """After finalize, short state resets to POSITION_STATE_WAIT."""
        pos = ShortPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_SHORT, [100.0], [90.0], 105.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.record_exit_fill(10.0, 90.0)
        pos.finalize()
        assert pos.state == POSITION_STATE_WAIT

    def test_long_reopen_after_finalize(self):
        """After finalize, position can be reopened."""
        pos = LongPosition(fee=0.001, full_position=10000.0)
        pos.open(STRATEGY_ACTION_OPEN_LONG, [100.0], [110.0], 95.0, 60, "test")
        pos.record_entry_fill(10.0, 100.0)
        pos.record_exit_fill(10.0, 110.0)
        pos.finalize()

        # Now reopen
        result = pos.open(STRATEGY_ACTION_OPEN_LONG, [105.0], [115.0], 100.0, 60, "test")
        assert result is True
        assert pos.state == POSITION_STATE_WAIT_BUY
