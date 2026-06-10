"""Tests for BasePosition abstract state machine (Phase 06, Task 02)."""

import time
import pytest
from constants import (
    POSITION_STATE_WAIT,
    POSITION_STATE_WAIT_BUY,
    POSITION_STATE_WAIT_SELL,
    POSITION_STATE_WAIT_SAFETY_BUY,
    POSITION_STATE_WAIT_SAFETY_SELL,
    POSITION_TYPE_UNKNOWN,
    POSITION_TYPE_LONG,
    POSITION_TYPE_SHORT,
    STRATEGY_ACTION_NOTHING,
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_OPEN_SHORT,
)
from position.base_position import BasePosition
from position.coin import Coin


# ---------------------------------------------------------------------------
# Minimal concrete subclasses for testing
# ---------------------------------------------------------------------------

class ConcreteLong(BasePosition):
    """Minimal long position for testing BasePosition."""

    def __init__(self, fee: float, thread_num: int = 0, full_position: float = 10000.0):
        super().__init__(fee, thread_num, full_position)
        self.position_type = POSITION_TYPE_LONG
        self.coinUse = Coin("usdt")
        self.coinGet = Coin("coin")

    def open(self, strategy_action, price_open, price_close, price_stop_loss, time_period, action_msg) -> bool:
        self.price_open = list(price_open)
        self.price_close = list(price_close)
        self.price_stop_loss = price_stop_loss
        self.safety_close_time = time_period
        self.open_time = time.time()
        self.state = POSITION_STATE_WAIT_BUY
        self.action = strategy_action
        return True

    def close(self, strategy_action, price_close, price_stop_loss, time_period, action_msg) -> None:
        self.price_close = list(price_close)
        self.price_stop_loss = price_stop_loss
        self.state = POSITION_STATE_WAIT_SELL
        self.action = strategy_action

    def record_entry_fill(self, coin_amount: float, price: float) -> None:
        self.executed_open.append(price)
        self.executed_open_amount.append(coin_amount * price)

    def record_exit_fill(self, coin_amount: float, price: float) -> None:
        self.executed_close.append(price)
        self.executed_close_amount.append(coin_amount)

    def avg_price_open(self) -> float:
        if not self.executed_open:
            return 0.0
        return sum(self.executed_open) / len(self.executed_open)

    def avg_price_close(self) -> float:
        if not self.executed_close:
            return 0.0
        return sum(self.executed_close) / len(self.executed_close)

    def direction_profit(self, val: float) -> float:
        return val  # Long: higher is profit

    def direction_loss(self, val: float) -> float:
        return -val  # Long: lower is loss

    def first_in_profit(self, a: float, b: float) -> bool:
        return a > b  # Long: higher stop-loss is better

    def is_stop_loss_triggered(self, cur_price: float) -> bool:
        return cur_price <= self.price_stop_loss


class ConcreteShort(BasePosition):
    """Minimal short position for testing BasePosition."""

    def __init__(self, fee: float, thread_num: int = 0, full_position: float = 10000.0):
        super().__init__(fee, thread_num, full_position)
        self.position_type = POSITION_TYPE_SHORT
        self.coinUse = Coin("coin")
        self.coinGet = Coin("usdt")

    def open(self, strategy_action, price_open, price_close, price_stop_loss, time_period, action_msg) -> bool:
        self.price_open = list(price_open)
        self.price_close = list(price_close)
        self.price_stop_loss = price_stop_loss
        self.safety_close_time = time_period
        self.open_time = time.time()
        self.state = POSITION_STATE_WAIT_SELL
        self.action = strategy_action
        return True

    def close(self, strategy_action, price_close, price_stop_loss, time_period, action_msg) -> None:
        self.price_close = list(price_close)
        self.price_stop_loss = price_stop_loss
        self.state = POSITION_STATE_WAIT_BUY
        self.action = strategy_action

    def record_entry_fill(self, coin_amount: float, price: float) -> None:
        self.executed_open.append(price)
        self.executed_open_amount.append(coin_amount)

    def record_exit_fill(self, coin_amount: float, price: float) -> None:
        self.executed_close.append(price)
        self.executed_close_amount.append(coin_amount * price)

    def avg_price_open(self) -> float:
        if not self.executed_open:
            return 0.0
        return sum(self.executed_open) / len(self.executed_open)

    def avg_price_close(self) -> float:
        if not self.executed_close:
            return 0.0
        return sum(self.executed_close) / len(self.executed_close)

    def direction_profit(self, val: float) -> float:
        return -val  # Short: lower is profit

    def direction_loss(self, val: float) -> float:
        return val  # Short: higher is loss

    def first_in_profit(self, a: float, b: float) -> bool:
        return a < b  # Short: lower stop-loss (buy-back) is better

    def is_stop_loss_triggered(self, cur_price: float) -> bool:
        return cur_price >= self.price_stop_loss


# ---------------------------------------------------------------------------
# Instantiation and initial state
# ---------------------------------------------------------------------------

class TestBasePositionInit:
    """Test BasePosition initialization."""

    def test_cannot_instantiate_directly(self):
        """BasePosition is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePosition(fee=0.001)

    def test_long_init_defaults(self):
        """ConcreteLong initializes with correct defaults."""
        pos = ConcreteLong(fee=0.001)
        assert pos.state == POSITION_STATE_WAIT
        assert pos.position_type == POSITION_TYPE_LONG
        assert pos.fee == 0.001
        assert pos.full_position == 10000.0
        assert pos.risk_per_trade == 0.01
        assert pos.safety_close_time == 0
        assert pos.close_idx == 0
        assert pos.action == STRATEGY_ACTION_NOTHING
        assert pos.price_stop_loss == 0.0
        assert pos.open_time == 0.0

    def test_short_init_defaults(self):
        """ConcreteShort initializes with correct defaults."""
        pos = ConcreteShort(fee=0.002)
        assert pos.state == POSITION_STATE_WAIT
        assert pos.position_type == POSITION_TYPE_SHORT
        assert pos.fee == 0.002
        assert pos.full_position == 10000.0
        assert pos.risk_per_trade == 0.01
        assert pos.action == STRATEGY_ACTION_NOTHING

    def test_fee_required(self):
        """fee is required (no default)."""
        with pytest.raises(TypeError):
            ConcreteLong()

    def test_full_position_custom(self):
        """full_position can be customized."""
        pos = ConcreteLong(fee=0.001, full_position=5000.0)
        assert pos.full_position == 5000.0

    def test_thread_num_default(self):
        """thread_num defaults to 0."""
        pos = ConcreteLong(fee=0.001)
        assert pos.thread_num == 0

    def test_thread_num_custom(self):
        """thread_num can be set."""
        pos = ConcreteLong(fee=0.001, thread_num=3)
        assert pos.thread_num == 3

    def test_lists_initialized_empty(self):
        """price_open, price_close, executed_* arrays start empty."""
        pos = ConcreteLong(fee=0.001)
        assert pos.price_open == []
        assert pos.price_close == []
        assert pos.executed_open == []
        assert pos.executed_open_amount == []
        assert pos.executed_close == []
        assert pos.executed_close_amount == []

    def test_coin_use_and_get_initialized(self):
        """coinUse and coinGet are Coin instances."""
        pos = ConcreteLong(fee=0.001)
        assert isinstance(pos.coinUse, Coin)
        assert isinstance(pos.coinGet, Coin)


# ---------------------------------------------------------------------------
# set_stop_loss
# ---------------------------------------------------------------------------

class TestSetStopLoss:
    """Test set_stop_loss one-directional enforcement."""

    def test_long_stop_loss_set_initially(self):
        """Long: first set_stop_loss with force=True sets value."""
        pos = ConcreteLong(fee=0.001)
        result = pos.set_stop_loss(100.0, "test", force=True)
        assert result is True
        assert pos.price_stop_loss == 100.0

    def test_long_stop_loss_higher_accepted(self):
        """Long: higher stop (more profitable) is accepted."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 100.0
        result = pos.set_stop_loss(105.0, "test")
        assert result is True
        assert pos.price_stop_loss == 105.0

    def test_long_stop_loss_lower_rejected(self):
        """Long: lower stop (less profitable for long) is silently ignored."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 100.0
        result = pos.set_stop_loss(95.0, "test")
        assert result is False
        assert pos.price_stop_loss == 100.0  # unchanged

    def test_long_stop_loss_lower_with_force(self):
        """Long: lower stop with force=True overrides."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 100.0
        result = pos.set_stop_loss(90.0, "test", force=True)
        assert result is True
        assert pos.price_stop_loss == 90.0

    def test_long_stop_loss_same_price_rejected(self):
        """Long: same price is not strictly more profitable, rejected."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 100.0
        result = pos.set_stop_loss(100.0, "test")
        assert result is False

    def test_short_stop_loss_lower_accepted(self):
        """Short: lower stop (better buy-back price) is accepted."""
        pos = ConcreteShort(fee=0.001)
        pos.price_stop_loss = 200.0
        result = pos.set_stop_loss(190.0, "test")
        assert result is True
        assert pos.price_stop_loss == 190.0

    def test_short_stop_loss_higher_rejected(self):
        """Short: higher stop is silently ignored."""
        pos = ConcreteShort(fee=0.001)
        pos.price_stop_loss = 200.0
        result = pos.set_stop_loss(210.0, "test")
        assert result is False
        assert pos.price_stop_loss == 200.0  # unchanged

    def test_short_stop_loss_higher_with_force(self):
        """Short: higher stop with force=True overrides."""
        pos = ConcreteShort(fee=0.001)
        pos.price_stop_loss = 200.0
        result = pos.set_stop_loss(220.0, "test", force=True)
        assert result is True
        assert pos.price_stop_loss == 220.0

    def test_set_stop_loss_no_exception_on_reject(self):
        """Rejected stop-loss does not raise exception."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 100.0
        # Should not raise
        pos.set_stop_loss(50.0, "test")


# ---------------------------------------------------------------------------
# check_stop_open
# ---------------------------------------------------------------------------

class TestCheckStopOpen:
    """Test check_stop_open (price drifted too far from entry target)."""

    def test_long_price_above_threshold_ok(self):
        """Long: price above threshold — fill still possible."""
        pos = ConcreteLong(fee=0.001)
        pos.price_open = [100.0]
        # price dropped only 0.1% — less than 4*0.001=0.4%
        assert pos.check_stop_open(99.9) is False

    def test_long_price_below_threshold_cancel(self):
        """Long: price dropped > 4*fee below entry — cancel."""
        pos = ConcreteLong(fee=0.001)
        pos.price_open = [100.0]
        # 4*fee = 0.4%, so threshold = 99.6; price 99.5 is below
        assert pos.check_stop_open(99.5) is True

    def test_long_price_at_threshold_not_cancel(self):
        """Long: price exactly at threshold — not yet cancel."""
        pos = ConcreteLong(fee=0.001)
        pos.price_open = [100.0]
        threshold = 100.0 * (1 - 4 * 0.001)
        assert pos.check_stop_open(threshold) is False

    def test_short_price_below_threshold_ok(self):
        """Short: price below threshold — fill still possible."""
        pos = ConcreteShort(fee=0.001)
        pos.price_open = [100.0]
        # price rose only 0.1% — less than 4*0.001=0.4%
        assert pos.check_stop_open(100.1) is False

    def test_short_price_above_threshold_cancel(self):
        """Short: price rose > 4*fee above entry — cancel."""
        pos = ConcreteShort(fee=0.001)
        pos.price_open = [100.0]
        # 4*fee = 0.4%, threshold = 100.4; price 100.5 exceeds
        assert pos.check_stop_open(100.5) is True

    def test_short_price_at_threshold_not_cancel(self):
        """Short: price exactly at threshold — not yet cancel."""
        pos = ConcreteShort(fee=0.001)
        pos.price_open = [100.0]
        threshold = 100.0 * (1 + 4 * 0.001)
        assert pos.check_stop_open(threshold) is False

    def test_long_price_above_entry_ok(self):
        """Long: price above entry is fine (not drifting against)."""
        pos = ConcreteLong(fee=0.001)
        pos.price_open = [100.0]
        assert pos.check_stop_open(110.0) is False


# ---------------------------------------------------------------------------
# close_by_time
# ---------------------------------------------------------------------------

class TestCloseByTime:
    """Test close_by_time safety timeout."""

    def test_close_by_time_within_period(self):
        """Returns False when within safety_close_time."""
        pos = ConcreteLong(fee=0.001)
        pos.open_time = 1000.0
        pos.safety_close_time = 3600  # 1 hour
        # current time 500s after open — within timeout
        assert pos.close_by_time(1500.0) is False

    def test_close_by_time_exceeded(self):
        """Returns True when elapsed > safety_close_time."""
        pos = ConcreteLong(fee=0.001)
        pos.open_time = 1000.0
        pos.safety_close_time = 3600
        # current time 4000s after open — exceeds 3600
        assert pos.close_by_time(5000.0) is True

    def test_close_by_time_exactly_at_limit(self):
        """Returns False when elapsed equals safety_close_time (not strictly greater)."""
        pos = ConcreteLong(fee=0.001)
        pos.open_time = 1000.0
        pos.safety_close_time = 3600
        assert pos.close_by_time(4600.0) is False

    def test_close_by_time_zero_timeout_never_triggers(self):
        """Returns False when safety_close_time is 0 (disabled)."""
        pos = ConcreteLong(fee=0.001)
        pos.open_time = 1000.0
        pos.safety_close_time = 0
        # close_by_time(1000 + epsilon) = 0 > 0 -> True?
        # Actually 0 > 0 is False — safety_close_time=0 should not trigger
        assert pos.close_by_time(1001.0) is False


# ---------------------------------------------------------------------------
# get_action
# ---------------------------------------------------------------------------

class TestGetAction:
    """Test get_action returns current action."""

    def test_get_action_initial(self):
        """Returns STRATEGY_ACTION_NOTHING initially."""
        pos = ConcreteLong(fee=0.001)
        assert pos.get_action() == STRATEGY_ACTION_NOTHING

    def test_get_action_after_set(self):
        """Returns updated action."""
        pos = ConcreteLong(fee=0.001)
        pos.action = STRATEGY_ACTION_OPEN_LONG
        assert pos.get_action() == STRATEGY_ACTION_OPEN_LONG


# ---------------------------------------------------------------------------
# get_action_amount
# ---------------------------------------------------------------------------

class TestGetActionAmount:
    """Test get_action_amount coin quantity calculation."""

    def test_long_wait_buy_converts_usd_to_coins(self):
        """Long WAIT_BUY: coinUse.action_amount / price."""
        pos = ConcreteLong(fee=0.001)
        pos.state = POSITION_STATE_WAIT_BUY
        pos.coinUse.action_amount = 1000.0
        amount = pos.get_action_amount(price=50.0)
        assert amount == pytest.approx(20.0)  # 1000/50

    def test_long_wait_sell_returns_coins_directly(self):
        """Long WAIT_SELL: coinGet.action_amount (already coins)."""
        pos = ConcreteLong(fee=0.001)
        pos.state = POSITION_STATE_WAIT_SELL
        pos.coinGet.action_amount = 25.5
        amount = pos.get_action_amount(price=50.0)
        assert amount == pytest.approx(25.5)

    def test_short_wait_sell_returns_coins_directly(self):
        """Short WAIT_SELL: coinUse.action_amount (already coins)."""
        pos = ConcreteShort(fee=0.001)
        pos.state = POSITION_STATE_WAIT_SELL
        pos.coinUse.action_amount = 10.0
        amount = pos.get_action_amount(price=100.0)
        assert amount == pytest.approx(10.0)

    def test_short_wait_buy_converts_usd_to_coins(self):
        """Short WAIT_BUY (exit): coinGet.action_amount / price."""
        pos = ConcreteShort(fee=0.001)
        pos.state = POSITION_STATE_WAIT_BUY
        pos.coinGet.action_amount = 500.0
        amount = pos.get_action_amount(price=50.0)
        assert amount == pytest.approx(10.0)  # 500/50

    def test_long_wait_safety_sell_falls_through(self):
        """WAIT_SAFETY_SELL: still returns coinGet.action_amount for long."""
        pos = ConcreteLong(fee=0.001)
        pos.state = POSITION_STATE_WAIT_SAFETY_SELL
        pos.coinGet.action_amount = 7.0
        amount = pos.get_action_amount(price=100.0)
        assert amount == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# get_target
# ---------------------------------------------------------------------------

class TestGetTarget:
    """Test get_target returns next exit price target."""

    def test_get_target_first_target(self):
        """Returns price_close[0] when close_idx=0."""
        pos = ConcreteLong(fee=0.001)
        pos.price_close = [110.0, 115.0, 120.0]
        pos.close_idx = 0
        assert pos.get_target() == 110.0

    def test_get_target_advances_with_close_idx(self):
        """Returns price_close[close_idx]."""
        pos = ConcreteLong(fee=0.001)
        pos.price_close = [110.0, 115.0, 120.0]
        pos.close_idx = 2
        assert pos.get_target() == 120.0

    def test_get_target_exhausted_long_transitions_state(self):
        """Long: when close_idx >= len(price_close), transitions to WAIT_SAFETY_SELL."""
        pos = ConcreteLong(fee=0.001)
        pos.price_close = [110.0, 115.0]
        pos.close_idx = 2  # exhausted
        pos.state = POSITION_STATE_WAIT_SELL
        target = pos.get_target()
        assert target == 115.0  # last price
        assert pos.state == POSITION_STATE_WAIT_SAFETY_SELL

    def test_get_target_exhausted_short_transitions_state(self):
        """Short: when close_idx >= len(price_close), transitions to WAIT_SAFETY_BUY."""
        pos = ConcreteShort(fee=0.001)
        pos.price_close = [90.0, 85.0]
        pos.close_idx = 2  # exhausted
        pos.state = POSITION_STATE_WAIT_BUY
        target = pos.get_target()
        assert target == 85.0  # last price
        assert pos.state == POSITION_STATE_WAIT_SAFETY_BUY


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

class TestFinalize:
    """Test finalize P&L computation and state reset."""

    def test_finalize_long_positive_pnl(self):
        """Long: bought at 100, sold at 105 — positive return."""
        pos = ConcreteLong(fee=0.001)
        pos.price_open = [100.0]
        pos.price_close = [105.0]
        pos.executed_open = [100.0]
        pos.executed_open_amount = [10000.0]
        pos.executed_close = [105.0]
        pos.executed_close_amount = [100.0]
        pos.full_position = 10000.0

        revenue_pct, revenue_abs = pos.finalize()

        # revenue_pct = (105/100 - 1) - 2*0.001 = 0.05 - 0.002 = 0.048
        assert revenue_pct == pytest.approx(0.048)
        assert revenue_abs == pytest.approx(480.0)

    def test_finalize_long_negative_pnl(self):
        """Long: bought at 100, sold at 95 — negative return."""
        pos = ConcreteLong(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [10000.0]
        pos.executed_close = [95.0]
        pos.executed_close_amount = [100.0]
        pos.full_position = 10000.0

        revenue_pct, revenue_abs = pos.finalize()

        # revenue_pct = (95/100 - 1) - 2*0.001 = -0.05 - 0.002 = -0.052
        assert revenue_pct == pytest.approx(-0.052)
        assert revenue_abs == pytest.approx(-520.0)

    def test_finalize_short_positive_pnl(self):
        """Short: sold at 100, bought back at 95 — positive return."""
        pos = ConcreteShort(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [100.0]
        pos.executed_close = [95.0]
        pos.executed_close_amount = [10000.0]
        pos.full_position = 10000.0

        revenue_pct, revenue_abs = pos.finalize()

        # revenue_pct = (1 - 95/100) - 2*0.001 = 0.05 - 0.002 = 0.048
        assert revenue_pct == pytest.approx(0.048)
        assert revenue_abs == pytest.approx(480.0)

    def test_finalize_short_negative_pnl(self):
        """Short: sold at 100, bought back at 105 — negative return."""
        pos = ConcreteShort(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [100.0]
        pos.executed_close = [105.0]
        pos.executed_close_amount = [10000.0]
        pos.full_position = 10000.0

        revenue_pct, revenue_abs = pos.finalize()

        # revenue_pct = (1 - 105/100) - 2*0.001 = -0.05 - 0.002 = -0.052
        assert revenue_pct == pytest.approx(-0.052)
        assert revenue_abs == pytest.approx(-520.0)

    def test_finalize_resets_state(self):
        """finalize() resets all state to idle."""
        pos = ConcreteLong(fee=0.001)
        pos.price_open = [100.0]
        pos.price_close = [105.0]
        pos.executed_open = [100.0]
        pos.executed_open_amount = [10000.0]
        pos.executed_close = [105.0]
        pos.executed_close_amount = [100.0]
        pos.state = POSITION_STATE_WAIT_SELL
        pos.action = STRATEGY_ACTION_OPEN_LONG
        pos.close_idx = 1

        pos.finalize()

        assert pos.state == POSITION_STATE_WAIT
        assert pos.action == STRATEGY_ACTION_NOTHING
        assert pos.close_idx == 0
        assert pos.price_open == []
        assert pos.price_close == []
        assert pos.executed_open == []
        assert pos.executed_close == []

    def test_finalize_resets_coins(self):
        """finalize() calls reset() on coinUse and coinGet."""
        pos = ConcreteLong(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [10000.0]
        pos.executed_close = [105.0]
        pos.executed_close_amount = [100.0]
        pos.coinUse.size = 1000.0
        pos.coinGet.size = 50.0

        pos.finalize()

        assert pos.coinUse.size == 0.0
        assert pos.coinGet.size == 0.0

    def test_finalize_returns_tuple(self):
        """finalize() returns (revenue_pct, revenue_abs)."""
        pos = ConcreteLong(fee=0.001)
        pos.executed_open = [100.0]
        pos.executed_open_amount = [10000.0]
        pos.executed_close = [100.0]
        pos.executed_close_amount = [100.0]
        result = pos.finalize()
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# to_dict / from_dict
# ---------------------------------------------------------------------------

class TestSerialization:
    """Test to_dict / from_dict roundtrip."""

    def _make_active_long(self) -> ConcreteLong:
        pos = ConcreteLong(fee=0.001, thread_num=1, full_position=5000.0)
        pos.state = POSITION_STATE_WAIT_SELL
        pos.action = STRATEGY_ACTION_OPEN_LONG
        pos.price_open = [100.0, 98.0]
        pos.price_close = [110.0, 115.0]
        pos.price_stop_loss = 95.0
        pos.safety_close_time = 3600
        pos.close_idx = 1
        pos.open_time = 1717000000.0
        pos.executed_open = [100.0]
        pos.executed_open_amount = [5000.0]
        pos.executed_close = [110.0]
        pos.executed_close_amount = [50.0]
        pos.coinUse.size = 5000.0
        pos.coinGet.size = 50.0
        return pos

    def test_to_dict_returns_dict(self):
        """to_dict returns a dict."""
        pos = self._make_active_long()
        d = pos.to_dict()
        assert isinstance(d, dict)

    def test_roundtrip_primitives(self):
        """to_dict/from_dict roundtrip preserves primitive state."""
        pos = self._make_active_long()
        d = pos.to_dict()

        pos2 = ConcreteLong(fee=0.001)
        pos2.from_dict(d)

        assert pos2.state == pos.state
        assert pos2.action == pos.action
        assert pos2.price_open == pos.price_open
        assert pos2.price_close == pos.price_close
        assert pos2.price_stop_loss == pos.price_stop_loss
        assert pos2.safety_close_time == pos.safety_close_time
        assert pos2.close_idx == pos.close_idx
        assert pos2.open_time == pos.open_time
        assert pos2.executed_open == pos.executed_open
        assert pos2.executed_open_amount == pos.executed_open_amount
        assert pos2.executed_close == pos.executed_close
        assert pos2.executed_close_amount == pos.executed_close_amount

    def test_roundtrip_coins(self):
        """to_dict/from_dict preserves Coin state."""
        pos = self._make_active_long()
        d = pos.to_dict()

        pos2 = ConcreteLong(fee=0.001)
        pos2.from_dict(d)

        assert pos2.coinUse.size == pos.coinUse.size
        assert pos2.coinGet.size == pos.coinGet.size

    def test_roundtrip_fee_and_position(self):
        """to_dict/from_dict preserves fee and full_position."""
        pos = self._make_active_long()
        d = pos.to_dict()

        pos2 = ConcreteLong(fee=0.001)
        pos2.from_dict(d)

        assert pos2.fee == pos.fee
        assert pos2.full_position == pos.full_position


# ---------------------------------------------------------------------------
# is_stop_loss_triggered (abstract — tested via concrete)
# ---------------------------------------------------------------------------

class TestStopLossTrigger:
    """Test is_stop_loss_triggered direction logic."""

    def test_long_triggered_when_price_at_or_below(self):
        """Long stop-loss triggers when cur_price <= price_stop_loss."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 95.0
        assert pos.is_stop_loss_triggered(95.0) is True
        assert pos.is_stop_loss_triggered(90.0) is True

    def test_long_not_triggered_when_above(self):
        """Long stop-loss does not trigger above stop."""
        pos = ConcreteLong(fee=0.001)
        pos.price_stop_loss = 95.0
        assert pos.is_stop_loss_triggered(96.0) is False
        assert pos.is_stop_loss_triggered(100.0) is False

    def test_short_triggered_when_price_at_or_above(self):
        """Short stop-loss triggers when cur_price >= price_stop_loss."""
        pos = ConcreteShort(fee=0.001)
        pos.price_stop_loss = 105.0
        assert pos.is_stop_loss_triggered(105.0) is True
        assert pos.is_stop_loss_triggered(110.0) is True

    def test_short_not_triggered_when_below(self):
        """Short stop-loss does not trigger below stop."""
        pos = ConcreteShort(fee=0.001)
        pos.price_stop_loss = 105.0
        assert pos.is_stop_loss_triggered(104.0) is False
        assert pos.is_stop_loss_triggered(95.0) is False
