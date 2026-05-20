import pytest
from simple_trader.position.base_position import BasePosition
from simple_trader.infrastructure.constants import PositionState, StrategyAction


class ConcretePosition(BasePosition):
    def _wait_buy_state(self):
        return PositionState.WAIT_BUY

    def _wait_sell_state(self):
        return PositionState.WAIT_SELL

    def _wait_safety_buy_state(self):
        return PositionState.WAIT_SAFETY_BUY

    def _open_action(self):
        return StrategyAction.OPEN_LONG

    def _close_action(self):
        return StrategyAction.CLOSE_LONG

    def _is_tighter_stop(self, price: float) -> bool:
        return price > self.price_stop_loss

    def finalize(self, entry_price: float, exit_price: float, fee: float):
        revenue_pct = (exit_price - entry_price) / entry_price - 2 * fee
        return revenue_pct, revenue_pct * entry_price


def test_position_starts_in_wait_state():
    pos = ConcretePosition(coin_use="USDT", coin_get="LINK", full_position=1000.0, fee=0.001)
    assert pos.state == PositionState.WAIT


def test_position_action_starts_as_nothing():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    assert pos.action == StrategyAction.NOTHING


def test_open_transitions_to_wait_buy():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(price_open=15.0, price_close=16.5, price_stop=14.5)
    assert pos.state == PositionState.WAIT_BUY


def test_open_sets_action_to_open_long():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 16.5, 14.5)
    assert pos.action == StrategyAction.OPEN_LONG


def test_open_stores_prices():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 16.5, 14.5)
    assert pos.price_open == [15.0]
    assert pos.price_close == 16.5
    assert pos.price_stop_loss == 14.5


def test_open_from_non_wait_state_raises():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 16.5, 14.5)
    with pytest.raises(AssertionError):
        pos.open(15.0, 16.5, 14.5)


def test_set_stop_loss_tighter_updates_price():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 16.5, 14.5)
    pos.set_stop_loss(15.0)
    assert pos.price_stop_loss == 15.0


def test_set_stop_loss_looser_ignored():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 16.5, 14.5)
    pos.set_stop_loss(13.0)
    assert pos.price_stop_loss == 14.5


def test_set_stop_loss_force_overrides():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 16.5, 14.5)
    pos.set_stop_loss(13.0, force=True)
    assert pos.price_stop_loss == 13.0


def test_finalize_returns_pnl():
    pos = ConcretePosition("USDT", "LINK", 1000.0, 0.001)
    pct, abs_rev = pos.finalize(entry_price=15.0, exit_price=16.5, fee=0.001)
    expected_pct = (16.5 - 15.0) / 15.0 - 2 * 0.001
    assert abs(pct - expected_pct) < 1e-9
