import pytest
from simple_trader.position.position import Position
from simple_trader.position.long_position import LongPosition
from simple_trader.position.short_position import ShortPosition
from simple_trader.infrastructure.constants import PositionState, StrategyAction, PositionType


def _long_pos() -> Position:
    impl = LongPosition("USDT", "LINK", 1000.0, 0.001)
    return Position(impl)


def _short_pos() -> Position:
    impl = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    return Position(impl)


def test_position_delegates_state():
    pos = _long_pos()
    assert pos.state == PositionState.WAIT


def test_position_delegates_open():
    pos = _long_pos()
    pos.open(15.0, 17.0, 14.0)
    assert pos.state == PositionState.WAIT_BUY


def test_position_delegates_action():
    pos = _long_pos()
    pos.open(15.0, 17.0, 14.0)
    assert pos.action == StrategyAction.OPEN_LONG


def test_position_get_action_returns_current_action():
    pos = _long_pos()
    pos.open(15.0, 17.0, 14.0)
    assert pos.get_action() == StrategyAction.OPEN_LONG


def test_position_set_action_id_stores_buy_id():
    pos = _long_pos()
    pos.set_action_id("12345")
    assert pos.buy_id == "12345"


def test_position_is_action_in_progress_true_when_id_set():
    pos = _long_pos()
    pos.set_action_id("99")
    assert pos.is_action_in_progress() is True


def test_position_is_action_in_progress_false_initially():
    pos = _long_pos()
    assert pos.is_action_in_progress() is False


def test_position_clear_action_id():
    pos = _long_pos()
    pos.set_action_id("99")
    pos.clear_action_id()
    assert pos.is_action_in_progress() is False


def test_position_short_open_action():
    pos = _short_pos()
    pos.open(15.0, 13.0, 16.0)
    assert pos.action == StrategyAction.OPEN_SHORT


def test_position_delegates_set_stop_loss():
    pos = _long_pos()
    pos.open(15.0, 17.0, 14.0)
    pos.set_stop_loss(14.5)
    assert pos.price_stop_loss == 14.5
