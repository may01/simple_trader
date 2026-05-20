from simple_trader.position.long_position import LongPosition
from simple_trader.infrastructure.constants import PositionState, StrategyAction, PositionType


def test_long_position_type():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    assert pos.position_type == PositionType.LONG


def test_long_opens_to_wait_buy():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 17.0, 14.0)
    assert pos.state == PositionState.WAIT_BUY


def test_long_close_transitions_to_wait_sell():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 17.0, 14.0)
    pos.record_entry_fill(10.0, 15.0)
    pos.close(17.0, 14.5)
    assert pos.state == PositionState.WAIT_SELL
    assert pos.action == StrategyAction.CLOSE_LONG


def test_long_stop_loss_tightens_upward():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 17.0, 14.0)
    pos.set_stop_loss(14.5)
    assert pos.price_stop_loss == 14.5


def test_long_stop_loss_cannot_go_lower():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    pos.open(15.0, 17.0, 14.0)
    pos.set_stop_loss(13.5)
    assert pos.price_stop_loss == 14.0


def test_long_finalize_profitable_trade():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    pct, _ = pos.finalize(entry_price=15.0, exit_price=16.5, fee=0.001)
    assert pct > 0


def test_long_finalize_losing_trade():
    pos = LongPosition("USDT", "LINK", 1000.0, 0.001)
    pct, _ = pos.finalize(entry_price=15.0, exit_price=14.0, fee=0.001)
    assert pct < 0
