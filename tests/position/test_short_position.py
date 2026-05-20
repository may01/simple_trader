from simple_trader.position.short_position import ShortPosition
from simple_trader.infrastructure.constants import PositionState, StrategyAction, PositionType


def test_short_position_type():
    pos = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    assert pos.position_type == PositionType.SHORT


def test_short_opens_to_wait_sell():
    pos = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    pos.open(15.0, 13.0, 16.0)
    assert pos.state == PositionState.WAIT_SELL


def test_short_open_action_is_open_short():
    pos = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    pos.open(15.0, 13.0, 16.0)
    assert pos.action == StrategyAction.OPEN_SHORT


def test_short_stop_loss_tightens_downward():
    pos = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    pos.open(15.0, 13.0, 16.0)
    pos.set_stop_loss(15.5)
    assert pos.price_stop_loss == 15.5


def test_short_stop_loss_cannot_go_higher():
    pos = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    pos.open(15.0, 13.0, 16.0)
    pos.set_stop_loss(16.5)
    assert pos.price_stop_loss == 16.0


def test_short_finalize_profitable_trade():
    pos = ShortPosition("LINK", "USDT", 1000.0, 0.001)
    pct, _ = pos.finalize(entry_price=15.0, exit_price=13.0, fee=0.001)
    assert pct > 0
