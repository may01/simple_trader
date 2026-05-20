from typing import Tuple
from .base_position import BasePosition
from ..infrastructure.constants import PositionState, StrategyAction, PositionType


class LongPosition(BasePosition):
    def __init__(self, coin_use: str, coin_get: str, full_position: float, fee: float, use_safety: bool = True) -> None:
        super().__init__(coin_use, coin_get, full_position, fee, use_safety)
        self.position_type = PositionType.LONG

    def _wait_buy_state(self) -> PositionState:
        return PositionState.WAIT_BUY

    def _wait_sell_state(self) -> PositionState:
        return PositionState.WAIT_SELL

    def _wait_safety_buy_state(self) -> PositionState:
        return PositionState.WAIT_SAFETY_BUY

    def _open_action(self) -> StrategyAction:
        return StrategyAction.OPEN_LONG

    def _close_action(self) -> StrategyAction:
        return StrategyAction.CLOSE_LONG

    def _is_tighter_stop(self, price: float) -> bool:
        return price > self.price_stop_loss

    def finalize(self, entry_price: float, exit_price: float, fee: float) -> Tuple[float, float]:
        revenue_pct = (exit_price - entry_price) / entry_price - 2 * fee
        return revenue_pct, revenue_pct * entry_price
