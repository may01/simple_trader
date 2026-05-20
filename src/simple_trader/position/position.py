from typing import Optional, Tuple
from .base_position import BasePosition
from ..infrastructure.constants import PositionState, StrategyAction


class Position:
    def __init__(self, pos_impl: BasePosition) -> None:
        self._impl = pos_impl
        self._action_id: Optional[str] = None
        self._loan_amount: float = 0.0

    @property
    def state(self) -> PositionState:
        return self._impl.state

    @property
    def action(self) -> StrategyAction:
        return self._impl.action

    @property
    def price_stop_loss(self) -> float:
        return self._impl.price_stop_loss

    @property
    def price_open(self) -> list:
        return self._impl.price_open

    @property
    def price_close(self) -> float:
        return self._impl.price_close

    def open(self, price_open: float, price_close: float, price_stop: float) -> None:
        self._impl.open(price_open, price_close, price_stop)

    def close(self, price_close: float, price_stop: float) -> None:
        self._impl.close(price_close, price_stop)

    def set_stop_loss(self, price: float, force: bool = False) -> None:
        self._impl.set_stop_loss(price, force=force)

    def record_entry_fill(self, amount: float, price: float) -> None:
        self._impl.record_entry_fill(amount, price)

    def record_exit_fill(self, amount: float, price: float) -> None:
        self._impl.record_exit_fill(amount, price)

    def finalize(self, entry_price: float, exit_price: float) -> Tuple[float, float]:
        return self._impl.finalize(entry_price, exit_price, self._impl.fee)

    def get_action(self) -> StrategyAction:
        return self._impl.action

    @property
    def buy_id(self) -> Optional[str]:
        return self._action_id

    @buy_id.setter
    def buy_id(self, value: Optional[str]) -> None:
        self._action_id = value

    def set_action_id(self, order_id: str) -> None:
        self._action_id = order_id

    def clear_action_id(self) -> None:
        self._action_id = None

    def is_action_in_progress(self) -> bool:
        return self._action_id is not None

    def set_loan(self, amount: float) -> None:
        self._loan_amount = amount
        self._impl.coinUse.borrow(amount)

    def repay_loan(self) -> float:
        amount = self._loan_amount
        self._loan_amount = 0.0
        return amount
