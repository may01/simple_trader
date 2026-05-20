from abc import ABC, abstractmethod
from typing import List, Tuple
from .coin import Coin
from ..infrastructure.constants import PositionState, StrategyAction, PositionType


class BasePosition(ABC):
    def __init__(self, coin_use: str, coin_get: str, full_position: float, fee: float, use_safety: bool = True) -> None:
        self.coin_use = coin_use
        self.coin_get = coin_get
        self.full_position = full_position
        self.fee = fee
        self.use_safety = use_safety
        self.position_type = PositionType.UNKNOWN
        self._do_init()

    def _do_init(self) -> None:
        self.state = PositionState.WAIT
        self.action = StrategyAction.NOTHING
        self.price_open: List[float] = []
        self.price_close: float = 0.0
        self.price_stop_loss: float = 0.0
        self.executed_open: List[float] = []
        self.executed_open_amount: List[float] = []
        self.executed_close: List[float] = []
        self.executed_close_amount: List[float] = []
        self.coinUse = Coin(self.coin_use)
        self.coinGet = Coin(self.coin_get)
        self.open_time: int = 0
        self.safety_close_time: int = 0

    def open(self, price_open: float, price_close: float, price_stop: float) -> None:
        assert self.state == PositionState.WAIT, f"open() requires WAIT state, got {self.state}"
        self.price_open = [price_open]
        self.price_close = price_close
        self.price_stop_loss = price_stop
        self.state = self._wait_buy_state()
        self.action = self._open_action()

    def close(self, price_close: float, price_stop: float) -> None:
        allowed = {self._wait_buy_state(), self._wait_safety_buy_state()}
        assert self.state in allowed, f"close() cannot be called from state {self.state}"
        self.price_close = price_close
        self.state = self._wait_sell_state()
        self.action = self._close_action()

    def set_stop_loss(self, price: float, force: bool = False) -> None:
        if force or self._is_tighter_stop(price):
            self.price_stop_loss = price

    def record_entry_fill(self, amount: float, price: float) -> None:
        self.executed_open.append(price)
        self.executed_open_amount.append(amount)
        if self.use_safety:
            self.state = self._wait_safety_buy_state()
        else:
            self.state = self._wait_sell_state()

    def record_exit_fill(self, amount: float, price: float) -> None:
        self.executed_close.append(price)
        self.executed_close_amount.append(amount)
        self.state = PositionState.WAIT
        self.action = StrategyAction.NOTHING

    @abstractmethod
    def _wait_buy_state(self) -> PositionState: ...

    @abstractmethod
    def _wait_sell_state(self) -> PositionState: ...

    @abstractmethod
    def _wait_safety_buy_state(self) -> PositionState: ...

    @abstractmethod
    def _open_action(self) -> StrategyAction: ...

    @abstractmethod
    def _close_action(self) -> StrategyAction: ...

    @abstractmethod
    def _is_tighter_stop(self, price: float) -> bool: ...

    @abstractmethod
    def finalize(self, entry_price: float, exit_price: float, fee: float) -> Tuple[float, float]: ...
