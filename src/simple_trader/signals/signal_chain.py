from typing import List
from .base_signal import BaseSignal
from ..data.data_point import DataPoint
from ..infrastructure.constants import StrategyAction


class SignalChain:
    def __init__(self, name: str, result_action: StrategyAction, tf: int, notify: bool = False) -> None:
        self.name = name
        self.result_action = result_action
        self.tf = tf
        self.notify = notify
        self.signals: List[BaseSignal] = []
        self.cur_pos = 0

    def add(self, signal: BaseSignal) -> "SignalChain":
        self.signals.append(signal)
        return self

    def check(self, data_point: DataPoint) -> bool:
        if self.cur_pos >= len(self.signals):
            return True
        if self.signals[self.cur_pos].check(data_point):
            self.cur_pos += 1
        return self.cur_pos >= len(self.signals)

    def reset(self) -> None:
        self.cur_pos = 0
        for s in self.signals:
            s.reset()

    @property
    def is_complete(self) -> bool:
        return self.cur_pos >= len(self.signals)
