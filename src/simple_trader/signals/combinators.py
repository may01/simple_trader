from collections import deque
from typing import Deque, List
from .base_signal import BaseSignal
from ..data.data_point import DataPoint


class And(BaseSignal):
    def __init__(self, signals: List[BaseSignal]) -> None:
        super().__init__(col="", tf=0)
        self.signals = signals

    def check(self, data_point: DataPoint) -> bool:
        return all(s.check(data_point) for s in self.signals)

    def reset(self) -> None:
        for s in self.signals:
            s.reset()


class Or(BaseSignal):
    def __init__(self, signals: List[BaseSignal]) -> None:
        super().__init__(col="", tf=0)
        self.signals = signals

    def check(self, data_point: DataPoint) -> bool:
        return any(s.check(data_point) for s in self.signals)

    def reset(self) -> None:
        for s in self.signals:
            s.reset()


class Not(BaseSignal):
    def __init__(self, signal: BaseSignal) -> None:
        super().__init__(col="", tf=0)
        self.signal = signal

    def check(self, data_point: DataPoint) -> bool:
        return not self.signal.check(data_point)

    def reset(self) -> None:
        self.signal.reset()


class Continuous(BaseSignal):
    def __init__(self, signal: BaseSignal, n: int) -> None:
        super().__init__(col="", tf=0)
        self.signal = signal
        self.n = n
        self._streak = 0

    def check(self, data_point: DataPoint) -> bool:
        if self.signal.check(data_point):
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self.n

    def reset(self) -> None:
        self._streak = 0
        self.signal.reset()


class History(BaseSignal):
    def __init__(self, signal: BaseSignal, window: int) -> None:
        super().__init__(col="", tf=0)
        self.signal = signal
        self.window = window
        self._history: Deque[bool] = deque(maxlen=window)

    def check(self, data_point: DataPoint) -> bool:
        result = self.signal.check(data_point)
        self._history.append(result)
        return any(self._history)

    def reset(self) -> None:
        self._history.clear()
        self.signal.reset()
