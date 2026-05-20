from .base_signal import BaseSignal
from ..data.data_point import DataPoint


class Less(BaseSignal):
    def __init__(self, col: str, tf: int, threshold: float, shift: int = 0) -> None:
        super().__init__(col, tf, shift)
        self.threshold = threshold

    def check(self, data_point: DataPoint) -> bool:
        return self.get_data(data_point) < self.threshold


class Greater(BaseSignal):
    def __init__(self, col: str, tf: int, threshold: float, shift: int = 0) -> None:
        super().__init__(col, tf, shift)
        self.threshold = threshold

    def check(self, data_point: DataPoint) -> bool:
        return self.get_data(data_point) > self.threshold


class CrossUp(BaseSignal):
    def __init__(self, col: str, tf: int, threshold: float) -> None:
        super().__init__(col, tf, shift=0)
        self.threshold = threshold

    def check(self, data_point: DataPoint) -> bool:
        current = data_point.get(self.col, self.tf, 0)
        previous = data_point.get(self.col, self.tf, 1)
        return previous < self.threshold <= current


class CrossDown(BaseSignal):
    def __init__(self, col: str, tf: int, threshold: float) -> None:
        super().__init__(col, tf, shift=0)
        self.threshold = threshold

    def check(self, data_point: DataPoint) -> bool:
        current = data_point.get(self.col, self.tf, 0)
        previous = data_point.get(self.col, self.tf, 1)
        return previous > self.threshold >= current


class Equal(BaseSignal):
    def __init__(self, col: str, tf: int, target: float, tolerance: float = 0.0, shift: int = 0) -> None:
        super().__init__(col, tf, shift)
        self.target = target
        self.tolerance = tolerance

    def check(self, data_point: DataPoint) -> bool:
        return abs(self.get_data(data_point) - self.target) <= self.tolerance
