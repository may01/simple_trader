from abc import ABC, abstractmethod
from ..data.data_point import DataPoint


class BaseSignal(ABC):
    def __init__(self, col: str, tf: int, shift: int = 0) -> None:
        self.col = col
        self.tf = tf
        self.shift = shift

    @abstractmethod
    def check(self, data_point: DataPoint) -> bool:
        """Evaluate this signal against the current data point."""

    def get_data(self, data_point: DataPoint) -> float:
        return data_point.get(self.col, self.tf, self.shift)

    def set_shift(self, shift: int) -> None:
        self.shift = shift

    def reset(self) -> None:
        """Reset any internal state. No-op for stateless signals."""


