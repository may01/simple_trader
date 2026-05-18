import pandas as pd
from .data_point import WideDataPoint


class SimulationData:
    """
    Replays pre-computed wide DataFrame for backtesting.

    Each call to step() returns a WideDataPoint at the next timestamp —
    O(1), no computation, no I/O during replay.
    """

    def __init__(self, df: pd.DataFrame, step_min: int = 1) -> None:
        self._df = df
        self._step_min = step_min
        self._pos = 0

    def step(self) -> WideDataPoint:
        if self._pos >= len(self._df):
            raise StopIteration("SimulationData exhausted")
        ts = self._df.index[self._pos]
        self._pos += self._step_min
        return WideDataPoint(self._df, ts)

    def is_done(self) -> bool:
        return self._pos >= len(self._df)

    def reset(self) -> None:
        self._pos = 0

    @property
    def current_ts(self) -> pd.Timestamp:
        return self._df.index[min(self._pos, len(self._df) - 1)]
