from typing import Dict, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class DataPoint(Protocol):
    """
    Unified read interface for market data.

    Both live (LiveDataPoint) and simulation (WideDataPoint) paths implement
    this protocol so that signals and strategies are path-agnostic.

    Column naming: columns in underlying DataFrames are stored as "{tf}_{col}".
    The get() method takes col WITHOUT the tf prefix.
    """

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        ...

    def get_df(self, tf: int) -> pd.DataFrame:
        ...


class LiveDataPoint:
    """DataPoint backed by per-tf DataFrames from the live data path."""

    def __init__(self, ohlc: Dict[int, pd.DataFrame]) -> None:
        self._ohlc = ohlc

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        if shift < 0:
            raise ValueError(f"shift must be >= 0, got {shift}")
        return float(self._ohlc[tf][f"{tf}_{col}"].iloc[-1 - shift])

    def get_df(self, tf: int) -> pd.DataFrame:
        return self._ohlc[tf]


class WideDataPoint:
    """
    DataPoint backed by a single wide DataFrame row.

    The wide DataFrame has all tf-prefixed columns in one flat structure,
    indexed by timestamp. Used in simulation/backtesting path.
    """

    def __init__(self, df: pd.DataFrame, ts: object) -> None:
        self._df = df
        self._ts = ts
        loc = df.index.get_loc(ts)  # type: ignore[arg-type]
        if not isinstance(loc, int):
            raise ValueError(f"Timestamp {ts} is ambiguous or not found in index")
        self._pos = loc

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        if shift < 0:
            raise ValueError(f"shift must be >= 0, got {shift}")
        target_pos = self._pos - shift
        if target_pos < 0:
            raise IndexError(f"shift={shift} exceeds available history at position {self._pos}")
        return float(self._df[f"{tf}_{col}"].iloc[target_pos])

    def get_df(self, tf: int) -> pd.DataFrame:
        raise NotImplementedError(
            "WideDataPoint does not expose per-tf DataFrames. "
            "Use LiveDataPoint for indicator computation."
        )
