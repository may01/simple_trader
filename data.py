# data.py — DataPoint protocol: LiveDataPoint and WideDataPoint.
#
# LiveDataPoint  — wraps per-TF DataFrames (used in live trading).
# WideDataPoint  — wraps a wide DataFrame at a fixed timestamp (used in simulation/backtesting).
#
# Column naming convention: always "{tf}_{col}", e.g. "5_rsi_14".

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataPoint(ABC):
    """Abstract base for accessing OHLC/indicator data at a point in time."""

    @abstractmethod
    def get(self, col: str, tf: int, shift: int = 0) -> float:
        """Return the value of column '{tf}_{col}' with the given shift.

        shift=0  → current (possibly partial) candle value.
        shift=N  → N-th previous *closed* candle value (for WideDataPoint)
                   or iloc[-1-N] row (for LiveDataPoint).
        Returns float('nan') when shift exceeds available history.
        """

    @abstractmethod
    def get_df(self, tf: int) -> pd.DataFrame:
        """Return the mutable per-TF DataFrame (for Indicators.compute() ONLY)."""

    @property
    @abstractmethod
    def timestamp(self) -> pd.Timestamp:
        """Current timestamp of this data point."""

    def cur_price(self, price_type: str) -> float:
        """Return the current price of the given type using tf=1.

        Valid price_type values: 'open', 'close', 'high', 'low'.
        """
        return self.get(price_type, tf=1)


# ---------------------------------------------------------------------------
# LiveDataPoint
# ---------------------------------------------------------------------------

class LiveDataPoint(DataPoint):
    """DataPoint backed by per-TF DataFrames (live trading)."""

    def __init__(self, ohlc: dict[int, pd.DataFrame]) -> None:
        """
        Args:
            ohlc: Mapping from timeframe (int, in minutes) to DataFrame.
                  Each DataFrame has columns named "{tf}_{col}" and a
                  DatetimeIndex sorted in ascending order.

        Raises:
            ValueError: If ``ohlc`` is empty or ``ohlc[1]`` has no rows.
                        An empty ohlc or empty tf=1 DataFrame is a programming
                        error — LiveDataPoint requires at least one row of data.
        """
        if not ohlc:
            raise ValueError("ohlc must not be empty")
        if 1 in ohlc and len(ohlc[1]) == 0:
            raise ValueError("ohlc[1] must have at least one row")
        self._ohlc = ohlc

    # ------------------------------------------------------------------
    # DataPoint interface
    # ------------------------------------------------------------------

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        """Return ``ohlc[tf][f"{tf}_{col}"].iloc[-1 - shift]``.

        Returns float('nan') when shift >= len(df) (not enough history).

        Raises:
            KeyError: If ``tf`` is not present in the ohlc mapping.
        """
        if tf not in self._ohlc:
            raise KeyError(f"tf={tf} not in LiveDataPoint")
        df = self._ohlc[tf][f"{tf}_{col}"]
        if shift >= len(df):
            return float("nan")
        return float(df.iloc[-1 - shift])

    def get_df(self, tf: int) -> pd.DataFrame:
        """Return the underlying per-TF DataFrame (mutable reference)."""
        return self._ohlc[tf]

    @property
    def timestamp(self) -> pd.Timestamp:
        """Last index entry of the tf=1 DataFrame."""
        return self._ohlc[1].index[-1]


# ---------------------------------------------------------------------------
# WideDataPoint
# ---------------------------------------------------------------------------

class WideDataPoint(DataPoint):
    """DataPoint backed by a wide DataFrame at a fixed timestamp (simulation)."""

    def __init__(self, df: pd.DataFrame, ts: pd.Timestamp) -> None:
        """
        Args:
            df: Wide DataFrame with columns "{tf}_{col}" and "{tf}_is_closed".
                Index is a DatetimeIndex.
            ts: The timestamp that represents "now" for this DataPoint.
        """
        self._df = df
        self._ts = ts

    # ------------------------------------------------------------------
    # DataPoint interface
    # ------------------------------------------------------------------

    def get(self, col: str, tf: int, shift: int = 0) -> float:
        """Return the value of column '{tf}_{col}'.

        shift=0  → df.loc[ts, "{tf}_{col}"]   (partial candle OK)
        shift=N  → value at the Nth-last row where "{tf}_is_closed" == True,
                   considering only rows up to and including ts.
                   Returns float('nan') when not enough history.
        """
        full_col = f"{tf}_{col}"

        if shift == 0:
            # Precondition: self._ts must be present in self._df.index.
            # SimulationData always uses timestamps from the index, so a missing
            # ts is a programming error — KeyError from .loc is intentional.
            return float(self._df.loc[self._ts, full_col])

        # shift >= 1: look at closed-candle rows up to ts
        closed_col = f"{tf}_is_closed"
        mask = (self._df.index <= self._ts) & self._df[closed_col].astype(bool)
        closed_rows = self._df.loc[mask]

        if len(closed_rows) < shift:
            return float("nan")

        # Nth-last means index -shift (1-indexed from the end)
        return float(closed_rows[full_col].iloc[-shift])

    def get_df(self, tf: int) -> pd.DataFrame:
        """Return a slice of the wide DataFrame up to ts (placeholder).

        This will be replaced in Task 03 with a call to build_indicator_input,
        which reconstructs a proper per-TF DataFrame for Indicators.compute().
        """
        return self._df[: self._ts]

    @property
    def timestamp(self) -> pd.Timestamp:
        """The fixed timestamp this DataPoint represents."""
        return self._ts
