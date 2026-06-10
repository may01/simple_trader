# data.py — DataPoint protocol: LiveDataPoint and WideDataPoint.
#
# LiveDataPoint  — wraps per-TF DataFrames (used in live trading).
# WideDataPoint  — wraps a wide DataFrame at a fixed timestamp (used in simulation/backtesting).
#
# Column naming convention: always "{tf}_{col}", e.g. "5_rsi_14".

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from config_loader import CANDLES


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
        """Return indicator input slice at ts for tf.

        Delegates to build_indicator_input() from indicators.py (lazy import
        to avoid the data.py ↔ indicators.py circular import).
        """
        from indicators import build_indicator_input  # lazy import: avoids circular dep
        return build_indicator_input(self._df, self._ts, tf)

    @property
    def timestamp(self) -> pd.Timestamp:
        """The fixed timestamp this DataPoint represents."""
        return self._ts


# ---------------------------------------------------------------------------
# Wide DataFrame builder
# ---------------------------------------------------------------------------


def _build_wide_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build wide DataFrame from renamed OHLCV df.

    Input columns: open, high, low, close, volume, taker_base_vol.
    For each tf in CANDLES, adds the following columns:
      {tf}_open_index, {tf}_open, {tf}_high, {tf}_low, {tf}_close,
      {tf}_volume, {tf}_buy_volume, {tf}_is_closed
    The input DataFrame is copied so the caller's copy is not mutated.
    """
    df = df.copy()

    for tf in CANDLES:
        open_index = df.index.floor(f"{tf}min")

        df[f"{tf}_open_index"] = open_index

        # open — first value of the candle period
        df[f"{tf}_open"] = df.groupby(open_index)["open"].transform("first")

        # high — cumulative max within candle
        df[f"{tf}_high"] = df.groupby(open_index)["high"].cummax()

        # low — cumulative min within candle
        df[f"{tf}_low"] = df.groupby(open_index)["low"].cummin()

        # close — raw 1-min close, never forward-looking
        df[f"{tf}_close"] = df["close"]

        # volume — cumulative sum within candle
        df[f"{tf}_volume"] = df.groupby(open_index)["volume"].cumsum()

        # buy_volume — cumulative sum of taker_base_vol within candle
        df[f"{tf}_buy_volume"] = df.groupby(open_index)["taker_base_vol"].cumsum()

        # is_closed — True at the last 1-min row of each tf-period candle
        idx = df.index
        if tf == 1:
            is_closed = pd.Series(True, index=idx)
        elif tf == 5:
            is_closed = idx.minute % 5 == 4
        elif tf == 15:
            is_closed = idx.minute % 15 == 14
        elif tf == 60:
            is_closed = idx.minute == 59
        elif tf == 240:
            is_closed = (idx.minute == 59) & (idx.hour % 4 == 3)
        elif tf == 1440:
            is_closed = (idx.minute == 59) & (idx.hour == 23)
        else:
            # Generic fallback: last minute of each tf-period
            is_closed = (idx + pd.Timedelta(minutes=1)).floor(f"{tf}min") != idx.floor(
                f"{tf}min"
            )

        df[f"{tf}_is_closed"] = is_closed

    return df


from stocks_holder import stock_holder  # noqa: E402 — after class definitions to avoid circular import
from indicators import Indicators  # noqa: E402


# ---------------------------------------------------------------------------
# LiveData — per-tick candle fetch + indicator compute
# ---------------------------------------------------------------------------

class LiveData:
    """Fetches live candles for all timeframes and computes indicators.

    Usage:
        data.item.build_candles()          # call each tick
        dp = data.item.get_data_point()    # access current OHLC + indicators
    """

    def __init__(self, nn_predictor=None) -> None:
        import os
        pair = os.environ.get("PAIR", "")
        # Derive coin from PAIR, e.g. "link_usdt" → "link"
        self.coin = pair.split("_")[0] if "_" in pair else pair
        self.candles = CANDLES
        self.nn_predictor = nn_predictor
        self.ohlc: dict[int, pd.DataFrame] = {}

    def build_candles(self, time_point: int = 0) -> None:
        """Fetch candles, rename to {tf}_* columns, compute indicators."""
        raw: dict[int, pd.DataFrame] = stock_holder.item.get_candles_history(
            self.candles, self.coin, time_point
        )

        # First loop: rename columns, compute indicators, store enriched dfs.
        # Track (point, tf) pairs so nn_predictor can be called after ALL
        # Indicators.compute calls complete.
        points: dict[int, LiveDataPoint] = {}

        for tf, df in raw.items():
            # Rename OHLCV columns to {tf}_{col} convention
            rename_cols = [
                "open", "high", "low", "close", "volume", "taker_base_vol",
                "open_time", "close_time", "qav", "num_trades",
                "taker_quote_vol", "ignore",
            ]
            rename_map = {col: f"{tf}_{col}" for col in rename_cols if col in df.columns}
            renamed_df = df.rename(columns=rename_map)

            # All historical candles are closed
            renamed_df[f"{tf}_is_closed"] = True

            # buy_volume is expected by volume indicator fields (VolBuyMAField etc.)
            # Source column is taker_base_vol (renamed to {tf}_taker_base_vol above)
            taker_col = f"{tf}_taker_base_vol"
            if taker_col in renamed_df.columns:
                renamed_df[f"{tf}_buy_volume"] = renamed_df[taker_col]

            # Wrap in LiveDataPoint so Indicators.compute can call get_df(tf)
            point = LiveDataPoint({tf: renamed_df})

            # Compute indicators — mutates renamed_df in place via point.get_df(tf)
            Indicators.compute(point, tf)

            # Store the enriched (mutated) DataFrame
            self.ohlc[tf] = renamed_df

            # Track point for nn_predictor pass below
            points[tf] = point

        # Second pass: call nn_predictor AFTER all Indicators.compute calls complete.
        if self.nn_predictor is not None:
            for tf, point in points.items():
                self.nn_predictor.compute(point, tf)

    def get_data_point(self) -> "LiveDataPoint":
        """Return LiveDataPoint wrapping the current self.ohlc."""
        return LiveDataPoint(self.ohlc)

    def get_depth_data(self, quantity: int):
        """Fetch order book depth from the exchange."""
        return stock_holder.item.depth(quantity)


# ---------------------------------------------------------------------------
# Module-level singleton — used by Robot
# ---------------------------------------------------------------------------

item: LiveData = LiveData()


def get_stock_data(pair: str) -> pd.DataFrame:
    """Load graber_data.pkl for *pair* and build a wide DataFrame.

    Reads the pickle produced by the data grabber, renames short column names
    (o→open, h→high, l→low, c→close, v→volume), then delegates to
    _build_wide_df() to add all per-TF columns.
    """
    import os
    from helpers import root_folder  # local import avoids circular deps at module level

    data_root = os.environ["DATA_ROOT"]
    data_set_name = os.environ["DATA_SET_NAME"]
    path = f"{root_folder()}/{data_root}/{data_set_name}_{pair}/graber_data.pkl"

    raw: pd.DataFrame = pd.read_pickle(path)

    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    # Only rename columns that actually exist in the pickle
    actual_rename = {k: v for k, v in rename_map.items() if k in raw.columns}
    if actual_rename:
        raw = raw.rename(columns=actual_rename)

    return _build_wide_df(raw)
