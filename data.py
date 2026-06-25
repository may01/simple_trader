# data.py — DataPoint protocol: LiveDataPoint and WideDataPoint.
#
# LiveDataPoint  — wraps per-TF DataFrames (used in live trading).
# WideDataPoint  — wraps a wide DataFrame at a fixed timestamp (used in simulation/backtesting).
#
# Column naming convention: always "{tf}_{col}", e.g. "5_rsi_14".

from __future__ import annotations

import os
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
        """Return ``ohlc[tf][name].iloc[-1 - shift]``.

        ``name`` is the bare column for NN result columns ("nn_res_*", which are
        timeframe-agnostic and carry no "{tf}_" prefix) and ``f"{tf}_{col}"``
        otherwise (data-class.md §2).

        Returns float('nan') when shift >= len(df) (not enough history).

        Raises:
            KeyError: If ``tf`` is not present in the ohlc mapping.
        """
        if tf not in self._ohlc:
            raise KeyError(f"tf={tf} not in LiveDataPoint")
        name = col if col.startswith("nn_res_") else f"{tf}_{col}"
        df = self._ohlc[tf][name]
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

        NN result columns ("nn_res_*") are timeframe-agnostic — they carry no
        "{tf}_" prefix and are looked up by their bare column name regardless of
        the requested tf (data-class.md §2/§8). The is_closed shift logic still
        uses the requested tf's candle boundaries.
        """
        full_col = col if col.startswith("nn_res_") else f"{tf}_{col}"

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

    # Collect all per-tf columns and append with one concat at the end —
    # 8 inserts × len(CANDLES) sequential setitems fragment the frame
    # (one memory block per insert) and pandas warns on every later insert.
    new_cols: dict[str, object] = {}

    for tf in CANDLES:
        open_index = df.index.floor(f"{tf}min")

        new_cols[f"{tf}_open_index"] = open_index

        # open — first value of the candle period
        new_cols[f"{tf}_open"] = df.groupby(open_index)["open"].transform("first")

        # high — cumulative max within candle
        new_cols[f"{tf}_high"] = df.groupby(open_index)["high"].cummax()

        # low — cumulative min within candle
        new_cols[f"{tf}_low"] = df.groupby(open_index)["low"].cummin()

        # close — raw 1-min close, never forward-looking
        new_cols[f"{tf}_close"] = df["close"]

        # volume — cumulative sum within candle
        new_cols[f"{tf}_volume"] = df.groupby(open_index)["volume"].cumsum()

        # buy_volume — cumulative sum of taker_base_vol within candle
        new_cols[f"{tf}_buy_volume"] = df.groupby(open_index)["taker_base_vol"].cumsum()

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

        new_cols[f"{tf}_is_closed"] = is_closed

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


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

    def __init__(self) -> None:
        pair = os.environ.get("PAIR", "")
        # Derive coin from PAIR, e.g. "link_usdt" → "link"
        self.coin = pair.split("_")[0] if "_" in pair else pair
        self.candles = CANDLES
        self.ohlc: dict[int, pd.DataFrame] = {}

    def build_candles(self, time_point: int = 0) -> None:
        """Fetch candles, rename to {tf}_* columns, compute indicators.

        After enriching the per-tf frames, left-joins the timeframe-agnostic
        ``nn_res_*`` columns from the live dataset's ``df_with_nn.pkl`` (produced
        by the same batch ``run_inference`` for live/backtest parity). The join
        is absence-safe: if the artifact is missing, ``nn_res_*`` are simply
        absent and reads fall back to the caller's default. The cadence of when
        the live artifact is (re)produced is out of scope here.
        """
        raw: dict[int, pd.DataFrame] = stock_holder.item.get_candles_history(
            self.candles, self.coin, time_point
        )

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

        # Left-join nn_res_* columns onto each per-tf frame (absence-safe no-op
        # when the live df_with_nn.pkl is absent). The columns are timeframe-
        # agnostic and shared across all tfs.
        dataset_dir = self._live_dataset_dir()
        if dataset_dir is not None:
            for tf in self.ohlc:
                self.ohlc[tf] = join_nn_results(self.ohlc[tf], dataset_dir)

    @staticmethod
    def _live_dataset_dir() -> "str | None":
        """Return the directory holding the live dataset's df_with_nn.pkl.

        Derived from the same env-based dataset folder used by the rest of the
        data layer. Returns None when the required env is not set (e.g. unit
        tests that exercise build_candles without a configured dataset), in
        which case the nn join is skipped entirely.
        """
        try:
            from helpers import dataset_folder  # lazy — avoids circular dep
            return dataset_folder()
        except KeyError:
            return None

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


# ---------------------------------------------------------------------------
# SimulationData — single-load replay with O(1) per-step access
# ---------------------------------------------------------------------------

def _wide_df_path_for_pair(pair: str) -> str:
    """Return path to df_with_indicators.pkl for the given pair.

    Reads ROOT_FOLDER, DATA_ROOT, DATA_SET_NAME from env and combines with
    the pair argument — does NOT mutate os.environ["PAIR"].
    """
    from helpers import root_folder  # local import — avoids circular dep at module level
    data_root = os.environ["DATA_ROOT"]
    data_set_name = os.environ["DATA_SET_NAME"]
    return f"{root_folder()}/{data_root}/{data_set_name}_{pair}/df_with_indicators.pkl"


def join_nn_results(df: pd.DataFrame, dataset_dir: str) -> pd.DataFrame:
    """Left-join {dataset_dir}/df_with_nn.pkl (timeframe-agnostic nn_res_* columns)
    onto ``df`` on the shared 1-min DatetimeIndex, and return the joined frame.

    The NN inference batch (NNOrchestrator.run_inference / Trainer.infer_nn)
    writes a separate, additive ``df_with_nn.pkl`` next to each dataset's
    ``df_with_indicators.pkl`` containing ONLY ``nn_res_*`` columns. Consumers
    merge it in at construction; the canonical ``df_with_indicators.pkl`` stays
    single-writer (DataPreparer) and is never mutated.

    Absence-safe: if ``df_with_nn.pkl`` does not exist, returns ``df`` unchanged
    (no-op) — the ``nn_res_*`` columns simply do not appear, and reads return
    the caller's default via ``DataPoint.get(col, tf, default=...)``.

    The join never overwrites existing ``df`` columns (only columns absent from
    ``df`` are taken from the pickle), and it does NOT mutate
    ``df_with_indicators.pkl`` on disk.

    Args:
        df:          The consumer's wide frame (1-min DatetimeIndex).
        dataset_dir: Directory containing df_with_nn.pkl (and df_with_indicators.pkl).

    Returns:
        ``df`` with ``nn_res_*`` columns left-joined on the index, or ``df``
        unchanged when the artifact is absent.
    """
    path = os.path.join(dataset_dir, "df_with_nn.pkl")
    if not os.path.exists(path):
        return df

    nn_df: pd.DataFrame = pd.read_pickle(path)

    # Take only columns not already present so the join never clobbers existing
    # df columns (the pickle is nn_res_*-only by construction, but guard anyway).
    new_cols = [c for c in nn_df.columns if c not in df.columns]
    if not new_cols:
        return df
    return df.join(nn_df[new_cols], how="left")


class SimulationData:
    """Single-load wide-DataFrame replay cursor for backtesting/simulation.

    Loads df_with_indicators.pkl once at construction time.  Each step is O(1):
    only the current timestamp index advances; no slice copies are made.
    """

    def __init__(self, pair: str, begin_ts: int, end_ts: int, step_min: int) -> None:
        """Load df_with_indicators.pkl and build a filtered timestamp range.

        Args:
            pair:      Trading pair string, e.g. "btc_usdt".
            begin_ts:  Start of simulation window as Unix seconds (inclusive).
            end_ts:    End of simulation window as Unix seconds (inclusive).
            step_min:  Step size in minutes between simulation ticks.

        Side-effects:
            Logs a warning via log_warning() if more than 1% of the requested
            timestamps are absent from the loaded DataFrame's index.
        """
        from logs import log_warning  # local import — avoids circular dep
        path = _wide_df_path_for_pair(pair)
        self._df: pd.DataFrame = pd.read_pickle(path)
        # Left-join nn_res_* columns from the dataset's df_with_nn.pkl (the dir
        # holding df_with_indicators.pkl). Absence-safe no-op; never mutates the
        # single-writer df_with_indicators.pkl.
        self._df = join_nn_results(self._df, os.path.dirname(path))

        begin = pd.Timestamp(begin_ts, unit="s", tz="UTC")
        end = pd.Timestamp(end_ts, unit="s", tz="UTC")
        full_range = pd.date_range(begin, end, freq=f"{step_min}min", tz="UTC")

        self._timestamps = full_range.intersection(self._df.index)
        self._cur_idx: int = 0

        if len(full_range) > 0:
            missing_ratio = (len(full_range) - len(self._timestamps)) / len(full_range)
            if missing_ratio > 0.01:
                log_warning(
                    f"SimulationData({pair}): {missing_ratio:.1%} of requested timestamps "
                    f"are missing from df_with_indicators.pkl "
                    f"({len(full_range) - len(self._timestamps)} / {len(full_range)})"
                )

    # ------------------------------------------------------------------
    # Core iteration interface
    # ------------------------------------------------------------------

    def get(self) -> WideDataPoint:
        """Return WideDataPoint for the current timestamp. O(1)."""
        return WideDataPoint(self._df, self._timestamps[self._cur_idx])

    def next(self) -> None:
        """Advance cursor by one step."""
        self._cur_idx += 1

    def is_end(self) -> bool:
        """Return True when all timestamps have been consumed."""
        return self._cur_idx >= len(self._timestamps)

    @property
    def steps(self) -> int:
        """Total number of timestamps in this simulation window."""
        return len(self._timestamps)

    @property
    def current_ts(self) -> pd.Timestamp:
        """Timestamp at the current cursor position."""
        return self._timestamps[self._cur_idx]

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _make_slice(self, timestamps) -> "SimulationData":
        """Create a new SimulationData sharing the same _df (no reload)."""
        obj = object.__new__(SimulationData)
        obj._df = self._df  # shared reference — no copy
        obj._timestamps = timestamps
        obj._cur_idx = 0
        return obj

    def split(self, n: int) -> list:
        """Divide timestamps into *n* roughly equal slices.

        Returns a list of *n* SimulationData instances that share the same
        underlying DataFrame object.  The last slice absorbs any remainder rows.

        Args:
            n: Number of slices to produce (must be >= 1).
        """
        size = len(self._timestamps) // n
        slices = []
        for i in range(n):
            start = i * size
            end = (i + 1) * size if i < n - 1 else len(self._timestamps)
            slices.append(self._make_slice(self._timestamps[start:end]))
        return slices


# ---------------------------------------------------------------------------
# FullData — read-only view over the full wide DataFrame
# ---------------------------------------------------------------------------

class FullData:
    """Read-only view over a wide DataFrame for batch/training access.

    Provides filtered slices by timeframe and single closed-candle lookup.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get(self, tf: int) -> pd.DataFrame:
        """Return all closed-candle rows for *tf*, keeping ``{tf}_*`` columns plus
        any timeframe-agnostic ``nn_res_*`` result columns (joined at load).

        Args:
            tf: Timeframe in minutes.

        Returns:
            DataFrame with 1-min timestamps at candle close (the DatetimeIndex of
            closed rows) and columns prefixed with ``f"{tf}_"`` or ``"nn_res_"``
            (the latter shared across all timeframes; data-class.md §8).
        """
        closed_col = f"{tf}_is_closed"
        mask = self._df[closed_col] == True  # noqa: E712 — explicit bool comparison
        closed_rows = self._df[mask]
        tf_cols = [
            c for c in closed_rows.columns
            if c.startswith(f"{tf}_") or c.startswith("nn_res_")
        ]
        return closed_rows[tf_cols]

    def get_candle(self, tf: int, open_time: pd.Timestamp) -> pd.Series:
        """Return the single closed-candle row for the candle that opened at *open_time*.

        Args:
            tf:        Timeframe in minutes.
            open_time: The candle open timestamp (stored in ``{tf}_open_index``).

        Returns:
            pd.Series for the matching row.

        Raises:
            KeyError: If no closed candle with the given open_time is found.
        """
        closed = self._df[self._df[f"{tf}_is_closed"].astype(bool)]
        tf_cols = [c for c in closed.columns if c.startswith(f"{tf}_")]
        matching = closed[closed[f"{tf}_open_index"] == open_time][tf_cols]
        if matching.empty:
            raise KeyError(f"No closed candle for tf={tf} at open_time={open_time}")
        return matching.iloc[0]


def get_stock_data(pair: str) -> pd.DataFrame:
    """Load graber_data.pkl for *pair* and build a wide DataFrame.

    Reads the pickle produced by the data grabber, renames short column names
    (o→open, h→high, l→low, c→close, v→volume), delegates to _build_wide_df()
    to add all per-TF columns, then left-joins the dataset's timeframe-agnostic
    ``nn_res_*`` columns (absence-safe). The joined frame is what callers wrap in
    ``FullData`` so ``FullData.get(tf)`` surfaces ``nn_res_*`` while ``FullData``
    stays a thin view.
    """
    from helpers import root_folder  # local import avoids circular deps at module level

    data_root = os.environ["DATA_ROOT"]
    data_set_name = os.environ["DATA_SET_NAME"]
    dataset_dir = f"{root_folder()}/{data_root}/{data_set_name}_{pair}"
    path = f"{dataset_dir}/graber_data.pkl"

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

    wide_df = _build_wide_df(raw)
    return join_nn_results(wide_df, dataset_dir)
