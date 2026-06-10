# stocks/binance_stock.py — Binance exchange implementation
# Implements candle fetching (get_candles_history, get_candles_range) for live trading
# and historical data retrieval.

import os
import time
import pandas as pd

from binance.client import Client

from stocks.base_stock import StockInterface

# Base interval → (binance interval constant, lookback string, list of TFs it covers)
_BASE_INTERVALS = [
    (Client.KLINE_INTERVAL_1MINUTE,  "10 hours ago UTC",  [1, 5]),
    (Client.KLINE_INTERVAL_15MINUTE, "4 days ago UTC",    [15]),
    (Client.KLINE_INTERVAL_1HOUR,    "43 days ago UTC",   [60, 240]),
    (Client.KLINE_INTERVAL_12HOUR,   "121 days ago UTC",  [1440]),
]

# Minutes per base interval (used when resampling)
_INTERVAL_MINUTES = {
    Client.KLINE_INTERVAL_1MINUTE:  1,
    Client.KLINE_INTERVAL_15MINUTE: 15,
    Client.KLINE_INTERVAL_1HOUR:    60,
    Client.KLINE_INTERVAL_12HOUR:   720,
}

_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore",
]

_FLOAT_COLS = ["open", "high", "low", "close", "volume", "taker_base_vol"]


def _parse_klines_to_df(klines: list) -> pd.DataFrame:
    """Parse raw Binance klines list to DataFrame with open/high/low/close/volume columns.
    Used internally by get_candles_history."""
    df = pd.DataFrame(klines, columns=_KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for col in _FLOAT_COLS:
        df[col] = df[col].astype(float)
    return df


def _parse_klines(klines: list) -> pd.DataFrame:
    """Parse raw Binance klines to DataFrame with short column aliases (o/h/l/c/v).
    Used internally by get_candles_range."""
    df = pd.DataFrame(klines, columns=_KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.rename(columns={
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }, inplace=True)
    for col in ["o", "h", "l", "c", "v", "taker_base_vol"]:
        df[col] = df[col].astype(float)
    return df


class Stock_Binance(StockInterface):
    """Binance exchange implementation — candle fetching."""

    def __init__(self):
        key = os.environ["BINANCE_API_KEY"]
        secret = os.environ["BINANCE_API_SECRET"]
        pair = os.environ["PAIR"]           # e.g. "link_usdt"
        fee_str = os.environ["EXCHANGE_FEE"]

        parts = pair.split("_")
        coin = parts[0]
        coin_base = parts[1]

        super().__init__(key, secret, coin, coin_base)

        self.fee = float(fee_str)
        self.client = Client(key, secret)
        self.was_init = True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_pair_name(self) -> str:
        """Return trading pair in Binance format, e.g. 'LINKUSDT'."""
        return (self.coin + self.coin_base).upper()

    # ------------------------------------------------------------------
    # Candle fetching — live trading
    # ------------------------------------------------------------------

    def get_candles_history(
        self, time_list: list, coin: str, time_point: int = 0
    ) -> dict:
        """Fetch recent OHLCV candles for all requested timeframes.

        Fetches 4 base intervals from Binance and resamples to every TF in
        time_list.  Returns at most 120 candles per TF.

        Args:
            time_list: List of timeframe minutes to return, e.g. [1, 5, 15, 60].
            coin: Coin symbol (unused internally — pair is taken from self).
            time_point: Unused reference point (kept for interface compat).

        Returns:
            dict mapping TF-minutes → DataFrame, filtered to keys in time_list.
        """
        result: dict[int, pd.DataFrame] = {}

        for interval, lookback, tfs_covered in _BASE_INTERVALS:
            # Only fetch if at least one TF in this group is requested
            requested_tfs = [tf for tf in tfs_covered if tf in time_list]
            if not requested_tfs:
                continue

            klines = self.client.get_historical_klines(
                self.get_pair_name(), interval, lookback
            )
            self.weight += 2
            if self.weight > self.overflow_weight:
                time.sleep(self.overflow_weight_time)
                self.weight = 0

            if not klines:
                continue

            base_df = _parse_klines_to_df(klines)
            base_df = base_df[~base_df.index.duplicated(keep="last")]
            # Forward-fill missing rows (zero-volume candles)
            base_min = _INTERVAL_MINUTES[interval]
            freq_str = f"{base_min}min"
            base_df = base_df.resample(freq_str).ffill()

            for tf in requested_tfs:
                if tf == base_min:
                    df_tf = base_df[["open", "high", "low", "close", "volume"]].copy()
                else:
                    df_tf = self._resample_to_tf(base_df, base_min, tf)

                df_tf = df_tf.dropna(how="all")
                # Keep last 120 candles
                result[tf] = df_tf.tail(120)

        return result

    # ------------------------------------------------------------------
    # Candle fetching — historical back-fill
    # ------------------------------------------------------------------

    def get_candles_range(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        """Fetch 1-min OHLCV in 30-day chunks from start_ms to end_ms.

        Args:
            symbol: Trading pair, e.g. 'LINKUSDT'.
            start_ms: Start timestamp in Unix milliseconds.
            end_ms: End timestamp in Unix milliseconds.

        Returns:
            DataFrame with columns open_time (index), o, h, l, c, v,
            close_time, taker_base_vol.
        """
        CHUNK_MS = 30 * 24 * 60 * 60 * 1000  # 30 days in ms
        all_dfs: list[pd.DataFrame] = []

        current_start = start_ms
        while current_start < end_ms:
            current_end = min(current_start + CHUNK_MS, end_ms)
            klines = self.client.get_historical_klines(
                symbol, "1m",
                str(current_start), str(current_end),
            )
            if klines:
                df = _parse_klines(klines)
                all_dfs.append(df)
            current_start = current_end

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs).drop_duplicates().sort_index()
        return result
