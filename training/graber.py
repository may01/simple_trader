"""training/graber.py — Programmatic OHLCV data acquisition for the training pipeline.

Graber is a thin wrapper around the fetch-and-save cycle used by DataPreparer.
It checks whether graber_data.pkl is present and covers the requested range,
fetches only the missing candles when needed, and saves atomically.
"""

import pandas as pd

from grabers.grab_binance import load_existing, merge_incremental, save_atomic
from stocks.base_stock import StockInterface


class Graber:
    """Ensure raw OHLCV data is present and up-to-date before indicator computation.

    Parameters
    ----------
    stock:
        Any StockInterface implementation (BinanceStock or mock).
        Must provide ``get_candles_range(symbol, start_ms, end_ms) -> pd.DataFrame``.
    output_path:
        Filesystem path to the ``graber_data.pkl`` pickle file.
    """

    def __init__(self, stock: StockInterface, output_path: str) -> None:
        self.stock = stock
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_data(self, symbol: str, start_ms: int, end_ms: int) -> None:
        """Ensure graber_data.pkl covers [start_ms, end_ms].

        - If file is present and already covers the range: no-op.
        - If file is missing: fetch full range, save.
        - If file exists but is partial (last timestamp < end_ms): fetch the
          missing tail, merge, save.

        Raises
        ------
        ValueError
            If ``stock.get_candles_range()`` returns an empty result when data
            was expected (non-empty date range with no file or stale file).
        """
        existing = load_existing(self.output_path)

        if existing is not None and not existing.empty:
            existing_end_ms = int(existing.index[-1].timestamp() * 1000)
            if existing_end_ms >= end_ms:
                # Data is already current — nothing to do.
                return

            # Fetch only the missing tail.
            fetch_start = existing_end_ms
            new_data = self.stock.get_candles_range(symbol, fetch_start, end_ms)
            if new_data is None or new_data.empty:
                raise ValueError(
                    f"stock.get_candles_range() returned empty result for "
                    f"symbol={symbol!r}, start_ms={fetch_start}, end_ms={end_ms}"
                )
            merged = merge_incremental(existing, new_data)
            save_atomic(merged, self.output_path)
        else:
            # No file present — fetch the full range.
            new_data = self.stock.get_candles_range(symbol, start_ms, end_ms)
            if new_data is None or new_data.empty:
                raise ValueError(
                    f"stock.get_candles_range() returned empty result for "
                    f"symbol={symbol!r}, start_ms={start_ms}, end_ms={end_ms}"
                )
            save_atomic(new_data, self.output_path)

    def load(self) -> pd.DataFrame:
        """Load and return the DataFrame from output_path.

        Raises
        ------
        FileNotFoundError
            If output_path does not exist.
        """
        df = load_existing(self.output_path)
        if df is None:
            raise FileNotFoundError(
                f"graber_data not found at {self.output_path!r}"
            )
        return df

    def is_current(self, end_ms: int) -> bool:
        """Return True if the last saved open_time >= end_ms.

        Returns False (not raises) if the file does not exist.
        """
        df = load_existing(self.output_path)
        if df is None or df.empty:
            return False
        last_ms = int(df.index[-1].timestamp() * 1000)
        return last_ms >= end_ms
