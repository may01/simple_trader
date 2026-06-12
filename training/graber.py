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
        """Ensure graber_data.pkl covers the range [start_ms, end_ms) — data is current if last candle open_time + 60000ms >= end_ms.

        - If file is present and already covers the range: no-op.
        - If file is missing: fetch full range, save.
        - If file exists but is partial: fetch the missing head (first
          timestamp > start_ms) and/or missing tail (last timestamp < end_ms),
          merge, save.

        Raises
        ------
        ValueError
            If ``stock.get_candles_range()`` returns an empty result when data
            was expected (non-empty date range with no file or stale file).
        """
        existing = load_existing(self.output_path)

        if existing is not None and not existing.empty:
            existing_start_ms = int(existing.index[0].timestamp() * 1000)
            existing_end_ms = int(existing.index[-1].timestamp() * 1000)

            merged = existing

            if start_ms < existing_start_ms:
                # Fetch the missing head: [start_ms, existing_start).
                head = self.stock.get_candles_range(symbol, start_ms, existing_start_ms)
                if head is None or head.empty:
                    raise ValueError(
                        f"stock.get_candles_range() returned empty result for "
                        f"symbol={symbol!r}, start_ms={start_ms}, end_ms={existing_start_ms}"
                    )
                merged = merge_incremental(head, merged)

            if existing_end_ms < end_ms:
                # Fetch the missing tail, skipping the already-saved last candle.
                fetch_start = existing_end_ms + 60_000
                tail = self.stock.get_candles_range(symbol, fetch_start, end_ms)
                if tail is None or tail.empty:
                    raise ValueError(
                        f"stock.get_candles_range() returned empty result for "
                        f"symbol={symbol!r}, start_ms={fetch_start}, end_ms={end_ms}"
                    )
                merged = merge_incremental(merged, tail)

            if merged is existing:
                # Range already covered — nothing to do.
                return
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
