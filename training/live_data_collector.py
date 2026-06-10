"""training/live_data_collector.py — Continuous live 1-minute candle collector.

Polls Binance every interval_seconds (default 60) for new 1-minute candles and
appends them incrementally to graber_data.pkl, crash-safe via atomic save.
"""

import logging
import time

import pandas as pd

from grabers.grab_binance import load_existing, save_atomic
from stocks.base_stock import StockInterface

logger = logging.getLogger(__name__)

# Column rename map: get_candles_history returns open/high/low/close/volume;
# graber_data.pkl schema uses o/h/l/c/v.
_RENAME_MAP = {
    "open": "o",
    "high": "h",
    "low": "l",
    "close": "c",
    "volume": "v",
}


class LiveDataCollector:
    """Poll Binance every interval_seconds for new 1-min candles and append to file.

    Parameters
    ----------
    stock:
        Any StockInterface implementation.  Must provide
        ``get_candles_history(time_list, coin) -> {int: pd.DataFrame}``.
    output_path:
        Filesystem path to the graber_data.pkl pickle file.
    coin:
        Lowercase coin name passed to get_candles_history, e.g. ``"link"``.
    interval_seconds:
        Sleep duration between polls (default 60).
    """

    def __init__(
        self,
        stock: StockInterface,
        output_path: str,
        coin: str,
        interval_seconds: int = 60,
    ) -> None:
        self.stock = stock
        self.output_path = output_path
        self.coin = coin
        self.interval_seconds = interval_seconds
        self.running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enter the poll loop.

        - Sets running=True.
        - Sleeps interval_seconds, then calls _poll().
        - Repeats until running is False.
        - Catches KeyboardInterrupt → sets running=False, exits cleanly.
        """
        self.running = True
        try:
            while self.running:
                time.sleep(self.interval_seconds)
                if self.running:
                    self._poll()
        except KeyboardInterrupt:
            logger.info("LiveDataCollector: KeyboardInterrupt received, shutting down.")
            self.running = False

    def stop(self) -> None:
        """Signal the poll loop to exit on its next iteration."""
        self.running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        """Fetch latest 1-min candles and append any new rows to output_path.

        Steps
        -----
        1. Load existing DataFrame (empty DataFrame if file missing).
        2. Determine last_timestamp from existing index.
        3. Call stock.get_candles_history([1], coin) → result[1].
        4. Rename open/high/low/close/volume → o/h/l/c/v.
        5. Filter rows where open_time > last_timestamp (all rows if empty).
        6. If new rows: call _append_save(existing, new_rows).
        """
        # Step 1: load existing
        existing = load_existing(self.output_path)
        if existing is None:
            existing = pd.DataFrame()

        # Step 2: determine last timestamp
        if not existing.empty:
            last_timestamp = existing.index[-1]
        else:
            last_timestamp = None

        # Step 3: fetch from exchange
        result = self.stock.get_candles_history([1], self.coin)
        raw_df: pd.DataFrame = result[1]

        # Step 4: rename columns
        new_data = raw_df.rename(columns=_RENAME_MAP)

        # Step 5: filter new rows only
        if last_timestamp is not None:
            new_rows = new_data[new_data.index > last_timestamp]
        else:
            new_rows = new_data

        # Step 6: persist if there are new candles
        if not new_rows.empty:
            logger.info(
                "LiveDataCollector: appending %d new candle(s) to %s",
                len(new_rows),
                self.output_path,
            )
            self._append_save(existing, new_rows)
        else:
            logger.debug(
                "LiveDataCollector: no new candles (last candle not yet closed)"
            )

    def _append_save(self, existing: pd.DataFrame, new_rows: pd.DataFrame) -> None:
        """Concatenate existing + new_rows, dedup on open_time, atomic save.

        Parameters
        ----------
        existing:
            Previously saved DataFrame (may be empty).
        new_rows:
            New candle rows to append.
        """
        if existing.empty:
            combined = new_rows.copy()
        else:
            combined = pd.concat([existing, new_rows])

        # Deduplicate on open_time index — keep last occurrence
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        save_atomic(combined, self.output_path)
