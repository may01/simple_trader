"""grab_binance.py — Raw 1-minute OHLCV kline collection from Binance.

Downloads Binance 1-min klines for DATA_START → DATA_END and writes
graber_data.pkl.  Supports incremental mode: if file exists, only the
missing date range is downloaded and merged.
"""

import os
import pandas as pd
from binance.client import Client

CHUNK_MS = 30 * 24 * 60 * 60 * 1000  # 30 days in milliseconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_klines(klines: list) -> pd.DataFrame:
    """Parse raw Binance kline list into a tidy DataFrame."""
    columns = [
        "open_time", "o", "h", "l", "c", "v",
        "close_time", "qav", "num_trades",
        "taker_base_vol", "taker_quote_vol", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.index.name = "open_time"
    for col in ["o", "h", "l", "c", "v", "taker_base_vol"]:
        df[col] = df[col].astype(float)
    return df[["o", "h", "l", "c", "v", "close_time", "taker_base_vol"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grab_data(pair: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch 1-min OHLCV klines from Binance for the given ms range.

    Parameters
    ----------
    pair:
        Coin pair in underscore-separated lowercase form, e.g. ``"link_usdt"``.
        Split on ``"_"``, uppercased, concatenated → Binance symbol
        (``"LINKUSDT"``).
    start_ms:
        Range start as Unix milliseconds (inclusive).
    end_ms:
        Range end as Unix milliseconds (exclusive / upper bound).

    Returns
    -------
    pd.DataFrame
        Columns: o, h, l, c, v, close_time, taker_base_vol.
        Index:   pd.DatetimeIndex (open_time), UTC, 1-min frequency.
    """
    api_key = os.environ["BINANCE_API_KEY"]
    api_secret = os.environ["BINANCE_API_SECRET"]
    client = Client(api_key, api_secret)

    parts = pair.split("_")
    symbol = (parts[0] + parts[1]).upper()  # "link_usdt" → "LINKUSDT"

    all_dfs = []
    current_start = start_ms
    while current_start < end_ms:
        current_end = min(current_start + CHUNK_MS, end_ms)
        klines = client.get_historical_klines(
            symbol, "1m", str(current_start), str(current_end)
        )
        if klines:
            df = _parse_klines(klines)
            all_dfs.append(df)
        current_start = current_end

    if not all_dfs:
        return pd.DataFrame(
            columns=["open_time", "o", "h", "l", "c", "v", "close_time", "taker_base_vol"]
        )

    result = pd.concat(all_dfs).drop_duplicates().sort_index()
    return result


def load_existing(path: str) -> "pd.DataFrame | None":
    """Load a pickle DataFrame from *path*, or return None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def merge_incremental(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Merge two DataFrames, dropping duplicate index entries, sorted by index."""
    combined = pd.concat([existing, new_data])
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.sort_index(inplace=True)
    return combined


def save_atomic(df: pd.DataFrame, path: str) -> None:
    """Write *df* to *path* atomically (write to .tmp then rename)."""
    tmp_path = path + ".tmp"
    df.to_pickle(tmp_path)
    os.rename(tmp_path, path)


def validate_1min_spacing(df: pd.DataFrame) -> None:
    """Assert that consecutive index entries are exactly 1 minute apart.

    Empty DataFrame is a no-op.  Raises ValueError on gap or duplicate.
    """
    if df.empty:
        return

    expected = pd.Timedelta("1min")
    diffs = df.index.to_series().diff().dropna()
    bad = diffs[diffs != expected]
    if not bad.empty:
        first_bad = bad.index[0]
        actual_diff = bad.iloc[0]
        raise ValueError(
            f"1-minute spacing violated at index {first_bad}: "
            f"expected {expected}, got {actual_diff}"
        )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from helpers import graber_data_path
    from grabers.init_folders import init_dataset_folders
    from config_loader import warmup_start_ms

    api_key = os.environ["BINANCE_API_KEY"]
    api_secret = os.environ["BINANCE_API_SECRET"]
    pair = os.environ["PAIR"]
    # Grab starts earlier than DATA_START so indicators have full warmup
    # history at the simulation start date.
    data_start = warmup_start_ms(int(os.environ["DATA_START"]))
    data_end = int(os.environ["DATA_END"])

    # Ensure folders exist
    init_dataset_folders()

    # Load existing data
    output_path = graber_data_path()
    existing = load_existing(output_path)

    if existing is not None:
        existing_start = int(existing.index[0].timestamp() * 1000)
        existing_end = int(existing.index[-1].timestamp() * 1000)

        new_dfs = []
        if data_start < existing_start:
            new_dfs.append(grab_data(pair, data_start, existing_start))
        if data_end > existing_end:
            new_dfs.append(grab_data(pair, existing_end, data_end))

        if new_dfs:
            new_data = pd.concat(new_dfs)
            merged = merge_incremental(existing, new_data)
            validate_1min_spacing(merged)
            save_atomic(merged, output_path)
            print(f"Saved {len(merged)} rows to {output_path}")
        else:
            print("No new data needed — existing range covers requested range")
    else:
        new_data = grab_data(pair, data_start, data_end)
        validate_1min_spacing(new_data)
        save_atomic(new_data, output_path)
        print(f"Saved {len(new_data)} rows to {output_path}")


if __name__ == "__main__":
    main()
