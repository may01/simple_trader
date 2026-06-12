"""Unit tests for grabers.grab_binance.main() — warmup-extended grab range."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


# DATA_START=1700000000000 minus 105 rows × 1440 min × 60_000 ms (= 105 days)
DATA_START_MS = 1_700_000_000_000
DATA_END_MS = 1_700_100_000_000
WARMUP_START_MS = DATA_START_MS - 105 * 1440 * 60_000  # 1_690_928_000_000


def _make_df(n: int, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    idx.name = "open_time"
    return pd.DataFrame(
        {
            "o": np.ones(n),
            "h": np.ones(n) * 1.1,
            "l": np.ones(n) * 0.9,
            "c": np.ones(n),
            "v": np.ones(n) * 100,
            "close_time": idx,
            "taker_base_vol": np.ones(n) * 50,
        },
        index=idx,
    )


def _set_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("DATA_START", str(DATA_START_MS))
    monkeypatch.setenv("DATA_END", str(DATA_END_MS))
    return str(tmp_path / "graber_data.pkl")


def test_main_grabs_warmup_extended_range_when_no_file(monkeypatch, tmp_path):
    """No existing file: grab_data called from warmup-extended start."""
    path = _set_env(monkeypatch, tmp_path)

    grabbed = _make_df(5, "2024-01-01")
    with patch("grabers.grab_binance.grab_data", return_value=grabbed) as mock_grab, \
         patch("helpers.graber_data_path", return_value=path), \
         patch("grabers.init_folders.init_dataset_folders"):
        from grabers.grab_binance import main
        main()

    mock_grab.assert_called_once_with("link_usdt", WARMUP_START_MS, DATA_END_MS)
    assert pd.read_pickle(path).equals(grabbed)


def test_main_no_fetch_when_warmup_range_covered(monkeypatch, tmp_path):
    """Existing file covering [warmup_start, end): no fetch triggered."""
    path = _set_env(monkeypatch, tmp_path)

    idx = pd.DatetimeIndex(
        [
            pd.Timestamp(WARMUP_START_MS, unit="ms", tz="UTC"),
            pd.Timestamp(DATA_END_MS, unit="ms", tz="UTC"),
        ],
        name="open_time",
    )
    existing = pd.DataFrame({"o": [1.0, 1.0], "c": [1.0, 1.0]}, index=idx)
    existing.to_pickle(path)

    with patch("grabers.grab_binance.grab_data") as mock_grab, \
         patch("helpers.graber_data_path", return_value=path), \
         patch("grabers.init_folders.init_dataset_folders"):
        from grabers.grab_binance import main
        main()

    mock_grab.assert_not_called()


def test_main_fetches_warmup_head_for_existing_file(monkeypatch, tmp_path):
    """Existing file starting at DATA_START: head [warmup_start, DATA_START) fetched."""
    path = _set_env(monkeypatch, tmp_path)

    # Existing spans [DATA_START, DATA_END] so only the warmup head is missing
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp(DATA_START_MS, unit="ms", tz="UTC"),
            pd.Timestamp(DATA_END_MS, unit="ms", tz="UTC"),
        ],
        name="open_time",
    )
    existing = pd.DataFrame({"o": [1.0, 1.0], "c": [1.0, 1.0]}, index=idx)
    existing.to_pickle(path)

    head = pd.DataFrame(
        {"o": [1.0], "c": [1.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp(WARMUP_START_MS, unit="ms", tz="UTC")], name="open_time"
        ),
    )
    with patch("grabers.grab_binance.grab_data", return_value=head) as mock_grab, \
         patch("grabers.grab_binance.validate_1min_spacing"), \
         patch("helpers.graber_data_path", return_value=path), \
         patch("grabers.init_folders.init_dataset_folders"):
        from grabers.grab_binance import main
        main()

    mock_grab.assert_called_once_with("link_usdt", WARMUP_START_MS, DATA_START_MS)
    loaded = pd.read_pickle(path)
    assert loaded.index[0] == pd.Timestamp(WARMUP_START_MS, unit="ms", tz="UTC")
