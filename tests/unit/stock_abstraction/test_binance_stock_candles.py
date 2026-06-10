# tests/unit/stock_abstraction/test_binance_stock_candles.py
# Unit tests for Stock_Binance candle fetching — no real network calls.

import time
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from stocks.binance_stock import Stock_Binance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_klines(n: int, interval_ms: int = 60_000) -> list:
    """Generate n synthetic Binance klines starting from a fixed timestamp."""
    base = 1_700_000_000_000  # arbitrary fixed ms timestamp
    return [
        [
            base + i * interval_ms,  # open_time
            "1.0",   # open
            "2.0",   # high
            "0.5",   # low
            "1.5",   # close
            "100.0", # volume
            base + i * interval_ms + interval_ms - 1,  # close_time
            "150.0", # qav
            "10",    # num_trades
            "50.0",  # taker_base_vol
            "75.0",  # taker_quote_vol
            "0",     # ignore
        ]
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_vars(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "test_key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("EXCHANGE_FEE", "0.001")


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestStock_BinanceInit:

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_SECRET", "s")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setenv("EXCHANGE_FEE", "0.001")
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        with pytest.raises(Exception):
            Stock_Binance()

    def test_missing_secret_raises(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "k")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.setenv("EXCHANGE_FEE", "0.001")
        monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
        with pytest.raises(Exception):
            Stock_Binance()

    def test_missing_pair_raises(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "k")
        monkeypatch.setenv("BINANCE_API_SECRET", "s")
        monkeypatch.setenv("EXCHANGE_FEE", "0.001")
        monkeypatch.delenv("PAIR", raising=False)
        with pytest.raises(Exception):
            Stock_Binance()

    def test_missing_fee_raises(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "k")
        monkeypatch.setenv("BINANCE_API_SECRET", "s")
        monkeypatch.setenv("PAIR", "link_usdt")
        monkeypatch.delenv("EXCHANGE_FEE", raising=False)
        with pytest.raises(Exception):
            Stock_Binance()

    def test_get_pair_name(self, env_vars):
        with patch("stocks.binance_stock.Client"):
            s = Stock_Binance()
        assert s.get_pair_name() == "LINKUSDT"

    def test_was_init_true(self, env_vars):
        with patch("stocks.binance_stock.Client"):
            s = Stock_Binance()
        assert s.was_init is True

    def test_fee_set(self, env_vars):
        with patch("stocks.binance_stock.Client"):
            s = Stock_Binance()
        assert s.fee == 0.001

    def test_coin_and_coin_base_parsed(self, env_vars):
        with patch("stocks.binance_stock.Client"):
            s = Stock_Binance()
        assert s.coin == "link"
        assert s.coin_base == "usdt"

    def test_inherits_rate_limit_defaults(self, env_vars):
        with patch("stocks.binance_stock.Client"):
            s = Stock_Binance()
        assert s.weight == 0
        assert s.overflow_weight == 5000
        assert s.overflow_weight_time == 60

    def test_pair_name_different_pair(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "k")
        monkeypatch.setenv("BINANCE_API_SECRET", "s")
        monkeypatch.setenv("PAIR", "btc_usdt")
        monkeypatch.setenv("EXCHANGE_FEE", "0.001")
        with patch("stocks.binance_stock.Client"):
            s = Stock_Binance()
        assert s.get_pair_name() == "BTCUSDT"


# ---------------------------------------------------------------------------
# get_candles_history tests
# ---------------------------------------------------------------------------

class TestGetCandlesHistory:

    def _make_stock(self, env_vars, klines_per_call: int = 120):
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            # 1-min klines
            mock_client.get_historical_klines.return_value = _mock_klines(
                klines_per_call, interval_ms=60_000
            )
            s = Stock_Binance()
        return s, mock_client

    def test_candles_history_keys_all_tfs(self, env_vars):
        """Result dict must contain all requested TFs."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(200)
            s = Stock_Binance()
            result = s.get_candles_history([1, 5, 15, 60, 240, 1440], "link")
        assert set(result.keys()) >= {1, 5, 15, 60, 240, 1440}

    def test_candles_history_returns_dataframes(self, env_vars):
        """Each value in result dict must be a DataFrame."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(200)
            s = Stock_Binance()
            result = s.get_candles_history([1, 5, 15, 60, 240, 1440], "link")
        for tf, df in result.items():
            assert isinstance(df, pd.DataFrame), f"TF {tf} is not a DataFrame"

    def test_candles_history_max_120_rows(self, env_vars):
        """Each TF must have at most 120 rows."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(200)
            s = Stock_Binance()
            result = s.get_candles_history([1, 5, 15, 60, 240, 1440], "link")
        for tf, df in result.items():
            assert len(df) <= 120, f"TF {tf} has {len(df)} rows (> 120)"

    def test_candles_history_ohlcv_columns(self, env_vars):
        """Each DataFrame must have open, high, low, close, volume columns."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(200)
            s = Stock_Binance()
            result = s.get_candles_history([1, 5], "link")
        for tf in [1, 5]:
            df = result[tf]
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in df.columns, f"TF {tf} missing column '{col}'"

    def test_candles_history_float_columns(self, env_vars):
        """Numeric columns must be float64."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(120)
            s = Stock_Binance()
            result = s.get_candles_history([1], "link")
        df = result[1]
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == "float64", f"Column {col} dtype is {df[col].dtype}"

    def test_candles_history_empty_klines_skipped(self, env_vars):
        """If Binance returns no klines, that TF is skipped (not in result)."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = []
            s = Stock_Binance()
            result = s.get_candles_history([1, 15, 60, 1440], "link")
        assert len(result) == 0

    def test_candles_history_weight_incremented(self, env_vars):
        """Weight should increase by 2 per klines call."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(10)
            s = Stock_Binance()
            # Request only one base interval group (TF=15 → 1 call)
            s.get_candles_history([15], "link")
        assert s.weight == 2

    def test_candles_history_only_requested_tfs_returned(self, env_vars):
        """Result contains only requested TFs, not extras."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(120)
            s = Stock_Binance()
            result = s.get_candles_history([1], "link")
        # Should only have TF=1, NOT 5 (even though 1-min base can produce 5)
        assert 1 in result
        assert 5 not in result

    def test_candles_history_subset_tfs(self, env_vars):
        """Requesting a subset of TFs returns only those TFs."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(200)
            s = Stock_Binance()
            result = s.get_candles_history([60, 240], "link")
        assert set(result.keys()) == {60, 240}

    def test_candles_history_datetime_index(self, env_vars):
        """DataFrames must have a DatetimeIndex."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(120)
            s = Stock_Binance()
            result = s.get_candles_history([1], "link")
        df = result[1]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_weight_resets_on_overflow(self, env_vars):
        """When weight exceeds overflow_weight, it resets to 0 after sleep."""
        with patch("stocks.binance_stock.Client") as mock_cls, \
             patch("stocks.binance_stock.time.sleep") as mock_sleep:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(10)
            s = Stock_Binance()
            # Manually set weight near overflow
            s.weight = 4999
            s.overflow_weight = 5000
            # Trigger one more call (adds 2 → 5001 > 5000)
            s.get_candles_history([15], "link")
        mock_sleep.assert_called_once_with(60)
        assert s.weight == 0


# ---------------------------------------------------------------------------
# get_candles_range tests
# ---------------------------------------------------------------------------

class TestGetCandlesRange:

    def test_returns_empty_when_no_klines(self, env_vars):
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = []
            s = Stock_Binance()
            result = s.get_candles_range("LINKUSDT", 1_000_000_000, 1_001_000_000)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_dataframe_with_short_cols(self, env_vars):
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(10)
            s = Stock_Binance()
            result = s.get_candles_range("LINKUSDT", 0, 60_000)
        for col in ["o", "h", "l", "c", "v"]:
            assert col in result.columns, f"Missing column '{col}'"

    def test_chunks_large_range(self, env_vars):
        """A range spanning 2 chunks produces 2 get_historical_klines calls."""
        CHUNK_MS = 30 * 24 * 60 * 60 * 1000
        start = 0
        end = CHUNK_MS + 1  # just over 1 chunk
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(5)
            s = Stock_Binance()
            result = s.get_candles_range("LINKUSDT", start, end)
        assert mock_client.get_historical_klines.call_count == 2

    def test_float_columns_in_range(self, env_vars):
        """o/h/l/c/v and taker_base_vol must be float64."""
        with patch("stocks.binance_stock.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_historical_klines.return_value = _mock_klines(10)
            s = Stock_Binance()
            result = s.get_candles_range("LINKUSDT", 0, 60_000)
        for col in ["o", "h", "l", "c", "v", "taker_base_vol"]:
            assert result[col].dtype == "float64", f"Column {col} dtype is {result[col].dtype}"
