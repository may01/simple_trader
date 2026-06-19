# tests/unit/stock_abstraction/test_base_stock.py
import pytest
import pandas as pd
import numpy as np
from stocks.base_stock import StockInterface
from constants import STATUS_FAIL


class TestStockInterfaceInitialization:
    """Test constructor and default attributes."""

    def test_default_init(self):
        """Test StockInterface with no arguments."""
        s = StockInterface()
        assert s.key == ""
        assert s.secret == ""
        assert s.coin == ""
        assert s.coin_base == ""
        assert s.was_init == False

    def test_init_with_credentials(self):
        """Test StockInterface with credentials."""
        s = StockInterface(key="test_key", secret="test_secret", coin="BTC", coin_base="USDT")
        assert s.key == "test_key"
        assert s.secret == "test_secret"
        assert s.coin == "BTC"
        assert s.coin_base == "USDT"

    def test_was_init_false(self):
        """Verify was_init defaults to False."""
        s = StockInterface()
        assert s.was_init is False

    def test_fee_not_set(self):
        """Verify fee raises AttributeError before initialization."""
        s = StockInterface()
        with pytest.raises(AttributeError):
            _ = s.fee

    def test_weight_defaults(self):
        """Verify rate-limit tracking defaults."""
        s = StockInterface()
        assert s.weight == 0
        assert s.weight_set_time == 0.0
        assert s.overflow_weight == 5000
        assert s.overflow_weight_time == 60


class TestStockInterfaceAbstractMethods:
    """Test all abstract methods return typed defaults and print 'INIT STOCK'."""

    def test_get_candles_history_returns_dict(self, capsys):
        """Verify get_candles_history prints and returns dict."""
        s = StockInterface()
        result = s.get_candles_history([1, 5], "link")
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_get_candles_history_with_time_point(self, capsys):
        """Verify get_candles_history with time_point argument."""
        s = StockInterface()
        result = s.get_candles_history([1000, 2000], "BTC", time_point=100)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert isinstance(result, dict)

    def test_get_candles_range_returns_dataframe(self, capsys):
        """Verify get_candles_range prints and returns empty DataFrame."""
        s = StockInterface()
        result = s.get_candles_range("BTCUSDT", 1000000, 2000000)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_trade_returns_status_fail(self, capsys):
        """Verify trade prints and returns (STATUS_FAIL, {})."""
        s = StockInterface()
        status, data = s.trade("TRADE_BUY", 20.0, 5.0)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL
        assert isinstance(data, dict)

    def test_trade_with_force(self, capsys):
        """Verify trade with force=True."""
        s = StockInterface()
        status, data = s.trade("TRADE_SELL", 30.0, 10.0, force=True)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL

    def test_order_info_returns_status_fail(self, capsys):
        """Verify order_info prints and returns (STATUS_FAIL, {})."""
        s = StockInterface()
        status, data = s.order_info("order123")
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL
        assert isinstance(data, dict)

    def test_cancel_order_returns_status_fail(self, capsys):
        """Verify cancel_order prints and returns (STATUS_FAIL, {})."""
        s = StockInterface()
        status, data = s.cancel_order("order456")
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL

    def test_depth_returns_empty_dataframes(self, capsys):
        """Verify depth prints and returns two empty DataFrames."""
        s = StockInterface()
        bids, asks = s.depth(10)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert isinstance(bids, pd.DataFrame)
        assert isinstance(asks, pd.DataFrame)
        assert len(bids) == 0
        assert len(asks) == 0

    def test_info_returns_dict(self, capsys):
        """Verify info prints and returns empty dict."""
        s = StockInterface()
        result = s.info()
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert isinstance(result, dict)

    def test_is_invalid_amount_returns_true(self, capsys):
        """Verify is_invalid_amount prints and returns True."""
        s = StockInterface()
        result = s.is_invalid_amount(0.001, 50000.0)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert result is True

    def test_funds_returns_status_fail(self, capsys):
        """Verify funds prints and returns (STATUS_FAIL, 0.0)."""
        s = StockInterface()
        status, amount = s.funds("BTC")
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL
        assert amount == 0.0

    def test_funds_with_asset_type(self, capsys):
        """Verify funds with asset_type argument."""
        s = StockInterface()
        status, amount = s.funds("USDT", asset_type="locked")
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL

    def test_get_aviable_loan_returns_status_fail(self, capsys):
        """Verify get_aviable_loan prints and returns (STATUS_FAIL, 0.0)."""
        s = StockInterface()
        status, amount = s.get_aviable_loan("BTC")
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL
        assert amount == 0.0

    def test_borrow_returns_status_fail(self, capsys):
        """Verify borrow prints and returns (STATUS_FAIL, 0.0)."""
        s = StockInterface()
        status, amount = s.borrow("BTC", 1.5)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL

    def test_repay_returns_status_fail(self, capsys):
        """Verify repay prints and returns (STATUS_FAIL, 0.0)."""
        s = StockInterface()
        status, amount = s.repay("BTC", 1.0)
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert status == STATUS_FAIL

    def test_get_pair_name_returns_empty_string(self, capsys):
        """Verify get_pair_name prints and returns empty string."""
        s = StockInterface()
        result = s.get_pair_name()
        captured = capsys.readouterr()
        assert "INIT STOCK" in captured.out
        assert isinstance(result, str)
        assert result == ""

    def test_set_operation_sleep_no_op(self, capsys):
        """Verify set_operation_sleep is a no-op (no print)."""
        s = StockInterface()
        s.set_operation_sleep()  # Should not print anything
        captured = capsys.readouterr()
        assert "INIT STOCK" not in captured.out


class TestResampleToTf:
    """Test the _resample_to_tf protected helper method."""

    def test_resample_1min_to_5min(self):
        """Resample 10 1-minute candles to 5-minute candles."""
        idx = pd.date_range("2024-01-01", periods=10, freq="1min")
        df = pd.DataFrame(
            {
                "open": [1.0] * 10,
                "high": [2.0] * 10,
                "low": [0.5] * 10,
                "close": [1.5] * 10,
                "volume": [100.0] * 10,
            },
            index=idx,
        )
        s = StockInterface()
        result = s._resample_to_tf(df, 1, 5)

        # 10 minutes of data should produce 2 5-minute candles
        assert len(result) == 2
        # First candle: open=1.0, high=2.0, low=0.5, close=1.5, volume=500.0
        assert result["open"].iloc[0] == 1.0
        assert result["high"].iloc[0] == 2.0
        assert result["low"].iloc[0] == 0.5
        assert result["close"].iloc[0] == 1.5
        assert result["volume"].iloc[0] == 500.0

    def test_resample_empty_dataframe(self):
        """Empty DataFrame should return empty."""
        df = pd.DataFrame()
        s = StockInterface()
        result = s._resample_to_tf(df, 1, 5)
        assert len(result) == 0

    def test_resample_5min_to_15min(self):
        """Resample 15 5-minute candles to 15-minute candles."""
        idx = pd.date_range("2024-01-01", periods=15, freq="5min")
        df = pd.DataFrame(
            {
                "open": list(range(1, 16)),
                "high": list(range(2, 17)),
                "low": list(range(0, 15)),
                "close": list(range(1, 16)),
                "volume": [10.0] * 15,
            },
            index=idx,
        )
        s = StockInterface()
        result = s._resample_to_tf(df, 5, 15)

        # 15 5-minute candles = 75 minutes, should produce 5 15-minute candles
        assert len(result) == 5
        # First 15-min candle aggregates 3 5-min candles (indices 0-2)
        assert result["open"].iloc[0] == 1  # first of first candle
        assert result["high"].iloc[0] == 4  # max of [2, 3, 4]
        assert result["close"].iloc[0] == 3  # last of third candle
        assert result["volume"].iloc[0] == 30.0  # sum of [10, 10, 10]

    def test_resample_with_varying_values(self):
        """Test resample with varying OHLC values."""
        idx = pd.date_range("2024-01-01", periods=6, freq="1min")
        df = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "high": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                "low": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "close": [15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                "volume": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            },
            index=idx,
        )
        s = StockInterface()
        result = s._resample_to_tf(df, 1, 3)

        # 6 minutes to 3-minute intervals = 2 candles
        assert len(result) == 2

        # First 3-minute candle (minutes 0-2)
        assert result["open"].iloc[0] == 10.0  # first open
        assert result["high"].iloc[0] == 22.0  # max of [20, 21, 22]
        assert result["low"].iloc[0] == 5.0  # min of [5, 6, 7]
        assert result["close"].iloc[0] == 17.0  # last close
        assert result["volume"].iloc[0] == 330.0  # sum of [100, 110, 120]

        # Second 3-minute candle (minutes 3-5)
        assert result["open"].iloc[1] == 13.0  # first open
        assert result["high"].iloc[1] == 25.0  # max of [23, 24, 25]
        assert result["low"].iloc[1] == 8.0  # min of [8, 9, 10]
        assert result["close"].iloc[1] == 20.0  # last close
        assert result["volume"].iloc[1] == 420.0  # sum of [130, 140, 150]

    def test_resample_preserves_columns(self):
        """Only OHLCV columns should be resampled; extra columns dropped."""
        idx = pd.date_range("2024-01-01", periods=4, freq="1min")
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [2.0, 3.0, 4.0, 5.0],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.5, 2.5, 3.5, 4.5],
                "volume": [100.0, 110.0, 120.0, 130.0],
            },
            index=idx,
        )
        s = StockInterface()
        result = s._resample_to_tf(df, 1, 2)

        # Should have all OHLCV columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_resample_preserves_taker_base_vol(self):
        """taker_base_vol must be summed through resampling (needed for
        buy_volume in the live indicator path)."""
        idx = pd.date_range("2024-01-01", periods=6, freq="1min")
        df = pd.DataFrame(
            {
                "open": [1.0] * 6,
                "high": [2.0] * 6,
                "low": [0.5] * 6,
                "close": [1.5] * 6,
                "volume": [100.0] * 6,
                "taker_base_vol": [10.0] * 6,
            },
            index=idx,
        )
        s = StockInterface()
        result = s._resample_to_tf(df, 1, 3)
        assert "taker_base_vol" in result.columns
        # Two 3-min candles, each summing three 10.0 source values.
        assert result["taker_base_vol"].iloc[0] == 30.0
        assert result["taker_base_vol"].iloc[1] == 30.0

    def test_resample_single_candle_unchanged(self):
        """Single candle should resample to itself."""
        idx = pd.date_range("2024-01-01", periods=1, freq="1min")
        df = pd.DataFrame(
            {
                "open": [10.0],
                "high": [20.0],
                "low": [5.0],
                "close": [15.0],
                "volume": [100.0],
            },
            index=idx,
        )
        s = StockInterface()
        result = s._resample_to_tf(df, 1, 5)

        # Single candle resampled to 5-min interval
        assert len(result) == 1
        assert result["open"].iloc[0] == 10.0
        assert result["close"].iloc[0] == 15.0
        assert result["volume"].iloc[0] == 100.0
