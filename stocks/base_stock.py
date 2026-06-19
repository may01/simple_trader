# stocks/base_stock.py — abstract stock/exchange interface
# All exchange implementations (Binance, etc.) inherit from StockInterface.
# Robot, LiveData, TrainRobot call stock.item.method() — never know the concrete impl.

import pandas as pd
from constants import STATUS_FAIL


class StockInterface:
    """Abstract interface for all exchange implementations.

    This base class defines the contract all concrete stock implementations must
    satisfy. Default no-op implementations print "INIT STOCK" to fail loudly if
    an uninitialized instance is used.

    Rate-limit tracking (weight, weight_set_time, overflow_weight, overflow_weight_time)
    is managed in the base class but used by concrete implementations.

    NOTE: fee is NOT set in __init__ — accessing it before do_stock_init() raises AttributeError.
    """

    def __init__(self, key: str = "", secret: str = "", coin: str = "", coin_base: str = ""):
        """Initialize a stock/exchange interface.

        Args:
            key: API key credential (empty string for testing/mocking)
            secret: API secret credential (empty string for testing/mocking)
            coin: Base asset (e.g., "BTC")
            coin_base: Quote asset (e.g., "USDT")
        """
        self.key = key
        self.secret = secret
        self.coin = coin
        self.coin_base = coin_base
        self.was_init = False
        self.weight = 0
        self.weight_set_time = 0.0
        self.overflow_weight = 5000
        self.overflow_weight_time = 60
        # NOTE: self.fee is NOT set here — accessing it before do_stock_init() raises AttributeError

    def get_candles_history(self, time_list: list[int], coin: str, time_point: int = 0) -> dict:
        """Fetch historical candles for a given list of timestamps.

        Default no-op: prints "INIT STOCK" and returns empty dict.

        Args:
            time_list: List of timestamps (ms) to fetch candles for
            coin: Coin symbol (e.g., "BTC")
            time_point: Reference time point (optional, default 0)

        Returns:
            dict: Empty dict (concrete impl returns {timestamp: candle_data})
        """
        print("INIT STOCK")
        return {}

    def get_candles_range(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch candles for a time range.

        Default no-op: prints "INIT STOCK" and returns empty DataFrame.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_ms: Start timestamp (milliseconds)
            end_ms: End timestamp (milliseconds)

        Returns:
            pd.DataFrame: Empty DataFrame (concrete impl returns OHLCV data)
        """
        print("INIT STOCK")
        return pd.DataFrame()

    def trade(self, trade_type: str, price: float, amount: float, force: bool = False) -> tuple:
        """Execute a buy or sell trade.

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, {}).

        Args:
            trade_type: "TRADE_BUY" or "TRADE_SELL"
            price: Price per unit
            amount: Quantity to trade
            force: Force execution (optional, default False)

        Returns:
            tuple: (status_code, order_data_dict)
                   status_code is STATUS_SUCCESS or STATUS_FAIL
        """
        print("INIT STOCK")
        return (STATUS_FAIL, {})

    def order_info(self, order_id: str) -> tuple:
        """Get information about a specific order.

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, {}).

        Args:
            order_id: Exchange order ID

        Returns:
            tuple: (status_code, order_info_dict)
        """
        print("INIT STOCK")
        return (STATUS_FAIL, {})

    def cancel_order(self, order_id: str) -> tuple:
        """Cancel a pending order.

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, {}).

        Args:
            order_id: Exchange order ID

        Returns:
            tuple: (status_code, cancel_response_dict)
        """
        print("INIT STOCK")
        return (STATUS_FAIL, {})

    def depth(self, quantity: int) -> tuple:
        """Get order book depth (bid/ask levels).

        Default no-op: prints "INIT STOCK" and returns (empty_bids_df, empty_asks_df).

        Args:
            quantity: Number of levels to retrieve

        Returns:
            tuple: (bids_dataframe, asks_dataframe)
                   Each is a pd.DataFrame with price/amount columns
        """
        print("INIT STOCK")
        return (pd.DataFrame(), pd.DataFrame())

    def info(self) -> dict:
        """Get exchange/account information.

        Default no-op: prints "INIT STOCK" and returns empty dict.

        Returns:
            dict: Exchange or account info
        """
        print("INIT STOCK")
        return {}

    def is_invalid_amount(self, amount: float, price: float) -> bool:
        """Check if a trade amount violates exchange minimums/maximums.

        Default no-op: prints "INIT STOCK" and returns True (invalid).

        Args:
            amount: Quantity to trade
            price: Price per unit

        Returns:
            bool: True if invalid, False if valid
        """
        print("INIT STOCK")
        return True

    def funds(self, coin: str, asset_type: str = "free") -> tuple:
        """Get available funds for a coin.

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, 0.0).

        Args:
            coin: Coin symbol (e.g., "BTC")
            asset_type: "free" for available, "locked" for in-order, etc. (default "free")

        Returns:
            tuple: (status_code, amount_float)
        """
        print("INIT STOCK")
        return (STATUS_FAIL, 0.0)

    def get_aviable_loan(self, coin: str) -> tuple:
        """Get available borrowing limit for margin trading.

        Note: Method name matches legacy spelling "aviable" (not "available").

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, 0.0).

        Args:
            coin: Coin symbol (e.g., "BTC")

        Returns:
            tuple: (status_code, borrowable_amount_float)
        """
        print("INIT STOCK")
        return (STATUS_FAIL, 0.0)

    def borrow(self, coin: str, amount: float) -> tuple:
        """Borrow funds for margin trading.

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, 0.0).

        Args:
            coin: Coin symbol (e.g., "BTC")
            amount: Amount to borrow

        Returns:
            tuple: (status_code, borrowed_amount_or_data)
        """
        print("INIT STOCK")
        return (STATUS_FAIL, 0.0)

    def repay(self, coin: str, amount: float) -> tuple:
        """Repay borrowed funds.

        Default no-op: prints "INIT STOCK" and returns (STATUS_FAIL, 0.0).

        Args:
            coin: Coin symbol (e.g., "BTC")
            amount: Amount to repay

        Returns:
            tuple: (status_code, repay_response)
        """
        print("INIT STOCK")
        return (STATUS_FAIL, 0.0)

    def get_pair_name(self) -> str:
        """Get the current trading pair as a string.

        Default no-op: prints "INIT STOCK" and returns empty string.

        Returns:
            str: Trading pair (e.g., "BTCUSDT") or empty string
        """
        print("INIT STOCK")
        return ""

    def set_operation_sleep(self) -> None:
        """Apply rate-limit sleep if weight exceeds overflow_weight.

        Base class provides no-op (no sleep).
        Concrete implementations override to check self.weight and time.sleep().
        """
        pass

    def _resample_to_tf(
        self, base_df: pd.DataFrame, source_interval_min: int, target_interval_min: int
    ) -> pd.DataFrame:
        """Resample OHLCV DataFrame from source to target timeframe.

        Protected helper (not overridden by subclasses).
        Uses pandas groupby/resample with proper OHLCV aggregation:
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum

        Args:
            base_df: Input OHLCV DataFrame (must have datetime index or 'timestamp' column)
            source_interval_min: Source timeframe in minutes (e.g., 1)
            target_interval_min: Target timeframe in minutes (e.g., 5)

        Returns:
            pd.DataFrame: Resampled OHLCV DataFrame
        """
        if base_df.empty:
            return base_df.copy()

        # Create a working copy and ensure datetime index
        df = base_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            # If no datetime index, assume there's a 'timestamp' column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
            else:
                # Already a DatetimeIndex or can't resample
                return df

        # Resample with proper OHLCV aggregation
        ohlcv_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            # taker_base_vol must survive resampling: LiveData derives
            # {tf}_buy_volume from it for the volume indicators.
            'taker_base_vol': 'sum',
        }

        # Build a resample rule: "5min", "1h", etc.
        target_str = f"{target_interval_min}min"

        # Only resample columns that exist in the dataframe
        agg_dict = {col: func for col, func in ohlcv_agg.items() if col in df.columns}

        result = df.resample(target_str).agg(agg_dict)

        return result
