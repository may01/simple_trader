# stocks/binance_stock.py — Binance exchange implementation
# Implements candle fetching (get_candles_history, get_candles_range) for live trading
# and historical data retrieval.

import logging
import os
import time
import pandas as pd

from binance.client import Client

from constants import STATUS_SUCCESS, STATUS_FAIL, TRADE_BUY, TRADE_SELL
from stocks.base_stock import StockInterface

log = logging.getLogger(__name__)

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
    """Binance exchange implementation — candle fetching and order management."""

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

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def trade(self, trade_type: str, price: float, amount: float, force: bool = False) -> tuple:
        """Place a margin limit order.

        Args:
            trade_type: TRADE_BUY or TRADE_SELL constant.
            price: Limit price (rounded to 2 decimal places).
            amount: Quantity (rounded to 2 decimal places).
            force: Unused flag kept for interface compatibility.

        Returns:
            (STATUS_SUCCESS, {"order_id": str}) on success,
            (STATUS_FAIL, {}) on invalid amount or exception.
        """
        if self.is_invalid_amount(amount, price):
            return (STATUS_FAIL, {})

        amount = round(amount, 2)
        price = round(price, 2)
        symbol = self.get_pair_name()
        side = "BUY" if trade_type == TRADE_BUY else "SELL"

        try:
            result = self.client.create_margin_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=amount,
                price=price,
            )
            self.weight += 6
            return (STATUS_SUCCESS, {"order_id": str(result["orderId"])})
        except Exception as exc:
            log.error("trade() error: %s", exc)
            # Network timeout — wait and retry once
            if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                time.sleep(60)
                try:
                    result = self.client.create_margin_order(
                        symbol=symbol,
                        side=side,
                        type="LIMIT",
                        timeInForce="GTC",
                        quantity=amount,
                        price=price,
                    )
                    self.weight += 6
                    return (STATUS_SUCCESS, {"order_id": str(result["orderId"])})
                except Exception as exc2:
                    log.error("trade() retry error: %s", exc2)
            return (STATUS_FAIL, {})

    def order_info(self, order_id: str) -> tuple:
        """Fetch current status of a margin order.

        Args:
            order_id: Exchange order ID string.

        Returns:
            (STATUS_SUCCESS, info_dict) or (STATUS_FAIL, {}).
            info_dict keys: status, start_amount, left_amount, rate.
        """
        try:
            result = self.client.get_margin_order(
                symbol=self.get_pair_name(), orderId=order_id
            )
            self.weight += 10
            return (STATUS_SUCCESS, {
                "status": result["status"],
                "start_amount": float(result["origQty"]),
                "left_amount": float(result["origQty"]) - float(result["executedQty"]),
                "rate": float(result["price"]),
            })
        except Exception as exc:
            log.error("order_info() error: %s", exc)
            return (STATUS_FAIL, {})

    def cancel_order(self, order_id: str) -> tuple:
        """Cancel a margin order, polling until cancellation is confirmed.

        Polls order_info() up to 10 times (2 s apart) when the exchange
        returns PENDING_CANCEL.

        Args:
            order_id: Exchange order ID string.

        Returns:
            Final (status, info_dict) from order_info(), or (STATUS_FAIL, {}).
        """
        try:
            result = self.client.cancel_margin_order(
                symbol=self.get_pair_name(), orderId=order_id
            )
            self.weight += 10
            if result.get("status") == "PENDING_CANCEL":
                for _ in range(10):
                    time.sleep(2)
                    status, info = self.order_info(order_id)
                    if status == STATUS_SUCCESS and info.get("status") != "PENDING_CANCEL":
                        return (status, info)
            return self.order_info(order_id)
        except Exception as exc:
            log.error("cancel_order() error: %s", exc)
            return (STATUS_FAIL, {})

    def depth(self, quantity: int) -> tuple:
        """Fetch order book depth.

        Args:
            quantity: Number of price levels to fetch.

        Returns:
            (asks_df, bids_df) — each a DataFrame with columns
            [price, quantity, cumulative], all float64, sorted by price.
        """
        result = self.client.get_order_book(
            symbol=self.get_pair_name(), limit=quantity
        )

        if quantity < 100:
            self.weight += 5
        elif quantity < 500:
            self.weight += 25
        elif quantity < 1000:
            self.weight += 50
        else:
            self.weight += 250

        def _build_df(entries: list) -> pd.DataFrame:
            df = pd.DataFrame(entries, columns=["price", "quantity"])
            df = df.astype({"price": float, "quantity": float})
            df = df.sort_values("price").reset_index(drop=True)
            df["cumulative"] = df["quantity"].cumsum()
            return df[["price", "quantity", "cumulative"]]

        asks_df = _build_df(result["asks"])
        bids_df = _build_df(result["bids"])
        return (asks_df, bids_df)

    def info(self) -> dict:
        """Return Binance symbol info for the current trading pair.

        Returns:
            Symbol info dict, or {} on error.
        """
        try:
            return self.client.get_symbol_info(self.get_pair_name())
        except Exception as exc:
            log.error("info() error: %s", exc)
            return {}

    def is_invalid_amount(self, amount: float, price: float) -> bool:
        """Check whether an amount violates exchange minimums.

        Checks LOT_SIZE (minQty) and MIN_NOTIONAL filters from symbol info.
        Uses 2× the minimums as a safety margin.

        Args:
            amount: Quantity to trade.
            price: Price per unit.

        Returns:
            True if the amount is too small or notional is too low; False otherwise.
        """
        try:
            symbol_info = self.info()
            if not symbol_info:
                return True
            filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
            lot = filters.get("LOT_SIZE")
            notional = filters.get("MIN_NOTIONAL")
            if lot is None or notional is None:
                return True
            min_qty = float(lot["minQty"])
            min_notional = float(notional["minNotional"])
            if amount < 2 * min_qty:
                return True
            if amount * price < 2 * min_notional:
                return True
            return False
        except Exception as exc:
            log.error("is_invalid_amount() error: %s", exc)
            return True

    def funds(self, coin: str, asset_type: str = "free") -> tuple:
        """Get margin account balance for a coin.

        Args:
            coin: Coin symbol (e.g., "LINK").
            asset_type: Asset sub-type key: "free", "locked", "borrowed", etc.

        Returns:
            (STATUS_SUCCESS, float) or (STATUS_FAIL, 0.0).
        """
        try:
            result = self.client.get_margin_account()
            for asset in result.get("userAssets", []):
                if asset.get("asset", "").upper() == coin.upper():
                    return (STATUS_SUCCESS, float(asset[asset_type]))
            return (STATUS_SUCCESS, 0.0)
        except Exception as exc:
            log.error("funds() error: %s", exc)
            return (STATUS_FAIL, 0.0)

    def get_aviable_loan(self, coin: str) -> tuple:
        """Query the maximum available margin loan for a coin.

        Note: Method name matches legacy spelling "aviable".

        Args:
            coin: Coin symbol (e.g., "LINK").

        Returns:
            (STATUS_SUCCESS, float) or (STATUS_FAIL, 0.0).
        """
        try:
            result = self.client.get_max_margin_loan(asset=coin.upper())
            return (STATUS_SUCCESS, float(result["amount"]))
        except Exception as exc:
            log.error("get_aviable_loan() error: %s", exc)
            return (STATUS_FAIL, 0.0)

    def borrow(self, coin: str, amount: float) -> tuple:
        """Borrow funds via margin loan.

        Args:
            coin: Coin symbol (e.g., "LINK").
            amount: Amount to borrow.

        Returns:
            (STATUS_SUCCESS, amount) or (STATUS_FAIL, 0.0).
        """
        try:
            self.client.create_margin_loan(asset=coin.upper(), amount=str(amount))
            return (STATUS_SUCCESS, amount)
        except Exception as exc:
            log.error("borrow() error: %s", exc)
            return (STATUS_FAIL, 0.0)

    def repay(self, coin: str, amount: float) -> tuple:
        """Repay a margin loan.

        Args:
            coin: Coin symbol (e.g., "LINK").
            amount: Amount to repay.

        Returns:
            (STATUS_SUCCESS, amount) or (STATUS_FAIL, 0.0).
        """
        try:
            self.client.repay_margin_loan(asset=coin.upper(), amount=str(amount))
            return (STATUS_SUCCESS, amount)
        except Exception as exc:
            log.error("repay() error: %s", exc)
            return (STATUS_FAIL, 0.0)

    def set_operation_sleep(self) -> None:
        """Sleep and reset weight counter if API weight limit is exceeded."""
        if self.weight > self.overflow_weight:
            time.sleep(self.overflow_weight_time)
            self.weight = 0
