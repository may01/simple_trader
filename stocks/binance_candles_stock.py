"""Keyless, read-only Binance stock: real market data, no account access at all.

STOCK_TYPE=binance_candles. Between the two existing options --
`mock_binance` (replays a frozen pickle fixture, so every indicator is a
constant) and `binance` (API keys, margin account, real orders) -- this is
the one that gives live indicators with no order path: candles, order-book
depth and symbol info come from Binance's public endpoints, and every
trading or account call is refused before it reaches the client.

It deliberately does not read BINANCE_API_KEY/BINANCE_API_SECRET even when
they are set, so it cannot act on the account whatever strategies are
registered.
"""
import logging
import os

from binance.client import Client

from constants import STATUS_FAIL
from stocks.base_stock import StockInterface
from stocks.binance_stock import Stock_Binance

log = logging.getLogger(__name__)


class Stock_BinanceCandles(Stock_Binance):
    """Stock_Binance's public-data half; every account/order method refuses."""

    stock_name = "binance_candles"

    def __init__(self):
        # Not Stock_Binance.__init__: that one requires and uses the keys.
        pair = os.environ["PAIR"]  # e.g. "link_usdt"
        coin, coin_base = pair.split("_")[:2]
        StockInterface.__init__(self, "", "", coin, coin_base)
        self.fee = float(os.environ["EXCHANGE_FEE"])
        self.client = Client()  # no key, no secret: public endpoints only
        self.was_init = True

    def _refuse(self, what: str, result):
        log.warning("%s refused: STOCK_TYPE=binance_candles is read-only", what)
        return (STATUS_FAIL, result)

    def trade(self, trade_type: str, price: float, amount: float, force: bool = False) -> tuple:
        return self._refuse("trade", {})

    def order_info(self, order_id: str) -> tuple:
        return self._refuse("order_info", {})

    def cancel_order(self, order_id: str) -> tuple:
        return self._refuse("cancel_order", {})

    def funds(self, coin: str, asset_type: str = "free") -> tuple:
        return self._refuse("funds", 0.0)

    def get_aviable_loan(self, coin: str) -> tuple:
        return self._refuse("get_aviable_loan", 0.0)

    def borrow(self, coin: str, amount: float) -> tuple:
        return self._refuse("borrow", 0.0)

    def repay(self, coin: str, amount: float) -> tuple:
        return self._refuse("repay", 0.0)
