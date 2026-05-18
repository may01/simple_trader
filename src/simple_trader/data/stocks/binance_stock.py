from typing import Any, Dict, List, Optional
from binance.client import Client
from .base_stock import BaseStock

_INTERVAL_MAP = {
    1: Client.KLINE_INTERVAL_1MINUTE,
    3: Client.KLINE_INTERVAL_3MINUTE,
    5: Client.KLINE_INTERVAL_5MINUTE,
    15: Client.KLINE_INTERVAL_15MINUTE,
    60: Client.KLINE_INTERVAL_1HOUR,
    240: Client.KLINE_INTERVAL_4HOUR,
    1440: Client.KLINE_INTERVAL_1DAY,
}


class BinanceStock(BaseStock):
    def __init__(
        self, api_key: str, api_secret: str, taker_fee: float = 0.001
    ) -> None:
        self._client = Client(api_key, api_secret)
        self._fee = taker_fee

    @property
    def fee(self) -> float:
        return self._fee

    def get_candles_history(
        self,
        count: int,
        symbol: str,
        time_point: int,
        tf_minutes: int = 1,
    ) -> List[Any]:
        interval = _INTERVAL_MAP.get(tf_minutes)
        if interval is None:
            raise ValueError(f"Unsupported tf_minutes: {tf_minutes}. Valid values: {sorted(_INTERVAL_MAP)}")
        return self._client.get_klines(
            symbol=symbol.upper(),
            interval=interval,
            limit=count,
        )

    def trade(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        kwargs: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type,
            "quantity": quantity,
        }
        if price is not None:
            kwargs["price"] = str(price)
            kwargs["timeInForce"] = "GTC"
        return self._client.create_order(**kwargs)

    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        return self._client.get_order(symbol=symbol.upper(), orderId=order_id)
