from typing import Optional
from .base_stock import BaseStock

_instance: Optional[BaseStock] = None


def set(stock: Optional[BaseStock]) -> None:
    global _instance
    _instance = stock


def get() -> BaseStock:
    if _instance is None:
        raise RuntimeError(
            "StockItem not initialized. Call stock_item.set(instance) before use."
        )
    return _instance
