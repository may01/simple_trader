from stocks.base_stock import StockInterface

class StockHolder:
    def __init__(self):
        self.item = StockInterface("", "", "", "")

stock_holder = StockHolder()  # module-level singleton

def do_stock_init(stock_name: str) -> None:
    """Initialize stock_holder.item with the named implementation.

    Args:
        stock_name: "binance", "mock_binance", or "mock"
    """
    global stock_holder
    if stock_name == "binance":
        from stocks.binance_stock import Stock_Binance
        stock_holder.item = Stock_Binance()
    elif stock_name == "mock_binance":
        from stocks.mock_stock import Stock_MockBinance
        stock_holder.item = Stock_MockBinance()
    elif stock_name == "mock":
        from stocks.mock_stock import Stock_Mock
        stock_holder.item = Stock_Mock()
    else:
        raise ValueError(f"Unknown stock_name: {stock_name}")
