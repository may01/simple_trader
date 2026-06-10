from stocks.base_stock import StockInterface

FIXTURE_PATH = "tests/fixtures/mock_candles.pkl"


class Stock_MockBinance(StockInterface):
    stock_name = "mock_binance"

    def __init__(self):
        super().__init__("", "", "link", "usdt")
        # Load fixture — raises FileNotFoundError if missing
        import pickle
        with open(FIXTURE_PATH, "rb") as f:
            self._candles = pickle.load(f)
        self.fee = 0.001
        self.was_init = True

    def get_candles_history(self, time_list, coin, time_point=0):
        return {k: v for k, v in self._candles.items() if k in time_list}

    def trade(self, trade_type, price, amount, force=False):
        from constants import STATUS_SUCCESS
        return (STATUS_SUCCESS, {"order_id": "mock-001"})

    def order_info(self, order_id):
        from constants import STATUS_SUCCESS
        return (STATUS_SUCCESS, {"status": "FILLED", "start_amount": 10.0, "left_amount": 0.0, "rate": 20.0})

    def cancel_order(self, order_id):
        from constants import STATUS_SUCCESS
        return (STATUS_SUCCESS, {"status": "CANCELED", "start_amount": 0.0, "left_amount": 0.0, "rate": 0.0})

    def funds(self, coin, asset_type="free"):
        from constants import STATUS_SUCCESS
        if coin.lower() == self.coin_base.lower():
            return (STATUS_SUCCESS, 10000.0)
        return (STATUS_SUCCESS, 0.0)


class Stock_Mock(StockInterface):
    stock_name = "mock"

    def __init__(self):
        super().__init__("", "", "link", "usdt")
        self.fee = 0.001
        self.was_init = True

    def get_candles_history(self, time_list, coin, time_point=0):
        return {}

    def trade(self, trade_type, price, amount, force=False):
        from constants import STATUS_SUCCESS
        return (STATUS_SUCCESS, {"order_id": "mock-001"})

    def order_info(self, order_id):
        from constants import STATUS_SUCCESS
        return (STATUS_SUCCESS, {"status": "FILLED", "start_amount": 10.0, "left_amount": 0.0, "rate": 20.0})

    def cancel_order(self, order_id):
        from constants import STATUS_SUCCESS
        return (STATUS_SUCCESS, {"status": "CANCELED", "start_amount": 0.0, "left_amount": 0.0, "rate": 0.0})

    def funds(self, coin, asset_type="free"):
        from constants import STATUS_SUCCESS
        if coin.lower() == self.coin_base.lower():
            return (STATUS_SUCCESS, 10000.0)
        return (STATUS_SUCCESS, 0.0)

    def get_pair_name(self):
        return (self.coin + self.coin_base).upper()

    def is_invalid_amount(self, amount, price):
        return amount <= 0 or price <= 0

    def info(self):
        return {}
