"""tests/integration/test_stock_operations.py — stock operation integration tests.

All tests use Stock_MockBinance or Stock_Mock — no real exchange credentials required.
Each test calls do_stock_init() independently to avoid shared state.
"""


def test_order_place_and_fill_long():
    import os
    from stocks_holder import do_stock_init, stock_holder as stock
    from position.position import Position
    from robots.live_order_tracker import LiveOrderTracker
    from constants import STATUS_SUCCESS, TRADE_BUY

    do_stock_init('mock_binance')

    _path = '/tmp/test_integration.json'
    os.remove(_path) if os.path.exists(_path) else None

    pos = Position(fee=0.001)
    pos.full_position = 1000.0
    tracker = LiveOrderTracker(position=pos, stock=stock.item, persist_path=_path)

    # Place buy order
    status, result = stock.item.trade(TRADE_BUY, 20.0, 50.0)
    assert status == STATUS_SUCCESS
    order_id = result["order_id"]
    assert order_id != ""

    tracker.set_buy_order(order_id)
    assert tracker.buy_id == order_id

    # Check fill
    status, fill = tracker.check_fill(order_id)
    assert status == STATUS_SUCCESS
    assert fill["status"] == "FILLED"
    assert fill["start_amount"] > 0
    assert fill["left_amount"] == 0.0
    assert fill["rate"] > 0.0

    coin_amount = fill["start_amount"] - fill["left_amount"]
    print(f"test_order_place_and_fill_long ok — filled {coin_amount} coins at {fill['rate']}")
    os.remove(_path) if os.path.exists(_path) else None


def test_order_cancel():
    import os
    from stocks_holder import do_stock_init, stock_holder as stock
    from position.position import Position
    from robots.live_order_tracker import LiveOrderTracker
    from constants import STATUS_SUCCESS, TRADE_BUY

    do_stock_init('mock_binance')

    _path = '/tmp/test_cancel.json'
    os.remove(_path) if os.path.exists(_path) else None

    pos = Position(fee=0.001)
    tracker = LiveOrderTracker(position=pos, stock=stock.item, persist_path=_path)

    status, result = stock.item.trade(TRADE_BUY, 20.0, 50.0)
    assert status == STATUS_SUCCESS
    tracker.set_buy_order(result["order_id"])

    tracker.cancel_buy()
    assert tracker.buy_id == ""
    os.remove(_path) if os.path.exists(_path) else None
    print("test_order_cancel ok")


def test_margin_borrow_repay():
    from stocks_holder import do_stock_init, stock_holder as stock
    from constants import STATUS_SUCCESS

    do_stock_init('mock_binance')

    status, amount = stock.item.borrow("LINK", 10.0)
    assert status == STATUS_SUCCESS
    assert amount > 0

    status, repaid = stock.item.repay("LINK", amount)
    assert status == STATUS_SUCCESS
    print(f"test_margin_borrow_repay ok — borrowed {amount}, repaid {repaid}")


def test_funds_query():
    from stocks_holder import do_stock_init, stock_holder as stock
    from constants import STATUS_SUCCESS

    do_stock_init('mock_binance')

    status, free = stock.item.funds("USDT", "free")
    assert status == STATUS_SUCCESS
    assert free >= 0.0

    status, locked = stock.item.funds("USDT", "locked")
    assert status == STATUS_SUCCESS
    print(f"test_funds_query ok — free: {free}, locked: {locked}")


def test_tracker_persist_load():
    import os
    from stocks_holder import do_stock_init, stock_holder as stock
    from position.position import Position
    from robots.live_order_tracker import LiveOrderTracker
    from constants import STRATEGY_ACTION_OPEN_LONG

    do_stock_init('mock')
    path = '/tmp/test_tracker_roundtrip.json'
    os.remove(path) if os.path.exists(path) else None

    try:
        pos = Position(fee=0.001)
        pos.full_position = 1000.0
        pos.open(STRATEGY_ACTION_OPEN_LONG, [20.0], [21.0], 19.0, 15, None)

        tracker = LiveOrderTracker(position=pos, stock=stock.item, persist_path=path)
        tracker.set_buy_order("order-abc")
        tracker.set_loan("loan-001", 500.0)

        # Reconstruct from JSON
        pos2 = Position(fee=0.001)
        pos2.full_position = 1000.0
        tracker2 = LiveOrderTracker(position=pos2, stock=stock.item, persist_path=path)
        loaded = tracker2.load()

        assert loaded is True
        assert tracker2.buy_id == "order-abc"
        assert tracker2.loan_id == "loan-001"
        assert tracker2.loan_amount == 500.0

        tracker2.clear()
        assert not os.path.exists(path)
        print("test_tracker_persist_load ok")
    finally:
        os.remove(path) if os.path.exists(path) else None
