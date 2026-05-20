import pytest
from simple_trader.position.coin import Coin


def test_coin_initial_state():
    coin = Coin("USDT")
    assert coin.name == "USDT"
    assert coin.size == 0.0
    assert coin.used == 0.0
    assert coin.loan == 0.0
    assert coin.returned == 0.0
    assert coin.want_to_use == 0.0
    assert coin.action_amount == 0.0


def test_coin_initializes_with_size():
    coin = Coin("USDT", size=1000.0)
    assert coin.size == 1000.0


def test_coin_use_transfers_from_size_to_used():
    coin = Coin("USDT", size=1000.0)
    coin.use(300.0)
    assert coin.size == 700.0
    assert coin.used == 300.0


def test_coin_use_multiple_times_accumulates():
    coin = Coin("USDT", size=1000.0)
    coin.use(200.0)
    coin.use(300.0)
    assert coin.used == 500.0
    assert coin.size == 500.0


def test_coin_free_transfers_from_used_to_size():
    coin = Coin("USDT", size=1000.0)
    coin.use(400.0)
    coin.free(200.0)
    assert coin.used == 200.0
    assert coin.size == 800.0


def test_coin_total_is_size_plus_used():
    coin = Coin("USDT", size=600.0)
    coin.use(400.0)
    assert coin.total == 1000.0


def test_coin_borrow_adds_to_loan_and_size():
    coin = Coin("USDT", size=100.0)
    coin.borrow(500.0)
    assert coin.loan == 500.0
    assert coin.size == 600.0


def test_coin_repay_reduces_loan_and_size():
    coin = Coin("USDT", size=600.0)
    coin.loan = 500.0
    coin.repay(500.0)
    assert coin.loan == 0.0
    assert coin.size == 100.0
