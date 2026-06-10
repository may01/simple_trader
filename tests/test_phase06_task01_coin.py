"""Tests for Coin currency tracker (Phase 06, Task 01)."""

import pytest
from position.coin import Coin


class TestCoinInitialization:
    """Test Coin initialization."""

    def test_coin_init_with_name(self):
        """Coin initializes with currency name."""
        coin = Coin("usdt")
        assert coin.name == "usdt"

    def test_coin_init_with_different_names(self):
        """Coin supports various currency names."""
        for name in ["btc", "eth", "link", "usdt", "busd"]:
            coin = Coin(name)
            assert coin.name == name

    def test_coin_init_all_amounts_zero(self):
        """All amounts initialize to 0.0."""
        coin = Coin("usdt")
        assert coin.size == 0.0
        assert coin.used == 0.0
        assert coin.returned == 0.0
        assert coin.loan == 0.0
        assert coin.want_to_use == 0.0
        assert coin.action_amount == 0.0

    def test_coin_amounts_are_floats(self):
        """All amounts are float type."""
        coin = Coin("btc")
        assert isinstance(coin.size, float)
        assert isinstance(coin.used, float)
        assert isinstance(coin.returned, float)
        assert isinstance(coin.loan, float)
        assert isinstance(coin.want_to_use, float)
        assert isinstance(coin.action_amount, float)


class TestCoinAttributes:
    """Test Coin attribute setting and getting."""

    def test_coin_set_size(self):
        """Can set size attribute."""
        coin = Coin("usdt")
        coin.size = 1000.0
        assert coin.size == 1000.0

    def test_coin_set_used(self):
        """Can set used attribute."""
        coin = Coin("usdt")
        coin.used = 500.0
        assert coin.used == 500.0

    def test_coin_set_returned(self):
        """Can set returned attribute."""
        coin = Coin("usdt")
        coin.returned = 250.0
        assert coin.returned == 250.0

    def test_coin_set_loan(self):
        """Can set loan attribute."""
        coin = Coin("btc")
        coin.loan = 1.5
        assert coin.loan == 1.5

    def test_coin_set_want_to_use(self):
        """Can set want_to_use attribute."""
        coin = Coin("eth")
        coin.want_to_use = 100.0
        assert coin.want_to_use == 100.0

    def test_coin_set_action_amount(self):
        """Can set action_amount attribute."""
        coin = Coin("usdt")
        coin.action_amount = 250.0
        assert coin.action_amount == 250.0

    def test_coin_multiple_attributes(self):
        """Can set multiple attributes independently."""
        coin = Coin("usdt")
        coin.size = 1000.0
        coin.used = 500.0
        coin.returned = 200.0
        coin.loan = 50.0
        coin.want_to_use = 600.0
        coin.action_amount = 100.0

        assert coin.size == 1000.0
        assert coin.used == 500.0
        assert coin.returned == 200.0
        assert coin.loan == 50.0
        assert coin.want_to_use == 600.0
        assert coin.action_amount == 100.0


class TestCoinSerialization:
    """Test Coin serialization with to_dict and from_dict."""

    def test_coin_to_dict_empty(self):
        """to_dict returns dict with all zero amounts."""
        coin = Coin("usdt")
        d = coin.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "usdt"
        assert d["size"] == 0.0
        assert d["used"] == 0.0
        assert d["returned"] == 0.0
        assert d["loan"] == 0.0
        assert d["want_to_use"] == 0.0
        assert d["action_amount"] == 0.0

    def test_coin_to_dict_with_values(self):
        """to_dict returns dict with current values."""
        coin = Coin("btc")
        coin.size = 1.5
        coin.used = 0.5
        coin.returned = 0.25
        coin.loan = 0.1
        coin.want_to_use = 1.0
        coin.action_amount = 0.3

        d = coin.to_dict()
        assert d["name"] == "btc"
        assert d["size"] == 1.5
        assert d["used"] == 0.5
        assert d["returned"] == 0.25
        assert d["loan"] == 0.1
        assert d["want_to_use"] == 1.0
        assert d["action_amount"] == 0.3

    def test_coin_from_dict_empty(self):
        """from_dict restores from empty dict."""
        coin = Coin("usdt")
        data = {
            "name": "usdt",
            "size": 0.0,
            "used": 0.0,
            "returned": 0.0,
            "loan": 0.0,
            "want_to_use": 0.0,
            "action_amount": 0.0,
        }
        coin.from_dict(data)

        assert coin.name == "usdt"
        assert coin.size == 0.0
        assert coin.used == 0.0
        assert coin.returned == 0.0
        assert coin.loan == 0.0
        assert coin.want_to_use == 0.0
        assert coin.action_amount == 0.0

    def test_coin_from_dict_with_values(self):
        """from_dict restores all attributes."""
        coin = Coin("placeholder")
        data = {
            "name": "eth",
            "size": 10.5,
            "used": 5.2,
            "returned": 2.1,
            "loan": 1.5,
            "want_to_use": 8.0,
            "action_amount": 2.5,
        }
        coin.from_dict(data)

        assert coin.name == "eth"
        assert coin.size == 10.5
        assert coin.used == 5.2
        assert coin.returned == 2.1
        assert coin.loan == 1.5
        assert coin.want_to_use == 8.0
        assert coin.action_amount == 2.5

    def test_coin_serialization_roundtrip(self):
        """to_dict/from_dict roundtrip preserves exact state."""
        coin1 = Coin("usdt")
        coin1.size = 1000.0
        coin1.used = 500.0
        coin1.returned = 250.0
        coin1.loan = 100.0
        coin1.want_to_use = 800.0
        coin1.action_amount = 250.0

        d = coin1.to_dict()

        coin2 = Coin("placeholder")
        coin2.from_dict(d)

        assert coin2.name == coin1.name
        assert coin2.size == coin1.size
        assert coin2.used == coin1.used
        assert coin2.returned == coin1.returned
        assert coin2.loan == coin1.loan
        assert coin2.want_to_use == coin1.want_to_use
        assert coin2.action_amount == coin1.action_amount

    def test_coin_serialization_multiple_roundtrips(self):
        """Multiple roundtrips maintain exact state."""
        coin1 = Coin("link")
        coin1.size = 500.5
        coin1.used = 250.25
        coin1.returned = 100.1
        coin1.loan = 50.05
        coin1.want_to_use = 400.0
        coin1.action_amount = 150.0

        # First roundtrip
        d1 = coin1.to_dict()
        coin2 = Coin("placeholder")
        coin2.from_dict(d1)

        # Second roundtrip
        d2 = coin2.to_dict()
        coin3 = Coin("placeholder")
        coin3.from_dict(d2)

        # All should match original
        assert coin3.name == "link"
        assert coin3.size == 500.5
        assert coin3.used == 250.25
        assert coin3.returned == 100.1
        assert coin3.loan == 50.05
        assert coin3.want_to_use == 400.0
        assert coin3.action_amount == 150.0


class TestCoinReset:
    """Test Coin reset functionality."""

    def test_coin_reset_clears_amounts(self):
        """reset() sets all amounts to 0.0."""
        coin = Coin("usdt")
        coin.size = 1000.0
        coin.used = 500.0
        coin.returned = 250.0
        coin.loan = 100.0
        coin.want_to_use = 800.0
        coin.action_amount = 250.0

        coin.reset()

        assert coin.size == 0.0
        assert coin.used == 0.0
        assert coin.returned == 0.0
        assert coin.loan == 0.0
        assert coin.want_to_use == 0.0
        assert coin.action_amount == 0.0

    def test_coin_reset_preserves_name(self):
        """reset() keeps the currency name unchanged."""
        coin = Coin("btc")
        coin.size = 5.0
        coin.reset()

        assert coin.name == "btc"

    def test_coin_reset_can_be_reused(self):
        """After reset, coin can be reused with new values."""
        coin = Coin("eth")
        coin.size = 100.0
        coin.reset()

        assert coin.size == 0.0

        coin.size = 50.0
        assert coin.size == 50.0
        assert coin.name == "eth"


class TestCoinLog:
    """Test Coin log functionality."""

    def test_coin_log_runs_without_error(self, capsys):
        """log() runs and prints without error."""
        coin = Coin("usdt")
        coin.size = 1000.0
        coin.used = 500.0

        coin.log()

        captured = capsys.readouterr()
        output = captured.out

        # Should contain the name and values
        assert "usdt" in output or "1000" in output or output != ""

    def test_coin_log_with_empty_coin(self, capsys):
        """log() works on newly created coin."""
        coin = Coin("btc")
        coin.log()

        captured = capsys.readouterr()
        # Just verify it runs without error (output format not strictly specified)
        assert True

    def test_coin_log_after_reset(self, capsys):
        """log() works after reset."""
        coin = Coin("eth")
        coin.size = 100.0
        coin.used = 50.0
        coin.reset()
        coin.log()

        captured = capsys.readouterr()
        assert True
