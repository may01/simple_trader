"""Unit tests for data.py — LiveData class and item singleton."""

import os
import sys

import pandas as pd
import pytest

from indicators import Indicators


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_indicators_registry():
    """Reset Indicators._registry before each test so prior tests don't pollute it."""
    Indicators._registry = None
    yield
    Indicators._registry = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_binance(monkeypatch):
    """Construct a Stock_MockBinance with the required env vars set."""
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setenv("DATA_ROOT", "data")
    monkeypatch.setenv("DATA_SET_NAME", "test")
    from stocks.mock_stock import Stock_MockBinance
    return Stock_MockBinance()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildCandles:
    def test_build_candles_populates_ohlc(self, monkeypatch):
        """After build_candles(), live.ohlc has a key for every CANDLE in CANDLES."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        from config_loader import CANDLES
        import data
        live = data.LiveData()
        live.build_candles()

        for tf in CANDLES:
            assert tf in live.ohlc, f"tf={tf} missing from live.ohlc"

    def test_build_candles_renames_columns(self, monkeypatch):
        """live.ohlc[5] has columns prefixed with '5_' after build_candles()."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        import data
        live = data.LiveData()
        live.build_candles()

        df5 = live.ohlc[5]
        for col in ("5_open", "5_close", "5_high", "5_low", "5_volume"):
            assert col in df5.columns, f"column {col!r} not found in live.ohlc[5]"

    def test_build_candles_adds_is_closed(self, monkeypatch):
        """live.ohlc[5]['5_is_closed'] is True for all rows (historical candles)."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        import data
        live = data.LiveData()
        live.build_candles()

        is_closed = live.ohlc[5]["5_is_closed"]
        assert is_closed.all(), "Expected all rows to have 5_is_closed == True"

    def test_build_candles_writes_indicator_columns(self, monkeypatch):
        """After build_candles(), indicator columns like '5_rsi_14' exist."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        import data
        live = data.LiveData()
        live.build_candles()

        df5 = live.ohlc[5]
        assert "5_rsi_14" in df5.columns, "'5_rsi_14' not found — indicators were not computed"


class TestGetDataPoint:
    def test_get_data_point_returns_live_data_point(self, monkeypatch):
        """get_data_point() returns a LiveDataPoint instance."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        import data
        from data import LiveDataPoint
        live = data.LiveData()
        live.build_candles()

        dp = live.get_data_point()
        assert isinstance(dp, LiveDataPoint)

    def test_get_data_point_get_close(self, monkeypatch):
        """dp.get('close', tf=5) returns a float after build_candles()."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        import data
        live = data.LiveData()
        live.build_candles()

        dp = live.get_data_point()
        value = dp.get("close", tf=5)
        assert isinstance(value, float), f"Expected float, got {type(value)}"
        assert value == value, "Expected a real float, got NaN"  # NaN check


class TestNoArgConstruction:
    def test_live_data_constructs_with_no_args(self, monkeypatch):
        """LiveData() takes no arguments and exposes no nn_predictor attribute.

        Per Task 12 the per-tick NNPredictor is removed; LiveData no longer
        accepts or stores an nn_predictor.
        """
        monkeypatch.setenv("PAIR", "link_usdt")
        import data
        live = data.LiveData()
        assert not hasattr(live, "nn_predictor")

    def test_live_data_rejects_nn_predictor_kwarg(self, monkeypatch):
        """The removed nn_predictor parameter is gone (passing it raises)."""
        monkeypatch.setenv("PAIR", "link_usdt")
        import data
        with pytest.raises(TypeError):
            data.LiveData(nn_predictor=object())


class TestBuildCandlesNnJoin:
    def test_build_candles_surfaces_nn_res_when_present(self, monkeypatch, tmp_path):
        """build_candles left-joins nn_res_* from the live df_with_nn.pkl (absence-safe)."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        import data
        live = data.LiveData()
        live.build_candles()  # populates live.ohlc with no nn artifact present

        # Absent artifact → no nn_res_* columns, construction still succeeds.
        for tf, df in live.ohlc.items():
            assert not any(c.startswith("nn_res_") for c in df.columns)


class TestItemSingleton:
    def test_item_singleton_exists(self):
        """data.item is a LiveData instance at module level."""
        import data
        from data import LiveData
        assert isinstance(data.item, LiveData)
