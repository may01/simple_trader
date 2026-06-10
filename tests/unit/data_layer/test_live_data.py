"""Unit tests for data.py — LiveData class and item singleton."""

import os
import sys

import pandas as pd
import pytest
from unittest.mock import MagicMock

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


class TestNnPredictorOrdering:
    def test_nn_predictor_called_after_indicators(self, monkeypatch):
        """nn_predictor.compute() must be called only AFTER all Indicators.compute() calls."""
        mock = _make_mock_binance(monkeypatch)
        monkeypatch.setattr("data.stock_holder.item", mock)

        call_order = []

        mock_nn = MagicMock()
        mock_nn.compute.side_effect = lambda dp, tf: call_order.append(("nn", tf))

        original_compute = Indicators.compute

        def tracked_compute(dp, tf):
            call_order.append(("ind", tf))
            original_compute(dp, tf)

        monkeypatch.setattr(Indicators, "compute", tracked_compute)

        import data
        live = data.LiveData(nn_predictor=mock_nn)
        live.build_candles()

        # Verify we have both indicator and nn calls recorded
        ind_indices = [i for i, (kind, _) in enumerate(call_order) if kind == "ind"]
        nn_indices = [i for i, (kind, _) in enumerate(call_order) if kind == "nn"]

        assert ind_indices, "No Indicators.compute calls recorded"
        assert nn_indices, "No nn_predictor.compute calls recorded"

        last_ind = max(ind_indices)
        first_nn = min(nn_indices)
        assert last_ind < first_nn, (
            f"nn_predictor.compute was called before all Indicators.compute calls finished: "
            f"last ind at position {last_ind}, first nn at position {first_nn}. "
            f"call_order={call_order}"
        )


class TestItemSingleton:
    def test_item_singleton_exists(self):
        """data.item is a LiveData instance at module level."""
        import data
        from data import LiveData
        assert isinstance(data.item, LiveData)
