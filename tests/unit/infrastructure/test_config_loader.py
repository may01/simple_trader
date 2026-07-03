"""Unit tests for config_loader.py — Phase 00 Task 03."""

from config_loader import load_candles_config, load_indicators_config, CANDLES


def test_candles_list():
    assert load_candles_config() == [1, 5, 15, 60, 240, 1440]


def test_candles_constant():
    assert CANDLES == [1, 5, 15, 60, 240, 1440]


def test_dependency_order():
    fields = load_indicators_config()
    names = [f.name for f in fields]
    assert "rsi_14" in names
    assert "rsi_ma8" in names
    assert names.index("rsi_ma8") > names.index("rsi_14")


def test_classification_applies_to_subset():
    fields = load_indicators_config()
    class_field = next(f for f in fields if f.name == "move_class")
    assert 1 not in class_field.applies_to
    assert 5 not in class_field.applies_to
    assert 15 in class_field.applies_to


def test_all_required_groups_present():
    fields = load_indicators_config()
    groups = {f.group for f in fields}
    required = {
        "momentum", "trend", "volatility", "oscillators", "volume",
        "price_derivatives", "classification", "targets", "trend_flags", "nn_features",
    }
    assert required.issubset(groups)
