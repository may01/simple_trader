from config_loader import load_shared_indicators_config


def test_loads_the_configured_allowlist(tmp_path):
    config_file = tmp_path / "shared_indicators_config.yaml"
    config_file.write_text(
        "indicators:\n"
        "  - name: ema_7\n"
        "    timeframes: [15, 60, 240]\n"
        "  - name: ema_14\n"
        "    timeframes: [15, 60, 240]\n"
    )
    result = load_shared_indicators_config(path=str(config_file))
    assert len(result) == 2
    assert result[0].name == "ema_7"
    assert result[0].timeframes == [15, 60, 240]


def test_empty_indicators_list_returns_empty(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("indicators: []\n")
    assert load_shared_indicators_config(path=str(config_file)) == []
