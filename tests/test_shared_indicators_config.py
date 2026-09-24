from pathlib import Path

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


def test_empty_file_returns_empty_instead_of_crashing(tmp_path):
    """yaml.safe_load returns None for an empty file; `.get` on None is an
    AttributeError. Robot.__init__ calls this loader unconditionally when a
    publisher is present, so that AttributeError aborted trader startup --
    an empty allowlist must simply mean "broadcast nothing"."""
    config_file = tmp_path / "blank.yaml"
    config_file.write_text("")
    assert load_shared_indicators_config(path=str(config_file)) == []


def test_comments_only_file_returns_empty(tmp_path):
    """Same None-from-safe_load case, via the shape an operator actually
    produces by commenting every entry out."""
    config_file = tmp_path / "commented.yaml"
    config_file.write_text("# indicators:\n#   - name: ema_7\n")
    assert load_shared_indicators_config(path=str(config_file)) == []


def test_file_without_an_indicators_key_returns_empty(tmp_path):
    config_file = tmp_path / "other.yaml"
    config_file.write_text("something_else: 1\n")
    assert load_shared_indicators_config(path=str(config_file)) == []


def test_null_indicators_key_returns_empty(tmp_path):
    """`indicators:` with nothing after it parses to None, not []."""
    config_file = tmp_path / "null.yaml"
    config_file.write_text("indicators:\n")
    assert load_shared_indicators_config(path=str(config_file)) == []



def test_shipped_config_publishes_sar_on_the_one_minute_timeframe():
    """trade_executor's SAR flip signal reads `1_sar_002_02` off the wire.

    The wire name is f"{tf}_{name}" (robot._publish_shared_indicators), so
    this entry in the *shipped* config is what makes that name exist at all
    -- an allowlist that omits it leaves the executor's check permanently
    quiet with nothing to warn about. Path is resolved from this file, not
    the CWD, so it holds wherever pytest is invoked from.
    """
    shipped = Path(__file__).resolve().parent.parent / "configs" / "shared_indicators_config.yaml"
    result = load_shared_indicators_config(path=str(shipped))
    sar = [c for c in result if c.name == "sar_002_02"]
    assert len(sar) == 1, "sar_002_02 must be published exactly once"
    assert sar[0].timeframes == [1]
