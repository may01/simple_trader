import pytest
from pathlib import Path
from simple_trader.infrastructure.config import Config
from simple_trader.infrastructure.logging import TradeLogger


@pytest.fixture
def tmp_config(monkeypatch, tmp_path):
    monkeypatch.setenv("PAIR", "link_usdt")
    return Config()


def test_log_writes_to_file(tmp_config, tmp_path):
    logger = TradeLogger(tmp_config, log_dir=tmp_path)
    logger.log("hello")
    log_file = tmp_path / "LOG_link_usdt.txt"
    assert log_file.read_text() == "hello\n"


def test_log_appends_multiple_messages(tmp_config, tmp_path):
    logger = TradeLogger(tmp_config, log_dir=tmp_path)
    logger.log("line1")
    logger.log("line2")
    content = (tmp_path / "LOG_link_usdt.txt").read_text()
    assert "line1" in content
    assert "line2" in content


def test_log_revenue_writes_to_per_thread_file(tmp_config, tmp_path):
    logger = TradeLogger(tmp_config, log_dir=tmp_path)
    logger.log_revenue("profit=5%", time_point=12345, thread=0)
    rev_file = tmp_path / "REVENUE_link_usdt_0.txt"
    assert rev_file.exists()
    assert "profit=5%" in rev_file.read_text()


def test_log_error_prefixes_error(tmp_config, tmp_path):
    logger = TradeLogger(tmp_config, log_dir=tmp_path)
    logger.log_error("something failed")
    content = (tmp_path / "LOG_link_usdt.txt").read_text()
    assert "ERROR" in content
    assert "something failed" in content
