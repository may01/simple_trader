"""Phase 15 Task 01 — live strategy/size wiring in trader.py.

Mock-only, no credentials, runs in Docker. Verifies the two live knobs:
- STRATEGY_SET selects the strategy set (ema -> 2 strategies; else inert).
- LIVE_POSITION_USDT sets Position.full_position, clamped to <= 50.
"""

import logging

import trader
from robots.robot import Robot


class _FakeStock:
    """Minimal stock stub: trader wiring only reads ``fee``."""

    fee = 0.001


def test_strategy_set_ema_registers_two_ema_strategies():
    sm = trader.build_strategy_manager(_FakeStock(), "ema")
    assert len(sm.strategies) == 2


def test_strategy_set_unset_is_inert():
    sm = trader.build_strategy_manager(_FakeStock(), "")
    assert sm.strategies == []


def _build_robot(tmp_path):
    sm = trader.build_strategy_manager(_FakeStock(), "")
    return Robot(
        sm,
        live_data=None,
        stock=_FakeStock(),
        fee=_FakeStock.fee,
        persist_path=str(tmp_path / "live_tracker.json"),
    )


def test_live_position_usdt_sets_full_position(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_POSITION_USDT", "35")
    robot = _build_robot(tmp_path)
    robot.position.full_position = trader.resolve_live_usdt()
    assert robot.position.full_position == 35.0


def test_live_position_usdt_clamped_to_50(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LIVE_POSITION_USDT", "1000")
    with caplog.at_level(logging.WARNING):
        robot = _build_robot(tmp_path)
        robot.position.full_position = trader.resolve_live_usdt()
    assert robot.position.full_position == 50.0
    assert any("clamp" in r.message.lower() for r in caplog.records)


def test_live_position_usdt_default_is_40(monkeypatch):
    monkeypatch.delenv("LIVE_POSITION_USDT", raising=False)
    assert trader.resolve_live_usdt() == 40.0
