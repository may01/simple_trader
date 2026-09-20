import uuid
from datetime import datetime, timezone, timedelta

from mq.indicator_publisher import build_indicator_update


def test_builds_a_none_kind_indicator_update():
    now = datetime(2026, 9, 20, 0, 0, 0, tzinfo=timezone.utc)
    msg = build_indicator_update(pair="BTCUSDT", name="15_ema_7", value=12345.0, now=now, ttl_seconds=30.0)

    assert msg["type"] == "indicator_update"
    assert msg["pair"] == "BTCUSDT"
    assert msg["name"] == "15_ema_7"
    assert msg["value"] == 12345.0
    assert msg["kind"] == "none"
    assert "volume" not in msg
    assert msg["expires_at"] == "2026-09-20T00:00:30+00:00"
    uuid.UUID(msg["id"])  # raises ValueError if not a valid uuid


def test_each_call_gets_a_fresh_id():
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    a = build_indicator_update(pair="BTCUSDT", name="x", value=1.0, now=now, ttl_seconds=30.0)
    b = build_indicator_update(pair="BTCUSDT", name="x", value=1.0, now=now, ttl_seconds=30.0)
    assert a["id"] != b["id"]


def test_expires_at_is_now_plus_ttl():
    now = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    msg = build_indicator_update(pair="BTCUSDT", name="x", value=1.0, now=now, ttl_seconds=120.0)
    expected = (now + timedelta(seconds=120.0)).isoformat()
    assert msg["expires_at"] == expected
