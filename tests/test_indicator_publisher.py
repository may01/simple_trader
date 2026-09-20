import json
import time
import uuid
from datetime import datetime, timezone

import zmq

from mq.indicator_publisher import IndicatorPublisher, build_indicator_update


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
    # Identical arguments must differ in `id` ONLY — every other field matches.
    assert {k: v for k, v in a.items() if k != "id"} == {k: v for k, v in b.items() if k != "id"}


def test_expires_at_is_now_plus_ttl():
    now = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    msg = build_indicator_update(pair="BTCUSDT", name="x", value=1.0, now=now, ttl_seconds=120.0)
    assert msg["expires_at"] == "2026-09-20T12:02:00+00:00"


def test_naive_now_raises_value_error():
    naive_now = datetime(2026, 9, 20, 12, 0, 0)  # no tzinfo
    try:
        build_indicator_update(pair="BTCUSDT", name="x", value=1.0, now=naive_now, ttl_seconds=30.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a naive `now`")


def test_publish_reaches_a_real_pull_socket():
    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    try:
        publisher = IndicatorPublisher(connect_addr=f"tcp://127.0.0.1:{port}", ttl_seconds=30.0)
        try:
            # connect() is asynchronous and the socket drops rather than queues
            # (IMMEDIATE=1), so the first publish can legitimately land before
            # the pipe is up. Retry the way the real per-tick heartbeat does,
            # instead of asserting on a single shot.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                publisher.publish(pair="BTCUSDT", name="15_ema_7", value=42.0)
                if pull.poll(timeout=200):
                    break
            else:
                raise AssertionError("no message arrived within 5s of repeated publishes")
            raw = pull.recv()
            msg = json.loads(raw)
            assert msg["pair"] == "BTCUSDT"
            assert msg["name"] == "15_ema_7"
            assert msg["value"] == 42.0
            assert msg["kind"] == "none"
        finally:
            publisher.close()
    finally:
        pull.close()


def test_publish_to_a_dead_address_returns_promptly_and_counts_the_drop():
    """The executor being down must never stall main/'s tick loop.

    Port 1 has nothing listening; connect() still succeeds (it is
    asynchronous), so this is exactly the "executor not running" case.
    A blocking PUSH send would hang here forever.
    """
    publisher = IndicatorPublisher(connect_addr="tcp://127.0.0.1:1", ttl_seconds=30.0)
    try:
        started = time.monotonic()
        for _ in range(100):
            publisher.publish(pair="BTCUSDT", name="15_ema_7", value=42.0)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"100 publishes to a dead peer took {elapsed:.2f}s -- publish is blocking"
        assert publisher.dropped == 100
    finally:
        publisher.close()


def test_publishing_resumes_after_the_executor_comes_back():
    """An executor restart must heal itself -- no publisher rebuild, no manual reconnect.

    libzmq retries the connect from its own IO thread; this test pins that
    behaviour so a future "add a reconnect loop" change can't quietly replace
    it with something worse.
    """
    ctx = zmq.Context.instance()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    publisher = IndicatorPublisher(connect_addr=f"tcp://127.0.0.1:{port}", ttl_seconds=30.0)

    def publish_until_delivered(sock, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            publisher.publish(pair="BTCUSDT", name="15_ema_7", value=42.0)
            if sock.poll(timeout=200):
                return sock.recv()
        raise AssertionError("nothing delivered within the timeout")

    try:
        publish_until_delivered(pull)

        # Executor goes away. Publishing must stay fast and lossy, not block.
        pull.close()
        started = time.monotonic()
        for _ in range(50):
            publisher.publish(pair="BTCUSDT", name="15_ema_7", value=42.0)
        assert time.monotonic() - started < 1.0, "publish blocked while the peer was gone"

        # Executor comes back on the same address; delivery resumes by itself.
        pull = ctx.socket(zmq.PULL)
        pull.bind(f"tcp://127.0.0.1:{port}")
        raw = publish_until_delivered(pull)
        assert json.loads(raw)["name"] == "15_ema_7"
    finally:
        publisher.close()
        pull.close()
