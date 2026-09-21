# indicator_publisher.py — builds indicator_update wire JSON for trade_executor.
#
# Task 7 scope: the pure, fully unit-testable JSON builder only. Task 8 adds the
# IndicatorPublisher class that wraps this with an actual zmq PUSH socket.

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

# NOTE: `zmq` is deliberately NOT imported at module level. This module is
# telemetry; the live trader must start and trade even in an image that has
# no pyzmq installed. A module-level `import zmq` would make every importer
# of this file (including robots/robot.py) die at import time on such an
# image, turning telemetry into a startup dependency of the money path.
# The import lives inside IndicatorPublisher.__init__ instead, so only code
# that actually asks for a publisher can raise ImportError -- and
# robots/robot.py guards even that. build_indicator_update below stays
# importable and fully usable with no pyzmq at all.

logger = logging.getLogger(__name__)


def build_indicator_update(pair: str, name: str, value: float, now: datetime, ttl_seconds: float) -> dict:
    """Build one indicator_update wire message, kind: none (v1 scope — no support/resistance/volume yet).

    Args:
        pair: e.g. "BTCUSDT".
        name: stable indicator identifier, e.g. "15_ema_7" (matches main/indicators/'s own {tf}_{name} column naming).
        value: the indicator's current reading.
        now: current time; must be timezone-aware (UTC). expires_at is computed from
            this, not read from any indicator metadata.
        ttl_seconds: how far past `now` this reading stays valid — must comfortably exceed one tick interval.

    Returns:
        dict ready for json.dumps and PUSH — kind is always "none" in this v1 builder.

    Raises:
        ValueError: if `now` is a naive datetime (no tzinfo). A naive datetime
            serializes via .isoformat() with no UTC offset (e.g. "...T12:02:00" instead
            of "...T12:02:00+00:00"), and the executor's wire.rs parses expires_at with
            chrono::DateTime::parse_from_rfc3339, which requires an explicit offset or
            "Z" — an offset-less string is rejected outright.
    """
    if now.tzinfo is None:
        raise ValueError(
            "now must be timezone-aware (UTC); a naive datetime serializes without a "
            "UTC offset and the executor rejects it"
        )
    return {
        "id": str(uuid.uuid4()),
        "type": "indicator_update",
        "pair": pair,
        "name": name,
        "value": value,
        "kind": "none",
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }


class IndicatorPublisher:
    """PUSH client onto trade_executor's inbound MQ socket, publishing indicator_update messages only (v1 scope)."""

    def __init__(self, connect_addr: str, ttl_seconds: float = 300.0) -> None:
        # Imported here, not at module scope: see the module-level note.
        # Constructing a publisher is the one operation that genuinely
        # needs pyzmq, so this is the only place an ImportError may be
        # raised -- and every caller treats that as "no telemetry", never
        # as "do not trade".
        import zmq

        self._zmq = zmq
        self._ttl_seconds = ttl_seconds
        self._connect_addr = connect_addr
        self.dropped = 0
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUSH)
        # A PUSH socket with no peer blocks forever on send by default, and
        # connect() succeeds even when nothing is listening -- these four
        # options are what keep a down executor from freezing the tick loop.
        self._socket.setsockopt(zmq.SNDTIMEO, 0)   # never wait for a peer
        self._socket.setsockopt(zmq.IMMEDIATE, 1)  # never queue for a peer that never connected
        self._socket.setsockopt(zmq.LINGER, 0)     # never block process exit on unsent messages
        self._socket.setsockopt(zmq.SNDHWM, 100)   # bounded backlog; drop past it
        # libzmq reconnects on its own, forever -- these only set the cadence.
        # Never call connect() a second time to "retry": it adds an endpoint
        # rather than repairing the existing one.
        self._socket.setsockopt(zmq.RECONNECT_IVL, 100)       # first retry after 100ms
        self._socket.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # exponential backoff, capped at 5s
        self._socket.connect(connect_addr)
        # connect() succeeds even with nothing listening, so this is the only
        # signal an operator gets of where telemetry is actually headed --
        # log it once here so a wrong/unset address is at least discoverable
        # from the startup log, not just a silent, permanent drop counter.
        logger.info("IndicatorPublisher: connecting to trade_executor at %s", connect_addr)

    def publish(self, pair: str, name: str, value: float) -> None:
        """Best-effort. Never blocks, never raises: an unreachable executor
        costs one tick of freshness, and the next tick republishes anyway."""
        msg = build_indicator_update(pair=pair, name=name, value=value, now=datetime.now(timezone.utc), ttl_seconds=self._ttl_seconds)
        try:
            self._socket.send_json(msg, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            self.dropped += 1
            # Rate-limited, not per-drop -- this fires on every tick while no
            # executor is reachable, so logging every occurrence would be a
            # tick-rate log in a live process. Still surfaces the blackout
            # (first drop, then every 1000th) rather than staying silent
            # forever behind an incrementing counter nothing ever reads.
            if self.dropped == 1 or self.dropped % 1000 == 0:
                logger.warning(
                    "indicator publish: no reachable executor at %s; dropped=%d so far",
                    self._connect_addr, self.dropped,
                )
        except self._zmq.ZMQError as e:
            self.dropped += 1
            # Same rate limit as the Again path above, and for the same
            # reason: a persistent socket-level fault recurs on every tick,
            # so logging each occurrence would be a tick-rate log stream in
            # a live process (I3). First failure, then every 1000th.
            if self.dropped == 1 or self.dropped % 1000 == 0:
                logger.warning(
                    "indicator publish failed: %s; dropped=%d so far", e, self.dropped
                )

    def close(self) -> None:
        self._socket.close()
