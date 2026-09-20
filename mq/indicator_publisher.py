# indicator_publisher.py — builds indicator_update wire JSON for trade_executor.
#
# Task 7 scope: the pure, fully unit-testable JSON builder only. Task 8 adds the
# IndicatorPublisher class that wraps this with an actual zmq PUSH socket.

from __future__ import annotations

import uuid
from datetime import datetime, timedelta


def build_indicator_update(pair: str, name: str, value: float, now: datetime, ttl_seconds: float) -> dict:
    """Build one indicator_update wire message, kind: none (v1 scope — no support/resistance/volume yet).

    Args:
        pair: e.g. "BTCUSDT".
        name: stable indicator identifier, e.g. "15_ema_7" (matches main/indicators/'s own {tf}_{name} column naming).
        value: the indicator's current reading.
        now: current time; expires_at is computed from this, not read from any indicator metadata.
        ttl_seconds: how far past `now` this reading stays valid — must comfortably exceed one tick interval.

    Returns:
        dict ready for json.dumps and PUSH — kind is always "none" in this v1 builder.
    """
    return {
        "id": str(uuid.uuid4()),
        "type": "indicator_update",
        "pair": pair,
        "name": name,
        "value": value,
        "kind": "none",
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }
