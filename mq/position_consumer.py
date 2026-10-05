# mq/position_consumer.py — receives position state from trade_executor.
#
# What main/ uses this for, and what it therefore does NOT do
# -----------------------------------------------------------
# main/ tracks a position for exactly two reasons (position-management
# design §6.4):
#
#   1. Record actions, so position levels can be visualised.
#   2. Hold `avg_price_open`, so risk can be recomputed and strategies can
#      run against the price actually paid rather than the price intended.
#
# Neither needs a fill. **main/ does not track fills**, and this module
# does not reconstruct them: the executor has already netted them against
# the authoritative source (`get_order_fills` + `settle`) and holds them
# in `exchange_fill`. A second reconstruction here, in another language,
# from a lossier stream, could only ever disagree with it. Per-fill data
# is not on the wire at all — this is "never sent", not "sent and thrown
# away".
#
# `zmq` is imported inside `PositionConsumer.__init__`, never at module
# level, so an image without pyzmq still imports this file and still
# trades. Telemetry must never be a startup dependency of the money path
# — the same rule `mq/indicator_publisher.py` follows and for the same
# reason.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: The wire contract version this module understands. A message carrying
#: a higher `schema` had a breaking change made to it and is skipped
#: rather than guessed at.
SUPPORTED_SCHEMA = 2

#: Event kinds the executor can send (mq_gateway's
#: `PositionStateEvent::kind`, which is total by construction).
KNOWN_EVENTS = frozenset({
    "not_placed",
    "opened",
    "partially_filled",
    "filled",
    "not_filled",
    "closed",
    "stopped_out",
    "target_hit",
    "already_closed",
    "sl_moved",
})

#: Logged once per distinct unknown kind, so an executor running ahead of
#: this consumer produces one line rather than one per message.
_warned_unknown: set[str] = set()


@dataclass(frozen=True)
class PositionEvent:
    """One decoded outbound message.

    Aggregates only — see the module docstring for why there is no fills
    field.
    """

    schema: int
    pair: str
    event: str
    position_id: str | None
    reason: str | None
    # --- position_state ---
    status: str
    side: str | None
    market_kind: str | None
    target_size: float | None
    net_size: float | None
    intended_open_price: float | None
    avg_entry_price: float | None
    avg_exit_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    realized_pnl: float | None
    unrealized_pnl: float | None
    settlement_complete: bool
    leverage: int | None
    liquidation_price: float | None
    created_at: int | None
    ts: str

    @property
    def is_flat(self) -> bool:
        return self.status == "flat"


def _num(payload: dict, key: str) -> float | None:
    """Read a decimal field.

    The executor serialises `Decimal` as a quoted string, never a JSON
    number, precisely so a price does not round-trip through a float on
    the way here. It becomes a float at this boundary because that is
    what main/'s `Position` is built from — but the conversion happens
    once, explicitly, rather than silently inside a JSON parser.

    Absent is `None`, not `0.0`: a zero average entry price is
    indistinguishable from a real one and would poison every risk
    calculation downstream.
    """
    raw = payload.get(key)
    if raw is None:
        return None
    return float(raw)


def parse_position_event(raw: dict) -> PositionEvent | None:
    """Decode one outbound message.

    Returns `None` for anything this consumer should skip — a newer
    schema, an unknown event kind, a malformed message. Never raises:
    this runs on the trading tick.
    """
    if not isinstance(raw, dict):
        logger.warning("position event is not an object: %r", type(raw))
        return None

    schema = raw.get("schema")
    if schema != SUPPORTED_SCHEMA:
        if schema not in _warned_unknown:
            _warned_unknown.add(schema)
            logger.warning(
                "skipping position event with schema=%r (this consumer speaks %d); "
                "trade_executor is running a newer wire contract",
                schema, SUPPORTED_SCHEMA,
            )
        return None

    event = raw.get("event")
    if event not in KNOWN_EVENTS:
        if event not in _warned_unknown:
            _warned_unknown.add(event)
            logger.warning("skipping unknown position event kind %r", event)
        return None

    pair = raw.get("pair")
    if not pair:
        logger.warning("position event %r has no pair; skipping", event)
        return None

    state = raw.get("position_state")
    if not isinstance(state, dict):
        logger.warning("position event %r for %s has no position_state; skipping", event, pair)
        return None

    status = state.get("status")
    if not status:
        logger.warning("position event %r for %s has no status; skipping", event, pair)
        return None

    try:
        return PositionEvent(
            schema=schema,
            pair=pair,
            event=event,
            position_id=raw.get("position_id"),
            reason=raw.get("reason"),
            status=status,
            side=state.get("side"),
            market_kind=state.get("market_kind"),
            target_size=_num(state, "target_size"),
            net_size=_num(state, "net_size"),
            intended_open_price=_num(state, "intended_open_price"),
            avg_entry_price=_num(state, "avg_entry_price"),
            avg_exit_price=_num(state, "avg_exit_price"),
            stop_loss_price=_num(state, "stop_loss_price"),
            take_profit_price=_num(state, "take_profit_price"),
            realized_pnl=_num(state, "realized_pnl"),
            unrealized_pnl=_num(state, "unrealized_pnl"),
            settlement_complete=bool(state.get("settlement_complete", False)),
            leverage=state.get("leverage"),
            liquidation_price=_num(state, "liquidation_price"),
            created_at=state.get("created_at"),
            ts=raw.get("ts", ""),
        )
    except (TypeError, ValueError) as exc:
        logger.warning("position event %r for %s could not be decoded: %s", event, pair, exc)
        return None


class PositionConsumer:
    """PULL socket on the executor's outbound topic.

    Args:
        connect_addr: e.g. ``tcp://executor:5556``.

    Raises:
        ImportError: from `__init__` only, when pyzmq is absent. The
            module itself still imports, so `trader.py` can degrade to no
            consumer exactly as it already does for `IndicatorPublisher`.
    """

    def __init__(self, connect_addr: str) -> None:
        import zmq  # noqa: PLC0415 — see the module docstring

        self._zmq = zmq
        self.connect_addr = connect_addr
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        # Bounded: a consumer that stalls must drop old position updates
        # rather than grow without limit. The newest state is what
        # matters, and a `position_query` recovers anything missed.
        self._sock.setsockopt(zmq.RCVHWM, 1000)
        self._sock.connect(connect_addr)

        self._req = self._ctx.socket(zmq.PUSH)
        self._req.setsockopt(zmq.SNDHWM, 10)

    def connect_request_socket(self, inbound_addr: str) -> None:
        """Connect the socket `request_snapshot` publishes on.

        Separate from the PULL connection because they are opposite ends
        of two different topics — the executor binds one for what it
        sends and one for what it receives.
        """
        self._req.connect(inbound_addr)

    def poll(self, timeout_ms: int = 0) -> list[PositionEvent]:
        """Drain everything queued. Never blocks past `timeout_ms`.

        Called from `Robot`'s 1 s tick, so it must return promptly with an
        empty list when nothing is waiting: telemetry must never block the
        trading path.
        """
        events: list[PositionEvent] = []
        try:
            while True:
                flags = self._zmq.NOBLOCK if not events and timeout_ms == 0 else self._zmq.NOBLOCK
                try:
                    raw = self._sock.recv(flags=flags)
                except self._zmq.Again:
                    break
                try:
                    decoded = parse_position_event(json.loads(raw))
                except (ValueError, TypeError) as exc:
                    logger.warning("position event is not valid JSON: %s", exc)
                    continue
                if decoded is not None:
                    events.append(decoded)
        except Exception:
            # A socket error must not reach the trading tick. Logged, not
            # raised; the next poll retries.
            logger.exception("position consumer poll failed")
        return events

    def request_snapshot(self, pair: str | None = None) -> None:
        """Ask the executor what it holds (`position_query`, spec §6.2).

        For a freshly started main/: the outbound topic carries *changes*,
        so a consumer that joins after the last one has nothing to rebuild
        from and could otherwise wait hours for the next.
        """
        import uuid

        message = {"id": str(uuid.uuid4()), "type": "position_query"}
        if pair is not None:
            message["pair"] = pair
        try:
            self._req.send_string(json.dumps(message), flags=self._zmq.NOBLOCK)
        except Exception:
            logger.exception("position_query could not be sent")

    def close(self) -> None:
        for sock in (self._sock, self._req):
            try:
                sock.close(linger=0)
            except Exception:
                logger.exception("position consumer socket close failed")
