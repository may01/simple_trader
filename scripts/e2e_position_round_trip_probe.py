"""Inside-container half of scripts/e2e_position_round_trip.py.

Runs in the `simple_trader` image on the `trader_mq` network, because it
needs pyzmq and main/'s own `PositionConsumer` -- the point of the check
is that *main/'s real decoder* understands what the executor really
sends, not that some test double does.

Prints one JSON object on the last stdout line.
"""
import argparse
import json
import sys
import time

from mq.position_consumer import KNOWN_EVENTS, PositionConsumer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outbound", required=True, help="executor's PUSH address")
    ap.add_argument("--inbound", required=True, help="executor's PULL address")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--snapshot-only", action="store_true")
    args = ap.parse_args()

    consumer = PositionConsumer(connect_addr=args.outbound)
    consumer.connect_request_socket(args.inbound)
    # A PUSH/PULL connection is not established synchronously; a query
    # sent the instant after connect can be dropped on the floor.
    time.sleep(1.0)

    consumer.request_snapshot(args.pair)

    seen: list[dict] = []
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        for event in consumer.poll():
            seen.append({
                "event": event.event,
                "pair": event.pair,
                "status": event.status,
                "side": event.side,
                "net_size": event.net_size,
                "avg_entry_price": event.avg_entry_price,
                "reason": event.reason,
                "settlement_complete": event.settlement_complete,
            })
        if args.snapshot_only and seen:
            break
        time.sleep(0.25)

    consumer.close()

    kinds = sorted({e["event"] for e in seen})
    print(json.dumps({
        "decoded": len(seen),
        "kinds": kinds,
        "missing_kinds": sorted(set(KNOWN_EVENTS) - set(kinds)),
        "events": seen[-20:],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
