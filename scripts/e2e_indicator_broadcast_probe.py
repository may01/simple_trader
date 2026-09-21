"""In-container half of the indicator-broadcast end-to-end check.

Runs inside the `simple_trader` image, attached to the `trader_mq` network,
and publishes through main/'s real `IndicatorPublisher` -- the point is to
exercise main/'s own builder and socket code, not a hand-rolled PUSH. The
only hand-built messages are the two the real publisher cannot produce:
a resent id (dedup) and a `kind: "support"` reading (v1 only emits "none").

Prints one JSON line on stdout describing what it sent; every assertion
is made by the host-side driver, `e2e_indicator_broadcast.py`, which is
the thing to run. See external/executor/specs/
2026-09-20-indicator-broadcast-e2e-check-design.md.
"""
import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import zmq

from mq.indicator_publisher import IndicatorPublisher, build_indicator_update

# zmq.IMMEDIATE=1 on the publisher: anything sent before libzmq finishes
# connecting is dropped, not queued, and logged exactly like a wrong
# address would be. Every socket here settles for this long before its
# first message that an assertion depends on.
SETTLE_S = 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--run", required=True, help="unique name prefix for this run's readings")
    ap.add_argument("--dead-addr", default="tcp://127.0.0.1:5999")
    ap.add_argument("--short-ttl", type=float, default=3.0)
    ap.add_argument("--warmup-only", action="store_true",
                    help="publish one reading and exit; the driver polls for it as its readiness check")
    args = ap.parse_args()
    run, pair = args.run, args.pair

    pub = IndicatorPublisher(connect_addr=args.addr, ttl_seconds=300.0)
    time.sleep(SETTLE_S)

    if args.warmup_only:
        pub.publish(pair=pair, name=f"{run}_warmup", value=1.0)
        time.sleep(0.5)
        print(json.dumps({"run": run, "dropped_live": pub.dropped}))
        pub.close()
        return 0

    # A1 arrival + A4 append-only: two values for one (pair, name).
    pub.publish(pair=pair, name=f"{run}_arrival", value=111.5)
    time.sleep(0.2)
    pub.publish(pair=pair, name=f"{run}_arrival", value=222.5)

    # A5 expiry: short TTL -- served now, gone from "current" after it lapses,
    # while its row stays in the table.
    short = IndicatorPublisher(connect_addr=args.addr, ttl_seconds=args.short_ttl)
    time.sleep(SETTLE_S)
    short.publish(pair=pair, name=f"{run}_shortttl", value=333.5)
    short_sent_at = time.time()

    raw = zmq.Context.instance().socket(zmq.PUSH)
    raw.setsockopt(zmq.IMMEDIATE, 1)
    raw.setsockopt(zmq.LINGER, 0)
    raw.connect(args.addr)
    time.sleep(SETTLE_S)

    # A3 dedup: the same id twice. The real publisher mints a fresh uuid per
    # call, so reuse one message built by main/'s own builder.
    dup = build_indicator_update(pair=pair, name=f"{run}_dedup", value=444.5,
                                 now=datetime.now(timezone.utc), ttl_seconds=300.0)
    raw.send_json(dup)
    time.sleep(0.2)
    raw.send_json(dup)

    # A6 support-kind round trip, wire -> store -> API.
    raw.send_json({
        "id": str(uuid.uuid4()), "type": "indicator_update", "pair": pair,
        "name": f"{run}_support", "value": 55555.5, "kind": "support", "volume": 3.25,
        "expires_at": dup["expires_at"],
    })

    # A7 negative control: nothing listening -> no raise, no row, counter moves.
    dead = IndicatorPublisher(connect_addr=args.dead_addr, ttl_seconds=300.0)
    time.sleep(SETTLE_S)
    dead.publish(pair=pair, name=f"{run}_deadaddr", value=666.5)

    time.sleep(0.5)
    print(json.dumps({
        "run": run,
        "dropped_live": pub.dropped + short.dropped,
        "dropped_dead": dead.dropped,
        "short_ttl_s": args.short_ttl,
        "short_sent_at": short_sent_at,
    }))
    for s in (pub, short, dead):
        s.close()
    raw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
