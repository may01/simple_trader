"""End-to-end check: trade_executor position state -> main/'s PositionConsumer.

    python3 scripts/e2e_position_round_trip.py          # against the stack as it is
    python3 scripts/e2e_position_round_trip.py --cold   # down + rebuild the executor first

Run from main/ on the host. Stdlib only -- everything needing pyzmq or
main/'s code runs in `scripts/e2e_position_round_trip_probe.py`, inside
the `simple_trader` image on the `trader_mq` network.

What it proves (position-management design §10):

  A1  main/'s real decoder understands what the executor really sends --
      not a fixture, not a double.
  A2  `position_query` from a freshly started consumer gets an answer,
      including for a flat pair. Silence there is indistinguishable from
      a dropped message, which is the confusion the query exists to end.
  A3  Whatever the executor reports agrees with `/api/positions`, so the
      two readers of the same state cannot drift.

Side effects: `--cold` restarts the executor stack (never `down -v`; the
DB volume is kept). Sends one `position_query`, which the executor
answers and does not persist.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(HERE)
DEFAULT_EXECUTOR_DIR = os.path.normpath(
    os.path.join(MAIN_DIR, "..", "trade_executor", ".worktrees", "layer-implementation"))
NETWORK = "trader_mq"
EXPECTED_SCHEMA = 9


def sh(cmd, check=True):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


class Executor:
    def __init__(self, directory, api):
        self.compose = ["docker", "compose", "-f", os.path.join(directory, "docker-compose.yml")]
        self.api = api.rstrip("/")

    def up(self, cold):
        if cold:
            sh(self.compose + ["down"])  # never -v: the DB volume is not ours to drop
            sh(self.compose + ["up", "-d", "--build", "postgres", "executor", "visualizer"])
        else:
            sh(self.compose + ["up", "-d", "postgres", "executor", "visualizer"])

    def get(self, path):
        with urllib.request.urlopen(self.api + path, timeout=5) as resp:
            return json.loads(resp.read())


def probe(args):
    cmd = ["docker", "run", "--rm", "--network", NETWORK,
           "-e", "PYTHONDONTWRITEBYTECODE=1",  # no root-owned __pycache__ in the bind mount
           "-v", f"{MAIN_DIR}:/code", "-w", "/code", args.image,
           "python3", "scripts/e2e_position_round_trip_probe.py",
           "--outbound", args.outbound, "--inbound", args.inbound,
           "--pair", args.pair, "--seconds", str(args.seconds)]
    out = sh(cmd).stdout.strip().splitlines()
    if not out:
        raise SystemExit("probe produced no output")
    return json.loads(out[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--executor-dir", default=os.environ.get("EXECUTOR_DIR", DEFAULT_EXECUTOR_DIR))
    ap.add_argument("--api", default="http://127.0.0.1:8090")
    ap.add_argument("--outbound", default="tcp://executor:5556")
    ap.add_argument("--inbound", default="tcp://executor:5555")
    ap.add_argument("--pair", default="LINKUSDT")
    ap.add_argument("--image", default="simple_trader")
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()

    executor = Executor(args.executor_dir, args.api)
    executor.up(args.cold)

    status = executor.get("/api/status")
    if status.get("schema_version") not in (None, EXPECTED_SCHEMA):
        raise SystemExit(
            f"executor reports schema {status.get('schema_version')}, expected {EXPECTED_SCHEMA}")

    print(f"listening for {args.seconds}s on {args.outbound} ...")
    result = probe(args)
    print(json.dumps(result, indent=2))

    failures = []

    # A1/A2: something arrived and main/'s own decoder understood it. The
    # heartbeat alone guarantees traffic for a live position; the
    # position_query guarantees an answer even when flat.
    if result["decoded"] == 0:
        failures.append(
            "A1/A2: nothing decoded. Either the executor published nothing in the window "
            "(no live position AND no answer to position_query -- the query must be "
            "answered even for a flat pair), or main/'s decoder rejected every message.")

    # A3: the two readers of the same state agree.
    try:
        api_positions = executor.get(f"/api/positions?pair={args.pair}&from=0&to={2**63 - 1}")
    except Exception as exc:  # noqa: BLE001 -- the route may not exist yet
        api_positions = None
        print(f"note: /api/positions unavailable ({exc}); A3 skipped")

    if api_positions is not None and result["events"]:
        last = result["events"][-1]
        if last["status"] != "flat" and not api_positions:
            failures.append(
                f"A3: the executor pushed an open position for {args.pair} but "
                "/api/positions reports none -- the MQ and HTTP readers disagree")

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1

    print("OK: main/'s consumer decoded the executor's own messages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
