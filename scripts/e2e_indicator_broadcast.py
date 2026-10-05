"""End-to-end check: main/'s real IndicatorPublisher -> trade_executor -> Postgres + API.

    python3 scripts/e2e_indicator_broadcast.py          # against the executor stack as it is
    python3 scripts/e2e_indicator_broadcast.py --cold   # down + rebuild both sides first

Run from main/ on the host. Stdlib only -- everything that needs pyzmq or
main/'s code runs in `scripts/e2e_indicator_broadcast_probe.py`, inside the
`simple_trader` image on the `trader_mq` network. Exits non-zero if any
assertion fails.

Spec: external/executor/specs/2026-09-20-indicator-broadcast-e2e-check-design.md
(assertion numbers below are that spec's §6). Assertion 2 -- the executor's
in-process `current_indicator` cache -- is deliberately not here: it is
unobservable from outside the process and is covered by state_store's own
tests.

Side effects: `--cold` restarts the executor stack (never `down -v`; the DB
volume is kept). Every run appends a handful of rows named `e2e_<run>_*` to
the append-only `indicators` table; they show in the visualiser's
Indicators panel until they expire (300 s, the sender's TTL).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(HERE)
DEFAULT_EXECUTOR_DIR = os.path.normpath(
    os.path.join(MAIN_DIR, "..", "trade_executor", ".worktrees", "layer-implementation"))
NETWORK = "trader_mq"
EXPECTED_SCHEMA = 7
SENDER_TTL_MS = 300_000
# expires_at is stamped by the sender's clock, received_at by the executor's,
# so expires_at - received_at is the TTL minus transit and skew, never exact
# (a 1 ms shortfall was the first cold run's A1 "failure").
TTL_TOLERANCE_MS = 1_000


def sh(cmd, check=True, capture=True):
    r = subprocess.run(cmd, text=True, capture_output=capture)
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

    def sql(self, query):
        r = sh(self.compose + ["exec", "-T", "postgres", "psql", "-U", "executor", "-d", "trader",
                               "-tA", "-F", "|", "-c", query])
        return [line.split("|") for line in r.stdout.strip().splitlines() if line]

    def get(self, path):
        with urllib.request.urlopen(self.api + path, timeout=5) as resp:
            return json.loads(resp.read())

    def current(self, pair):
        return {row["name"]: row for row in self.get(f"/api/current_indicators?pair={pair}")}


def probe(args, run, warmup=False):
    cmd = ["docker", "run", "--rm", "--network", NETWORK,
           "-e", "PYTHONDONTWRITEBYTECODE=1",  # no root-owned __pycache__ in the bind mount
           "-v", f"{MAIN_DIR}:/code", "-w", "/code", args.image,
           "python3", "scripts/e2e_indicator_broadcast_probe.py",
           "--addr", args.addr, "--pair", args.pair, "--run", run,
           "--short-ttl", str(args.short_ttl)]
    if warmup:
        cmd.append("--warmup-only")
    out = sh(cmd).stdout.strip().splitlines()
    return json.loads(out[-1])


def wait_until(what, fn, timeout_s, every_s=2.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            if fn():
                return
        except Exception as e:  # noqa: BLE001 -- anything short of success means "not yet"
            last = e
        time.sleep(every_s)
    raise SystemExit(f"timed out after {timeout_s}s waiting for {what} (last error: {last})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cold", action="store_true", help="down + rebuild the executor stack and main's image first")
    ap.add_argument("--executor-dir", default=os.environ.get("EXECUTOR_DIR", DEFAULT_EXECUTOR_DIR))
    ap.add_argument("--api", default="http://127.0.0.1:8090")
    ap.add_argument("--addr", default="tcp://executor:5555")
    ap.add_argument("--pair", default="BTCUSDT")
    ap.add_argument("--image", default="simple_trader")
    ap.add_argument("--short-ttl", type=float, default=15.0)
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    args = ap.parse_args()

    if sh(["docker", "network", "inspect", NETWORK], check=False).returncode != 0:
        raise SystemExit(f"docker network {NETWORK} missing -- run once: docker network create {NETWORK}")

    ex = Executor(args.executor_dir, args.api)
    if args.cold:
        print("cold start: rebuilding main image + executor stack ...", flush=True)
        sh(["docker", "compose", "-f", os.path.join(MAIN_DIR, "docker-compose.yml"), "build", "live"])
    ex.up(args.cold)

    def api_ready():
        s = ex.get("/api/status")
        if s.get("schema_version") != EXPECTED_SCHEMA:
            raise SystemExit(f"schema_version {s.get('schema_version')} != {EXPECTED_SCHEMA}: stale executor build")
        return s.get("ready")
    wait_until("visualizer /api/status ready", api_ready, args.ready_timeout)

    # Readiness of the ingest path itself: keep publishing until one lands.
    # A healthy API says nothing about whether indicator_ingest is running.
    run = f"e2e_{uuid.uuid4().hex[:8]}"
    wait_until("a warm-up reading to reach the indicators table",
               lambda: probe(args, run, warmup=True) is not None
               and ex.sql(f"SELECT 1 FROM indicators WHERE name = '{run}_warmup' LIMIT 1"),
               args.ready_timeout, every_s=1.0)

    sent = probe(args, run)
    t_probe_done = time.time()
    rows = {}
    for name, value, kind, volume, expires_at, received_at in ex.sql(
            f"SELECT name, value, kind, coalesce(volume::text, 'NULL'), expires_at, received_at "
            f"FROM indicators WHERE name LIKE '{run}\\_%' ORDER BY received_at, seq"):
        rows.setdefault(name[len(run) + 1:], []).append(
            {"value": float(value), "kind": kind, "volume": volume,
             "ttl_ms": int(expires_at) - int(received_at)})
    api_now = ex.current(args.pair)

    results = []

    def check(num, label, ok, detail):
        results.append((num, label, bool(ok), detail))

    arrival = rows.get("arrival", [])
    check(1, "arrival", len(arrival) >= 1 and arrival[0]["kind"] == "none"
          and arrival[0]["volume"] == "NULL" and abs(arrival[0]["ttl_ms"] - SENDER_TTL_MS) <= TTL_TOLERANCE_MS
          and arrival[0]["value"] == 111.5, arrival[:1])
    check(3, "dedup on a resent id", len(rows.get("dedup", [])) == 1, f"{len(rows.get('dedup', []))} row(s)")
    api_arrival = api_now.get(f"{run}_arrival")
    check(4, "append-only, newest wins", [r["value"] for r in arrival] == [111.5, 222.5]
          and api_arrival is not None and float(api_arrival["value"]) == 222.5,
          {"rows": [r["value"] for r in arrival], "api": api_arrival and api_arrival["value"]})
    sup = rows.get("support", [])
    api_sup = api_now.get(f"{run}_support")
    check(6, "support + volume round trip", len(sup) == 1 and sup[0]["kind"] == "support"
          and float(sup[0]["volume"]) == 3.25 and api_sup is not None
          and api_sup["kind"] == "support" and float(api_sup["volume"]) == 3.25,
          {"row": sup, "api": api_sup and {k: api_sup[k] for k in ("kind", "volume")}})
    check(7, "negative control (dead address)", "deadaddr" not in rows and sent["dropped_dead"] >= 1
          and sent["dropped_live"] == 0, {"dropped_dead": sent["dropped_dead"], "dropped_live": sent["dropped_live"]})

    short_row = rows.get("shortttl", [])
    served_before = f"{run}_shortttl" in api_now
    lapse_at = sent["short_sent_at"] + args.short_ttl + 1.0
    if served_before:
        time.sleep(max(0.0, lapse_at - time.time()))
    gone_after = f"{run}_shortttl" not in ex.current(args.pair)
    row_kept = bool(ex.sql(f"SELECT 1 FROM indicators WHERE name = '{run}_shortttl'"))
    check(5, "expiry: served, then not, row kept", len(short_row) == 1
          and abs(short_row[0]["ttl_ms"] - args.short_ttl * 1000) <= TTL_TOLERANCE_MS and served_before and gone_after and row_kept,
          {"served_before": served_before, "gone_after": gone_after, "row_kept": row_kept,
           "checked_s_after_send": round(t_probe_done - sent["short_sent_at"], 1)})

    print(f"\nindicator broadcast e2e -- run {run}, pair {args.pair}, via {args.addr}")
    for num, label, ok, detail in sorted(results):
        print(f"  [{'PASS' if ok else 'FAIL'}] A{num} {label}: {detail}")
    failed = [r for r in results if not r[2]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
