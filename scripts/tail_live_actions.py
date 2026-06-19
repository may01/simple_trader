"""tail_live_actions.py — observe the live Robot (Phase 15 Task 03).

Two read-only modes:

  --preflight   Run the pre-trade checks so a multi-hour live wait is not
                wasted on a misconfiguration: live creds resolve, the EMA
                columns the strategies need are present, the configured size
                is tradeable, and exactly 2 EMA strategies register.

  (default)     Tail the most recent live Action records from the live store
                (shared_folder()/live_actions.jsonl), without stopping the bot.

⚠️  --preflight needs live Binance creds (it fetches candles). It places no
orders. Tail mode needs no creds — it only reads the local action store.

Usage:
    docker compose run --rm trader python3 scripts/tail_live_actions.py --preflight
    docker compose run --rm trader python3 scripts/tail_live_actions.py -n 20
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tf the EMA test strategies run on (StrategyTest1Long/Short default tf=5).
EMA_TF = 5
EMA_COLUMNS = [f"{EMA_TF}_ema_7", f"{EMA_TF}_ema_14"]


def _action_store_path() -> str:
    from helpers import shared_folder

    return shared_folder() + "live_actions.jsonl"


def tail(n: int) -> int:
    from robots.live_action_log import LiveActionLog

    path = _action_store_path()
    records = LiveActionLog(path).tail(n)
    if not records:
        print(f"No live Action records at {path}")
        return 0
    print(f"Last {len(records)} live Action records ({path}):")
    for a in records:
        print(
            f"  ts={a.timestamp:.0f} tick={a.tick_index} {a.event:<10} "
            f"{a.position_type:<8} target={a.target_price} executed={a.executed_price} "
            f"rev_pct={a.revenue_pct}"
        )
    return 0


def preflight() -> int:
    import trader
    from data import LiveData
    from stocks_holder import do_stock_init, stock_holder

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'}: {label}{(' — ' + detail) if detail else ''}")

    # 1. Live creds / stock init -----------------------------------------
    try:
        do_stock_init("binance")
        stock = stock_holder.item
        check("live stock init (binance)", True, stock.get_pair_name())
    except Exception as exc:
        check("live stock init (binance)", False, repr(exc))
        return 1  # nothing else is meaningful without a stock

    # 2. EMA columns present + non-NaN on closed candles -----------------
    try:
        live_data = LiveData()
        live_data.build_candles()
        df = live_data.get_data_point().get_df(EMA_TF)
        present = all(c in df.columns for c in EMA_COLUMNS)
        last_vals = {
            c: [float(v) for v in df[c].tail(2)] if c in df.columns else []
            for c in EMA_COLUMNS
        }
        non_nan = present and all(
            vals and not any(math.isnan(v) for v in vals) for vals in last_vals.values()
        )
        check("EMA columns present + non-NaN", non_nan, str(last_vals))
        price = float(df[f"{EMA_TF}_close"].iloc[-1])
    except Exception as exc:
        check("EMA columns present + non-NaN", False, repr(exc))
        price = 0.0

    # 3. Size resolves in [30, 50] and is tradeable ----------------------
    try:
        usdt = trader.resolve_live_usdt()
        in_range = 30.0 <= usdt <= 50.0
        tradeable = price > 0 and not stock.is_invalid_amount(usdt / price, price)
        check(
            "size in [30,50] and tradeable",
            in_range and tradeable,
            f"usdt={usdt} price={price}",
        )
    except Exception as exc:
        check("size in [30,50] and tradeable", False, repr(exc))

    # 4. Exactly 2 EMA strategies register -------------------------------
    try:
        sm = trader.build_strategy_manager(stock, "ema")
        check("EMA strategies registered == 2", len(sm.strategies) == 2,
              f"count={len(sm.strategies)}")
    except Exception as exc:
        check("EMA strategies registered == 2", False, repr(exc))

    print("\nPRE-FLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true",
                        help="Run pre-trade checks (needs creds; sends no orders).")
    parser.add_argument("-n", "--num", type=int, default=20,
                        help="Number of recent Action records to tail (default 20).")
    args = parser.parse_args()
    return preflight() if args.preflight else tail(args.num)


if __name__ == "__main__":
    sys.exit(main())
