"""verify_stock_ops.py — attended mainnet margin verification (Phase 15 Task 02).

Exercises the six spec operations against the REAL Binance mainnet margin
account, individually and attributably, with a hard <=50 USDT cap:

    get_info | buy | get_order | sell | borrow | repay

Buy/sell are placed as non-filling LIMIT orders far from market and cancelled.
borrow takes a tiny loan and immediately repays it, asserting the outstanding
loan returns to its pre-test level. Every placed order is cancelled and every
borrow repaid in a finally block, even on failure.

⚠️  REAL FUNDS. Run attended only. Never imported into pytest/CI.

Usage:
    docker compose run --rm trader python3 scripts/verify_stock_ops.py --dry-run
    LIVE_POSITION_USDT=40 docker compose run --rm -e LIVE_POSITION_USDT \
        trader python3 scripts/verify_stock_ops.py
"""

import argparse
import os
import sys

# scripts/ is one level below the repo root; ensure imports resolve when run
# directly as `python3 scripts/verify_stock_ops.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import STATUS_SUCCESS, TRADE_BUY, TRADE_SELL  # noqa: E402

USDT_CAP = 50.0
DEFAULT_USDT = 40.0

# Non-filling offsets: buy well below best bid, sell well above best ask.
BUY_OFFSET = 0.80
SELL_OFFSET = 1.20

# Outstanding loan is considered "returned to zero" within this coin tolerance.
LOAN_EPSILON = 1e-6


def resolve_usdt() -> float:
    """Read LIVE_POSITION_USDT (default 40). Refuse to run above the 50 cap."""
    usdt = float(os.environ.get("LIVE_POSITION_USDT", str(DEFAULT_USDT)))
    if usdt > USDT_CAP:
        print(f"REFUSE: LIVE_POSITION_USDT={usdt} exceeds hard cap {USDT_CAP}.")
        sys.exit(2)
    return usdt


def _best_prices(item):
    """Return (best_ask, best_bid) from the order book.

    depth() returns (asks, bids), each sorted by price ascending, so the best
    ask is the cheapest sell (first ask row) and the best bid is the highest
    buy (last bid row).
    """
    asks, bids = item.depth(100)
    best_ask = float(asks.iloc[0]["price"])
    best_bid = float(bids.iloc[-1]["price"])
    return best_ask, best_bid


class Recorder:
    """Collects PASS/FAIL lines and prints a final summary table."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def record(self, op: str, ok: bool, detail: str = "") -> bool:
        tag = "PASS" if ok else "FAIL"
        print(f"{tag}: {op}{(' — ' + detail) if detail else ''}")
        self.results.append((op, ok, detail))
        return ok

    def all_passed(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def summary(self) -> None:
        print("\n=== Summary ===")
        for op, ok, detail in self.results:
            print(f"  {'PASS' if ok else 'FAIL'}  {op:<10} {detail}")


def dry_run() -> int:
    """Print the intended call sequence without sending anything."""
    pair = os.environ.get("PAIR", "<PAIR unset>")
    usdt = resolve_usdt()
    print("DRY RUN — no orders sent, no loans taken.")
    print(f"  pair           : {pair}")
    print(f"  notional cap   : {usdt} USDT (hard cap {USDT_CAP})")
    print("  planned calls (borrow funds the sell, so no pre-held inventory needed):")
    print("    1. item.info()                                  # get_info")
    print("    2. item.depth(100) -> best_ask, best_bid")
    print("    3. item.borrow(coin, usdt/best_ask)             # borrow (<= get_aviable_loan)")
    print(f"    4. item.trade(TRADE_SELL, best_ask*{SELL_OFFSET}, usdt/price) # sell (uses borrowed coin)")
    print("    5. item.order_info(order_id)                    # get_order (on sell)")
    print("    6. item.cancel_order(order_id)                  # cancel sell, frees coin")
    print(f"    7. item.trade(TRADE_BUY,  best_bid*{BUY_OFFSET}, usdt/price)  # buy (non-filling) + cancel")
    print("    8. item.repay(coin, amount); assert loan back to pre-test level  # repay")
    return 0


def real_run() -> int:
    from stocks_holder import do_stock_init, stock_holder

    usdt = resolve_usdt()
    do_stock_init("binance")
    item = stock_holder.item
    coin = item.coin

    rec = Recorder()
    open_order_ids: list[str] = []
    borrowed_amount = 0.0
    loan_before = 0.0

    try:
        # 1. get_info -----------------------------------------------------
        info = item.info()
        symbol = item.get_pair_name()
        rec.record(
            "get_info",
            bool(info) and info.get("symbol", "").upper() == symbol,
            f"symbol={info.get('symbol') if info else None}",
        )

        # 2. reference prices --------------------------------------------
        best_ask, best_bid = _best_prices(item)
        print(f"INFO: best_ask={best_ask} best_bid={best_bid}")

        # 3. borrow ------------------------------------------------------
        # Borrow first so the loaned coin funds the sell test below — the
        # harness is then self-sufficient and does not require pre-held
        # inventory. Borrow enough to cover the sell amount (best_ask < the
        # 1.20x sell price, so usdt/best_ask > usdt/sell_price).
        _, loan_before = item.funds(coin, "borrowed")
        _, available = item.get_aviable_loan(coin)
        borrow_amount = round(min(usdt / best_ask, available), 2)
        if borrow_amount <= 0:
            rec.record("borrow", False, f"no borrowable amount (available={available})")
        else:
            status, got = item.borrow(coin, borrow_amount)
            if status == STATUS_SUCCESS and got > 0:
                borrowed_amount = borrow_amount
            rec.record(
                "borrow",
                status == STATUS_SUCCESS and got > 0,
                f"coin={coin} amount={borrow_amount}",
            )

        # 4. sell (non-filling) — uses the borrowed coin as inventory -----
        sell_price = round(best_ask * SELL_OFFSET, 2)
        sell_amount = usdt / sell_price
        sell_id = ""
        if item.is_invalid_amount(sell_amount, sell_price):
            rec.record("sell", False, "amount below exchange minimums at offset price")
        else:
            status, res = item.trade(TRADE_SELL, sell_price, sell_amount)
            sell_id = res.get("order_id", "")
            if status == STATUS_SUCCESS and sell_id:
                open_order_ids.append(sell_id)
            rec.record(
                "sell",
                status == STATUS_SUCCESS and bool(sell_id),
                f"order_id={sell_id} price={sell_price} amount={round(sell_amount, 2)}",
            )

        # 5. get_order — inspect the resting sell order ------------------
        if sell_id:
            status, oinfo = item.order_info(sell_id)
            rec.record(
                "get_order",
                status == STATUS_SUCCESS
                and oinfo.get("status") in {"NEW", "PARTIALLY_FILLED"}
                and oinfo.get("start_amount", 0) > 0,
                f"status={oinfo.get('status')} start_amount={oinfo.get('start_amount')}",
            )
        else:
            rec.record("get_order", False, "skipped — no sell order placed")

        # 6. cancel sell (cleanup, frees the coin for repay) -------------
        if sell_id:
            status, cinfo = item.cancel_order(sell_id)
            cancelled = status == STATUS_SUCCESS and cinfo.get("status") == "CANCELED"
            if cancelled and sell_id in open_order_ids:
                open_order_ids.remove(sell_id)
            rec.record("cancel_sell", cancelled, f"status={cinfo.get('status')}")

        # 7. buy (non-filling) + cancel ----------------------------------
        buy_price = round(best_bid * BUY_OFFSET, 2)
        buy_amount = usdt / buy_price
        if item.is_invalid_amount(buy_amount, buy_price):
            rec.record("buy", False, "amount below exchange minimums at offset price")
        else:
            status, res = item.trade(TRADE_BUY, buy_price, buy_amount)
            buy_id = res.get("order_id", "")
            if status == STATUS_SUCCESS and buy_id:
                open_order_ids.append(buy_id)
            rec.record(
                "buy",
                status == STATUS_SUCCESS and bool(buy_id),
                f"order_id={buy_id} price={buy_price} amount={round(buy_amount, 2)}",
            )
            if buy_id:
                status, cinfo = item.cancel_order(buy_id)
                if status == STATUS_SUCCESS and cinfo.get("status") == "CANCELED" \
                        and buy_id in open_order_ids:
                    open_order_ids.remove(buy_id)

        # 8. repay -------------------------------------------------------
        if borrowed_amount > 0:
            status, _ = item.repay(coin, borrowed_amount)
            _, loan_after = item.funds(coin, "borrowed")
            returned = abs(loan_after - loan_before) <= max(LOAN_EPSILON, borrowed_amount * 0.01)
            if returned:
                borrowed_amount = 0.0
            rec.record(
                "repay",
                status == STATUS_SUCCESS and returned,
                f"loan_before={loan_before} loan_after={loan_after}",
            )
        else:
            rec.record("repay", False, "skipped — borrow did not succeed")

    finally:
        # Safety net: cancel any order still resting, repay any open loan.
        for oid in list(open_order_ids):
            print(f"CLEANUP: cancelling residual order {oid}")
            item.cancel_order(oid)
        if borrowed_amount > 0:
            print(f"CLEANUP: repaying residual loan {borrowed_amount} {coin}")
            item.repay(coin, borrowed_amount)

    rec.summary()
    residual = bool(open_order_ids) or borrowed_amount > 0
    if residual:
        print("FAIL: residual orders or loan left open after cleanup attempt.")
    return 0 if (rec.all_passed() and not residual) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended calls without sending anything (no creds/network).",
    )
    args = parser.parse_args()
    return dry_run() if args.dry_run else real_run()


if __name__ == "__main__":
    sys.exit(main())
