# stocks/disarmed_stock.py — refuses every order-placing call, passes reads through.
#
# Why this exists
# ---------------
# trade_executor now owns the live position (position-management design,
# decision D2). Until that cutover is complete, `main/` still contains a
# complete order path — robots/robot.py's _open_position/_close_position/
# _stop_loss and robots/live_order_tracker.py — and running it against the
# same account as the executor would double-trade.
#
# That code is deliberately NOT deleted. LiveOrderTracker's loan bookkeeping
# and its crash-recovery JSON describe positions that may be open on the
# exchange right now, and a disabled path can be re-armed with one
# environment variable if the executor turns out not to be ready, whereas a
# deleted one cannot.
#
# So the refusal happens here, at the StockInterface boundary, in one place
# rather than as scattered `if` statements at each of the five call sites.
# This is the same shape as the executor's own NoTradeAccount
# (crates/orchestrator/src/no_trade.rs) and for the same stated reason: the
# refusal must not depend on API-key permissions, on the exchange, or on
# anything outside this process being configured correctly.
#
# What replaced each refused call, on the executor side:
#   trade()        -> execution::Executor::open_position / place_exit
#   borrow()       -> the executor's own margin handling for a short entry
#   cancel_order() -> execution::Executor::do_close / apply_stop_loss_move
#
# To re-arm: MAIN_ORDER_PLACEMENT=enabled (see trader.py). Do not do that
# while the executor is running with EXECUTION_MODE=live.

from __future__ import annotations

import logging

from constants import STATUS_FAIL

logger = logging.getLogger(__name__)

# Present in the dict every refused call returns, so a caller can tell a
# policy refusal from a genuine exchange failure. Both return STATUS_FAIL —
# the callers already handle that — but only one of them is worth an ERROR
# in the log, and telling them apart is what keeps a disarmed run quiet.
DISARMED_KEY = "disarmed"
DISARMED_REASON = "order placement disabled in main/: trade_executor owns the live position"


def is_disarmed_result(payload) -> bool:
    """True when `payload` is the dict a refused call returned.

    Used by robots/robot.py to log a refusal at DEBUG instead of ERROR.
    """
    return isinstance(payload, dict) and payload.get(DISARMED_KEY) is True


class DisarmedStock:
    """Wraps a real stock, refusing every call that could place or cancel an order.

    Reads pass through untouched — disarming a write path must not blind the
    process. Order placement, borrowing, repayment and cancellation are
    refused locally, before any request is constructed.

    **Deliberately not a StockInterface subclass.** StockInterface defines a
    concrete no-op for every method (printing "INIT STOCK" and returning
    STATUS_FAIL), so inheriting it would mean normal attribute lookup always
    succeeds and `__getattr__` is never consulted — every read would silently
    resolve to the base class's no-op instead of reaching the real stock.
    A test caught exactly that. Duck typing is what the rest of the codebase
    relies on anyway: nothing does `isinstance(x, StockInterface)`.

    Args:
        inner: The real StockInterface this delegates every read to.
    """

    def __init__(self, inner: StockInterface) -> None:
        self.inner = inner

    # ------------------------------------------------------------------
    # Refused — no request is constructed, nothing reaches the exchange
    # ------------------------------------------------------------------

    def trade(self, trade_type: str, price: float, amount: float, force: bool = False) -> tuple:
        """Refuse to place an order.

        `force` is accepted and deliberately ignored. A `force` flag that
        armed a disarmed stock would do so on the call path most likely to
        be used in an emergency, which is the worst possible place for it.
        """
        logger.debug(
            "DisarmedStock.trade refused: %s price=%s amount=%s force=%s",
            trade_type, price, amount, force,
        )
        return (STATUS_FAIL, self._refusal())

    def borrow(self, coin: str, amount: float) -> tuple:
        logger.debug("DisarmedStock.borrow refused: %s %s", coin, amount)
        return (STATUS_FAIL, self._refusal())

    def repay(self, coin: str, amount: float) -> tuple:
        logger.debug("DisarmedStock.repay refused: %s %s", coin, amount)
        return (STATUS_FAIL, self._refusal())

    def cancel_order(self, order_id: str) -> tuple:
        logger.debug("DisarmedStock.cancel_order refused: %s", order_id)
        return (STATUS_FAIL, self._refusal())

    @staticmethod
    def _refusal() -> dict:
        return {DISARMED_KEY: True, "reason": DISARMED_REASON}

    # ------------------------------------------------------------------
    # Everything else delegates, unchanged
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Delegate any attribute this class does not define to `inner`.

        Deliberately a catch-all rather than one forwarding method per read:
        StockInterface grows, and a method added there later must keep
        working through this wrapper without anyone remembering to forward
        it. The four refused methods above are defined explicitly, so they
        shadow this and are never delegated — which is the only property
        that matters for safety.

        `__getattr__` is only consulted for attributes normal lookup did not
        find, so it cannot accidentally intercept the refusals.
        """
        return getattr(self.inner, name)
