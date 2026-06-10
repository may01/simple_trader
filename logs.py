# logs.py — structured logging utilities
# All functions print to stdout with a timestamp prefix.

import traceback
from datetime import datetime


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    """Standard operation log with timestamp prefix."""
    print(f"[{_timestamp()}] INFO  {message}", flush=True)


def log_error(message: str) -> None:
    """Error log with traceback context when available."""
    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        print(f"[{_timestamp()}] ERROR {message}\n{tb}", flush=True)
    else:
        print(f"[{_timestamp()}] ERROR {message}", flush=True)


def log_warning(message: str) -> None:
    """Warning log with timestamp prefix."""
    print(f"[{_timestamp()}] WARN  {message}", flush=True)


def log_revenue(revenue_pct: float, revenue_abs: float, details: dict) -> None:
    """P&L accounting log."""
    print(
        f"[{_timestamp()}] REVENUE  pct={revenue_pct:.4f}  abs={revenue_abs:.4f}  details={details}",
        flush=True,
    )


def log_stock(operation: str, weight: int) -> None:
    """API weight tracking log."""
    print(
        f"[{_timestamp()}] STOCK  operation={operation}  weight={weight}",
        flush=True,
    )
