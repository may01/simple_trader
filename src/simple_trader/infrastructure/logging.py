from pathlib import Path
from .config import Config


class TradeLogger:
    def __init__(self, config: Config, log_dir: Path = Path("output")) -> None:
        self._config = config
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"LOG_{config.pair}.txt"

    def log(self, msg: str, do_print: bool = False) -> None:
        with open(self._log_path, "a") as f:
            f.write(f"{msg}\n")
        if do_print:
            print(msg)

    def log_error(self, msg: str) -> None:
        self.log(f"ERROR: {msg}", do_print=True)

    def log_warn(self, msg: str) -> None:
        self.log(f"WARN: {msg}", do_print=True)

    def log_revenue(self, msg: str, time_point: int, thread: int) -> None:
        line = f"[t={time_point} th={thread}] {msg}"
        print(line)
        rev_path = self._log_dir / f"REVENUE_{self._config.pair}_{thread}.txt"
        with open(rev_path, "a") as f:
            f.write(f"{line}\n")
