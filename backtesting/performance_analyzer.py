"""PerformanceAnalyzer for backtesting results (Phase 08, Task 03)."""


class PerformanceAnalyzer:
    """Analyzes backtesting performance metrics from worker results."""

    def __init__(self, results: list[dict]) -> None:
        """
        Initialize with worker results and flatten trades.

        Args:
            results: List of worker result dicts, each containing:
                - revenue_history: list of (revenue_pct, revenue_abs) tuples
                - trade_count: number of trades in this worker
        """
        self.all_trades: list[tuple[float, float]] = []

        # Flatten all trades from all workers
        for worker_result in results:
            revenue_history = worker_result.get('revenue_history', [])
            self.all_trades.extend(revenue_history)

        # Calculate basic statistics
        self.total_trades: int = len(self.all_trades)
        self.winning_trades: int = sum(1 for pct, _ in self.all_trades if pct > 0)
        self.losing_trades: int = sum(1 for pct, _ in self.all_trades if pct <= 0)

    def analyze(self) -> dict:
        """
        Compute and return performance metrics.

        Returns:
            Dict with keys:
            - total_trades: int
            - win_rate: float (0.0 if no trades)
            - avg_revenue_pct: float (0.0 if no trades)
            - total_revenue_abs: float
            - max_drawdown: float (maximum peak-to-trough drop)
            - max_consecutive_losses: int
            - profit_factor: float (inf if only wins, 0.0 if only losses)
        """
        metrics = {
            'total_trades': self.total_trades,
            'win_rate': self._calculate_win_rate(),
            'avg_revenue_pct': self._calculate_avg_revenue_pct(),
            'total_revenue_abs': self._calculate_total_revenue_abs(),
            'max_drawdown': self._calculate_max_drawdown(),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(),
            'profit_factor': self._calculate_profit_factor(),
        }
        return metrics

    def _calculate_win_rate(self) -> float:
        """Win rate = winning_trades / total_trades."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def _calculate_avg_revenue_pct(self) -> float:
        """Mean of all revenue_pct values."""
        if self.total_trades == 0:
            return 0.0
        total_pct = sum(pct for pct, _ in self.all_trades)
        return total_pct / self.total_trades

    def _calculate_total_revenue_abs(self) -> float:
        """Sum of all revenue_abs values."""
        return sum(abs_rev for _, abs_rev in self.all_trades)

    def _calculate_max_drawdown(self) -> float:
        """
        Maximum peak-to-trough drop in cumulative revenue_abs.

        Tracks running peak and computes drawdown as max(peak - value)
        where value is after the peak.
        """
        if self.total_trades == 0:
            return 0.0

        # Build cumulative revenue series
        cumulative = 0.0
        running_peak = 0.0
        max_dd = 0.0

        for _, abs_rev in self.all_trades:
            cumulative += abs_rev
            if cumulative > running_peak:
                running_peak = cumulative
            drawdown = running_peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _calculate_max_consecutive_losses(self) -> int:
        """Longest streak of trades with revenue_pct <= 0."""
        if self.total_trades == 0:
            return 0

        max_streak = 0
        current_streak = 0

        for pct, _ in self.all_trades:
            if pct <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _calculate_profit_factor(self) -> float:
        """
        Profit factor = sum(winning_abs) / abs(sum(losing_abs)).

        Returns:
        - float('inf') if no losing trades and there are winning trades
        - 0.0 if no winning trades
        - 0.0 if no trades at all
        """
        if self.total_trades == 0:
            return 0.0

        winning_sum = sum(abs_rev for pct, abs_rev in self.all_trades if pct > 0)
        losing_sum = sum(abs_rev for pct, abs_rev in self.all_trades if pct <= 0)

        # No winning trades
        if winning_sum == 0:
            return 0.0

        # No losing trades
        if losing_sum == 0:
            return float('inf')

        # Both exist
        return winning_sum / abs(losing_sum)

    def print_report(self) -> None:
        """Print formatted summary of metrics to stdout."""
        metrics = self.analyze()

        print("=" * 60)
        print("PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)
        print(f"Total Trades:           {metrics['total_trades']}")
        print(f"Win Rate:               {metrics['win_rate']:.2%}")
        print(f"Avg Revenue (pct):      {metrics['avg_revenue_pct']:.4f}")
        print(f"Total Revenue (abs):    {metrics['total_revenue_abs']:.2f}")
        print(f"Max Drawdown:           {metrics['max_drawdown']:.2f}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        profit_factor = metrics['profit_factor']
        if profit_factor == float('inf'):
            print(f"Profit Factor:          inf")
        else:
            print(f"Profit Factor:          {profit_factor:.4f}")
        print("=" * 60)

    def to_dict(self) -> dict:
        """
        Return dict with all_trades and all analyze() keys.

        Returns:
            Dict with:
            - all_trades: list of (pct, abs) tuples
            - Plus all keys from analyze()
        """
        metrics = self.analyze()
        metrics['all_trades'] = self.all_trades
        return metrics
