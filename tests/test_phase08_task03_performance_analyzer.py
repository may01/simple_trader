"""Tests for PerformanceAnalyzer (Phase 08, Task 03)."""

import pytest
from backtesting.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzerInit:
    """Test constructor and initialization."""

    def test_empty_results(self):
        """Empty results list should handle gracefully."""
        pa = PerformanceAnalyzer([])
        assert pa.all_trades == []
        assert pa.total_trades == 0
        assert pa.winning_trades == 0
        assert pa.losing_trades == 0

    def test_single_worker_single_trade_win(self):
        """Single winning trade."""
        results = [
            {'revenue_history': [(0.01, 10.0)], 'trade_count': 1}
        ]
        pa = PerformanceAnalyzer(results)
        assert pa.all_trades == [(0.01, 10.0)]
        assert pa.total_trades == 1
        assert pa.winning_trades == 1
        assert pa.losing_trades == 0

    def test_single_worker_single_trade_loss(self):
        """Single losing trade (revenue_pct < 0)."""
        results = [
            {'revenue_history': [(-0.01, -10.0)], 'trade_count': 1}
        ]
        pa = PerformanceAnalyzer(results)
        assert pa.all_trades == [(-0.01, -10.0)]
        assert pa.total_trades == 1
        assert pa.winning_trades == 0
        assert pa.losing_trades == 1

    def test_single_worker_zero_revenue_pct_is_loss(self):
        """Trade with revenue_pct == 0 counts as losing trade."""
        results = [
            {'revenue_history': [(0.0, 0.0)], 'trade_count': 1}
        ]
        pa = PerformanceAnalyzer(results)
        assert pa.all_trades == [(0.0, 0.0)]
        assert pa.total_trades == 1
        assert pa.winning_trades == 0
        assert pa.losing_trades == 1

    def test_multiple_workers_flattened(self):
        """Multiple workers with multiple trades each should be flattened."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0)], 'trade_count': 2},
            {'revenue_history': [(0.02, 20.0)], 'trade_count': 1},
        ]
        pa = PerformanceAnalyzer(results)
        assert pa.all_trades == [(0.01, 10.0), (-0.005, -5.0), (0.02, 20.0)]
        assert pa.total_trades == 3
        assert pa.winning_trades == 2
        assert pa.losing_trades == 1

    def test_worker_with_zero_trades(self):
        """Worker with empty revenue_history."""
        results = [
            {'revenue_history': [], 'trade_count': 0},
            {'revenue_history': [(0.01, 10.0)], 'trade_count': 1},
        ]
        pa = PerformanceAnalyzer(results)
        assert pa.all_trades == [(0.01, 10.0)]
        assert pa.total_trades == 1
        assert pa.winning_trades == 1
        assert pa.losing_trades == 0


class TestAnalyze:
    """Test analyze() method."""

    def test_analyze_empty_results(self):
        """Empty results should return zero-valued metrics."""
        pa = PerformanceAnalyzer([])
        metrics = pa.analyze()
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_revenue_pct'] == 0.0
        assert metrics['total_revenue_abs'] == 0.0
        assert metrics['max_drawdown'] == 0.0
        assert metrics['max_consecutive_losses'] == 0
        assert metrics['profit_factor'] == 0.0

    def test_analyze_single_winning_trade(self):
        """Single winning trade."""
        results = [{'revenue_history': [(0.01, 10.0)], 'trade_count': 1}]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['total_trades'] == 1
        assert metrics['win_rate'] == 1.0
        assert metrics['avg_revenue_pct'] == 0.01
        assert metrics['total_revenue_abs'] == 10.0
        assert metrics['max_drawdown'] == 0.0  # No loss after peak
        assert metrics['max_consecutive_losses'] == 0
        assert metrics['profit_factor'] == float('inf')  # Win only

    def test_analyze_single_losing_trade(self):
        """Single losing trade."""
        results = [{'revenue_history': [(-0.01, -10.0)], 'trade_count': 1}]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['total_trades'] == 1
        assert metrics['win_rate'] == 0.0
        assert metrics['avg_revenue_pct'] == -0.01
        assert metrics['total_revenue_abs'] == -10.0
        # Peak starts at 0, first trade goes to -10, drawdown = 0 - (-10) = 10
        assert metrics['max_drawdown'] == pytest.approx(10.0)
        assert metrics['max_consecutive_losses'] == 1
        assert metrics['profit_factor'] == 0.0  # Loss only

    def test_analyze_win_rate(self):
        """Win rate = winning_trades / total_trades."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0), (0.02, 20.0)], 'trade_count': 3},
            {'revenue_history': [(0.015, 15.0)], 'trade_count': 1},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['total_trades'] == 4
        assert metrics['win_rate'] == pytest.approx(0.75)

    def test_analyze_avg_revenue_pct(self):
        """Average of all revenue_pct values."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.01, -10.0)], 'trade_count': 2},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['avg_revenue_pct'] == pytest.approx(0.0)

    def test_analyze_total_revenue_abs(self):
        """Sum of all revenue_abs values."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0), (0.02, 20.0)], 'trade_count': 3},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['total_revenue_abs'] == pytest.approx(25.0)

    def test_analyze_max_drawdown_simple(self):
        """Max drawdown: peak at 20, trough at 5, drawdown = 15."""
        results = [
            {'revenue_history': [(0.01, 10.0), (0.01, 20.0), (-0.015, -15.0)], 'trade_count': 3},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        # Cumulative: [10, 30, 15]. Peak: 30. Trough after peak: 15. Drawdown: 30 - 15 = 15.
        assert metrics['max_drawdown'] == pytest.approx(15.0)

    def test_analyze_max_drawdown_multiple_peaks(self):
        """Max drawdown tracks across multiple peaks."""
        results = [
            {'revenue_history': [(0.01, 10.0), (0.01, 20.0), (-0.02, -10.0), (0.01, 30.0), (-0.015, -15.0)], 'trade_count': 5},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        # Cumulative: [10, 20, 10, 40, 25]
        # After peak 20: min is 10, drawdown = 10
        # After peak 40: min is 25, drawdown = 15
        # Max drawdown = 15
        assert metrics['max_drawdown'] == pytest.approx(15.0)

    def test_analyze_max_drawdown_no_trades(self):
        """Max drawdown is 0 when no trades."""
        pa = PerformanceAnalyzer([])
        metrics = pa.analyze()
        assert metrics['max_drawdown'] == 0.0

    def test_analyze_max_consecutive_losses_none(self):
        """No consecutive losses when all trades win."""
        results = [
            {'revenue_history': [(0.01, 10.0), (0.02, 20.0), (0.01, 10.0)], 'trade_count': 3},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['max_consecutive_losses'] == 0

    def test_analyze_max_consecutive_losses_single_streak(self):
        """Longest streak of consecutive losses."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.01, -10.0), (-0.01, -10.0), (0.01, 10.0)], 'trade_count': 4},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['max_consecutive_losses'] == 2

    def test_analyze_max_consecutive_losses_multiple_streaks(self):
        """Should return longest streak."""
        results = [
            {'revenue_history': [
                (-0.01, -10.0),  # streak: 1
                (-0.01, -10.0),  # streak: 2
                (0.01, 10.0),    # reset
                (-0.01, -10.0),  # streak: 1
                (-0.01, -10.0),  # streak: 2
                (-0.01, -10.0),  # streak: 3
                (0.01, 10.0),    # reset
            ], 'trade_count': 7},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['max_consecutive_losses'] == 3

    def test_analyze_max_consecutive_losses_all_losses(self):
        """All trades are losses."""
        results = [
            {'revenue_history': [(-0.01, -10.0), (-0.01, -10.0), (-0.01, -10.0)], 'trade_count': 3},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['max_consecutive_losses'] == 3

    def test_analyze_profit_factor_inf(self):
        """Profit factor is inf when no losing trades but winning trades exist."""
        results = [
            {'revenue_history': [(0.01, 10.0), (0.02, 20.0)], 'trade_count': 2},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['profit_factor'] == float('inf')

    def test_analyze_profit_factor_zero(self):
        """Profit factor is 0 when no winning trades."""
        results = [
            {'revenue_history': [(-0.01, -10.0), (-0.02, -20.0)], 'trade_count': 2},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['profit_factor'] == 0.0

    def test_analyze_profit_factor_normal(self):
        """Profit factor = sum(winning_abs) / abs(sum(losing_abs))."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.01, -10.0), (0.02, 20.0), (-0.01, -5.0)], 'trade_count': 4},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        # Winning: 10 + 20 = 30
        # Losing: -10 + -5 = -15
        # Profit factor: 30 / 15 = 2.0
        assert metrics['profit_factor'] == pytest.approx(2.0)

    def test_analyze_profit_factor_with_zero_revenue_loss(self):
        """Trade with revenue_pct == 0 is treated as loss in profit factor."""
        results = [
            {'revenue_history': [(0.01, 10.0), (0.0, 0.0)], 'trade_count': 2},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        # Winning: 10
        # Losing: 0 (zero revenue_pct is loss, but abs(0) = 0, so abs(sum) = 0)
        assert metrics['profit_factor'] == float('inf')  # No actual losing abs value

    def test_analyze_returns_dict(self):
        """analyze() returns dict with all required keys."""
        pa = PerformanceAnalyzer([])
        metrics = pa.analyze()
        required_keys = {
            'total_trades', 'win_rate', 'avg_revenue_pct',
            'total_revenue_abs', 'max_drawdown', 'max_consecutive_losses', 'profit_factor'
        }
        assert set(metrics.keys()) == required_keys


class TestPrintReport:
    """Test print_report() method."""

    def test_print_report_runs(self, capsys):
        """print_report() should print to stdout."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0)], 'trade_count': 2},
        ]
        pa = PerformanceAnalyzer(results)
        pa.print_report()
        captured = capsys.readouterr()
        output = captured.out

        # Check that all required metrics appear in output (case-insensitive for values)
        assert '2' in output  # total_trades
        assert '50' in output or '0.5' in output  # win_rate (50% or 0.5)
        assert 'Revenue' in output  # avg_revenue_pct or total_revenue_abs
        assert 'Drawdown' in output  # max_drawdown
        assert 'Losses' in output  # max_consecutive_losses
        assert 'Profit Factor' in output

    def test_print_report_empty_results(self, capsys):
        """print_report() with empty results."""
        pa = PerformanceAnalyzer([])
        pa.print_report()
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should print something


class TestToDict:
    """Test to_dict() method."""

    def test_to_dict_contains_all_trades(self):
        """to_dict() should include all_trades."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0)], 'trade_count': 2},
        ]
        pa = PerformanceAnalyzer(results)
        data = pa.to_dict()
        assert data['all_trades'] == [(0.01, 10.0), (-0.005, -5.0)]

    def test_to_dict_contains_analyze_keys(self):
        """to_dict() should include all analyze() keys plus all_trades."""
        results = [
            {'revenue_history': [(0.01, 10.0)], 'trade_count': 1},
        ]
        pa = PerformanceAnalyzer(results)
        data = pa.to_dict()
        required_keys = {
            'all_trades', 'total_trades', 'win_rate', 'avg_revenue_pct',
            'total_revenue_abs', 'max_drawdown', 'max_consecutive_losses', 'profit_factor'
        }
        assert set(data.keys()) == required_keys

    def test_to_dict_values_match_analyze(self):
        """to_dict() values should match analyze() for overlapping keys."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0), (0.02, 20.0)], 'trade_count': 3},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        data = pa.to_dict()

        for key in metrics:
            assert data[key] == metrics[key], f"Mismatch in key {key}"


class TestIntegration:
    """Integration tests matching the verification command."""

    def test_verification_example(self):
        """Test case from the verification command."""
        results = [
            {'revenue_history': [(0.01, 10.0), (-0.005, -5.0), (0.02, 20.0)], 'trade_count': 3},
            {'revenue_history': [(0.015, 15.0)], 'trade_count': 1},
        ]
        pa = PerformanceAnalyzer(results)
        metrics = pa.analyze()
        assert metrics['total_trades'] == 4
        assert metrics['win_rate'] == pytest.approx(0.75)
