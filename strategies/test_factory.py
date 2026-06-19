"""TestStrategyFactory — picklable factory for the Phase 14 test strategies.

Registers all four test strategies (EMA-cross + RSI-zone, long + short) into a
fresh StrategyManager. Top-level class so it crosses the ProcessPoolExecutor
boundary (a closure cannot be pickled).
"""


class TestStrategyFactory:
    """Zero-arg-callable factory producing a StrategyManager with the 4 test strategies.

    Args:
        fee: Trading fee fraction forwarded to every strategy and the manager.
    """

    def __init__(self, fee: float) -> None:
        self.fee: float = fee

    def __call__(self):
        from strategies.strategy_manager import StrategyManager
        from strategies.strategy_test_1_long import StrategyTest1Long
        from strategies.strategy_test_1_short import StrategyTest1Short
        from strategies.strategy_test_2_long import StrategyTest2Long
        from strategies.strategy_test_2_short import StrategyTest2Short

        sm = StrategyManager(self.fee)
        sm.register(StrategyTest1Long(self.fee))
        sm.register(StrategyTest1Short(self.fee))
        sm.register(StrategyTest2Long(self.fee))
        sm.register(StrategyTest2Short(self.fee))
        return sm


class EmaStrategyFactory:
    """Picklable factory registering only the EMA-cross test strategies (long + short)."""

    def __init__(self, fee: float) -> None:
        self.fee: float = fee

    def __call__(self):
        from strategies.strategy_manager import StrategyManager
        from strategies.strategy_test_1_long import StrategyTest1Long
        from strategies.strategy_test_1_short import StrategyTest1Short

        sm = StrategyManager(self.fee)
        sm.register(StrategyTest1Long(self.fee))
        sm.register(StrategyTest1Short(self.fee))
        return sm
