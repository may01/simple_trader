import pytest
from simple_trader.signals.base_signal import BaseSignal
from simple_trader.data.data_point import LiveDataPoint
from tests.signals.conftest import _make_live_point


def test_base_signal_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseSignal("rsi_14", tf=1)  # type: ignore[abstract]


def test_subclass_without_check_raises():
    class Bad(BaseSignal):
        pass
    with pytest.raises(TypeError):
        Bad("rsi_14", tf=1)


def test_base_signal_get_data_reads_data_point():
    class Probe(BaseSignal):
        def check(self, data_point: object) -> bool:
            return self.get_data(data_point) > 0  # type: ignore[arg-type]

    probe = Probe("rsi_14", tf=1)
    point = _make_live_point({1: {"rsi_14": [50.0]}})
    assert probe.get_data(point) == 50.0


def test_base_signal_set_shift_updates_shift():
    class Probe(BaseSignal):
        def check(self, data_point: object) -> bool:
            return True

    probe = Probe("rsi_14", tf=1, shift=0)
    probe.set_shift(2)
    assert probe.shift == 2


def test_base_signal_reset_is_noop_by_default():
    class Probe(BaseSignal):
        def check(self, data_point: object) -> bool:
            return True

    probe = Probe("rsi_14", tf=1)
    probe.reset()  # should not raise
