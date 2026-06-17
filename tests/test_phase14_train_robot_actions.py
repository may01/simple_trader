"""Phase 14 Task 04 — TrainRobot Action assembly."""

from robots.train_robot import TrainRobot
from constants import (
    STRATEGY_ACTION_OPEN_LONG,
    STRATEGY_ACTION_CLOSE_LONG,
    STRATEGY_ACTION_NOTHING,
)


class _FakeTS:
    def timestamp(self):
        return 1000.0


class _DP:
    timestamp = _FakeTS()

    def get(self, *a, **k):
        return 20.0

    def cur_price(self, _):
        return 20.0


class _ScriptedSM:
    """Returns a scripted action per step and exposes last_fired."""

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.last_fired = []

    def check(self, dp, state, t, action_msg):
        act, of, cf, sp, tf = self.script[self.i]
        self.i += 1
        self.last_fired = [act] if act != STRATEGY_ACTION_NOTHING else []
        return (act, of, cf, sp, tf)


def _run_open_close():
    sm = _ScriptedSM([
        (STRATEGY_ACTION_OPEN_LONG, [20.0], [20.16], 19.5, 5),
        (STRATEGY_ACTION_CLOSE_LONG, [], [20.16], 19.5, 5),
    ])
    r = TrainRobot(sm, 0.001)
    r.set_sim_context(42)
    r.position.full_position = 1000.0
    r.step(_DP())
    r.step(_DP())
    return r


def test_records_signal_open_close():
    r = _run_open_close()
    events = [a.event for a in r.get_action_log().actions]
    assert "SIGNAL_FIRED" in events
    assert "OPEN" in events
    assert "CLOSE" in events


def test_sim_id_stamped():
    r = _run_open_close()
    assert r.get_action_log().sim_id == 42
    assert all(a.sim_id == 42 for a in r.get_action_log().actions)


def test_open_carries_target_price_and_close_revenue():
    r = _run_open_close()
    opens = [a for a in r.get_action_log().actions if a.event == "OPEN"]
    closes = [a for a in r.get_action_log().actions if a.event == "CLOSE"]
    # OPEN: executed=entry (20.0), target=take-profit (price_close[0]=20.16).
    assert opens[0].executed_price == 20.0
    assert opens[0].target_price == 20.16
    assert closes[0].revenue_pct != 0.0
    assert r.trade_count == 1


def test_no_phantom_close_when_flat():
    sm = _ScriptedSM([(STRATEGY_ACTION_CLOSE_LONG, [], [20.0], 0.0, 5)])
    r = TrainRobot(sm, 0.001)
    r.set_sim_context(1)
    r.step(_DP())
    assert r.trade_count == 0
    assert not any(a.event == "CLOSE" for a in r.get_action_log().actions)
