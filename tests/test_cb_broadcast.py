"""Phase 18 task 10: MQ broadcast of the candle-bound columns.

on_change entries whose value differs from the last published one are
published alone, immediately; the 30 s heartbeat still publishes the whole
allowlist on its own cadence (a flip never resets it).
"""
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

from config_loader import SharedIndicatorConfig, load_shared_indicators_config
from tests.test_robot import FakeIndicatorPublisher, _fake_clock, make_robot_with_publisher

SHIPPED = Path(__file__).resolve().parent.parent / "configs" / "shared_indicators_config.yaml"
TFS = [15, 60, 240]
PLAIN = ["cb_high", "cb_low", "cb_zone_long", "cb_zone_short", "cbnc_high", "cbnc_low"]
ONCHG = ["cb_inzone_long", "cb_inzone_short", "extdone_high", "extdone_low",
         "cb_f_extdone_long", "cb_f_extdone_short", "cb_f_trend_long", "cb_f_trend_short",
         "cb_f_overshoot_long", "cb_f_overshoot_short"]


def test_loader_reads_on_change(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        "indicators:\n"
        "  - name: a\n    timeframes: [15]\n    on_change: true\n"
        "  - name: b\n    timeframes: [15]\n"
    )
    r = load_shared_indicators_config(path=str(f))
    assert [c.on_change for c in r] == [True, False]


def test_loader_default_file_contains_cb_entries():
    by_name = {c.name: c for c in load_shared_indicators_config(path=str(SHIPPED))}
    for n in PLAIN:
        assert by_name[n].timeframes == TFS and by_name[n].on_change is False
    for n in ONCHG:
        assert by_name[n].timeframes == TFS and by_name[n].on_change is True
    for n in ("cb_f_mcx_long", "cbx_inzone_long", "cbnc_inzone_long",
              "cb_overshoot_long", "cb_overshoot_short"):
        assert n not in by_name


def test_allowlist_cb_name_count():
    cb = [c for c in load_shared_indicators_config(path=str(SHIPPED))
          if c.name.startswith(("cb_", "cbnc_", "extdone_"))]
    assert len(cb) == 16 == len(PLAIN) + len(ONCHG)
    assert sum(len(c.timeframes) for c in cb) == 48


def _setup(values, names=("cb_high", "cb_inzone_long"), interval=30.0):
    """values: dict name -> float|Exception, mutated by the test."""
    shared = [SharedIndicatorConfig(name=n, timeframes=[15], on_change=n in ONCHG) for n in names]

    def fake_get(col, tf, shift=0):
        v = values[col]
        if isinstance(v, Exception):
            raise v
        return v

    pub = FakeIndicatorPublisher()
    robot, dp, sm = make_robot_with_publisher(pub, shared, fake_get, publish_interval_sec=interval)
    fake_time, clock = _fake_clock()

    def tick(dt=1.0):
        clock["t"] += dt
        with patch("robots.robot.time", fake_time):
            robot._publish_shared_indicators(dp)

    return robot, pub, tick, fake_time, clock, dp


def _first(robot, dp, fake_time):
    with patch("robots.robot.time", fake_time):
        robot._publish_shared_indicators(dp)


def test_interval_still_throttles_plain_entries():
    v = {"cb_high": 1.0, "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    n = len(pub.published)
    v["cb_high"] = 2.0
    tick(1.0)
    assert len(pub.published) == n


def test_on_change_bypasses_interval():
    v = {"cb_high": 1.0, "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    n = len(pub.published)
    v["cb_inzone_long"] = 1.0
    tick(1.0)
    batch = pub.published[n:]
    assert batch == [("BTCUSDT", "15_cb_inzone_long", 1.0)]


def test_flip_publish_does_not_reset_heartbeat():
    v = {"cb_high": 1.0, "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)                      # t = 0: full heartbeat
    v["cb_inzone_long"] = 1.0
    tick(1.0)                                  # t = 1: flip -> one reading
    n = len(pub.published)
    tick(28.0)                                 # t = 29: nothing
    assert len(pub.published) == n
    tick(1.0)                                  # t = 30: heartbeat, full list
    assert pub.published[n:] == [("BTCUSDT", "15_cb_high", 1.0),
                                 ("BTCUSDT", "15_cb_inzone_long", 1.0)]


def test_flapping_value_publishes_one_reading_per_tick():
    v = {"cb_high": 1.0, "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    assert len(pub.published) == 2
    for i in range(10):
        v["cb_inzone_long"] = float((i + 1) % 2)
        tick(1.0)
    flips = pub.published[2:]
    assert len(flips) == 10
    assert flips == [("BTCUSDT", "15_cb_inzone_long", float((i + 1) % 2)) for i in range(10)]


def test_no_change_no_extra_publish():
    v = {"cb_high": 1.0, "cb_inzone_long": 1.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    n = len(pub.published)
    for _ in range(5):
        tick(1.0)
    assert len(pub.published) == n


def test_change_back_publishes_again():
    v = {"cb_high": 1.0, "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    v["cb_inzone_long"] = 1.0
    tick()
    v["cb_inzone_long"] = 0.0
    tick()
    flags = [val for _, name, val in pub.published if name == "15_cb_inzone_long"]
    assert flags == [0.0, 1.0, 0.0]


def test_nan_is_not_a_change():
    v = {"cb_high": 1.0, "cb_inzone_long": 1.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    n = len(pub.published)
    v["cb_inzone_long"] = float("nan")
    tick()
    assert len(pub.published) == n
    v["cb_inzone_long"] = 1.0
    tick()
    assert len(pub.published) == n


def test_nan_then_different_finite_publishes():
    v = {"cb_high": 1.0, "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    n = len(pub.published)
    v["cb_inzone_long"] = float("nan")
    tick()
    assert len(pub.published) == n
    v["cb_inzone_long"] = 1.0
    tick()
    assert pub.published[n:] == [("BTCUSDT", "15_cb_inzone_long", 1.0)]


def test_unreadable_on_change_entry_does_not_break_tick():
    v = {"cb_high": 1.0, "cb_inzone_long": KeyError("15_cb_inzone_long")}
    robot, pub, tick, ft, clock, dp = _setup(v)
    _first(robot, dp, ft)  # must not raise
    assert ("BTCUSDT", "15_cb_high", 1.0) in pub.published
    assert not any(name == "15_cb_inzone_long" for _, name, _ in pub.published)
    n = len(pub.published)
    tick(1.0)
    assert len(pub.published) == n
    tick(31.0)
    assert len(pub.published) == n + 1


def test_keyerror_read_logs_warning_once_without_traceback(caplog):
    import logging
    v = {"cb_high": 1.0, "cb_inzone_long": KeyError("15_cb_inzone_long")}
    robot, pub, tick, ft, _, dp = _setup(v)
    with caplog.at_level(logging.WARNING, logger="robots.robot"):
        _first(robot, dp, ft)
        for _ in range(5):
            tick(31.0)
    recs = [r for r in caplog.records if r.name == "robots.robot"]
    assert len(recs) == 1
    assert recs[0].levelno == logging.WARNING and recs[0].exc_info is None
    assert "cb_inzone_long" in recs[0].getMessage()


def test_other_read_error_keeps_traceback(caplog):
    import logging
    v = {"cb_high": RuntimeError("boom"), "cb_inzone_long": 0.0}
    robot, pub, tick, ft, _, dp = _setup(v)
    with caplog.at_level(logging.WARNING, logger="robots.robot"):
        _first(robot, dp, ft)
        tick(31.0)
    recs = [r for r in caplog.records if r.name == "robots.robot"]
    assert len(recs) == 1 and recs[0].exc_info is not None


def test_overshoot_flag_flip_publishes_inside_interval():
    v = {"cb_high": 1.0, "cb_f_overshoot_long": 1.0}
    robot, pub, tick, ft, _, dp = _setup(v, names=("cb_high", "cb_f_overshoot_long"))
    _first(robot, dp, ft)
    assert [(n, x) for _, n, x in pub.published if n == "15_cb_f_overshoot_long"] == [("15_cb_f_overshoot_long", 1.0)]
    n = len(pub.published)
    v["cb_f_overshoot_long"] = 0.0
    tick(1.0)                                   # inside the 30 s interval
    assert pub.published[n:] == [(pub.published[0][0], "15_cb_f_overshoot_long", 0.0)]


def test_bools_published_as_floats():
    v = {"cb_high": 1.0, "cb_inzone_long": True}
    robot, pub, tick, ft, _, dp = _setup(v)
    _first(robot, dp, ft)
    val = [x for _, name, x in pub.published if name == "15_cb_inzone_long"][0]
    assert val == 1.0 and type(val) is float
