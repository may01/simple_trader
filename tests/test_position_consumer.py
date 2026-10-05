"""Decoder tests for mq/position_consumer.py.

Every test here runs against the **golden fixtures** copied from
`trade_executor/crates/mq_gateway/tests/fixtures/`, which that crate
generates and asserts byte-stable. That is the point: this consumer is
Python in another repository and cannot compile against the encoder, so
a field renamed there has to surface as a failing test here rather than
as a silently dropped value in production.
"""

import json
import logging
from pathlib import Path

import pytest

from mq.position_consumer import (
    KNOWN_EVENTS,
    SUPPORTED_SCHEMA,
    PositionEvent,
    parse_position_event,
)

FIXTURES = Path(__file__).parent / "fixtures" / "wire_v2"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


# ---------------------------------------------------------------------
# One test per golden fixture — the cross-repo contract
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", fixture_names())
def test_every_golden_fixture_decodes(name):
    event = parse_position_event(load(name))
    assert event is not None, f"{name}.json must decode"
    assert isinstance(event, PositionEvent)
    assert event.schema == SUPPORTED_SCHEMA
    assert event.event == name
    assert event.pair == "LINKUSDT"


def test_fixtures_cover_every_event_kind_the_executor_can_send():
    """A fixture set that stops covering a kind is worse than none: this
    consumer would keep passing against a stale contract."""
    assert set(fixture_names()) == set(KNOWN_EVENTS)


def test_opened_carries_the_aggregate_main_actually_uses():
    event = parse_position_event(load("opened"))
    assert event.status == "open"
    assert event.side == "long"
    # Purpose 2 of main/'s position tracking: the price actually paid.
    assert event.avg_entry_price == pytest.approx(15.0150)
    assert event.intended_open_price == pytest.approx(15.00)
    assert event.net_size == pytest.approx(0.8)
    assert event.target_size == pytest.approx(1.2)
    assert event.stop_loss_price == pytest.approx(14.85)
    assert event.take_profit_price == pytest.approx(15.60)
    assert event.settlement_complete is False


def test_slippage_is_reconstructible_from_one_message():
    """avg_entry_price against intended_open_price -- main/ can compute
    what the fill actually cost without asking for anything else."""
    event = parse_position_event(load("opened"))
    slippage = event.avg_entry_price - event.intended_open_price
    assert slippage == pytest.approx(0.015)


def test_not_placed_carries_its_reason():
    """The v1 regression this closes: the rejection reason was computed,
    persisted, and then thrown away at the wire boundary, leaving main/
    unable to learn its decision was refused."""
    event = parse_position_event(load("not_placed"))
    assert event.event == "not_placed"
    assert event.reason == "computed size exceeds account limits"
    assert event.is_flat


def test_closed_carries_its_close_reason():
    event = parse_position_event(load("closed"))
    assert event.reason == "target"
    assert event.settlement_complete is True


def test_no_fixture_contains_fill_detail():
    """main/ does not track fills and the wire does not carry them
    (design §6.4). If this fails, the contract grew a trade ledger."""
    for name in fixture_names():
        raw = (FIXTURES / f"{name}.json").read_text()
        for forbidden in ('"fills"', '"entries"', '"exits"', '"trade_id"'):
            assert forbidden not in raw, f"{name}.json carries {forbidden}"


# ---------------------------------------------------------------------
# Skipping, never raising
# ---------------------------------------------------------------------

def test_a_newer_schema_is_skipped_not_guessed_at(caplog):
    raw = load("opened")
    raw["schema"] = 3
    with caplog.at_level(logging.WARNING):
        assert parse_position_event(raw) is None
    assert any("schema" in r.getMessage() for r in caplog.records)


def test_an_unknown_event_kind_is_skipped(caplog):
    raw = load("opened")
    raw["event"] = "teleported"
    with caplog.at_level(logging.WARNING):
        assert parse_position_event(raw) is None


def test_unknown_kinds_are_logged_once_not_once_per_message(caplog):
    """An executor running ahead of this consumer must produce one line,
    not one per message on every tick."""
    import mq.position_consumer as pc

    pc._warned_unknown.clear()
    raw = load("opened")
    raw["event"] = "brand_new_kind"
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            parse_position_event(raw)
    matching = [r for r in caplog.records if "brand_new_kind" in r.getMessage()]
    assert len(matching) == 1


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("pair"),
    lambda r: r.pop("position_state"),
    lambda r: r["position_state"].pop("status"),
    lambda r: r.update({"position_state": "not-an-object"}),
])
def test_a_malformed_message_is_skipped_not_raised(mutate):
    raw = load("opened")
    mutate(raw)
    assert parse_position_event(raw) is None


def test_a_non_object_is_skipped():
    assert parse_position_event([]) is None
    assert parse_position_event("nope") is None


# ---------------------------------------------------------------------
# Absent vs zero
# ---------------------------------------------------------------------

def test_absent_average_entry_price_decodes_as_none_not_zero():
    """A zero average entry price is indistinguishable from a real one
    and would poison every risk calculation downstream."""
    raw = load("opened")
    del raw["position_state"]["avg_entry_price"]
    event = parse_position_event(raw)
    assert event.avg_entry_price is None


def test_a_flat_state_has_no_prices_rather_than_zeroes():
    event = parse_position_event(load("already_closed"))
    assert event.is_flat
    assert event.avg_entry_price is None
    assert event.net_size is None
    assert event.side is None


def test_decimals_arrive_as_strings_and_keep_their_precision():
    """The executor serialises Decimal as a quoted string precisely so a
    price does not round-trip through a float on the way here."""
    raw = load("opened")
    assert isinstance(raw["position_state"]["avg_entry_price"], str)
    event = parse_position_event(raw)
    assert event.avg_entry_price == pytest.approx(15.0150, abs=1e-9)
