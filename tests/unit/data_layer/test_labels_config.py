"""Unit tests for config_loader.load_labels_config — labels: section parsing."""

import pytest

from config_loader import load_labels_config


def _write_config(tmp_path, labels_yaml: str) -> str:
    path = tmp_path / "indicators_config.yaml"
    path.write_text("fields: []\n" + labels_yaml)
    return str(path)


def test_load_labels_profit_entry(tmp_path):
    path = _write_config(tmp_path, """
labels:
  - type: profit
    tfs: [15]
    n: 1
    m: 1
    x: 0.4
""")
    specs = load_labels_config(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.type == "profit"
    assert s.tfs == [15]
    assert (s.n, s.m, s.x) == (1, 1.0, 0.4)
    assert s.atr_period == 14  # default
    assert s.l is None and s.y is None


def test_load_labels_strict_entry(tmp_path):
    path = _write_config(tmp_path, """
labels:
  - type: profit_strict
    tfs: [60, 240]
    n: 2
    m: 1
    x: 0.3
    l: 15
    y: 0.3
    atr_period: 7
""")
    specs = load_labels_config(path)
    assert len(specs) == 1
    s = specs[0]
    assert s.type == "profit_strict"
    assert s.tfs == [60, 240]
    assert (s.l, s.y) == (15, 0.3)
    assert s.atr_period == 7


def test_load_labels_missing_section_returns_empty(tmp_path):
    path = _write_config(tmp_path, "")
    assert load_labels_config(path) == []


def test_load_labels_strict_without_l_y_raises(tmp_path):
    path = _write_config(tmp_path, """
labels:
  - type: profit_strict
    tfs: [15]
    n: 1
    m: 1
    x: 0.4
""")
    with pytest.raises(ValueError, match="l.*y|y.*l"):
        load_labels_config(path)


def test_load_labels_unknown_type_raises(tmp_path):
    path = _write_config(tmp_path, """
labels:
  - type: bogus
    tfs: [15]
    n: 1
    m: 1
    x: 0.4
""")
    with pytest.raises(ValueError, match="bogus"):
        load_labels_config(path)


def test_real_config_ships_approved_label_set():
    """indicators_config.yaml carries the approved 14-spec set (6 profit +
    8 strict over tfs 15/60/240 for profit, 5/15/60/240 for strict; n in {1,2})."""
    specs = load_labels_config("configs/indicators_config.yaml")
    assert len(specs) == 14
    plain = [s for s in specs if s.type == "profit"]
    strict = [s for s in specs if s.type == "profit_strict"]
    assert len(plain) == 6 and len(strict) == 8

    # Verify plain profit entries: 6 entries over TF15/60/240
    plain_combos = {(s.tfs[0], s.n) for s in plain}
    assert plain_combos == {(15, 1), (15, 2), (60, 1), (60, 2), (240, 1), (240, 2)}
    expected_x_profit = {15: 0.3, 60: 0.2, 240: 0.1}
    for s in plain:
        assert s.m == 1
        assert s.x == expected_x_profit[s.tfs[0]]

    # Verify strict entries: 8 entries over TF5/15/60/240
    strict_combos = {(s.tfs[0], s.n) for s in strict}
    assert strict_combos == {(5, 1), (5, 2), (15, 1), (15, 2), (60, 1), (60, 2), (240, 1), (240, 2)}
    expected_x_strict = {5: 0.3, 15: 0.3, 60: 0.2, 240: 0.1}
    expected_y_strict = {5: 0.2, 15: 0.2, 60: 0.1, 240: 0.1}
    for s in strict:
        assert s.m == 1
        assert s.l == 15
        assert s.x == expected_x_strict[s.tfs[0]]
        assert s.y == expected_y_strict[s.tfs[0]]


def test_labels_never_registered_as_fields():
    """Contract: label columns are not indicator fields — neither in the
    registry nor in the fields: section."""
    from indicators import _FIELD_REGISTRY
    from config_loader import load_indicators_config

    assert not any("plong" in k or "pshort" in k for k in _FIELD_REGISTRY)
    names = [f.name for f in load_indicators_config()]
    assert not any("plong" in n or "pshort" in n for n in names)


def test_tf5_profit_strict_entries_present():
    specs = load_labels_config("configs/indicators_config.yaml")
    tf5_strict = [s for s in specs if s.type == "profit_strict" and s.tfs == [5]]
    assert len(tf5_strict) == 2, "expected exactly two profit_strict tfs:[5] entries (n1, n2)"
    by_n = {s.n: s for s in tf5_strict}
    assert set(by_n) == {1, 2}
    for s in tf5_strict:
        assert s.m == 1.0 and s.x == 0.3 and s.l == 15 and s.y == 0.2


def test_no_nonstrict_profit_tf5():
    specs = load_labels_config("configs/indicators_config.yaml")
    assert not [s for s in specs if s.type == "profit" and s.tfs == [5]], \
        "strict only — no non-strict profit tf5 entries"
