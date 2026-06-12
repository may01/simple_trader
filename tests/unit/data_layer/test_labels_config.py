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
    """indicators_config.yaml carries the approved 12-spec set (6 profit +
    6 strict over tfs 15/60/240, n in {1,2})."""
    specs = load_labels_config("configs/indicators_config.yaml")
    assert len(specs) == 12
    plain = [s for s in specs if s.type == "profit"]
    strict = [s for s in specs if s.type == "profit_strict"]
    assert len(plain) == 6 and len(strict) == 6

    expected_x = {15: 0.4, 60: 0.3, 240: 0.2}
    for group in (plain, strict):
        combos = {(s.tfs[0], s.n) for s in group}
        assert combos == {(15, 1), (15, 2), (60, 1), (60, 2), (240, 1), (240, 2)}
        for s in group:
            assert s.m == 1
            assert s.x == expected_x[s.tfs[0]]
    for s in strict:
        assert s.l == 15
        assert s.y == expected_x[s.tfs[0]]


def test_labels_never_registered_as_fields():
    """Contract: label columns are not indicator fields — neither in the
    registry nor in the fields: section."""
    from indicators import _FIELD_REGISTRY
    from config_loader import load_indicators_config

    assert not any("plong" in k or "pshort" in k for k in _FIELD_REGISTRY)
    names = [f.name for f in load_indicators_config()]
    assert not any("plong" in n or "pshort" in n for n in names)
