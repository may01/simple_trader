from experiments.swing_sinusoid.cb_filter import allowed, table


def test_240_strong_rows():
    for tf in (15, 60, 240):
        assert not allowed(tf, "long", 0, 0, -3) and allowed(tf, "short", 0, 0, -3)
        assert not allowed(tf, "long", 2, 2, -2)
        assert allowed(tf, "long", 0, 0, 3) and not allowed(tf, "short", 0, 0, 3)   # +2 / +3 disable shorts
        assert not allowed(tf, "short", -2, -2, 2)


def test_240_minus1_rows():
    assert not allowed(15, "long", 0, 0, -1)
    assert allowed(60, "long", 0, 2, -1) and not allowed(60, "long", 0, 0, -1)
    assert allowed(60, "long", 0, 1, -1)               # z240 −1 and z60 +1 cancel → the 60 rule governs
    assert not allowed(240, "long", 0, 0, -1)          # zone −1 → shorts only for its own tf
    assert all(allowed(tf, "short", 0, 0, -1) for tf in (15, 60, 240))


def test_240_plus1_rows():
    assert not allowed(15, "short", 0, 0, 1)
    assert allowed(60, "short", 0, -2, 1) and not allowed(60, "short", 0, 0, 1)
    assert allowed(60, "short", 0, -1, 1)              # cancel → the 60 rule governs
    assert not allowed(240, "short", 0, 0, 1)
    assert all(allowed(tf, "long", 0, 0, 1) for tf in (15, 60, 240))


def test_60_rule_when_240_neutral():
    assert not allowed(15, "long", 0, -2, 0) and not allowed(60, "long", 0, -2, 0)
    assert allowed(240, "long", 0, -2, 0)              # the 60 rule does not address 240 entries
    assert allowed(15, "long", 1, -1, 0) and not allowed(15, "long", 0, -1, 0)   # 15 zone ≥ +1
    assert not allowed(60, "long", 0, -1, 0) and allowed(60, "short", 0, -1, 0)  # own tf: direction only
    assert allowed(15, "short", -1, 1, 0) and not allowed(15, "short", 0, 1, 0)
    assert not allowed(60, "short", 0, 1, 0) and allowed(60, "long", 0, 1, 0)
    assert allowed(15, "long", 0, 0, 0) and allowed(15, "short", 0, 0, 0)
    assert allowed(60, "long", 0, 3, 0) and not allowed(60, "short", 0, 3, 0)


def test_60_rule_replaces_240_when_they_cancel():
    assert allowed(15, "long", 1, 1, -1)               # 240 alone would disable the 15 long
    assert allowed(15, "short", -1, -1, 1)


def test_15_rule():
    assert not allowed(15, "long", -2, 0, 0) and allowed(15, "short", -2, 0, 0)
    assert allowed(15, "short", -1, 0, 0) and not allowed(15, "long", -1, 0, 0)
    assert allowed(15, "long", -1, 2, 0)               # higher tf (60) ≥ +2
    assert allowed(15, "long", -1, 0, 2)               # 60 neutral → 240 is the higher tf
    assert allowed(15, "long", 1, 0, 0) and not allowed(15, "short", 1, 0, 0)
    assert allowed(15, "short", 1, -2, 0)
    assert allowed(15, "long", 3, 0, 0) and not allowed(15, "short", 3, 0, 0)


def test_table_size():
    assert len(table()) == 7 ** 3
