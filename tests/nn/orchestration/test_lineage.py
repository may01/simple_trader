from nn.orchestration.lineage import LineageDecision, decide


def test_first_version_promotes():
    d = decide(
        best_score=None,
        candidate_score=0.40,
        margin=0.01,
        strike_count=0,
        K=2,
        version_n=1,
        max_versions=6,
        elapsed_s=0.0,
        budget_s=28800.0,
    )
    assert isinstance(d, LineageDecision)
    assert d.action == "promote"
    assert d.new_best is True
    assert d.strike_count == 0
    assert d.stop is False


def test_real_improvement_promotes_and_resets_strikes():
    d = decide(
        best_score=0.40,
        candidate_score=0.42,
        margin=0.01,
        strike_count=1,          # there was a prior strike; promote must clear it
        K=2,
        version_n=3,
        max_versions=6,
        elapsed_s=1000.0,
        budget_s=28800.0,
    )
    assert d.action == "promote"
    assert d.new_best is True
    assert d.strike_count == 0   # reset on promote
    assert d.stop is False
    assert "promote" in d.reason


def test_exactly_margin_promotes():
    # candidate - best == margin exactly -> must promote because comparison is >=
    d = decide(
        best_score=0.40,
        candidate_score=0.41,    # 0.41 - 0.40 == 0.01 == margin
        margin=0.01,
        strike_count=0,
        K=2,
        version_n=2,
        max_versions=6,
        elapsed_s=0.0,
        budget_s=28800.0,
    )
    assert d.action == "promote"
    assert d.new_best is True
    assert d.strike_count == 0
    assert d.stop is False


def test_noise_below_margin_reverts_first_strike():
    d = decide(
        best_score=0.40,
        candidate_score=0.404,   # +0.004 < margin 0.01
        margin=0.01,
        strike_count=0,
        K=2,
        version_n=2,
        max_versions=6,
        elapsed_s=1000.0,
        budget_s=28800.0,
    )
    assert d.action == "revert"
    assert d.new_best is False
    assert d.strike_count == 1   # 0 -> 1
    assert d.stop is False        # one strike of two; lineage continues
    assert "strike 1/2" in d.reason


def test_second_strike_stops():
    d = decide(
        best_score=0.40,
        candidate_score=0.404,   # same sub-margin candidate
        margin=0.01,
        strike_count=1,          # already one strike on the board
        K=2,
        version_n=3,
        max_versions=6,
        elapsed_s=1000.0,
        budget_s=28800.0,
    )
    assert d.action == "revert"
    assert d.strike_count == 2   # 1 -> 2 == K
    assert d.stop is True
    assert "strike" in d.reason  # strikes are the dominant stop cause
    assert "2/2" in d.reason


def test_max_versions_backstop_stops_even_when_promoting():
    d = decide(
        best_score=0.40,
        candidate_score=0.50,    # a clear improvement -> promote
        margin=0.01,
        strike_count=0,
        K=2,
        version_n=6,             # already at the cap
        max_versions=6,
        elapsed_s=1000.0,
        budget_s=28800.0,
    )
    assert d.action == "promote"     # it still promoted the winner
    assert d.new_best is True
    assert d.strike_count == 0
    assert d.stop is True            # but the lineage ends here
    assert "max_versions" in d.reason


def test_budget_exceeded_stops_regardless_of_action():
    d = decide(
        best_score=0.40,
        candidate_score=0.42,    # an improvement -> promote
        margin=0.01,
        strike_count=0,
        K=2,
        version_n=3,             # below the version cap
        max_versions=6,
        elapsed_s=28800.0,       # elapsed == budget -> exceeded (>=)
        budget_s=28800.0,
    )
    assert d.action == "promote"
    assert d.stop is True
    assert "budget" in d.reason
    assert "28800" in d.reason
