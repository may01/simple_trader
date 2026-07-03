from dataclasses import dataclass


@dataclass
class LineageDecision:
    action: str        # "promote" | "revert"
    new_best: bool
    strike_count: int  # updated count AFTER this version
    stop: bool
    reason: str


def decide(
    *,
    best_score: float | None,
    candidate_score: float,
    margin: float,
    strike_count: int,
    K: int,
    version_n: int,
    max_versions: int,
    elapsed_s: float,
    budget_s: float,
) -> LineageDecision:
    # --- promote vs revert ------------------------------------------------
    if best_score is None:
        is_improvement = True
        delta = None
    else:
        delta = candidate_score - best_score
        # Use epsilon to handle IEEE-754 float noise at the boundary (>= margin)
        is_improvement = delta >= margin - 1e-9

    if is_improvement:
        action = "promote"
        new_best = True
        new_strike = 0
    else:
        action = "revert"
        new_best = False
        new_strike = strike_count + 1

    # --- stop conditions (plain OR) ---------------------------------------
    stop_strikes = new_strike >= K
    stop_versions = version_n >= max_versions
    stop_budget = elapsed_s >= budget_s
    stop = stop_strikes or stop_versions or stop_budget

    # --- reason: dominant cause, priority strikes > versions > budget -----
    if stop_strikes:
        reason = f"stop: {new_strike}/{K} strikes"
    elif stop_versions:
        reason = f"stop: max_versions {max_versions} reached"
    elif stop_budget:
        reason = f"stop: budget {int(budget_s)}s exceeded"
    elif action == "promote":
        if delta is None:
            reason = "promote: first version"
        else:
            reason = f"promote: {delta:+.4f} ≥ margin {margin:.4f}"
    else:
        reason = f"revert: {delta:+.4f} < margin (strike {new_strike}/{K})"

    return LineageDecision(
        action=action,
        new_best=new_best,
        strike_count=new_strike,
        stop=stop,
        reason=reason,
    )
