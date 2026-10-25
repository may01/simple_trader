# Copied 2026-10-25 from worktrees/swing-sinusoid-experiment/experiments/swing_sinusoid/cb_filter.py (branch swing-sinusoid-experiment, uncommitted) — provenance for the cb_class_bounds experiment; unchanged.
"""Combined-zone filter for CB zone position OPENING (user rule 2026-10-24, confirmed 2026-10-25).

Inputs at the opening minute: the combined zone of each tf,
    z = cci_ma8 zone (+1 buy / −1 sell / 0) + 0_rsi (RSI move class with ±3 → 0),  z ∈ {−3 … +3}
and the candidate entry: CB zone tf ∈ {15, 60, 240}, side ∈ {"long", "short"}.

Three levels; an entry is allowed only if every applicable level allows it (AND), with one
exception: when z240 == 0, or z240 and z60 are both in {−1, +1} and cancel (z240 + z60 == 0),
the 60 rule REPLACES the 240 rule. Confirmed readings: +2 / +3 disable shorts (mirror of
−2 / −3 disabling longs); a zone of −1 / +1 allows only the entry in its own direction for
entries of its own tf; the 60 rule's "15 long if … ≥ +1" refers to the 15 zone; the 15 rule's
"higher tf" is z60, or z240 when z60 == 0.
"""
from __future__ import annotations

LONG, SHORT = "long", "short"


def _rule_240(tf: int, side: str, z240: int, z60: int) -> bool:
    if z240 <= -2:
        return side == SHORT                       # longs disabled on 15 / 60 / 240
    if z240 == -1:
        if side == SHORT:
            return True
        if tf == 15:
            return False                           # 15 long disabled
        if tf == 60:
            return z60 > 1                         # 60 long only if z60 ≥ +2
        return False                               # 240 long: zone −1 allows shorts only
    if z240 == 0:
        return True
    if z240 == 1:
        if side == LONG:
            return True
        if tf == 15:
            return False                           # 15 short disabled
        if tf == 60:
            return z60 < -1                        # 60 short only if z60 ≤ −2
        return False                               # 240 short: zone +1 allows longs only
    return side == LONG                            # z240 ≥ +2: shorts disabled


def _rule_60(tf: int, side: str, z60: int, z15: int) -> bool:
    if tf == 240:
        return True                                # the 60 rule only addresses 15 / 60 entries
    if z60 <= -2:
        return side == SHORT
    if z60 == -1:
        if side == SHORT:
            return True
        if tf == 15:
            return z15 >= 1                        # 15 long only if the 15 zone ≥ +1
        return False                               # 60 long: zone −1 allows shorts only
    if z60 == 0:
        return True
    if z60 == 1:
        if side == LONG:
            return True
        if tf == 15:
            return z15 <= -1                       # 15 short only if the 15 zone ≤ −1
        return False                               # 60 short: zone +1 allows longs only
    return side == LONG


def _rule_15(side: str, z15: int, higher: int) -> bool:
    if z15 <= -2:
        return side == SHORT
    if z15 == -1:
        return True if side == SHORT else higher > 1     # 15 long only if the higher tf ≥ +2
    if z15 == 0:
        return True
    if z15 == 1:
        return True if side == LONG else higher < -1     # 15 short only if the higher tf ≤ −2
    return side == LONG


def allowed(tf: int, side: str, z15: int, z60: int, z240: int) -> bool:
    """True if the CB zone entry (tf, side) may open given the three combined zones."""
    assert tf in (15, 60, 240) and side in (LONG, SHORT)
    neutralised = z240 == 0 or (abs(z240) == 1 and abs(z60) == 1 and z240 + z60 == 0)
    ok = True
    if not neutralised:
        ok &= _rule_240(tf, side, z240, z60)
    ok &= _rule_60(tf, side, z60, z15)             # governs alone when neutralised, AND otherwise
    if tf == 15:
        higher = z60 if z60 != 0 else z240
        ok &= _rule_15(side, z15, higher)
    return bool(ok)


def table():
    """Full decision table as a list of dicts (for the spec / tests)."""
    return [{"z240": z240, "z60": z60, "z15": z15,
             **{f"{tf}_{s}": allowed(tf, s, z15, z60, z240) for tf in (15, 60, 240) for s in (LONG, SHORT)}}
            for z240 in range(-3, 4) for z60 in range(-3, 4) for z15 in range(-3, 4)]
