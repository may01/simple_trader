"""Frozen constants of the candle-bounds (CB) stage.

Values mirror ``notebooks/candle_bounds_nc/cbnc.py`` and the 2y extdone ladder
fit (``fit_extdone_ladder.py``); they are the contract of the stored models.
"""

TFS: tuple[int, ...] = (15, 60, 240)
SIDES: tuple[str, ...] = ("high", "low")
DIRECTIONS: tuple[str, ...] = ("long", "short")
FEATURES: list[str] = [
    "R_position", "R_slope", "R_distance",
    "M_position", "M_slope", "M_distance",
    "A_position", "A_slope",
    "V_std12", "V_absmean12", "V_range", "V_body",
    "P_z1", "P_z2", "P_rm",
]
ZONE_FRAC: float = 0.05
TGT_SHRINK_PCT: float = 0.15
ATR_COL: str = "1_atr_14_ma_5"
# (progress_from, d) steps per TF and side, from the 2y fit.
EXTDONE_LADDER: dict[int, dict[str, tuple[tuple[float, float], ...]]] = {
    15:  {"high": ((0.25, 5.0), (0.50, 2.5), (0.75, 1.5)), "low": ((0.25, 4.0), (0.50, 2.5), (0.75, 1.5))},
    60:  {"high": ((0.25, 10.0), (0.50, 8.0), (0.75, 4.0)), "low": ((0.25, 10.0), (0.50, 6.0), (0.75, 4.0))},
    240: {"high": ((0.75, 8.0),), "low": ((0.75, 8.0),)},
}
# p75 of the marker overshoot (band sd) over the 2y filtered markers, frozen
# from the oos2m sidecar df_with_cb_overshoot.pkl (run_cb_overshoot.py,
# OVER_Q=0.75; spec 2026-10-17-cb-overshoot-main-flow-design.md D1). The
# sidecar's training-side veto used the oos2m-picked extdone ladder;
# scripts/cb_export_overshoot.py prints a refit with the main-flow veto.
OVERSHOOT_Q: float = 0.75
OVERSHOOT_THR: dict[int, dict[str, float]] = {
    15:  {"long": 0.3477, "short": 0.3808},
    60:  {"long": 0.3756, "short": 0.3796},
    240: {"long": 0.4693, "short": 0.5178},
}
