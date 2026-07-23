"""tdlib/config.py -- Task 1: SLIM-frame column whitelist + shared constants.

Three things live here:

1. ``ANALYSIS_TFS`` / ``CONTEXT_TFS`` -- the timeframe sets every later
   trend_detection task keys off. ``ANALYSIS_TFS`` (15/60/240) are the tfs
   this experiment actually classifies trend/zone/move on; ``CONTEXT_TFS``
   adds 5 (fine-grained recent context) and 1440 (daily regime context) --
   every tf the SLIM frame carries at all.

2. ``MOVE_CUTS`` / ``LABEL_COLS`` -- literal constants copied from the
   task-1 brief. ``MOVE_CUTS`` was VERIFIED against its stated source
   (worktrees/rsi-quantile-sym0/stats/train/link_usdt/rsi_classification.json,
   keys "15"/"60"/"240" -> ``move_cuts``) rather than trusted from the
   brief's own inline literals -- the two did NOT match byte-for-byte (they
   diverge starting the ~6th decimal digit, e.g. brief's 15[0]
   -1.3642937922290073 vs the file's -1.3642919195372998). Per the brief's
   own "if they differ ... use the file's values" instruction, the values
   below are the FILE's, not the brief's inline copy. See task-1-report.md
   for the full comparison.

3. ``slim_columns`` / ``slim_out_path`` / ``artifacts_dir`` -- the SLIM
   whitelist-reduction function and its two output-path helpers (built on
   ``helpers.dataset_folder()``, env-driven: ROOT_FOLDER/DATA_ROOT/
   DATA_SET_NAME/PAIR, read at CALL time -- see helpers.py's own "ZERO
   business logic" convention, mirrored here: no os.environ read happens at
   import time).
"""

from __future__ import annotations

from helpers import dataset_folder

# ---------------------------------------------------------------------------
# Timeframe sets
# ---------------------------------------------------------------------------

# tfs this experiment actually classifies trend/zone/move on (task-1 brief,
# exact values).
ANALYSIS_TFS = [15, 60, 240]

# Every tf the SLIM frame carries: ANALYSIS_TFS plus 5 (fine recent context)
# and 1440 (daily regime context) (task-1 brief, exact values).
CONTEXT_TFS = [5, 15, 60, 240, 1440]

# ---------------------------------------------------------------------------
# move_cuts -- 2y fit thresholds on {tf}_rsi_ma8_diff for move_class_sym0
# bucketing. See module docstring point 2: these are the verified SOURCE
# FILE's values, not the brief's own (slightly different) inline literals.
# ---------------------------------------------------------------------------

MOVE_CUTS = {
    15: [-1.3642919195372998, -0.4092875758611899, 0.4092875758611899, 1.3642919195372998],
    60: [-1.4096495558610285, -0.4228948667583085, 0.4228948667583085, 1.4096495558610285],
    240: [-1.4597374604647457, -0.4379212381394237, 0.4379212381394237, 1.4597374604647457],
}

# ---------------------------------------------------------------------------
# Label columns -- copied verbatim from the task-1 brief.
# ---------------------------------------------------------------------------

LABEL_COLS = {
    15: {
        "pslong_n1": "15_pslong_n1_m1_x0p3_l15_y0p2", "psshort_n1": "15_psshort_n1_m1_x0p3_l15_y0p2",
        "pslong_n2": "15_pslong_n2_m1_x0p3_l15_y0p2", "psshort_n2": "15_psshort_n2_m1_x0p3_l15_y0p2",
        "plong_n1": "15_plong_n1_m1_x0p3", "pshort_n1": "15_pshort_n1_m1_x0p3",
    },
    60: {
        "pslong_n1": "60_pslong_n1_m1_x0p2_l15_y0p1", "psshort_n1": "60_psshort_n1_m1_x0p2_l15_y0p1",
        "pslong_n2": "60_pslong_n2_m1_x0p2_l15_y0p1", "psshort_n2": "60_psshort_n2_m1_x0p2_l15_y0p1",
        "plong_n1": "60_plong_n1_m1_x0p2", "pshort_n1": "60_pshort_n1_m1_x0p2",
    },
    240: {
        "pslong_n1": "240_pslong_n1_m1_x0p1_l15_y0p1", "psshort_n1": "240_psshort_n1_m1_x0p1_l15_y0p1",
        "pslong_n2": "240_pslong_n2_m1_x0p1_l15_y0p1", "psshort_n2": "240_psshort_n2_m1_x0p1_l15_y0p1",
        "plong_n1": "240_plong_n1_m1_x0p1", "pshort_n1": "240_pshort_n1_m1_x0p1",
    },
}


def _label_columns_flat() -> list[str]:
    """Every LABEL_COLS value, in tf-then-field insertion order."""
    out: list[str] = []
    for tf in LABEL_COLS:
        out.extend(LABEL_COLS[tf].values())
    return out


# ---------------------------------------------------------------------------
# SLIM column whitelist (task-1 brief, "Whitelist per tf in CONTEXT_TFS").
#
# Every group below is applied per tf in CONTEXT_TFS as "{tf}_{suffix}".
# Not every suffix exists for every tf in the real wide df (e.g. sin_tod is
# tf=15-only; align_* only ever matches ONE suffix per tf -- see
# _ALIGN_SUFFIXES below) -- slim_columns() intersects this whitelist with
# the caller's actually-available columns, so absent optional suffixes are
# silently dropped; only _required_columns() (further down) raises.
# ---------------------------------------------------------------------------

_OHLCV_SUFFIXES = ["open_index", "open", "high", "low", "close", "volume", "is_closed"]

_RSI_SUFFIXES = [
    "rsi_14", "rsi_ma8", "rsi_ma8_diff", "rsi_ma8_slope",
    "rsi_ma12", "rsi_ma12_diff", "rsi_ma24", "rsi_ma24_diff",
    "nn_rsi_ma8_norm_mean_20",
]

_ZONE_SUFFIXES = ["zone_class", "move_class", "over_low", "over_high", "zone_class_q", "move_class_sym0"]

_EMA_TREND_SUFFIXES = [
    "ema_7_minus_close", "ema_14_minus_close", "ema_25_minus_close", "ema_50_minus_close", "ema_100_minus_close",
    "ema_7_minus_ema_14", "ema_7_minus_ema_25", "ema_14_minus_ema_25", "ema_7_minus_ema_50", "ema_14_minus_ema_50",
    "ema_25_minus_ema_50",
    "ema_7_slope", "ema_14_slope", "ema_25_slope", "ema_50_slope", "ema_100_slope",
    "trend_up_50", "trend_down_50", "adx_14", "adx_14_slope",
]

_MACD_SUFFIXES = [
    "macd_12_26_9", "macd_signal_12_26_9", "macd_hist_12_26_9",
    "macd_5_13_9", "macd_signal_5_13_9",
    "macd_12_26_9_slope", "macd_signal_12_26_9_slope", "macd_5_13_9_slope", "macd_signal_5_13_9_slope",
]

_CCI_SAR_SUFFIXES = ["cci_14", "cci_14_ma_5", "cci_diff", "sar_002_02"]

_VOLATILITY_SUFFIXES = [
    "atr_14", "natr_14", "atr_14_ma_5", "natr_14_ma_5", "range_atr", "vol_regime",
    "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2",
    "bb_upper_10_15", "bb_lower_10_15",
    "bb_upper_20_3", "bb_lower_20_3",
    "bb_upper_20_2_minus_close", "bb_middle_20_2_minus_close", "bb_lower_20_2_minus_close",
]

_VOLUME_SUFFIXES = ["vol_ma_20", "vol_ma_20_minus_volume"]

_CANDLE_SHAPE_SUFFIXES = [
    "logret", "body_ratio", "wick_up", "wick_dn",
    "close_diff_prc", "close_diff_prc_rm_6", "high_diff_prc_rm_6", "low_diff_prc_rm_6",
    "close_diff_prc_rm_6_std_above", "close_diff_prc_rm_6_std_below",
    "high_diff_prc_rm_6_std_above", "high_diff_prc_rm_6_std_below",
    "low_diff_prc_rm_6_std_above", "low_diff_prc_rm_6_std_below",
]

_TARGET_SUFFIXES = ["tgt_long", "tgt_short", "sl_long", "sl_short", "ZB", "ZS"]

# Cross-TF align ladder (configs/indicators_config.yaml; verified against
# tests/unit/data_layer/test_nn_features.py::TestAlignFieldScheduling): each
# tf's OWN align column is named "{tf}_align_{next_tf}" where next_tf is the
# next rung up (5->15->60->240->1440; tf=1440 has none). The brief's literal
# whitelist prose lists suffixes align_5/15/60/240; used alone that set
# would (a) never match anything for any tf in CONTEXT_TFS (align_5 only
# ever applies to tf=1, which CONTEXT_TFS excludes) and (b) never capture
# tf=240's real align column (240_align_1440 -- suffix "align_1440" isn't in
# the brief's list). Both gaps are additive-safe to close: an extra
# candidate suffix that never matches in production is simply absent from
# `available` and silently dropped by slim_columns' normal
# whitelist-intersect-available semantics. So this list is the brief's four
# suffixes UNION align_1440, not a replacement -- see task-1-report.md.
_ALIGN_SUFFIXES = ["align_5", "align_15", "align_60", "align_240", "align_1440"]

# tf=15 only (task-1 brief).
_TF15_ONLY_SUFFIXES = ["sin_tod", "cos_tod", "sin_dow", "cos_dow"]

_PER_TF_SUFFIX_GROUPS = [
    _OHLCV_SUFFIXES,
    _RSI_SUFFIXES,
    _ZONE_SUFFIXES,
    _EMA_TREND_SUFFIXES,
    _MACD_SUFFIXES,
    _CCI_SAR_SUFFIXES,
    _VOLATILITY_SUFFIXES,
    _VOLUME_SUFFIXES,
    _CANDLE_SHAPE_SUFFIXES,
    _TARGET_SUFFIXES,
    _ALIGN_SUFFIXES,
]


def _whitelist() -> list[str]:
    """Full SLIM whitelist, order-stable, no duplicates. Order: per tf in
    CONTEXT_TFS (suffix groups in the module-level order above, then tf=15's
    extra sin/cos columns), followed once by every LABEL_COLS value."""
    cols: list[str] = []
    seen: set[str] = set()

    def _add(col: str) -> None:
        if col not in seen:
            seen.add(col)
            cols.append(col)

    for tf in CONTEXT_TFS:
        for group in _PER_TF_SUFFIX_GROUPS:
            for suffix in group:
                _add(f"{tf}_{suffix}")
        if tf == 15:
            for suffix in _TF15_ONLY_SUFFIXES:
                _add(f"{tf}_{suffix}")

    for col in _label_columns_flat():
        _add(col)

    return cols


# ---------------------------------------------------------------------------
# Required subset (task-1 brief, "REQUIRED subset (raise if missing)").
# ---------------------------------------------------------------------------

_REQUIRED_PER_CONTEXT_TF_SUFFIXES = ["open_index", "open", "high", "low", "close", "is_closed"]
_REQUIRED_PER_ANALYSIS_TF_SUFFIXES = ["rsi_ma8_diff", "atr_14", "bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"]
_REQUIRED_ANALYSIS_TFS = [15, 60, 240, 1440]  # brief: "for tf in [15,60,240,1440]"


def _required_columns() -> list[str]:
    """Every column slim_columns() must find in `available`, or it raises."""
    cols: list[str] = []
    seen: set[str] = set()

    def _add(col: str) -> None:
        if col not in seen:
            seen.add(col)
            cols.append(col)

    for tf in CONTEXT_TFS:
        for suffix in _REQUIRED_PER_CONTEXT_TF_SUFFIXES:
            _add(f"{tf}_{suffix}")

    for tf in _REQUIRED_ANALYSIS_TFS:
        for suffix in _REQUIRED_PER_ANALYSIS_TF_SUFFIXES:
            _add(f"{tf}_{suffix}")

    for col in _label_columns_flat():
        _add(col)

    return cols


def slim_columns(available: list) -> list:
    """Return ``whitelist ∩ available``, in the whitelist's own canonical
    order (NOT `available`'s order -- so the SLIM frame's column order is
    deterministic regardless of what order the source df's columns happen
    to be in), with no duplicates.

    Raises ``KeyError`` if any REQUIRED column (see _required_columns()) is
    absent from `available` -- the message names EVERY missing required
    column, not just the first.
    """
    available_set = set(available)

    missing = [col for col in _required_columns() if col not in available_set]
    if missing:
        raise KeyError(f"slim_columns: missing required columns: {missing!r}")

    return [col for col in _whitelist() if col in available_set]


# ---------------------------------------------------------------------------
# Output paths -- built on helpers.dataset_folder() (env-driven; read at
# CALL time, never at import time -- mirrors helpers.py's own convention).
# ---------------------------------------------------------------------------


def slim_out_path() -> str:
    """Return ``{dataset_folder()}trend_detection/slim.pkl``."""
    return f"{dataset_folder()}trend_detection/slim.pkl"


def artifacts_dir() -> str:
    """Return ``{dataset_folder()}trend_detection/``."""
    return f"{dataset_folder()}trend_detection/"
