"""tdlib/features.py -- Task 3: feature engineering (Layer 3).

Builds every hypothesis metric the experiment will later screen (Layer 4).
Leak-safety is this module's entire mandate: every formula below reads only
CURRENT and PAST information relative to the row it is computed at --
running/forming values at each row, or explicitly ``shift(1)``-lagged
rolling aggregates that exclude the current bar. Nothing here ever reads a
future close, a future candle, or a future label.

Four things live here:

1. ``HIGHER_TF`` / ``LOWER_TF`` -- per (analysis) tf, which OTHER tfs count
   as "higher" (coarser, used for both engineered cross-tf features and as
   raw-column context) and "lower" (finer, raw-column context ONLY -- no
   engineered features are ever computed on a lower tf, see
   ``engineered_features``'s docstring). ``CUTS_FALLBACK`` maps a tf absent
   from ``tdlib.config.MOVE_CUTS`` (which only carries 15/60/240) to the
   nearest tf that IS present, mirroring
   ``indicators/library/classification.py::_get_tf_classification``'s
   "nearest available tf" fallback convention.

2. ``engineered_features(slim, tf, side)`` -- the actual hypothesis metrics:
   Bollinger-band position/out-of-band/distance, swing-level distance,
   near-level flag, time-in-candle/time-left, "need speed" (required
   ATR-per-remaining-candle drift), higher-tf move classification, and
   higher-tf/side move agreement. Returns ONLY the new columns (never the
   input's own columns), on the input's own index, as a freshly built frame
   (the input is never mutated -- see its docstring for the exact
   dict-then-``pd.concat`` construction discipline, matching data.py:241's
   "collect columns, single concat" convention).

3. ``default_feature_cols`` / ``LEAK_BLOCKLIST_RE`` -- the candidate X
   column whitelist-by-blocklist: every ``{tf}_*`` column belonging to the
   analyzed tf, its higher tfs, or its lower-tf context, MINUS anything the
   blocklist regex flags as a label, a future/truth column, or a bookkeeping
   / absolute-price / raw-size column. ``LEAK_BLOCKLIST_RE`` was built and
   cross-checked against ``tdlib.config``'s ACTUAL slim whitelist (the real
   inventory of what a slim frame carries), not assumed -- see
   ``LEAK_BLOCKLIST_RE``'s own docstring for the specific verifications
   (e.g. raw ``ema_*`` levels are confirmed ABSENT from slim so no ``ema``
   blocklist entry is needed -- only ``_minus_close``/``_slope`` variants
   exist, and those are legitimate relative features, not leaks).

4. ``feature_matrix`` / ``FreezeStats`` -- the actual X/y assembly, and a
   train-frozen median-impute + robust-z-score transform.

   **``feature_matrix`` takes the FULL slim frame, not a pre-selected
   subset.** It runs ``tdlib.points.strong_points`` internally and calls
   ``engineered_features`` on the full frame BEFORE row-subsetting to the
   kept points -- never the other way around. An earlier version took an
   already-``strong_points``-filtered frame directly, which silently broke
   the swing-level feature family (see ``engineered_features``'s
   docstring): rolling over a strong-points-only subset means "the last
   ``w`` closed bars that also happened to be strong points" instead of
   ``w`` truly consecutive closed candles -- generally wrong, not just
   noisier, and with few strong points total could never even reach a
   non-NaN value at all.

   **Imputation lives in ``FreezeStats``, not in ``feature_matrix``.** An
   earlier plan draft had ``feature_matrix`` itself median-impute X. That
   was moved into ``FreezeStats.fit``/``.transform`` instead: OOS/live
   scoring (e.g. the oos2m holdout) must reuse the TRAIN split's medians,
   never recompute its own -- an OOS frame median-imputing against ITS OWN
   distribution would leak that frame's own statistics into what is
   supposed to be an honest out-of-sample evaluation. So ``feature_matrix``
   returns X with raw NaN preserved (only whole-NaN columns are dropped, as
   an actual absence-of-signal fact, not an imputation), and callers fit
   ONE ``FreezeStats`` on the train split's X, freeze it (``to_json``), and
   ``.transform`` every split (train and OOS alike) through that same
   frozen object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tdlib.config import MOVE_CUTS
from tdlib.points import strong_points, sym0_class
from tdlib.truth import mark_truth

# ---------------------------------------------------------------------------
# tf relationships (task-3 brief, verbatim).
# ---------------------------------------------------------------------------

HIGHER_TF = {15: [60, 240, 1440], 60: [240, 1440], 240: [1440]}
LOWER_TF = {15: [5], 60: [5, 15], 240: [15, 60]}  # context only; no engineered features on lower tfs
CUTS_FALLBACK = {1440: 240, 5: 15}  # nearest-tf cuts convention (classification.py _get_tf_classification analog)

# ---------------------------------------------------------------------------
# Leak blocklist -- see default_feature_cols()'s docstring for the full
# per-group rationale and the verification against tdlib.config's real slim
# whitelist.
# ---------------------------------------------------------------------------

LEAK_BLOCKLIST_RE = (
    r"^(?:"
    # labels (profit_strict + plain long/short pairs): the token appears
    # ANYWHERE in the column (e.g. "60_pslong_n1_m1_x0p2_l15_y0p1"), so this
    # branch is wrapped in .* on both sides rather than anchored to the
    # bare-suffix form the bookkeeping branch below uses.
    r".*(?:pslong|psshort|plong|pshort).*"
    # future/truth columns (e.g. a caller-added fwd_log_return diagnostic).
    r"|.*fwd_.*"
    # bookkeeping / absolute-price / raw-size columns: these are BARE
    # "{tf}_{suffix}" columns (digits, underscore, then EXACTLY this suffix
    # and nothing more) -- anchored as a full match (not a trailing
    # substring) specifically so this does NOT catch legitimate relative
    # features that happen to end in the same word, e.g.
    # "60_ema_25_minus_close" (a diff, kept) vs "60_close" (absolute price,
    # blocked), or "60_vol_ma_20_minus_volume" (a diff, kept) vs
    # "60_vol_ma_20" (absolute size, blocked).
    r"|\d+_(?:"
    r"open_index|is_closed|open|high|low|close|volume|buy_volume|close_time"
    r"|tgt_long|tgt_short|sl_long|sl_short"
    r"|bb_upper_20_2|bb_middle_20_2|bb_lower_20_2"
    r"|bb_upper_10_15|bb_lower_10_15|bb_upper_20_3|bb_lower_20_3"
    r"|sar_002_02|vol_ma_20"
    r")"
    r")$"
)

_LEAK_RE = re.compile(LEAK_BLOCKLIST_RE)


# ---------------------------------------------------------------------------
# engineered_features
# ---------------------------------------------------------------------------


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """``numer / denom``, with ``denom == 0`` positions forced to NaN instead
    of the +-inf a raw division would produce there. ``denom`` positions
    that are themselves NaN already yield NaN "for free" through the
    division (``x / nan == nan`` for any ``x``, no explicit handling
    needed) -- ``denom != 0`` is True for a NaN denom (NaN compares unequal
    to everything, including 0), so the ``.where`` guard below leaves those
    positions' already-NaN raw result untouched either way.
    """
    raw = numer / denom
    return raw.where(denom != 0, np.nan)


def engineered_features(slim: pd.DataFrame, tf: int, side: int) -> pd.DataFrame:
    """Every hypothesis feature for analyzed tf ``tf`` / strong-move side
    ``side`` (+2 or -2).

    ``slim`` MUST be the FULL slim frame (every ``{ctf}_*`` column's
    complete, chronologically-ordered row set) -- NOT a ``strong_points``
    selection or any other row subset. Every formula except the
    swing-level family (``swing_dist_hi``/``swing_dist_lo``, and
    ``near_level`` which derives from them) is purely ROW-LOCAL and would
    in fact be unaffected by a subset. But the swing-level family rolls
    over ctf's CLOSED-candle subseries of WHATEVER frame it is given (see
    below): fed anything other than the full frame, "the last ``w`` closed
    bars" would silently mean "the last ``w`` closed bars present IN THAT
    SUBSET" instead of ``w`` truly consecutive closed candles -- a wrong
    feature, not merely a noisier one. ``feature_matrix`` (this module's
    only caller) enforces this by construction: it takes the full slim
    frame itself and calls this function on it directly, row-subsetting
    the RESULT to the kept points afterward, never the other way around.

    Returns ONLY the new columns (never ``slim``'s own), indexed exactly
    like ``slim``. ``slim`` is never mutated -- every value read from it is
    used to build a brand new Series, never written back.

    ``side`` must be +2 or -2 (mirrors ``tdlib.points.strong_points``'s own
    validation) -- anything else raises ``ValueError`` before any column is
    touched.

    Leak-safety by construction:

    - Every per-ctf formula (``bb_pos``, ``oob_*``, ``bb_dist_*``) reads
      ONLY ``ctf``'s own columns AT THE SAME ROW -- a running/forming value,
      never a future one.
    - ``swing_dist_hi``/``swing_dist_lo`` roll over ctf's CLOSED-candle
      subseries with an explicit ``.shift(1)`` -- the rolling window ending
      at the CURRENT closed bar is shifted one position forward, so the
      value attached to a given closed row is the swing extreme over the
      PRIOR ``w`` closed bars, excluding the current one. A brand new
      all-time-high row therefore still measures its distance against the
      swing high BEFORE that row -- see the module's test suite for the
      concrete "negative dist_hi on a new high" case this guarantees.
    - The higher-tf-only block (``time_in_candle`` etc.) reads each row's
      own timestamp and ``{htf}_open_index`` (itself a deterministic
      function of that same timestamp, ``index.floor(f"{htf}min")`` per
      data.py:197) -- again, nothing beyond the current row.

    For ``ctf`` in ``{tf} union HIGHER_TF[tf]`` (formulas keyed on ctf's own
    columns):

    - ``bb_pos_{ctf}`` = ``(close - bb_lower_20_2) / (bb_upper_20_2 -
      bb_lower_20_2)``; a degenerate band (``upper == lower``) yields NaN,
      not +-inf (``_safe_div``).
    - ``oob_up2_{ctf}`` / ``oob_dn2_{ctf}`` = ``close > bb_upper_20_2`` /
      ``close < bb_lower_20_2`` (int8, strict); ``oob_up3_{ctf}`` /
      ``oob_dn3_{ctf}`` same with the ``_20_3`` band.
    - ``bb_dist_up_{ctf}`` = ``(bb_upper_20_2 - close) / atr_14``;
      ``bb_dist_dn_{ctf}`` = ``(close - bb_lower_20_2) / atr_14``. ``atr_14
      == 0`` or NaN both yield NaN (``_safe_div``).
    - ``swing_dist_hi_{ctf}_{w}`` / ``swing_dist_lo_{ctf}_{w}`` for ``w`` in
      ``(20, 50)``: computed on the subseries where ``{ctf}_is_closed`` is
      True, then reindexed onto ``slim.index`` with ``ffill`` (a row
      between two ctf closes carries the last closed value forward; rows
      before ctf's first-ever close stay NaN -- ffill never invents a value
      before the first real one).
    - ``near_level_{ctf}`` = ``min(|swing_dist_hi_{ctf}_20|,
      |swing_dist_lo_{ctf}_20|) < 1.0`` (int8) -- the
      ``levels.py::is_price_near_level`` "within one ATR" convention,
      applied to the swing_dist family's already-ATR-normalized distance.
      NaN inputs fall out of the ``<`` comparison as False (numpy: any
      comparison against NaN is False) -- no explicit NaN branch needed, it
      naturally becomes 0.

    For ``htf`` in ``HIGHER_TF[tf]`` ONLY (the analyzed tf itself has no
    "higher tf" version of these):

    - ``time_in_candle_{htf}`` = ``(row_ts - {htf}_open_index).total_seconds()
      / (htf * 60)``, always in ``[0, 1)`` by construction (floor's own
      defining property); ``time_left_{htf}`` = ``1 - time_in_candle_{htf}``.
    - ``need_speed_up_{htf}`` = ``bb_dist_up_{htf} / clip(time_left_{htf},
      0.05, None)``; ``need_speed_dn_{htf}`` analog with ``bb_dist_dn_{htf}``
      -- the clip keeps the denominator from exploding to near-zero right at
      candle close.
    - ``htf_move_{htf}`` = ``sym0_class(htf_rsi_ma8_diff, cuts)`` where
      ``cuts = MOVE_CUTS[htf]`` if present, else
      ``MOVE_CUTS[CUTS_FALLBACK[htf]]`` (only ``htf == 1440`` ever needs the
      fallback, to 240's cuts, among the values ``HIGHER_TF`` ever produces).
    - ``htf_diff_sign_{htf}`` = ``sign(htf_rsi_ma8_diff)`` (int8, -1/0/+1);
      NaN maps to 0 (``fillna(0.0)`` BEFORE ``sign`` -- ``sign(0) == 0``
      too, so this single fillna correctly covers both the "genuinely zero"
      and "missing" cases without a separate NaN branch).
    - ``htf_move_agree_{htf}`` = ``htf_diff_sign_{htf} == sign(side)`` (int8)
      -- whether the higher tf is moving the SAME direction as the analyzed
      strong move.
    """
    if side not in (2, -2):
        raise ValueError(f"engineered_features: side must be one of (2, -2), got {side!r}")

    ctfs = [tf] + HIGHER_TF[tf]  # KeyError propagates naturally for an unknown tf, not caught here

    ctf_cols: dict[str, pd.Series] = {}
    for ctf in ctfs:
        close = slim[f"{ctf}_close"]
        atr = slim[f"{ctf}_atr_14"]
        bb_up2 = slim[f"{ctf}_bb_upper_20_2"]
        bb_lo2 = slim[f"{ctf}_bb_lower_20_2"]
        bb_up3 = slim[f"{ctf}_bb_upper_20_3"]
        bb_lo3 = slim[f"{ctf}_bb_lower_20_3"]

        ctf_cols[f"bb_pos_{ctf}"] = _safe_div(close - bb_lo2, bb_up2 - bb_lo2)

        ctf_cols[f"oob_up2_{ctf}"] = (close > bb_up2).astype(np.int8)
        ctf_cols[f"oob_dn2_{ctf}"] = (close < bb_lo2).astype(np.int8)
        ctf_cols[f"oob_up3_{ctf}"] = (close > bb_up3).astype(np.int8)
        ctf_cols[f"oob_dn3_{ctf}"] = (close < bb_lo3).astype(np.int8)

        ctf_cols[f"bb_dist_up_{ctf}"] = _safe_div(bb_up2 - close, atr)
        ctf_cols[f"bb_dist_dn_{ctf}"] = _safe_div(close - bb_lo2, atr)

        closed = slim[f"{ctf}_is_closed"] == True  # noqa: E712 - explicit bool-column compare
        high_c = slim.loc[closed, f"{ctf}_high"]
        low_c = slim.loc[closed, f"{ctf}_low"]
        close_c = slim.loc[closed, f"{ctf}_close"]
        atr_c = slim.loc[closed, f"{ctf}_atr_14"]

        for w in (20, 50):
            roll_hi = high_c.rolling(w, min_periods=w).max().shift(1)
            roll_lo = low_c.rolling(w, min_periods=w).min().shift(1)

            dist_hi = _safe_div(roll_hi - close_c, atr_c).reindex(slim.index).ffill()
            dist_lo = _safe_div(close_c - roll_lo, atr_c).reindex(slim.index).ffill()

            ctf_cols[f"swing_dist_hi_{ctf}_{w}"] = dist_hi
            ctf_cols[f"swing_dist_lo_{ctf}_{w}"] = dist_lo

            if w == 20:
                near = np.minimum(dist_hi.abs(), dist_lo.abs()) < 1.0
                ctf_cols[f"near_level_{ctf}"] = near.astype(np.int8)

    htf_cols: dict[str, pd.Series] = {}
    row_ts = slim.index.to_series()
    side_sign = 1 if side > 0 else -1

    for htf in HIGHER_TF[tf]:
        open_idx = slim[f"{htf}_open_index"]
        elapsed = (row_ts - open_idx).dt.total_seconds()
        time_in_candle = elapsed / (htf * 60)
        time_left = 1.0 - time_in_candle
        htf_cols[f"time_in_candle_{htf}"] = time_in_candle
        htf_cols[f"time_left_{htf}"] = time_left

        denom = time_left.clip(lower=0.05)
        htf_cols[f"need_speed_up_{htf}"] = ctf_cols[f"bb_dist_up_{htf}"] / denom
        htf_cols[f"need_speed_dn_{htf}"] = ctf_cols[f"bb_dist_dn_{htf}"] / denom

        cuts = MOVE_CUTS[htf] if htf in MOVE_CUTS else MOVE_CUTS[CUTS_FALLBACK[htf]]
        diff = slim[f"{htf}_rsi_ma8_diff"]
        htf_cols[f"htf_move_{htf}"] = sym0_class(diff, cuts)

        sign_series = np.sign(diff.fillna(0.0)).astype(np.int8)
        htf_cols[f"htf_diff_sign_{htf}"] = sign_series
        htf_cols[f"htf_move_agree_{htf}"] = (sign_series == side_sign).astype(np.int8)

    return pd.concat([pd.DataFrame(ctf_cols, index=slim.index), pd.DataFrame(htf_cols, index=slim.index)], axis=1)


# ---------------------------------------------------------------------------
# default_feature_cols
# ---------------------------------------------------------------------------


def default_feature_cols(slim_cols: list[str], tf: int) -> list[str]:
    """Candidate X column names from a slim frame's own column list (raw
    slim columns ONLY -- the engineered columns are added separately by
    ``feature_matrix``, this function knows nothing about them).

    Every column prefixed ``{ctf}_`` for ``ctf`` in ``{tf} union
    HIGHER_TF[tf] union LOWER_TF[tf]`` (i.e. the analyzed tf, its higher-tf
    context, AND its lower-tf context -- unlike ``engineered_features``,
    which never touches lower tfs, the RAW lower-tf columns are still
    legitimate context features), except any column matching
    ``LEAK_BLOCKLIST_RE``. Order matches ``slim_cols``'s own order (not
    grouped by ctf), so the result is deterministic given a deterministic
    input column order.

    ``LEAK_BLOCKLIST_RE`` was verified against ``tdlib.config``'s actual
    slim whitelist (the authoritative inventory of what a slim frame
    carries), not assumed from the task brief's prose alone:

    - Raw ``ema_*`` LEVEL columns (e.g. a bare ``{tf}_ema_25``) are
      confirmed ABSENT from the slim whitelist (``config.py``'s
      ``_EMA_TREND_SUFFIXES`` carries only ``*_minus_close`` diffs and
      ``*_slope`` slopes) -- so no ``ema`` blocklist entry exists here; one
      would incorrectly catch the legitimate diff/slope columns too (e.g. a
      naive ``ema`` substring match would block
      ``60_ema_25_minus_close``, a real relative feature, right along with
      any bare level).
    - ``sar_002_02`` and every ``bb_upper``/``bb_middle``/``bb_lower`` band
      column ARE present and absolute-price-scale -- blocked.
    - ``vol_ma_20`` (raw absolute volume size) IS present and blocked, but
      ``vol_ma_20_minus_volume`` (a relative diff) is NOT blocked -- the
      bookkeeping half of ``LEAK_BLOCKLIST_RE`` matches the BARE
      ``{tf}_{suffix}`` form only (anchored full match), which is what
      makes this distinction possible; see ``LEAK_BLOCKLIST_RE``'s own
      comment for the general form of this trap (it also applies to
      ``close`` vs ``ema_25_minus_close``).
    - ``ZB``/``ZS`` (target.py's Zone-Buy/Zone-Sell fields) are bounded 0/1
      classification flags derived from a threshold compare, not an
      absolute-price/raw-size column (the same category as the kept
      ``zone_class``/``move_class`` columns) -- intentionally NOT
      blocklisted, matching the brief's own blocklist enumeration (which
      names the ``tgt``/``sl`` pair but not ``ZB``/``ZS``).
    """
    ctfs = [tf] + HIGHER_TF[tf] + LOWER_TF[tf]  # KeyError propagates naturally for an unknown tf
    prefixes = tuple(f"{ctf}_" for ctf in ctfs)
    return [c for c in slim_cols if c.startswith(prefixes) and not _LEAK_RE.fullmatch(c)]


# ---------------------------------------------------------------------------
# feature_matrix
# ---------------------------------------------------------------------------


def feature_matrix(
    slim: pd.DataFrame,
    tf: int,
    side: int,
    feature_cols: list[str] | None = None,
    horizon: str = "n1",
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble (X, y) from ``slim`` -- the FULL slim frame, not a
    pre-selected subset (see ``engineered_features``'s docstring for why:
    the swing-level feature family needs the complete, chronologically
    consecutive closed-candle history to mean what it says it means).

    - ``pts = strong_points(slim, tf, side)`` -- the L2 strong-point
      selection, run internally so every caller gets it applied
      consistently and ``engineered_features`` still gets the full frame
      (see below).
    - ``marked = mark_truth(pts, tf, horizon)``; rows are kept only where
      ``marked`` is "long" or "short" ("both"/"neither"/"nan" are dropped --
      an ambiguous or missing label is not a usable training example here).
    - ``y`` = 1 for "long", 0 for "short" (int8), one entry per kept row.
    - ``X`` = ``slim[feature_cols or default_feature_cols(slim.columns, tf)]``
      (booleans among those raw columns cast to int8; any remaining
      non-numeric-dtype column dropped) joined with
      ``engineered_features(slim, tf, side)`` -- computed ONCE on the full
      ``slim``, so the swing-level family rolls over every truly-consecutive
      closed candle, not just the (generally scattered, non-contiguous)
      strong points -- with BOTH restricted to the kept rows only
      afterward, via a single ``pd.concat(axis=1)`` (the two column groups
      never collide by name -- raw slim columns are always ``{tf}_...``,
      engineered columns never carry a leading ``{tf}_`` token).
    - NO imputation happens here -- raw NaN is preserved column-by-column,
      EXCEPT a column that is entirely NaN across every kept row is dropped
      outright (there is no signal left to impute; see the module docstring
      for why imputation itself belongs in ``FreezeStats``, not here).
    - Belt-and-braces: every final X column is checked against
      ``LEAK_BLOCKLIST_RE`` and a match raises ``ValueError`` naming the
      offending column(s) -- this catches a caller-supplied ``feature_cols``
      that bypasses ``default_feature_cols``'s own filtering (e.g. a planted
      label column), even though ``engineered_features``'s own output never
      matches the blocklist by construction.

    Returns ``(X, y)`` sharing the same row order as ``pts``'s kept rows
    (``pts``'s own row order is ``slim``'s chronological order, since
    ``strong_points`` boolean-masks rather than reorders). ``slim`` itself
    is never mutated (every selection is copied before any in-place dtype
    cast).
    """
    pts = strong_points(slim, tf, side)
    marked = mark_truth(pts, tf, horizon)
    keep_mask = marked.isin(["long", "short"])
    kept_index = marked.index[keep_mask.to_numpy()]

    y = (marked.loc[kept_index] == "long").astype(np.int8)

    cols = list(feature_cols) if feature_cols is not None else default_feature_cols(list(slim.columns), tf)

    raw = slim.loc[kept_index, cols].copy()
    for col in raw.columns:
        if raw[col].dtype == bool:
            raw[col] = raw[col].astype(np.int8)
    numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
    raw = raw[numeric_cols]

    engineered = engineered_features(slim, tf, side).loc[kept_index]

    X = pd.concat([raw, engineered], axis=1)
    X = X.dropna(axis=1, how="all")

    leaked = [c for c in X.columns if _LEAK_RE.fullmatch(c)]
    if leaked:
        raise ValueError(f"feature_matrix: leak-blocklisted columns present in X: {leaked!r}")

    return X, y


# ---------------------------------------------------------------------------
# FreezeStats
# ---------------------------------------------------------------------------


@dataclass
class FreezeStats:
    """Train-frozen per-column (median, robust-scale), JSON round-trippable.

    ``stats`` maps each column name straight to ``{"median": ..., "scale":
    ...}`` -- unlike ``azlib.indicators.FreezeStats`` (whose keys are
    3-tuples and need a records-list JSON workaround), plain string column
    names are already JSON-native, so ``to_json``/``from_json`` here are a
    direct ``json.dump``/``json.load`` of ``self.stats``, no reshaping
    needed.

    ``fit`` is the ONLY method allowed to look at a frame's own
    distribution (see the module docstring's imputation-ownership note);
    ``transform`` always reuses whatever was frozen at the last ``fit``
    call and never recomputes anything from the frame it is scoring.
    """

    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "FreezeStats":
        """Per column: ``median`` = the impute value; ``scale`` = robust
        IQR-based scale (``(q75 - q25) / 1.349``, the constant that makes
        this estimator consistent with a normal distribution's std).

        Degenerate columns fall back in two stages: if the IQR scale is not
        finite or below the ``1e-9`` floor (e.g. a constant column, IQR ==
        0), fall back to the plain sample std; if THAT is also not finite or
        below the floor (e.g. a single-row column, ``ddof=1`` std is NaN),
        fall back to a flat ``1.0``. Either fallback avoids a
        divide-by-zero/near-zero blowup in ``transform`` without ever
        raising on a degenerate/constant input column.

        All NaN values are excluded before computing median/quantiles/std
        (``.dropna()``) -- a column's missingness does not skew its own
        impute/scale statistics.
        """
        stats: dict[str, dict[str, float]] = {}
        for col in X.columns:
            series = X[col].dropna()
            if series.empty:
                median = 0.0
                scale = 1.0
            else:
                median = float(series.median())
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                scale = (q3 - q1) / 1.349
                if not np.isfinite(scale) or scale < 1e-9:
                    std = float(series.std())
                    scale = std if (np.isfinite(std) and std >= 1e-9) else 1.0
            stats[col] = {"median": median, "scale": max(float(scale), 1e-9)}
        self.stats = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """NaN cells impute to the FROZEN median, then every value is
        centered/scaled: ``(x - median) / scale``.

        Any column in ``X`` that was NOT seen at ``fit`` time raises
        ``ValueError`` naming every such column -- transforming an unknown
        column would silently skip the one normalization step this class
        exists to enforce. Conversely, a column that WAS seen at fit but is
        simply ABSENT from ``X`` is not an error: it is treated as if every
        one of its values were the frozen median (which, run through the
        same center/scale step, works out to exactly 0.0 for every row --
        no separate "fill with 0" branch needed, this falls out of the same
        formula used for a present-but-NaN cell).

        Never mutates ``self.stats`` -- this method looks only at ``X``'s
        values, never its distribution, so calling it on however many
        different frames never changes what a later call sees.
        """
        unseen = [c for c in X.columns if c not in self.stats]
        if unseen:
            raise ValueError(f"FreezeStats.transform: column(s) not seen at fit time: {unseen!r}")

        cols: dict[str, pd.Series] = {}
        for col, params in self.stats.items():
            median = params["median"]
            scale = params["scale"]
            filled = X[col].fillna(median) if col in X.columns else pd.Series(median, index=X.index)
            cols[col] = (filled - median) / scale
        return pd.DataFrame(cols, index=X.index)

    def to_json(self, path: str) -> None:
        """Write ``self.stats`` (``{col: {"median", "scale"}}``) as JSON,
        sorted keys, 2-space indent."""
        with open(path, "w") as f:
            json.dump(self.stats, f, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FreezeStats":
        """Load a ``FreezeStats`` from a ``to_json`` file -- exact round
        trip (dict equality) since plain ``{str: {str: float}}`` JSON needs
        no reshaping on the way back in."""
        with open(path) as f:
            stats = json.load(f)
        return cls(stats)
