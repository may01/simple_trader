"""azlib/config.py — Task 1.2: default label params per timeframe.

``default_label_params`` returns a ``LabelParams`` (azlib.loader) filled
with this experiment's default knobs for a given ``tf``. These are
experiment knobs, not "correct" values — a documented starting point to run
the pipeline end to end before any tuning happens. Meaning of each field
(see ``indicators.labels`` for the exact math each one drives):

- ``n`` (default 1): how many *higher-TF* candles of forward window to
  check for target-before-stop. The actual 1-minute lookahead window is
  ``n * tf`` rows (see ``indicators.labels._forward_labels``); ``n=1`` means
  "one candle of the entry's own timeframe".
- ``m`` (default 2.0): target distance from entry, in units of
  ``{tf}_atr_{atr_period}_ma_{ma_length}``. Target = entry +/- ``m`` *
  atr_ma (sign depends on direction).
- ``x`` (default 2.0): stop distance from entry, same atr_ma units.
  Symmetric with ``m`` by default (1:1 reward/risk).
- ``l`` (default ``tf``): how many 1-minute rows to look *backward* (entry
  row inclusive) for the profit_strict "clean entry" condition — no prior
  dip/spike through ``y`` * atr_ma (see ``indicators.labels._past_clean``).
  Defaulting to ``tf`` itself means "one full own-timeframe candle of
  history must have been clean".
- ``y`` (default 1.0): backward clean-entry tolerance, in the same atr_ma
  units as ``m``/``x``.
- ``atr_period``/``ma_length`` (defaults 14/5, from ``LabelParams``' own
  dataclass defaults): match the precomputed ``{tf}_atr_14_ma_5`` column
  every wide df already ships, so labels can be computed with zero extra
  indicator passes.
"""

from __future__ import annotations

from azlib.loader import LabelParams


def default_label_params(tf: int) -> LabelParams:
    """``LabelParams`` with this experiment's default knobs for one ``tf``.

    ``l`` defaults to ``tf`` itself (see module docstring); every other
    knob is a tf-independent constant. ``atr_period``/``ma_length`` fall
    through to ``LabelParams``' own dataclass defaults (14/5).
    """
    return LabelParams(tf=tf, n=1, m=2.0, x=2.0, l=tf, y=1.0)


# --- Task 3: indicator attribute column names (azlib/indicators.py) --------
#
# Exact wide-df column suffixes (after the "{tf}_" prefix each wide-df
# column carries) that azlib.indicators.raw_attribute reads for each
# indicator's position/slope/distance attrs (design spec §7). Values
# confirmed against configs/indicators_config.yaml's `name:` entries — every
# one of these is an `applies_to: all` field the production indicator
# pipeline already computes for every tf, so no new indicator computation is
# introduced here, only column-name selection:
#
#   - rsi_14      (configs/indicators_config.yaml: momentum, talib RSI period=14)
#   - rsi_ma8     (configs/indicators_config.yaml: EMA-of-RSI, depends_on [rsi_14];
#                  the "8" period is the one this experiment standardizes on
#                  out of the available rsi_ma8/rsi_ma12/rsi_ma24 family)
#   - macd_12_26_9 / macd_hist_12_26_9
#                 (configs/indicators_config.yaml: the 12/26/9 MACD family,
#                  not the alternate macd_5_13_9 family)
#   - ema_25      (configs/indicators_config.yaml: the "ma" indicator's
#                  reference EMA, out of the ema_7/14/25/50/100 family)
#
# Kept as plain module-level constants (not a dict) so callers/tests can
# import exactly the name they need without an extra lookup indirection —
# mirrors this module's existing flat-constant style.
RSI_COL = "rsi_14"
RSI_MA_COL = "rsi_ma8"
MACD_COL = "macd_12_26_9"
MACD_HIST_COL = "macd_hist_12_26_9"
MA_COL = "ema_25"
