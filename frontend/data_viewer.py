"""DataViewer — renders historical OHLCV + indicators + levels for analysis."""

from __future__ import annotations

import colorsys
import re
from typing import Optional, Any

import pandas as pd
import plotly.graph_objects as go

import constants
from frontend.chart_renderer import ChartRenderer

# ---------------------------------------------------------------------------
# Indicator subplot routing
# ---------------------------------------------------------------------------

# Indicators whose name starts with one of these prefixes go on the price axis.
_PRICE_AXIS_PREFIXES = ("ema", "sma", "bb", "vwap", "tgt_", "sl_")

# Maps indicator base name (without _NN suffix) → subplot name.
# Indicators not found here AND not on the price axis get a subplot named
# after their base name (e.g. "rsi_14" → subplot "rsi").
_INDICATOR_SUBPLOT = {
    "rsi": "rsi",
    "cci": "cci",
    "macd_12_26_9": "macd_12_26_9",
    "stoch": "stoch",
}

# Full-name routing checked before the base-name split. The split would send
# every macd_* field to one "macd" subplot; the two configured MACD variants
# need their own subplots, with signal/hist joining their line.
_INDICATOR_SUBPLOT_EXACT = {
    "macd_12_26_9": "macd_12_26_9",
    "macd_signal_12_26_9": "macd_12_26_9",
    "macd_hist_12_26_9": "macd_12_26_9",
    "macd_5_13_9": "macd_5_13_9",
    "macd_signal_5_13_9": "macd_5_13_9",
}

# Price-derivative (diff field) groups: subplot name → fields sharing it.
# Routed via the exact map — the base-name split would wrongly send
# "close_diff_prc" to a subplot named "close". Charts show the diff, its
# rolling mean, and the mean ± rolling-std band.
_DERIVATIVE_SUBPLOTS = {
    "close_diff": [
        "close_diff_prc", "close_diff_prc_rm_6",
        "close_diff_prc_rm_6_std_above", "close_diff_prc_rm_6_std_below",
    ],
    "high_diff": [
        "high_diff_prc", "high_diff_prc_rm_6",
        "high_diff_prc_rm_6_std_above", "high_diff_prc_rm_6_std_below",
    ],
    "low_diff": [
        "low_diff_prc", "low_diff_prc_rm_6",
        "low_diff_prc_rm_6_std_above", "low_diff_prc_rm_6_std_below",
    ],
    "rsi_diff": ["rsi_ma8_diff", "rsi_ma12_diff", "rsi_ma24_diff"],
}

_INDICATOR_SUBPLOT_EXACT.update(
    {f: sp for sp, fields in _DERIVATIVE_SUBPLOTS.items() for f in fields}
)

# The one-signed mean bands are no longer drawn by default but stay routable
# for explicit indicator lists.
_INDICATOR_SUBPLOT_EXACT.update({
    f"{src}_diff_prc_rm_6_mean_{side}": f"{src}_diff"
    for src in ("close", "high", "low")
    for side in ("above", "below")
})


def _indicator_subplot(indicator: str) -> str | None:
    """Return the subplot name for an indicator, or None for price-axis indicators.

    Examples:
        "rsi_14"  → "rsi"
        "cci_14"  → "cci"
        "ema_20"  → None  (price axis)
        "sma_50"  → None  (price axis)
    """
    lower = indicator.lower()
    # Price-axis indicators
    for prefix in _PRICE_AXIS_PREFIXES:
        if lower.startswith(prefix):
            return None
    # Exact-name routing (MACD variants)
    if lower in _INDICATOR_SUBPLOT_EXACT:
        return _INDICATOR_SUBPLOT_EXACT[lower]
    # Oscillator indicators: use the part before the first underscore
    base = lower.split("_")[0]
    return _INDICATOR_SUBPLOT.get(base, base)


def empty_figure(message: str = "no data in range") -> go.Figure:
    """Return an empty figure with a centered annotation instead of raising."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    return fig


# ---------------------------------------------------------------------------
# FullData
# ---------------------------------------------------------------------------

class FullData:
    """Container for a wide DataFrame covering all timeframes.

    The wide DataFrame has columns named ``{tf}_{col}`` (e.g. ``15_close``,
    ``15_rsi_14``) and a DatetimeIndex.  ``DataViewer`` treats this as
    read-only.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


# ---------------------------------------------------------------------------
# DataViewer
# ---------------------------------------------------------------------------

class DataViewer:
    """Renders historical OHLCV data, indicators, and price levels.

    Args:
        full_data: Wide DataFrame container (read-only).
        tf: Primary timeframe in minutes (default 15).
    """

    _DEFAULT_INDICATORS = ["rsi_14", "cci_14"]

    # Toggleable price-axis overlay groups (skip-if-absent). Each group is one
    # dashboard checkbox; its value is the list of field base names it draws.
    # A Bollinger group ("bb_x_...") bundles the band pair/triple that share a
    # colour; "ema" bundles every EMA line; "sar" renders as markers. Order
    # drives checkbox order. Targets/stop-losses are always drawn (see below).
    _OVERLAY_GROUPS = {
        "bb_x_20_2": ["bb_upper_20_2", "bb_middle_20_2", "bb_lower_20_2"],
        "bb_x_10_15": ["bb_upper_10_15", "bb_lower_10_15"],
        "bb_x_20_3": ["bb_upper_20_3", "bb_lower_20_3"],
        "ema": ["ema_7", "ema_14", "ema_25", "ema_50", "ema_100"],
        "sar": ["sar_002_02"],
    }

    # Overlay groups unchecked on first load (still drawable via their toggle).
    _DEFAULT_OVERLAYS_OFF = {"bb_x_10_15", "bb_x_20_3", "sar"}

    # Target/stop-loss overlays — always drawn (at the chart tf), never toggleable.
    _TARGET_OVERLAYS = ["tgt_long", "sl_long", "tgt_short", "sl_short"]

    # Higher-TF target/SL overlay groups. The four target fields are computed
    # only for these TFs (applies_to [15,60,240,1440]); on a chart finer than 15
    # (tf 1/5) the native columns are absent, so each group re-draws its pinned
    # source TF's target lines onto the finer chart. Toggleable; gated to tf<15
    # (see _draw_price_overlays). group name -> source tf.
    _TARGET_TF_GROUPS = {
        "tgtsl_15m": 15, "tgtsl_60m": 60, "tgtsl_240m": 240, "tgtsl_1440m": 1440,
    }

    # Only the nearest (15m) group is on by first load; coarser TFs are one
    # click away so the finer chart is not swamped with 16 lines.
    _TARGET_TF_DEFAULT_OFF = {"tgtsl_60m", "tgtsl_240m", "tgtsl_1440m"}

    # Source TF -> (line dash, width): nearest solid+bold, farther dashed+thin,
    # so on a 1m chart the 15m levels read strongest and 1440m faintest.
    _TARGET_TF_STYLE = {
        15: (None, 2.0), 60: ("dash", 1.6), 240: ("dot", 1.3), 1440: ("dashdot", 1.0),
    }

    _OVERLAY_COLORS = {
        "bb_upper_20_2": "royalblue", "bb_middle_20_2": "royalblue",
        "bb_lower_20_2": "royalblue",
        "bb_upper_10_15": "darkorange", "bb_lower_10_15": "darkorange",
        "bb_upper_20_3": "purple", "bb_lower_20_3": "purple",
        "ema_7": "gold", "ema_14": "orange", "ema_25": "magenta",
        "ema_50": "teal", "ema_100": "brown",
        "sar_002_02": "black",
        # Targets profit-side greens, stop-losses loss-side reds; long darker,
        # short lighter so direction reads at a glance.
        "tgt_long": "darkgreen", "sl_long": "darkred",
        "tgt_short": "mediumseagreen", "sl_short": "indianred",
    }

    # Oscillator set for window figures (skip-if-absent). Each oscillator's
    # MA lines share the subplot of their source series; macd_hist_* renders
    # as bars. Order drives subplot order under price/volume.
    _OSCILLATORS = [
        "rsi_14", "rsi_ma8", "rsi_ma12", "rsi_ma24",
        "cci_14", "cci_14_ma_5",
        "macd_12_26_9", "macd_signal_12_26_9", "macd_hist_12_26_9",
        "macd_5_13_9", "macd_signal_5_13_9",
        "adx_14",
    ]

    # ATR / volatility fields. Routed by base name (_indicator_subplot): the
    # price-unit ATR and its MA share the "atr" subplot; the normalized %
    # variants share "natr". Order drives subplot order under the oscillators.
    _ATR_FIELDS = [
        "atr_14", "atr_14_ma_5",
        "natr_14", "natr_14_ma_5",
    ]

    _OSC_COLORS = {
        "rsi_14": "blue", "rsi_ma8": "orange", "rsi_ma12": "green",
        "rsi_ma24": "red",
        "cci_14": "blue", "cci_14_ma_5": "orange",
        "macd_12_26_9": "blue", "macd_signal_12_26_9": "orange",
        "macd_hist_12_26_9": "gray",
        "macd_5_13_9": "blue", "macd_signal_5_13_9": "orange",
        "adx_14": "purple",
        "atr_14": "blue", "atr_14_ma_5": "orange",
        "natr_14": "blue", "natr_14_ma_5": "orange",
        # Derivatives: raw diff and rm_20 stand out; std bands muted.
        "close_diff_prc": "blue", "close_diff_prc_rm_6": "orange",
        "close_diff_prc_rm_6_std_above": "lightgreen",
        "close_diff_prc_rm_6_std_below": "lightcoral",
        "high_diff_prc": "blue", "high_diff_prc_rm_6": "orange",
        "high_diff_prc_rm_6_std_above": "lightgreen",
        "high_diff_prc_rm_6_std_below": "lightcoral",
        "low_diff_prc": "blue", "low_diff_prc_rm_6": "orange",
        "low_diff_prc_rm_6_std_above": "lightgreen",
        "low_diff_prc_rm_6_std_below": "lightcoral",
        "rsi_ma8_diff": "orange", "rsi_ma12_diff": "green",
        "rsi_ma24_diff": "red",
    }

    # Subplot → fields view of the derivative groups (module-level routing).
    _DERIVATIVES = _DERIVATIVE_SUBPLOTS

    def __init__(self, full_data: FullData, tf: int = 15) -> None:
        self.full_data = full_data
        self.tf = tf
        self.renderer = ChartRenderer()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slice(self, start_idx: int, end_idx: int | None) -> pd.DataFrame:
        """Return ``df.iloc[start_idx:end_idx]`` (read-only view)."""
        return self.full_data.df.iloc[start_idx:end_idx]

    def _resolve_indicators(self, indicators: list[str] | None) -> list[str]:
        if indicators is None:
            return list(self._DEFAULT_INDICATORS)
        return list(indicators)

    def _dedup_tf_rows(self, df_slice: pd.DataFrame, tf: int) -> pd.DataFrame:
        """Keep one row per *tf*-minute period (the last — completed candle state),
        re-indexed to the period open timestamp.

        Higher TFs repeat values on every base-frequency row of the wide df;
        without dedup their candles would render once per base row. The index
        is floored to the period start so each candle plots at its open time;
        the resulting even spacing makes plotly render it across the full
        period width. Warm-up rows with NaN close for this TF are dropped too.
        """
        if df_slice.empty:
            return df_slice
        floored = df_slice.index.floor(f"{tf}min")
        keep = ~floored.duplicated(keep="last")
        df_slice = df_slice[keep].set_axis(floored[keep])
        close_col = f"{tf}_close"
        if close_col in df_slice.columns:
            df_slice = df_slice[df_slice[close_col].notna()]
        return df_slice

    def _subplot_list(
        self, indicators: list[str], show_nn: bool = False
    ) -> list[str]:
        """Build the ordered subplot name list for create_figure.

        ``show_nn`` appends a dedicated "nn" subplot for the timeframe-agnostic
        ``nn_res_*`` inference columns (drawn by ``_draw_nn_results``).
        """
        subplots = ["price", "volume"]
        seen: set[str] = set()
        for ind in indicators:
            sp = _indicator_subplot(ind)
            if sp is not None and sp not in seen:
                subplots.append(sp)
                seen.add(sp)
        if show_nn:
            subplots.append("nn")
        return subplots

    # NN inference result columns are timeframe-agnostic (no ``{tf}_`` prefix),
    # produced by NNOrchestrator.run_inference and left-joined by
    # ``data.join_nn_results``. Colour by semantic suffix so a direction head
    # reads at a glance (up green / neutral gray / down red); every other head
    # (label prob, regression value) gets its own distinct per-head colour
    # (``_nn_distinct_color``) so a many-head coexist view stays legible.
    _NN_RES_COLORS = {
        "prob_up": "green",
        "prob_neutral": "gray",
        "prob_down": "red",
    }

    @staticmethod
    def _nn_distinct_color(index: int) -> str:
        """Deterministic distinct colour for the *index*-th non-semantic NN head.

        Golden-angle hue stepping (0.618 turns per step) spreads consecutive
        heads far apart on the colour wheel, so a 32-head coexist view reads as
        distinct lines instead of one single-colour blob. Fixed saturation and
        value keep every line legible on the subplot.
        """
        hue = (index * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _nn_res_cols(df: pd.DataFrame) -> list[str]:
        """Sorted ``nn_res_*`` columns present in *df* (empty if none)."""
        return sorted(c for c in df.columns if str(c).startswith("nn_res_"))

    def _draw_nn_results(self, fig: go.Figure, df_slice: pd.DataFrame) -> None:
        """Draw ``nn_res_*`` inference outputs on the "nn" subplot. Skip-if-absent.

        One line per ``nn_res_*`` column over the displayed candles. Direction
        heads yield three probability lines in [0, 1]; label/regression heads
        yield their single value. Does nothing if the "nn" subplot is hidden
        or no ``nn_res_*`` column is present.
        """
        if "nn" not in getattr(fig, "_subplot_rows", {}):
            return
        times = list(df_slice.index)
        palette_i = 0
        for col in self._nn_res_cols(df_slice):
            color = None
            for suffix, c in self._NN_RES_COLORS.items():
                if str(col).endswith(suffix):
                    color = c
                    break
            if color is None:
                color = self._nn_distinct_color(palette_i)
                palette_i += 1
            self.renderer.draw_line(
                fig, "nn", times, list(df_slice[col]), label=str(col), color=color
            )

    def _draw_indicators(
        self,
        fig: go.Figure,
        df_slice: pd.DataFrame,
        indicators: list[str],
        tf: int | None = None,
    ) -> None:
        """Draw each indicator as a line on the appropriate subplot."""
        tf = self.tf if tf is None else tf
        times = list(df_slice.index)
        for ind in indicators:
            col = f"{tf}_{ind}"
            if col not in df_slice.columns:
                continue
            values = list(df_slice[col])
            sp = _indicator_subplot(ind)
            subplot = sp if sp is not None else "price"
            color = self._OSC_COLORS.get(ind, "blue")
            if ind.lower().startswith("macd_hist"):
                self.renderer.draw_bar(
                    fig, subplot, times, values, label=ind, color=color
                )
            else:
                self.renderer.draw_line(
                    fig, subplot, times, values, label=ind, color=color
                )

    def _build_figure(
        self,
        df_slice: pd.DataFrame,
        indicators: list[str],
        tf: int | None = None,
        range_row: bool = False,
        show_nn: bool = False,
    ) -> go.Figure:
        """Create and populate a figure (candles + indicators). Does not show/save.

        ``range_row=True`` adds the navigator row holding a second OHLC
        candlestick copy — the rangeslider preview then always shows price,
        never an indicator. ``show_nn=True`` adds the ``nn_res_*`` subplot.
        """
        tf = self.tf if tf is None else tf
        subplots = self._subplot_list(indicators, show_nn=show_nn)
        fig = self.renderer.create_figure(subplots, range_row=range_row)

        times = list(df_slice.index)
        opens = list(df_slice[f"{tf}_open"])
        highs = list(df_slice[f"{tf}_high"])
        lows = list(df_slice[f"{tf}_low"])
        closes = list(df_slice[f"{tf}_close"])

        self.renderer.draw_candles(fig, times, opens, highs, lows, closes)
        if range_row:
            self.renderer.draw_candles(
                fig, times, opens, highs, lows, closes, subplot="range"
            )

        # Draw volume if column exists
        vol_col = f"{tf}_volume"
        if vol_col in df_slice.columns:
            vol_vals = list(df_slice[vol_col])
            self.renderer.draw_bar(fig, "volume", times, vol_vals, label="volume")

        self._draw_indicators(fig, df_slice, indicators, tf=tf)
        if show_nn:
            self._draw_nn_results(fig, df_slice)

        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_tfs(self) -> list[int]:
        """Timeframes present in the wide df — every ``{tf}_close`` column, ascending."""
        tfs = []
        for col in self.full_data.df.columns:
            m = re.match(r"^(\d+)_close$", str(col))
            if m:
                tfs.append(int(m.group(1)))
        return sorted(tfs)

    def available_subplots(self) -> list[str]:
        """Oscillator/derivative subplot names available in the wide df.

        Order follows the default indicator set (``_OSCILLATORS`` then the
        derivative groups), deduped — indicators sharing a subplot (e.g.
        ``rsi_14``/``rsi_ma8``) yield one entry. Only subplots whose backing
        column exists for at least one available TF are included, so the
        list is TF-independent and usable as one control for all TFs.
        """
        defaults = self._OSCILLATORS + self._ATR_FIELDS + [
            f for fields in self._DERIVATIVES.values() for f in fields
        ]
        tfs = self.available_tfs()
        cols = set(self.full_data.df.columns)
        subplots: list[str] = []
        for ind in defaults:
            sp = _indicator_subplot(ind)
            if sp is None or sp in subplots:
                continue
            if any(f"{tf}_{ind}" in cols for tf in tfs):
                subplots.append(sp)
        # NN inference results (timeframe-agnostic) get their own toggleable subplot.
        if self._nn_res_cols(self.full_data.df):
            subplots.append("nn")
        return subplots

    def available_overlays(self) -> list[str]:
        """Price-overlay group names whose backing columns exist in the wide df.

        Order follows ``_OVERLAY_GROUPS``. A group is included when any of its
        fields exists for at least one available TF, so the list is
        TF-independent and usable as one control for all TFs.
        """
        cols = set(self.full_data.df.columns)
        tfs = self.available_tfs()
        groups = [
            group
            for group, fields in self._OVERLAY_GROUPS.items()
            if any(f"{tf}_{f}" in cols for tf in tfs for f in fields)
        ]
        # Higher-TF target/SL groups: included when the source TF's target
        # columns exist AND there is a chart finer than 15min to draw them on
        # (they are gated to tf<15). Without a sub-15 TF the toggle would be a
        # dead checkbox, so it is omitted.
        if any(tf < min(self._TARGET_TF_GROUPS.values()) for tf in tfs):
            groups += [
                group
                for group, src_tf in self._TARGET_TF_GROUPS.items()
                if any(f"{src_tf}_{f}" in cols for f in self._TARGET_OVERLAYS)
            ]
        return groups

    def default_overlays(self) -> list[str]:
        """Overlay groups checked on first load — available minus the off sets."""
        off = self._DEFAULT_OVERLAYS_OFF | self._TARGET_TF_DEFAULT_OFF
        return [g for g in self.available_overlays() if g not in off]

    def build_window_figure(
        self,
        start: pd.Timestamp,
        days: int,
        indicators: list[str] | None = None,
        tf: int | None = None,
        subplots: list[str] | None = None,
        show_actions: bool = False,
        show_labels: bool = False,
        overlays: list[str] | None = None,
    ) -> go.Figure:
        """Build a figure for a date window of the wide DataFrame.

        Slices by DatetimeIndex — ``[start, start + days)`` — unlike the
        iloc-based ``view_full()`` API. Returns the figure without showing it.
        An empty window returns an annotated empty figure, never raises.

        Args:
            start: Window start (inclusive).
            days: Window length in days from *start*.
            indicators: Same contract as ``view_full()``, except ``None``
                defaults to the full oscillator set (``_OSCILLATORS``) filtered
                to columns present for this TF.
            tf: Timeframe override for this figure; ``None`` uses ``self.tf``.
            subplots: Subplot names (see ``available_subplots()``) to keep;
                indicators routed to any other subplot are dropped. ``None``
                keeps everything; ``[]`` leaves only price and volume.
                Price-axis indicators are never filtered by this.
            overlays: Price-overlay group names (see ``available_overlays()``)
                to draw. ``None`` draws every group; ``[]`` draws none.
                Target/stop-loss overlays are always drawn regardless.
        """
        tf = self.tf if tf is None else int(tf)
        start = pd.Timestamp(start)
        df = self.full_data.df
        # The wide df index is tz-aware (UTC); date-picker values are naive.
        idx_tz = getattr(df.index, "tz", None)
        if idx_tz is not None and start.tz is None:
            start = start.tz_localize(idx_tz)
        end = start + pd.Timedelta(days=days)
        window = df[(df.index >= start) & (df.index < end)]
        df_slice = self._dedup_tf_rows(window, tf)
        if df_slice.empty:
            return empty_figure()
        if indicators is None:
            defaults = self._OSCILLATORS + self._ATR_FIELDS + [
                f for fields in self._DERIVATIVES.values() for f in fields
            ]
            indicators = [
                i for i in defaults if f"{tf}_{i}" in df_slice.columns
            ]
        else:
            indicators = list(indicators)
        if subplots is not None:
            allowed = set(subplots)
            indicators = [
                i for i in indicators
                if _indicator_subplot(i) is None or _indicator_subplot(i) in allowed
            ]
        # Show the nn_res_* subplot when the columns are present and "nn" is
        # selected (or no explicit subplot selection was made).
        show_nn = bool(self._nn_res_cols(df_slice)) and (
            subplots is None or "nn" in subplots
        )
        fig = self._build_figure(
            df_slice, indicators, tf=tf, range_row=True, show_nn=show_nn
        )
        self._draw_price_overlays(fig, df_slice, tf, overlays)
        self._draw_rsi_class_markers(fig, window, tf)
        if show_labels:
            self._draw_label_markers(fig, window, tf)
        if show_actions:
            self._draw_action_markers(fig, window)
        self._draw_zero_lines(fig)
        n_rows = len(getattr(fig, "_subplot_rows", {})) or 1
        # 264 = 220 * 1.2: total grows with the extra weight of the non-price
        # rows (their ratios went 1.5x) so the price row keeps its pixel size.
        fig.update_layout(height=max(600, 264 * n_rows))
        return fig

    # Class markers on the rsi subplot: field → (marker symbol, size,
    # class value → colour). move_class buckets rsi_ma8_diff momentum (-2..2),
    # zone_class buckets the rsi_ma8 level (0..4) — five spec tiers each
    # (see indicators/library/classification.py); both are drawn at the
    # rsi_ma8 y-value, the diamond ringing the dot.
    _RSI_CLASS_MARKERS = {
        "move_class": ("circle", 6, {
            -2: "red", -1: "orange", 0: "silver",
            1: "lightgreen", 2: "green",
        }),
        "zone_class": ("diamond-open", 11, {
            0: "red", 1: "orange", 2: "silver",
            3: "lightgreen", 4: "green",
        }),
    }

    # The rsi_ma line both class fields are computed from.
    _RSI_CLASS_SOURCE = "rsi_ma8"

    def _draw_rsi_class_markers(
        self,
        fig: go.Figure,
        window: pd.DataFrame,
        tf: int,
    ) -> None:
        """Draw move_class/zone_class markers on the rsi_ma8 line. Skip-if-absent.

        One marker trace per (field, class value), plotted per-minute on the
        raw (un-deduped) window: the class fields and rsi_ma8 recompute on
        every base-frequency row, so a marker is drawn at each minute's
        rsi_ma8 y-value rather than once per displayed candle — exposing
        intra-candle class evolution. Skipped when the rsi subplot is hidden
        or the source/class columns are absent.
        """
        rows = getattr(fig, "_subplot_rows", {})
        if "rsi" not in rows:
            return
        src_col = f"{tf}_{self._RSI_CLASS_SOURCE}"
        if src_col not in window.columns:
            return
        src = window[src_col]
        for field, (symbol, size, colors) in self._RSI_CLASS_MARKERS.items():
            col = f"{tf}_{field}"
            if col not in window.columns:
                continue
            cls = window[col]
            for value, color in colors.items():
                mask = cls == value
                if not mask.any():
                    continue
                self.renderer.draw_marker(
                    fig,
                    list(window.index[mask]),
                    list(src[mask]),
                    marker_symbol=symbol,
                    color=color,
                    label=f"{field}={value}",
                    subplot="rsi",
                    size=size,
                )

    # Profit-label column prefixes → (side, marker colour). Strict variants
    # darker than their plain counterpart. Matches the column names written
    # by indicators/labels.py (add_profit_labels / add_profit_strict_labels).
    _LABEL_PREFIXES = {
        "plong": ("long", "limegreen"),
        "pslong": ("long", "green"),
        "pshort": ("short", "orange"),
        "psshort": ("short", "red"),
    }

    # Marker offset from the candle extreme, as a fraction of price, stepped
    # per variant so several label sets stack instead of overlapping.
    _LABEL_OFFSET_STEP = 0.002

    _LABEL_COL_RE = re.compile(r"^(\d+)_(plong|pslong|pshort|psshort)_.+$")

    def _draw_label_markers(
        self,
        fig: go.Figure,
        window: pd.DataFrame,
        tf: int,
    ) -> None:
        """Draw profit-label markers of every TF on the price subplot.

        *window* is the raw (un-deduped) date-window slice. A profit label
        holds constant across its whole label-TF candle, so every
        base-frequency (per-minute) row of a flagged candle gets a marker — a
        band spanning the candle width rather than one marker at its open. One
        trace per column, markers on rows where the label is 1: longs as
        triangles-up below the label-TF candle low, shorts as triangles-down
        above its high, each variant on its own offset step. Trace name = full
        column name. Skip-if-absent.
        """
        side_counts = {"long": 0, "short": 0}
        for col in window.columns:
            m = self._LABEL_COL_RE.match(str(col))
            if not m:
                continue
            label_tf, prefix = int(m.group(1)), m.group(2)
            side, color = self._LABEL_PREFIXES[prefix]
            step = self._LABEL_OFFSET_STEP * (side_counts[side] + 1)
            side_counts[side] += 1
            # Per-minute: keep every base-frequency row where the label is set.
            mask = window[col] == 1
            if not mask.any():
                continue
            if side == "long":
                ys = window.loc[mask, f"{label_tf}_low"] * (1.0 - step)
                symbol = "triangle-up"
            else:
                ys = window.loc[mask, f"{label_tf}_high"] * (1.0 + step)
                symbol = "triangle-down"
            self.renderer.draw_marker(
                fig,
                list(window.index[mask]),
                list(ys),
                marker_symbol=symbol,
                color=color,
                label=str(col),
                subplot="price",
            )

    # Action-marker styling: event → (symbol, colour). OPEN direction is
    # resolved from the action's position_type. SIGNAL_FIRED is never drawn.
    _ACTION_OPEN = {
        "POSITION_TYPE_LONG": ("triangle-up", "limegreen"),
        "POSITION_TYPE_SHORT": ("triangle-down", "orange"),
    }
    _ACTION_OFFSET = 0.003

    def _load_latest_actions(self) -> list[dict]:
        """Load the newest simulation's actions.jsonl. Skip-if-absent → []."""
        import json

        try:
            from helpers import latest_simulation_folder
            folder = latest_simulation_folder()
            if not folder:
                return []
            path = folder + "actions.jsonl"
            with open(path) as fh:
                return [json.loads(line) for line in fh if line.strip()]
        except Exception as exc:  # noqa: BLE001 — viewer must never crash
            print(f"[DataViewer] _load_latest_actions skipped: {exc}")
            return []

    def _draw_action_markers(self, fig: go.Figure, window: pd.DataFrame) -> None:
        """Draw latest-simulation OPEN/CLOSE/STOP_LOSS markers on the price subplot.

        x = action timestamp (epoch s → tz-aware UTC), y = executed_price
        (falling back to target_price). Only actions inside the visible window
        are drawn. SIGNAL_FIRED actions are intentionally skipped (too noisy).
        Skip-if-absent: no simulation / empty window renders unchanged.
        """
        actions = self._load_latest_actions()
        if not actions or window.empty:
            return

        idx_tz = getattr(window.index, "tz", None)
        start, end = window.index.min(), window.index.max()

        # Bucket markers by (symbol, colour, label) so each is one trace.
        # Each action yields one or more (symbol, color, label, y) points; an
        # OPEN yields three: entry, take-profit target, and stop-loss.
        buckets: dict[tuple, tuple[list, list, list]] = {}
        for a in actions:
            event = a.get("event")
            ts = pd.Timestamp(a["timestamp"], unit="s")
            if idx_tz is not None:
                ts = ts.tz_localize(idx_tz)
            if ts < start or ts > end:
                continue

            hover = self._action_hover(a, ts)
            entry = a.get("executed_price") or a.get("target_price") or 0.0
            points: list[tuple] = []  # (symbol, color, label, y, hover)

            if event == "OPEN":
                spec = self._ACTION_OPEN.get(a.get("position_type"))
                if spec is None:
                    continue
                symbol, color = spec
                side = a.get("position_type", "").replace("POSITION_TYPE_", "").lower()
                # Entry marker sits exactly at the execution price.
                points.append((symbol, color, f"OPEN {side}", entry, hover))
                tp = a.get("target_price") or 0.0
                sl = a.get("stop_loss_price") or 0.0
                if tp > 0:
                    points.append(("triangle-right", "seagreen", "target", tp, hover))
                if sl > 0:
                    points.append(("triangle-right", "crimson", "open_stop", sl, hover))
            elif event == "CLOSE":
                points.append(("x", "royalblue", "CLOSE", entry, hover))
            elif event == "STOP_LOSS":
                points.append(("x", "red", "STOP_LOSS", entry, hover))
            else:
                continue  # SIGNAL_FIRED / MOVE_STOP_LOSS not plotted

            for symbol, color, label, y, hov in points:
                xs, ys, hs = buckets.setdefault((symbol, color, label), ([], [], []))
                xs.append(ts)
                ys.append(y)
                hs.append(hov)

        for (symbol, color, label), (xs, ys, hs) in buckets.items():
            if not xs:
                continue
            self.renderer.draw_marker(
                fig, xs, ys,
                marker_symbol=symbol, color=color, label=label,
                subplot="price", size=11, hovertext=hs,
            )

    @staticmethod
    def _action_hover(a: dict, ts: pd.Timestamp) -> str:
        """Multi-line hover string exposing every meaningful field of an action."""
        lines = [
            f"<b>{a.get('event', '')}</b>",
            f"time: {ts:%Y-%m-%d %H:%M}",
            f"action: {a.get('action_type', '')}",
            f"position: {str(a.get('position_type', '')).replace('POSITION_TYPE_', '')}",
            f"target: {a.get('target_price', 0.0):.4f}",
            f"executed: {a.get('executed_price', 0.0):.4f}",
            f"stop_loss: {a.get('stop_loss_price', 0.0):.4f}",
        ]
        if a.get("event") in ("CLOSE", "STOP_LOSS"):
            lines.append(f"revenue: {a.get('revenue_pct', 0.0) * 100:.2f}% "
                         f"({a.get('revenue_abs', 0.0):.2f})")
            lines.append(f"stopped_out: {a.get('was_stop_loss', False)}")
        return "<br>".join(lines)

    def _draw_zero_lines(self, fig: go.Figure) -> None:
        """Add a zero reference line on every derivative subplot.

        Diff series oscillate around 0. Uses add_shape with per-row yref —
        same plotly 6 compatibility approach as ChartRenderer.draw_level.
        """
        for sp, row in getattr(fig, "_subplot_rows", {}).items():
            if sp not in self._DERIVATIVES:
                continue
            yref = "y" if row == 1 else f"y{row}"
            fig.add_shape(
                type="line",
                x0=0, x1=1, y0=0, y1=0,
                xref="paper", yref=yref,
                line={"color": "gray", "width": 1},
            )

    def _draw_price_overlays(
        self,
        fig: go.Figure,
        df_slice: pd.DataFrame,
        tf: int,
        overlays: list[str] | None = None,
    ) -> None:
        """Draw Bollinger/EMA/SAR overlays on the price subplot. Skip-if-absent.

        *overlays* selects which toggleable groups to draw (``None`` = all,
        ``[]`` = none); target/stop-loss overlays are always drawn.
        """
        if overlays is None:
            groups = list(self._OVERLAY_GROUPS)
        else:
            groups = [g for g in overlays if g in self._OVERLAY_GROUPS]
        names = [f for g in groups for f in self._OVERLAY_GROUPS[g]]
        names += self._TARGET_OVERLAYS
        times = list(df_slice.index)
        for name in names:
            col = f"{tf}_{name}"
            if col not in df_slice.columns:
                continue
            values = list(df_slice[col])
            color = self._OVERLAY_COLORS.get(name, "gray")
            if name.startswith("sar"):
                self.renderer.draw_marker(
                    fig, times, values,
                    marker_symbol="circle", color=color, label=name,
                )
            else:
                self.renderer.draw_line(
                    fig, "price", times, values, label=name, color=color,
                )

        self._draw_higher_tf_targets(fig, df_slice, tf, overlays, times)

    def _draw_higher_tf_targets(
        self,
        fig: go.Figure,
        df_slice: pd.DataFrame,
        tf: int,
        overlays: list[str] | None,
        times: list,
    ) -> None:
        """Overlay higher-TF target/SL step lines on charts finer than 15min.

        Only fires for tf < 15 (tf 1/5); tf>=15 keeps its native always-on
        target rendering untouched. Each enabled ``tgtsl_{H}m`` group draws its
        pinned source TF's four target columns (already flat-per-row in the wide
        df) as TF-suffixed lines styled by source TF. ``overlays=None`` draws all
        groups; a list restricts to the named ones; ``[]`` draws none.
        Skip-if-absent per column.
        """
        if tf >= min(self._TARGET_TF_GROUPS.values()):
            return
        # Draw descending TF so the nearest (15m) sits on top.
        for group, src_tf in sorted(
            self._TARGET_TF_GROUPS.items(), key=lambda kv: -kv[1]
        ):
            if overlays is not None and group not in overlays:
                continue
            dash, width = self._TARGET_TF_STYLE[src_tf]
            for name in self._TARGET_OVERLAYS:
                col = f"{src_tf}_{name}"
                if col not in df_slice.columns:
                    continue
                self.renderer.draw_line(
                    fig, "price", times, list(df_slice[col]),
                    label=f"{name}·{src_tf}m",
                    color=self._OVERLAY_COLORS.get(name, "gray"),
                    dash=dash, width=width,
                )

    def view_full(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
        indicators: list[str] | None = None,
    ) -> None:
        """Render historical OHLCV + indicators and open in a browser.

        Args:
            start_idx: Start row (inclusive, iloc-based).
            end_idx: End row (exclusive, iloc-based). None means end of data.
            indicators: List of indicator column base names (without TF prefix),
                e.g. ``["rsi_14", "cci_14"]``. ``None`` defaults to RSI-14 and CCI-14.
        """
        indicators = self._resolve_indicators(indicators)
        df_slice = self._slice(start_idx, end_idx)
        if df_slice.empty:
            return
        fig = self._build_figure(df_slice, indicators)
        fig.show()

    def view_full_levels(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
        levels: Optional[Any] = None,
        indicators: list[str] | None = None,
    ) -> None:
        """Render OHLCV + indicators + price-level overlays.

        Args:
            start_idx: Start row (inclusive, iloc-based).
            end_idx: End row (exclusive, iloc-based). None means end of data.
            levels: Levels object with ``get_active_levels(level_type, cur_time) →
                list[(price, label)]``. If ``None``, behaves like ``view_full()``.
            indicators: List of indicator column base names (without TF prefix),
                e.g. ``["rsi_14", "cci_14"]``. ``None`` defaults to RSI-14 and CCI-14.
        """
        indicators = self._resolve_indicators(indicators)
        df_slice = self._slice(start_idx, end_idx)
        fig = self._build_figure(df_slice, indicators)

        if levels is not None:
            if df_slice.empty:
                fig.show()
                return

            last_row = df_slice.index[-1]
            cur_time = int(pd.Timestamp(last_row).timestamp())

            level_types = [
                v for k, v in vars(constants).items()
                if k.startswith("LEVEL_TYPE_")
            ]
            for lt in level_types:
                active = levels.get_active_levels(lt, cur_time)
                for price, label in active:
                    self.renderer.draw_level(fig, price, label)

        fig.show()

    def save_chart(
        self,
        path: str,
        start_idx: int = 0,
        end_idx: int | None = None,
        indicators: list[str] | None = None,
    ) -> None:
        """Build the default chart (RSI + CCI) and save to a PNG file.

        Args:
            path: Destination file path (PNG).
            start_idx: Start row (inclusive, iloc-based).
            end_idx: End row (exclusive, iloc-based). None means end of data.
            indicators: List of indicator column base names (without TF prefix),
                e.g. ``["rsi_14", "cci_14"]``. ``None`` defaults to RSI-14 and CCI-14.
        """
        indicators = self._resolve_indicators(indicators)
        df_slice = self._slice(start_idx, end_idx)
        if df_slice.empty:
            return
        fig = self._build_figure(df_slice, indicators)
        self.renderer.save(fig, path)
