"""DataViewer — renders historical OHLCV + indicators + levels for analysis."""

from __future__ import annotations

import pandas as pd

from frontend.chart_renderer import ChartRenderer

# ---------------------------------------------------------------------------
# Indicator subplot routing
# ---------------------------------------------------------------------------

# Indicators whose name starts with one of these prefixes go on the price axis.
_PRICE_AXIS_PREFIXES = ("ema", "sma", "bb", "vwap")

# Maps indicator base name (without _NN suffix) → subplot name.
# Indicators not found here AND not on the price axis get a subplot named
# after their base name (e.g. "rsi_14" → subplot "rsi").
_INDICATOR_SUBPLOT = {
    "rsi": "rsi",
    "cci": "cci",
    "macd": "macd",
    "stoch": "stoch",
}


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
    # Oscillator indicators: use the part before the first underscore
    base = lower.split("_")[0]
    return _INDICATOR_SUBPLOT.get(base, base)


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

    def _subplot_list(self, indicators: list[str]) -> list[str]:
        """Build the ordered subplot name list for create_figure."""
        subplots = ["price", "volume"]
        seen: set[str] = set()
        for ind in indicators:
            sp = _indicator_subplot(ind)
            if sp is not None and sp not in seen:
                subplots.append(sp)
                seen.add(sp)
        return subplots

    def _draw_indicators(
        self,
        fig: object,
        df_slice: pd.DataFrame,
        indicators: list[str],
    ) -> None:
        """Draw each indicator as a line on the appropriate subplot."""
        times = list(df_slice.index)
        for ind in indicators:
            col = f"{self.tf}_{ind}"
            if col not in df_slice.columns:
                continue
            values = list(df_slice[col])
            sp = _indicator_subplot(ind)
            subplot = sp if sp is not None else "price"
            self.renderer.draw_line(fig, subplot, times, values, label=ind)

    def _build_figure(
        self,
        df_slice: pd.DataFrame,
        indicators: list[str],
    ) -> object:
        """Create and populate a figure (candles + indicators). Does not show/save."""
        subplots = self._subplot_list(indicators)
        fig = self.renderer.create_figure(subplots)

        times = list(df_slice.index)
        opens = list(df_slice[f"{self.tf}_open"])
        highs = list(df_slice[f"{self.tf}_high"])
        lows = list(df_slice[f"{self.tf}_low"])
        closes = list(df_slice[f"{self.tf}_close"])

        self.renderer.draw_candles(fig, times, opens, highs, lows, closes)

        # Draw volume if column exists
        vol_col = f"{self.tf}_volume"
        if vol_col in df_slice.columns:
            vol_vals = list(df_slice[vol_col])
            self.renderer.draw_line(fig, "volume", times, vol_vals, label="volume")

        self._draw_indicators(fig, df_slice, indicators)

        return fig

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        fig = self._build_figure(df_slice, indicators)
        fig.show()

    def view_full_levels(
        self,
        start_idx: int = 0,
        end_idx: int | None = None,
        levels=None,
    ) -> None:
        """Render OHLCV + indicators + price-level overlays.

        Args:
            start_idx: Start row (inclusive, iloc-based).
            end_idx: End row (exclusive, iloc-based). None means end of data.
            levels: Levels object with ``get_active_levels(level_type, cur_time) →
                list[(price, label)]``. If ``None``, behaves like ``view_full()``.
        """
        indicators = self._resolve_indicators(None)
        df_slice = self._slice(start_idx, end_idx)
        fig = self._build_figure(df_slice, indicators)

        if levels is not None:
            import constants

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
    ) -> None:
        """Build the default chart (RSI + CCI) and save to a PNG file.

        Args:
            path: Destination file path (PNG).
            start_idx: Start row (inclusive, iloc-based).
            end_idx: End row (exclusive, iloc-based). None means end of data.
        """
        indicators = self._resolve_indicators(None)
        df_slice = self._slice(start_idx, end_idx)
        fig = self._build_figure(df_slice, indicators)
        self.renderer.save(fig, path)
