"""ChartRenderer — shared chart layout and primitive drawing for all viewer subclasses."""

from plotly.subplots import make_subplots
import plotly.graph_objects as go


class ChartRenderer:
    """Base class providing shared chart layout, OHLCV candlestick rendering,
    indicator overlay helpers, and marker placement API."""

    def __init__(self, title: str = "") -> None:
        self.title = title

    def create_figure(
        self, subplots: list[str], range_row: bool = False
    ) -> go.Figure:
        """Create a multi-subplot figure.

        Row height ratios: "price" weighs 0.60, every other subplot weighs
        0.90/(n-1) — 1.5× the previous 0.60/(n-1) split (plotly normalizes
        the ratios). `fig._subplot_rows` maps subplot name → 1-indexed row
        number.

        The candlestick auto-rangeslider is disabled on every axis and a slim
        one (thickness 0.05 — a third of the plotly 0.15 default) is enabled
        on the bottom row, so it can never overlap subplot rows.

        With ``range_row=True`` an extra short, untitled "range" subplot is
        appended after the regular rows and the slim rangeslider attaches to
        it instead. Its y-axis is hidden — the caller draws the navigator
        content (OHLC candles) there, so the slider preview always shows
        price regardless of which indicator subplot ends up last.
        """
        n = len(subplots)

        # Relative row weights: price 0.60, others 1.5 * 0.60/(n-1) each.
        if n == 1:
            row_heights = [1.0]
        else:
            price_share = 0.60
            other_share = 1.5 * (1.0 - price_share) / (n - 1)
            row_heights = [
                price_share if name == "price" else other_share
                for name in subplots
            ]

        names = list(subplots)
        if range_row:
            names.append("range")
            row_heights.append(0.15)

        fig = make_subplots(
            rows=len(names),
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03,
            # No title over the navigator row.
            subplot_titles=[n if n != "range" else "" for n in names],
        )

        fig.update_layout(title_text=self.title)

        # Rangeslider: off everywhere, slim one on the bottom row only.
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05,
            row=len(names),
            col=1,
        )
        if range_row:
            fig.update_yaxes(visible=False, row=len(names), col=1)

        # Attach subplot name → row mapping as a plain dict attribute.
        fig._subplot_rows = {name: idx + 1 for idx, name in enumerate(names)}

        return fig

    def draw_candles(
        self,
        fig: go.Figure,
        times: list,
        opens: list,
        highs: list,
        lows: list,
        closes: list,
        subplot: str = "price",
    ) -> None:
        """Add a Candlestick trace to the given subplot (price by default)."""
        row = fig._subplot_rows[subplot]
        fig.add_trace(
            go.Candlestick(
                x=times,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                increasing_line_color="green",
                decreasing_line_color="red",
                showlegend=subplot == "price",
            ),
            row=row,
            col=1,
        )

    def draw_line(
        self,
        fig: go.Figure,
        subplot: str,
        times: list,
        values: list,
        label: str,
        color: str = "blue",
    ) -> None:
        """Add a line (Scatter) trace to the specified subplot."""
        row = fig._subplot_rows[subplot]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=label,
                line={"color": color},
            ),
            row=row,
            col=1,
        )

    def draw_bar(
        self,
        fig: go.Figure,
        subplot: str,
        times: list,
        values: list,
        label: str,
        color: str = "gray",
    ) -> None:
        """Add a bar trace to the specified subplot."""
        row = fig._subplot_rows[subplot]
        fig.add_trace(
            go.Bar(
                x=times,
                y=values,
                name=label,
                marker_color=color,
            ),
            row=row,
            col=1,
        )

    def draw_marker(
        self,
        fig: go.Figure,
        times: list,
        prices: list,
        marker_symbol: str,
        color: str,
        label: str = "",
        subplot: str = "price",
        size: int | None = None,
    ) -> None:
        """Add a marker (Scatter) trace to the given subplot (price by default)."""
        row = fig._subplot_rows[subplot]
        marker = {"symbol": marker_symbol, "color": color}
        if size is not None:
            marker["size"] = size
        fig.add_trace(
            go.Scatter(
                x=times,
                y=prices,
                mode="markers",
                name=label,
                marker=marker,
            ),
            row=row,
            col=1,
        )

    def draw_level(
        self,
        fig: go.Figure,
        price: float,
        label: str,
        color: str = "gray",
    ) -> None:
        """Add a horizontal line at the given price on the price subplot.

        Uses add_shape (xref='paper', yref per price row) for full plotly 6
        compatibility — add_hline does not populate layout.shapes on subplot
        figures in plotly 6+.
        """
        row = fig._subplot_rows["price"]
        yref = "y" if row == 1 else f"y{row}"
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=price,
            y1=price,
            xref="paper",
            yref=yref,
            line={"color": color},
            label={"text": label},
        )

    def save(self, fig: go.Figure, path: str) -> None:
        """Save the figure as a PNG image (requires kaleido)."""
        fig.write_image(path)
