"""ChartRenderer — shared chart layout and primitive drawing for all viewer subclasses."""

from plotly.subplots import make_subplots
import plotly.graph_objects as go


class ChartRenderer:
    """Base class providing shared chart layout, OHLCV candlestick rendering,
    indicator overlay helpers, and marker placement API."""

    def __init__(self, title: str = "") -> None:
        self.title = title

    def create_figure(self, subplots: list[str]) -> go.Figure:
        """Create a multi-subplot figure.

        The "price" subplot receives 60% of vertical space; all other subplots
        share the remainder equally. `fig._subplot_rows` maps subplot name →
        1-indexed row number.
        """
        n = len(subplots)

        # Build row heights: price gets 0.60, others share 0.40 equally.
        if n == 1:
            row_heights = [1.0]
        else:
            price_share = 0.60
            other_share = (1.0 - price_share) / (n - 1)
            row_heights = [
                price_share if name == "price" else other_share
                for name in subplots
            ]

        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03,
            subplot_titles=subplots,
        )

        fig.update_layout(title_text=self.title)

        # Attach subplot name → row mapping as a plain dict attribute.
        fig._subplot_rows = {name: idx + 1 for idx, name in enumerate(subplots)}

        return fig

    def draw_candles(
        self,
        fig: go.Figure,
        times: list,
        opens: list,
        highs: list,
        lows: list,
        closes: list,
    ) -> None:
        """Add a Candlestick trace to the price subplot."""
        row = fig._subplot_rows["price"]
        fig.add_trace(
            go.Candlestick(
                x=times,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                increasing_line_color="green",
                decreasing_line_color="red",
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

    def draw_marker(
        self,
        fig: go.Figure,
        times: list,
        prices: list,
        marker_symbol: str,
        color: str,
        label: str = "",
    ) -> None:
        """Add a marker (Scatter) trace to the price subplot."""
        row = fig._subplot_rows["price"]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=prices,
                mode="markers",
                name=label,
                marker={"symbol": marker_symbol, "color": color},
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
