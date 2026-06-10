"""LiveDashboard — Dash app for real-time trading state visualisation.

Polls ``shared/live_state.pkl`` every 1 000 ms and renders:
- Streaming price candles (rolling window)
- Current position overlay (green=long, red=short, gray=flat)
- Stop-loss level line
- Trade markers (buy / sell / stop_loss)
- Running P&L curve
"""

from __future__ import annotations

import pickle
from collections import deque
from typing import Any

import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from frontend.chart_renderer import ChartRenderer


class LiveDashboard:
    """Renders live trading state: price candles, position, stop-loss, trade markers, P&L."""

    def __init__(self, window_size: int = 200, tf: int = 15) -> None:
        self.window_size = window_size
        self.tf = tf
        self.renderer = ChartRenderer(title="Live Trading Dashboard")
        self.candle_buffer: deque = deque(maxlen=window_size)
        self.trade_markers: list[dict] = []
        self._state_path = "shared/live_state.pkl"
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state_path: str = "shared/live_state.pkl") -> None:
        """Start the Dash app, polling *state_path* every 1 000 ms. Blocking."""
        self._state_path = state_path
        self._app.run(debug=False)

    def _update_charts(self, state: dict) -> list[go.Figure]:
        """Return a list of Plotly figures derived from *state*.

        Never raises — missing or unexpected keys produce empty figures.
        """
        try:
            return self._build_figures(state)
        except Exception as e:
            print(f"[LiveDashboard] _update_charts error: {e}")
            return [go.Figure(), go.Figure()]

    # ------------------------------------------------------------------
    # Internal: figure construction
    # ------------------------------------------------------------------

    def _build_figures(self, state: dict) -> list[go.Figure]:
        candles = list(state.get("candles") or [])
        position = state.get("position") or {}
        raw_markers = list(state.get("trade_markers") or [])
        revenue_history = list(state.get("revenue_history") or [])

        # Sync candle_buffer and trade_markers from state.
        self._ingest_candles(candles)
        self.trade_markers = raw_markers

        # Window of candles to display.
        display_candles = list(self.candle_buffer)

        price_fig = self._build_price_figure(display_candles, position, self.trade_markers)
        pnl_fig = self._build_pnl_figure(revenue_history)

        return [price_fig, pnl_fig]

    def _ingest_candles(self, candles: list[dict]) -> None:
        """Extend the rolling buffer with new candles, capped at window_size."""
        for c in candles:
            self.candle_buffer.append(c)

    def _build_price_figure(
        self,
        candles: list[dict],
        position: dict,
        markers: list[dict],
    ) -> go.Figure:
        fig = self.renderer.create_figure(["price"])

        if candles:
            times = [c["time"] for c in candles]
            opens = [c["open"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]
            closes = [c["close"] for c in candles]
            self.renderer.draw_candles(fig, times, opens, highs, lows, closes)

        # Position overlay via add_vrect.
        is_open: bool = position.get("is_open", False)
        direction: str = position.get("direction", "flat")
        stop_loss = position.get("stop_loss")

        if is_open:
            if direction == "long":
                fillcolor = "rgba(0,128,0,0.15)"
            elif direction == "short":
                fillcolor = "rgba(255,0,0,0.15)"
            else:
                fillcolor = "rgba(128,128,128,0.15)"
        else:
            # Flat: draw gray vrect to indicate no position.
            fillcolor = "rgba(128,128,128,0.10)"

        fig.add_vrect(
            x0=0,
            x1=1,
            xref="paper",
            fillcolor=fillcolor,
            layer="below",
            line_width=0,
        )

        # Stop-loss level — only when position is open and stop_loss is set.
        if is_open and stop_loss is not None:
            self.renderer.draw_level(fig, price=stop_loss, label="SL", color="orange")

        # Trade markers.
        for m in markers:
            mtype = m.get("type", "")
            mtime = m.get("time")
            mprice = m.get("price")
            if mtime is None or mprice is None:
                continue
            if mtype == "buy":
                self.renderer.draw_marker(
                    fig,
                    times=[mtime],
                    prices=[mprice],
                    marker_symbol="triangle-up",
                    color="green",
                    label="Buy",
                )
            elif mtype == "sell":
                self.renderer.draw_marker(
                    fig,
                    times=[mtime],
                    prices=[mprice],
                    marker_symbol="triangle-down",
                    color="red",
                    label="Sell",
                )
            elif mtype == "stop_loss":
                self.renderer.draw_marker(
                    fig,
                    times=[mtime],
                    prices=[mprice],
                    marker_symbol="x",
                    color="orange",
                    label="Stop Loss",
                )

        return fig

    def _build_pnl_figure(self, revenue_history: list) -> go.Figure:
        """Build a running P&L % line figure."""
        fig = go.Figure()
        if not revenue_history:
            fig.update_layout(title_text="Running P&L")
            return fig

        pct_values = [
            r[0]
            for r in revenue_history
            if isinstance(r, (list, tuple)) and len(r) >= 1
        ]
        indices = list(range(1, len(pct_values) + 1))

        fig.add_trace(
            go.Scatter(
                x=indices,
                y=pct_values,
                mode="lines",
                name="P&L %",
                line={"color": "steelblue"},
            )
        )
        fig.update_layout(
            title_text="Running P&L (%)",
            xaxis_title="Trade #",
            yaxis_title="Return (%)",
        )
        return fig

    # ------------------------------------------------------------------
    # Dash app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(__name__)

        app.layout = html.Div(
            [
                html.H1("Live Trading Dashboard"),
                dcc.Interval(id="interval", interval=1000, n_intervals=0),
                dcc.Graph(id="chart-price"),
                dcc.Graph(id="chart-pnl"),
            ]
        )

        @app.callback(
            [Output("chart-price", "figure"), Output("chart-pnl", "figure")],
            [Input("interval", "n_intervals")],
        )
        def refresh(_n: int):  # type: ignore[return]
            state = self._load_state()
            figs = self._update_charts(state)
            # Ensure exactly two figures.
            if len(figs) < 2:
                figs = figs + [go.Figure()] * (2 - len(figs))
            return figs[0], figs[1]

        return app

    def _load_state(self) -> dict[str, Any]:
        """Read *_state_path* atomically; return empty dict on any error."""
        path = getattr(self, "_state_path", "shared/live_state.pkl")
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
