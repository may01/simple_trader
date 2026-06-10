"""TrainingDashboard — Dash app for monitoring training pipeline progress."""

from __future__ import annotations

import pickle
from typing import Any

import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


class TrainingDashboard:
    """Renders training pipeline progress: loss curves, P&L distribution, win rate."""

    def __init__(self) -> None:
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state_path: str = "shared/training_state.pkl") -> None:
        """Start the Dash app, polling *state_path* every 2000 ms.  Blocking."""
        self._state_path = state_path
        self._app.run(debug=False)

    def _update_charts(self, state: dict) -> list[go.Figure]:
        """Return a list of Plotly figures derived from *state*.

        Never raises — missing or unexpected keys produce empty figures.
        """
        try:
            return self._build_figures(state)
        except Exception:
            return [go.Figure(), go.Figure()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_figures(self, state: dict) -> list[go.Figure]:
        phase = state.get("phase", "")

        if phase == "nn_train":
            return self._nn_train_figures(state)
        elif phase == "simulate":
            return self._simulate_figures(state)
        else:
            # Unknown or missing phase → return two empty placeholder figures.
            return [go.Figure(), go.Figure()]

    def _nn_train_figures(self, state: dict) -> list[go.Figure]:
        """Build loss-curve figure for the nn_train phase."""
        loss_history = state.get("loss_history") or []
        val_loss_history = state.get("val_loss_history") or []

        epochs = list(range(1, len(loss_history) + 1))
        val_epochs = list(range(1, len(val_loss_history) + 1))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=list(loss_history),
                mode="lines",
                name="Train Loss",
                line={"color": "royalblue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=val_epochs,
                y=list(val_loss_history),
                mode="lines",
                name="Val Loss",
                line={"color": "orange"},
            )
        )
        fig.update_layout(
            title_text="Loss Curve",
            xaxis_title="Epoch",
            yaxis_title="Loss",
        )

        # Second figure is empty during nn_train (placeholder for P&L slot).
        return [fig, go.Figure()]

    def _simulate_figures(self, state: dict) -> list[go.Figure]:
        """Build P&L histogram + win-rate annotation figure for simulate phase."""
        revenue_history: list[tuple] = state.get("revenue_history") or []
        win_rate = state.get("win_rate")
        total_trades: int = state.get("total_trades", 0)

        pct_values = [r[0] for r in revenue_history] if revenue_history else []

        # Build histogram figure.
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=pct_values,
                name="P&L %",
                marker_color="steelblue",
            )
        )
        hist_fig.update_layout(
            title_text="Trade P&L Distribution (%)",
            xaxis_title="Return (%)",
            yaxis_title="Count",
        )

        # Build win-rate figure (uses annotation to display win rate prominently).
        wr_text = f"{win_rate:.2f}" if win_rate is not None else "N/A"
        wr_fig = go.Figure()
        wr_fig.update_layout(
            title_text=f"Win Rate: {wr_text} | Total Trades: {total_trades}",
            annotations=[
                dict(
                    text=f"Win Rate: {wr_text}",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"size": 32},
                )
            ],
        )

        return [hist_fig, wr_fig]

    # ------------------------------------------------------------------
    # Dash app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(__name__)

        app.layout = html.Div(
            [
                html.H1("Training Dashboard"),
                dcc.Interval(id="interval", interval=2000, n_intervals=0),
                dcc.Graph(id="chart-primary"),
                dcc.Graph(id="chart-secondary"),
            ]
        )

        @app.callback(
            [Output("chart-primary", "figure"), Output("chart-secondary", "figure")],
            [Input("interval", "n_intervals")],
        )
        def refresh(_n: int):  # type: ignore[return]
            state = self._load_state()
            figs = self._update_charts(state)
            # Ensure we always return exactly two figures.
            if len(figs) < 2:
                figs = figs + [go.Figure()] * (2 - len(figs))
            return figs[0], figs[1]

        return app

    def _load_state(self) -> dict[str, Any]:
        """Read *state_path* atomically; return empty dict on any error."""
        path = getattr(self, "_state_path", "shared/training_state.pkl")
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
