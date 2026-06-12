"""HistoryDashboard — interactive Dash viewer for the backtesting dataset.

Owns the Dash app and callbacks; ``DataViewer`` stays the figure factory —
the same split ``LiveDashboard`` uses.
"""

from __future__ import annotations

import os

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

from frontend.data_viewer import DataViewer, FullData, empty_figure


class HistoryDashboard:
    """Interactive dataset viewer: start-date + days-count window controls."""

    def __init__(self, full_data: FullData, tf: int = 15) -> None:
        self.full_data = full_data
        self.tf = tf
        self.viewer = DataViewer(full_data, tf)
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Dash app. Blocking.

        Binds 0.0.0.0:$DASH_PORT (default 8080) — the compose port mapping
        cannot reach the Dash default of 127.0.0.1:8050 inside the container.
        """
        self._app.run(
            host="0.0.0.0",
            port=int(os.environ.get("DASH_PORT", "8080")),
            debug=False,
        )

    # ------------------------------------------------------------------
    # Internal: rendering
    # ------------------------------------------------------------------

    def _date_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        idx = self.full_data.df.index
        return idx.min(), idx.max()

    def _render_window(self, start_date: str | None, days: int | None) -> go.Figure:
        """Return the figure for the selected window. Never raises."""
        try:
            if start_date is None or days is None or int(days) < 1:
                return empty_figure()
            return self.viewer.build_window_figure(
                pd.Timestamp(start_date), int(days)
            )
        except Exception as e:
            print(f"[HistoryDashboard] _render_window error: {e}")
            return empty_figure()

    # ------------------------------------------------------------------
    # Dash app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(__name__)
        dmin, dmax = self._date_bounds()

        app.layout = html.Div(
            [
                html.H1("History Dashboard"),
                html.Div(
                    [
                        dcc.DatePickerSingle(
                            id="start-date",
                            date=dmin.date(),
                            min_date_allowed=dmin.date(),
                            max_date_allowed=dmax.date(),
                        ),
                        dcc.Input(id="days", type="number", min=1, value=7),
                    ]
                ),
                dcc.Graph(id="history-chart"),
            ]
        )

        @app.callback(
            Output("history-chart", "figure"),
            [Input("start-date", "date"), Input("days", "value")],
        )
        def update(start_date, days):  # type: ignore[return]
            return self._render_window(start_date, days)

        return app
