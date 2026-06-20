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

_TF_LABELS = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h", 1440: "1d"}


def _tf_label(tf: int) -> str:
    """Human-readable label for a timeframe in minutes."""
    if tf in _TF_LABELS:
        return _TF_LABELS[tf]
    if tf % 1440 == 0:
        return f"{tf // 1440}d"
    if tf % 60 == 0:
        return f"{tf // 60}h"
    return f"{tf}m"


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

    def _render_groups(
        self,
        start_date: str | None,
        days: int | None,
        tfs: list | None,
        subplots: list | None = None,
        overlays: list | None = None,
        show_actions: bool = False,
    ) -> list:
        """Return one dcc.Graph per selected TF, ascending TF order. Never raises.

        *subplots* is the shared oscillator-subplot selection and *overlays*
        the shared price-overlay-group selection, both applied to every TF's
        figure. ``None`` keeps all; ``[]`` hides them all.
        *show_actions* overlays the latest simulation's actions on each chart.
        """
        try:
            if start_date is None or days is None or int(days) < 1 or not tfs:
                return []
            graphs = []
            for tf in sorted(int(t) for t in tfs):
                fig = self.viewer.build_window_figure(
                    pd.Timestamp(start_date), int(days), tf=tf, subplots=subplots,
                    overlays=overlays, show_actions=show_actions,
                )
                graphs.append(
                    dcc.Graph(id={"type": "tf-chart", "tf": tf}, figure=fig)
                )
            return graphs
        except Exception as e:
            print(f"[HistoryDashboard] _render_groups error: {e}")
            return [dcc.Graph(figure=empty_figure())]

    # ------------------------------------------------------------------
    # Dash app construction
    # ------------------------------------------------------------------

    def _day_options(self) -> list[int]:
        """Day-count presets bounded to the dataset span (always sorted, unique)."""
        dmin, dmax = self._date_bounds()
        span = max(1, int((dmax - dmin).total_seconds() // 86400) + 1)
        presets = [1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180, 365]
        opts = sorted({d for d in presets if d <= span} | {span})
        return opts

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(__name__)
        dmin, dmax = self._date_bounds()
        tfs = self.viewer.available_tfs()
        initial = [self.tf] if self.tf in tfs else tfs[:1]
        subplot_names = self.viewer.available_subplots()
        overlay_names = self.viewer.available_overlays()
        default_overlays = self.viewer.default_overlays()
        day_opts = self._day_options()
        # Prefer a 7-day window; for shorter datasets show the full span.
        default_days = 7 if 7 in day_opts else day_opts[-1]

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
                        html.Label("days:", style={"marginLeft": "10px"}),
                        dcc.Dropdown(
                            id="days",
                            options=[{"label": f"{d}d", "value": d} for d in day_opts],
                            value=default_days,
                            clearable=False,
                            style={"width": "100px", "display": "inline-block",
                                   "verticalAlign": "middle"},
                        ),
                        dcc.Checklist(
                            id="timeframes",
                            options=[
                                {"label": _tf_label(tf), "value": tf} for tf in tfs
                            ],
                            value=initial,
                            inline=True,
                        ),
                        dcc.Checklist(
                            id="subplots",
                            options=[
                                {"label": sp, "value": sp} for sp in subplot_names
                            ],
                            value=list(subplot_names),
                            inline=True,
                        ),
                        dcc.Checklist(
                            id="overlays",
                            options=[
                                {"label": ov, "value": ov} for ov in overlay_names
                            ],
                            value=default_overlays,
                            inline=True,
                        ),
                        dcc.Checklist(
                            id="actions",
                            options=[{"label": "actions", "value": "actions"}],
                            value=[],  # off by default
                            inline=True,
                        ),
                    ]
                ),
                html.Div(id="chart-groups"),
            ]
        )

        @app.callback(
            Output("chart-groups", "children"),
            [
                Input("start-date", "date"),
                Input("days", "value"),
                Input("timeframes", "value"),
                Input("subplots", "value"),
                Input("overlays", "value"),
                Input("actions", "value"),
            ],
        )
        def update(start_date, days, tfs_selected, subplots_selected,
                   overlays_selected, actions_selected):  # type: ignore[return]
            return self._render_groups(
                start_date, days, tfs_selected, subplots_selected,
                overlays=overlays_selected,
                show_actions=bool(actions_selected and "actions" in actions_selected),
            )

        return app
