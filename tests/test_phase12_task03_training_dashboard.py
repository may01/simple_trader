"""Tests for TrainingDashboard (Phase 12, Task 03)."""

import pytest
import plotly.graph_objects as go

from frontend.training_dashboard import TrainingDashboard


@pytest.fixture
def dashboard():
    return TrainingDashboard()


@pytest.fixture
def nn_train_state():
    return {
        "phase": "nn_train",
        "loss_history": [1.0, 0.8, 0.6, 0.4],
        "val_loss_history": [1.1, 0.9, 0.7, 0.5],
    }


@pytest.fixture
def simulate_state():
    return {
        "phase": "simulate",
        "revenue_history": [(1.5, 100.0), (-0.5, -30.0), (2.0, 150.0), (-1.0, -70.0)],
        "win_rate": 0.65,
        "total_trades": 42,
    }


class TestUpdateChartsNnTrain:
    def test_returns_list_of_figures(self, dashboard, nn_train_state):
        figs = dashboard._update_charts(nn_train_state)
        assert isinstance(figs, list)
        assert len(figs) > 0
        assert all(isinstance(f, go.Figure) for f in figs)

    def test_loss_trace_present(self, dashboard, nn_train_state):
        figs = dashboard._update_charts(nn_train_state)
        all_traces = [t for fig in figs for t in fig.data]
        scatter_traces = [t for t in all_traces if isinstance(t, go.Scatter)]
        names = [t.name for t in scatter_traces]
        assert any("loss" in (n or "").lower() for n in names), (
            f"Expected a trace with 'loss' in name, got: {names}"
        )

    def test_val_loss_trace_present(self, dashboard, nn_train_state):
        figs = dashboard._update_charts(nn_train_state)
        all_traces = [t for fig in figs for t in fig.data]
        scatter_traces = [t for t in all_traces if isinstance(t, go.Scatter)]
        names = [t.name for t in scatter_traces]
        assert any("val" in (n or "").lower() for n in names), (
            f"Expected a trace with 'val' in name, got: {names}"
        )

    def test_loss_trace_has_correct_values(self, dashboard, nn_train_state):
        figs = dashboard._update_charts(nn_train_state)
        all_traces = [t for fig in figs for t in fig.data]
        loss_trace = next(
            (t for t in all_traces if isinstance(t, go.Scatter) and "loss" in (t.name or "").lower() and "val" not in (t.name or "").lower()),
            None,
        )
        assert loss_trace is not None, "Expected a train loss trace"
        assert list(loss_trace.y) == [1.0, 0.8, 0.6, 0.4]

    def test_val_loss_trace_has_correct_values(self, dashboard, nn_train_state):
        figs = dashboard._update_charts(nn_train_state)
        all_traces = [t for fig in figs for t in fig.data]
        val_trace = next(
            (t for t in all_traces if isinstance(t, go.Scatter) and "val" in (t.name or "").lower()),
            None,
        )
        assert val_trace is not None, "Expected a val loss trace"
        assert list(val_trace.y) == [1.1, 0.9, 0.7, 0.5]

    def test_no_histogram_in_nn_train(self, dashboard, nn_train_state):
        figs = dashboard._update_charts(nn_train_state)
        all_traces = [t for fig in figs for t in fig.data]
        histograms = [t for t in all_traces if isinstance(t, go.Histogram)]
        assert len(histograms) == 0, "nn_train phase should not show histogram"


class TestUpdateChartsSimulate:
    def test_returns_list_of_figures(self, dashboard, simulate_state):
        figs = dashboard._update_charts(simulate_state)
        assert isinstance(figs, list)
        assert len(figs) > 0
        assert all(isinstance(f, go.Figure) for f in figs)

    def test_histogram_present(self, dashboard, simulate_state):
        figs = dashboard._update_charts(simulate_state)
        all_traces = [t for fig in figs for t in fig.data]
        histograms = [t for t in all_traces if isinstance(t, go.Histogram)]
        assert len(histograms) >= 1, "simulate phase must include a go.Histogram"

    def test_histogram_uses_pct_values(self, dashboard, simulate_state):
        figs = dashboard._update_charts(simulate_state)
        all_traces = [t for fig in figs for t in fig.data]
        hist = next((t for t in all_traces if isinstance(t, go.Histogram)), None)
        assert hist is not None
        # revenue_history pct values: [1.5, -0.5, 2.0, -1.0]
        assert list(hist.x) == [1.5, -0.5, 2.0, -1.0]

    def test_win_rate_visible_in_figure(self, dashboard, simulate_state):
        figs = dashboard._update_charts(simulate_state)
        # Win rate (0.65) should appear somewhere — in layout annotations or title
        combined_text = ""
        for fig in figs:
            combined_text += fig.layout.title.text or ""
            annotations = fig.layout.annotations or []
            for ann in annotations:
                combined_text += ann.text or ""
        assert "0.65" in combined_text or "65" in combined_text, (
            f"Win rate 0.65 not found in figure text: {combined_text!r}"
        )

    def test_win_rate_none_no_crash(self, dashboard):
        state = {
            "phase": "simulate",
            "revenue_history": [(1.0, 50.0)],
            "win_rate": None,
            "total_trades": 1,
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_empty_revenue_history_no_crash(self, dashboard):
        state = {
            "phase": "simulate",
            "revenue_history": [],
            "win_rate": 0.5,
            "total_trades": 0,
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)


class TestUpdateChartsEdgeCases:
    def test_empty_state_returns_empty_figures_no_crash(self, dashboard):
        figs = dashboard._update_charts({})
        assert isinstance(figs, list)
        assert all(isinstance(f, go.Figure) for f in figs)

    def test_none_loss_history_no_crash(self, dashboard):
        state = {
            "phase": "nn_train",
            "loss_history": None,
            "val_loss_history": None,
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_empty_loss_history_no_crash(self, dashboard):
        state = {
            "phase": "nn_train",
            "loss_history": [],
            "val_loss_history": [],
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_unknown_phase_no_crash(self, dashboard):
        state = {"phase": "unknown_phase"}
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_missing_phase_key_no_crash(self, dashboard):
        state = {"loss_history": [0.5, 0.4]}
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_extra_keys_no_crash(self, dashboard):
        state = {
            "phase": "nn_train",
            "loss_history": [0.5],
            "val_loss_history": [0.6],
            "unexpected_key": "unexpected_value",
        }
        figs = dashboard._update_charts(state)
        assert isinstance(figs, list)

    def test_returns_go_figures_always(self, dashboard):
        for state in [{}, {"phase": "nn_train"}, {"phase": "simulate"}]:
            figs = dashboard._update_charts(state)
            assert all(isinstance(f, go.Figure) for f in figs), (
                f"All items must be go.Figure for state={state}"
            )
