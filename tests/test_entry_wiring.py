"""Tests for root entry-point wiring (phase 10 task 05 / phase 12 task 05).

trader.py → Robot live loop (paper mode by default);
view_*.py → frontend dashboards; dashboards bind 0.0.0.0:$DASH_PORT.
"""

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def _set_env(monkeypatch, extra: dict = None):
    env = {
        "PAIR": "link_usdt",
        "ROOT_FOLDER": "short",
        "DATA_ROOT": "live",
        "DATA_SET_NAME": "live",
        "EXCHANGE_FEE": "0.001",
    }
    if extra:
        env.update(extra)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


@contextmanager
def _inject(mods: dict):
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield
    finally:
        for key, original in saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


class TestTraderEntry:
    def _run_main(self, monkeypatch, tmp_path, extra_env=None):
        _set_env(monkeypatch, extra_env)

        mock_stock = MagicMock()
        mock_stock.fee = 0.001
        mock_holder_mod = MagicMock()
        mock_holder_mod.stock_holder.item = mock_stock

        mock_robot_instance = MagicMock()
        mock_robot_cls = MagicMock(return_value=mock_robot_instance)

        mods = {
            "stocks_holder": mock_holder_mod,
            "data": MagicMock(),
            "robots.robot": MagicMock(Robot=mock_robot_cls),
            "strategies.strategy_manager": MagicMock(),
        }
        with _inject(mods), patch("helpers.shared_folder", return_value=str(tmp_path) + "/"):
            import trader
            importlib.reload(trader)
            trader.main()
        return mock_holder_mod, mock_stock, mock_robot_cls, mock_robot_instance

    def test_paper_mode_is_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("STOCK_TYPE", raising=False)
        holder, _, _, _ = self._run_main(monkeypatch, tmp_path)
        holder.do_stock_init.assert_called_once_with("mock_binance")

    def test_robot_constructed_and_started(self, monkeypatch, tmp_path):
        _, stock, robot_cls, robot = self._run_main(monkeypatch, tmp_path)
        robot_cls.assert_called_once()
        args = robot_cls.call_args
        assert args.args[2] is stock
        assert args.args[3] == stock.fee
        assert args.kwargs["persist_path"].endswith("live_tracker.json")
        robot.run_instantly.assert_called_once()

    def test_explicit_binance_respected(self, monkeypatch, tmp_path):
        holder, _, _, _ = self._run_main(
            monkeypatch, tmp_path, extra_env={"STOCK_TYPE": "binance"}
        )
        holder.do_stock_init.assert_called_once_with("binance")


class TestViewEntries:
    def test_view_online_point_runs_live_dashboard(self, monkeypatch, tmp_path):
        _set_env(monkeypatch)
        mock_dash = MagicMock()
        mock_cls = MagicMock(return_value=mock_dash)
        with _inject({"frontend.live_dashboard": MagicMock(LiveDashboard=mock_cls)}), \
             patch("helpers.shared_folder", return_value=str(tmp_path) + "/"):
            import view_online_point
            importlib.reload(view_online_point)
            view_online_point.main()
        mock_cls.assert_called_once()
        state_path = mock_dash.run.call_args.kwargs["state_path"]
        assert state_path.endswith("live_state.pkl")

    def test_view_onlineB_runs_live_dashboard(self, monkeypatch, tmp_path):
        _set_env(monkeypatch)
        mock_dash = MagicMock()
        mock_cls = MagicMock(return_value=mock_dash)
        with _inject({"frontend.live_dashboard": MagicMock(LiveDashboard=mock_cls)}), \
             patch("helpers.shared_folder", return_value=str(tmp_path) + "/"):
            import view_onlineB
            importlib.reload(view_onlineB)
            view_onlineB.main()
        state_path = mock_dash.run.call_args.kwargs["state_path"]
        assert state_path.endswith("live_state.pkl")

    def test_view_training_runs_training_dashboard(self, monkeypatch, tmp_path):
        _set_env(monkeypatch)
        mock_dash = MagicMock()
        mock_cls = MagicMock(return_value=mock_dash)
        with _inject({"frontend.training_dashboard": MagicMock(TrainingDashboard=mock_cls)}), \
             patch("helpers.shared_folder", return_value=str(tmp_path) + "/"):
            import view_training
            importlib.reload(view_training)
            view_training.main()
        state_path = mock_dash.run.call_args.kwargs["state_path"]
        assert state_path.endswith("training_state.pkl")

    def test_view_full_runs_history_dashboard(self, monkeypatch, tmp_path):
        _set_env(monkeypatch)
        mock_dashboard = MagicMock()
        mock_dashboard_cls = MagicMock(return_value=mock_dashboard)
        mock_hd_mod = MagicMock(HistoryDashboard=mock_dashboard_cls)
        mock_dv_mod = MagicMock()
        with _inject({
            "frontend.history_dashboard": mock_hd_mod,
            "frontend.data_viewer": mock_dv_mod,
        }), \
             patch("helpers.wide_df_path", return_value="/data/wide.pkl"), \
             patch("pandas.read_pickle", return_value=MagicMock()):
            import view_full
            importlib.reload(view_full)
            view_full.main()
        mock_dashboard_cls.assert_called_once()
        mock_dashboard.run.assert_called_once()


class TestDashboardBinding:
    """Dash apps must bind 0.0.0.0:$DASH_PORT — compose maps container ports;
    the default 127.0.0.1:8050 is unreachable through the port mapping."""

    def test_history_dashboard_binds_env_port(self, monkeypatch):
        monkeypatch.setenv("DASH_PORT", "8080")
        import pandas as pd
        import numpy as np
        from frontend.data_viewer import FullData
        from frontend.history_dashboard import HistoryDashboard

        idx = pd.date_range("2024-01-01", periods=30, freq="1min")
        df = pd.DataFrame(
            {f"15_{c}": np.linspace(1, 2, 30)
             for c in ("open", "high", "low", "close", "volume")},
            index=idx,
        )
        dashboard = HistoryDashboard(FullData(df))
        with patch.object(dashboard._app, "run") as mock_run:
            dashboard.run()
        assert mock_run.call_args.kwargs["host"] == "0.0.0.0"
        assert mock_run.call_args.kwargs["port"] == 8080

    def test_live_dashboard_binds_env_port(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DASH_PORT", "8080")
        from frontend.live_dashboard import LiveDashboard

        dashboard = LiveDashboard()
        with patch.object(dashboard._app, "run") as mock_run:
            dashboard.run(state_path=str(tmp_path / "live_state.pkl"))
        assert mock_run.call_args.kwargs["host"] == "0.0.0.0"
        assert mock_run.call_args.kwargs["port"] == 8080

    def test_training_dashboard_binds_env_port(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DASH_PORT", "8060")
        from frontend.training_dashboard import TrainingDashboard

        dashboard = TrainingDashboard()
        with patch.object(dashboard._app, "run") as mock_run:
            dashboard.run(state_path=str(tmp_path / "training_state.pkl"))
        assert mock_run.call_args.kwargs["host"] == "0.0.0.0"
        assert mock_run.call_args.kwargs["port"] == 8060
