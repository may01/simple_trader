"""Tests for the cb marker overshoot veto ({tf}_cbover_{side}) and its sidecar join."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from data import join_cb_overshoot
from frontend.data_viewer import DataViewer, FullData


def _make_df(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    data = {}
    for tf in (1, 15):
        data[f"{tf}_open"] = np.full(n, 100.0)
        data[f"{tf}_high"] = np.full(n, 105.0)
        data[f"{tf}_low"] = np.full(n, 95.0)
        data[f"{tf}_close"] = np.full(n, 102.0)
        data[f"{tf}_volume"] = np.full(n, 1000.0)
    inzone = np.zeros(n, dtype=bool)
    inzone[[3, 4, 5]] = True
    data["15_cb_inzone_long"] = inzone
    return pd.DataFrame(data, index=idx)


def _marker_times(df: pd.DataFrame) -> list:
    v = DataViewer(FullData(df), tf=1)
    v.renderer = MagicMock()
    v._draw_zone_markers(MagicMock(), df, None)
    calls = [
        c for c in v.renderer.draw_marker.call_args_list
        if c.kwargs.get("label") == "cbzone_long_15m"
    ]
    return list(calls[0][0][1]) if calls else []


class TestMarkerVeto:
    def test_all_markers_drawn_without_sidecar(self):
        df = _make_df()
        assert _marker_times(df) == list(df.index[[3, 4, 5]])

    def test_overshoot_rows_dropped(self):
        df = _make_df()
        over = np.zeros(len(df), dtype=bool)
        over[4] = True
        df["15_cbover_long"] = over
        assert _marker_times(df) == list(df.index[[3, 5]])

    def test_no_trace_when_every_marker_vetoed(self):
        df = _make_df()
        df["15_cbover_long"] = True
        assert _marker_times(df) == []


class TestJoinCbOvershoot:
    def test_absent_artifact_is_noop(self, tmp_path):
        df = _make_df()
        assert join_cb_overshoot(df, str(tmp_path)) is df

    def test_joins_new_columns_only(self, tmp_path):
        df = _make_df()
        side = pd.DataFrame(
            {"15_cbover_long": True, "15_cb_inzone_long": True}, index=df.index
        )
        side.to_pickle(tmp_path / "df_with_cb_overshoot.pkl")
        out = join_cb_overshoot(df, str(tmp_path))
        assert out["15_cbover_long"].all()
        assert out["15_cb_inzone_long"].equals(df["15_cb_inzone_long"])
