"""Tests for the NN inference visualisation in the full-view dashboard (Phase 11).

The viewer surfaces the timeframe-agnostic ``nn_res_*`` columns (left-joined from
``df_with_nn.pkl`` by ``data.join_nn_results``) on a dedicated "nn" subplot.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from frontend.data_viewer import DataViewer, FullData


def _make_df(n: int = 60, tf: int = 15, with_nn: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
    }
    if with_nn:
        up = np.linspace(0.2, 0.5, n)
        dn = np.linspace(0.5, 0.2, n)
        data["nn_res_dir15_prob_up"] = up
        data["nn_res_dir15_prob_down"] = dn
        data["nn_res_dir15_prob_neutral"] = 1.0 - up - dn
    return pd.DataFrame(data, index=idx)


def _viewer(df) -> DataViewer:
    return DataViewer(FullData(df), tf=15)


def test_available_subplots_includes_nn_when_present():
    assert "nn" in _viewer(_make_df(with_nn=True)).available_subplots()


def test_available_subplots_omits_nn_when_absent():
    assert "nn" not in _viewer(_make_df(with_nn=False)).available_subplots()


def test_nn_res_cols_helper():
    df = _make_df(with_nn=True)
    cols = DataViewer._nn_res_cols(df)
    assert cols == [
        "nn_res_dir15_prob_down",
        "nn_res_dir15_prob_neutral",
        "nn_res_dir15_prob_up",
    ]
    assert DataViewer._nn_res_cols(_make_df(with_nn=False)) == []


def test_build_window_figure_draws_nn_subplot_and_traces():
    """End-to-end with the real renderer: an 'nn' subplot row exists and the
    three nn_res_* probability traces are drawn on it."""
    df = _make_df(with_nn=True)
    fig = _viewer(df).build_window_figure(df.index[0], days=1, tf=15)
    rows = getattr(fig, "_subplot_rows", {})
    assert "nn" in rows, rows
    nn_traces = sorted(
        t.name for t in fig.data
        if getattr(t, "name", "") and t.name.startswith("nn_res_")
    )
    assert nn_traces == [
        "nn_res_dir15_prob_down",
        "nn_res_dir15_prob_neutral",
        "nn_res_dir15_prob_up",
    ]


def test_build_window_figure_no_nn_subplot_when_absent():
    df = _make_df(with_nn=False)
    fig = _viewer(df).build_window_figure(df.index[0], days=1, tf=15)
    assert "nn" not in getattr(fig, "_subplot_rows", {})


def test_build_window_figure_respects_subplot_selection():
    """When an explicit subplot selection excludes 'nn', it is not drawn."""
    df = _make_df(with_nn=True)
    fig = _viewer(df).build_window_figure(df.index[0], days=1, tf=15, subplots=[])
    assert "nn" not in getattr(fig, "_subplot_rows", {})


def test_draw_nn_results_colors_by_suffix():
    """up→green, down→red, neutral→gray on the 'nn' subplot."""
    df = _make_df(with_nn=True)
    v = _viewer(df)
    renderer = MagicMock()
    fig = MagicMock()
    fig._subplot_rows = {"price": 1, "nn": 2}
    v.renderer = renderer
    v._draw_nn_results(fig, df)
    colors = {
        c.kwargs["label"]: c.kwargs["color"]
        for c in renderer.draw_line.call_args_list
    }
    assert colors["nn_res_dir15_prob_up"] == "green"
    assert colors["nn_res_dir15_prob_down"] == "red"
    assert colors["nn_res_dir15_prob_neutral"] == "gray"


def _draw_and_collect_colors(df):
    v = _viewer(df)
    renderer = MagicMock()
    fig = MagicMock()
    fig._subplot_rows = {"price": 1, "nn": 2}
    v.renderer = renderer
    v._draw_nn_results(fig, df)
    return {
        c.kwargs["label"]: c.kwargs["color"]
        for c in renderer.draw_line.call_args_list
    }


def _df_with_label_heads(heads, n=30, tf=15):
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    data = {
        f"{tf}_open": np.linspace(100, 110, n),
        f"{tf}_high": np.linspace(105, 115, n),
        f"{tf}_low": np.linspace(95, 105, n),
        f"{tf}_close": np.linspace(102, 112, n),
        f"{tf}_volume": np.linspace(1000, 2000, n),
    }
    for i, h in enumerate(heads):
        data[h] = np.linspace(0.1 + 0.01 * i, 0.5, n)
    return pd.DataFrame(data, index=idx)


def test_label_heads_get_distinct_colors():
    """Every non-semantic (label _prob) head draws its own distinct colour —
    not the single shared fallback."""
    heads = [
        "nn_res_ps15_n1_long_prob", "nn_res_ps15_n1_short_prob",
        "nn_res_ps60_n2_long_prob", "nn_res_psh8_5_n1_short_prob",
        "nn_res_psh8_240_n2_long_prob",
    ]
    colors = _draw_and_collect_colors(_df_with_label_heads(heads))
    vals = [colors[h] for h in heads]
    assert len(set(vals)) == len(heads), vals          # all distinct
    assert all(c != "mediumpurple" for c in vals)      # not the old blob colour


def test_label_head_colors_are_deterministic():
    heads = ["nn_res_ps15_n1_long_prob", "nn_res_ps60_n2_short_prob"]
    df = _df_with_label_heads(heads)
    assert _draw_and_collect_colors(df) == _draw_and_collect_colors(df)


def test_semantic_heads_keep_colors_alongside_label_heads():
    """Direction heads keep green/red/gray even when label heads are present."""
    heads = [
        "nn_res_dir15_prob_up", "nn_res_dir15_prob_down",
        "nn_res_dir15_prob_neutral", "nn_res_ps15_n1_long_prob",
    ]
    colors = _draw_and_collect_colors(_df_with_label_heads(heads))
    assert colors["nn_res_dir15_prob_up"] == "green"
    assert colors["nn_res_dir15_prob_down"] == "red"
    assert colors["nn_res_dir15_prob_neutral"] == "gray"
    assert colors["nn_res_ps15_n1_long_prob"] not in ("green", "red", "gray")


def test_draw_nn_results_skips_when_subplot_hidden():
    df = _make_df(with_nn=True)
    v = _viewer(df)
    renderer = MagicMock()
    fig = MagicMock()
    fig._subplot_rows = {"price": 1}  # no "nn" row
    v.renderer = renderer
    v._draw_nn_results(fig, df)
    renderer.draw_line.assert_not_called()
