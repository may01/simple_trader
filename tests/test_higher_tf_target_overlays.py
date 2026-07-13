"""Tests for higher-TF target/stop-loss overlays on finer charts (tf 1, 5).

Spec: docs/superpowers/specs/2026-07-13-higher-tf-target-overlays-design.md

On sub-15min charts the four target fields (tgt_long, sl_long, tgt_short,
sl_short) are sourced from the higher TFs that compute them (15/60/240/1440)
via toggleable ``tgtsl_{H}m`` overlay groups, gated to tf < 15, drawn as
TF-suffixed step lines styled by source TF.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from frontend.chart_renderer import ChartRenderer
from frontend.data_viewer import DataViewer, FullData

TARGET_FIELDS = ["tgt_long", "sl_long", "tgt_short", "sl_short"]
HIGHER_TFS = [15, 60, 240, 1440]


def _make_df(target_tfs, n=60, base_tf=1, fields=TARGET_FIELDS):
    """1-min OHLCV for base_tf plus flat-step target columns for each higher TF.

    ``target_tfs`` maps source TF -> list of target field base names present.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    idx.name = "open_time"
    data = {
        f"{base_tf}_open": np.linspace(100, 110, n),
        f"{base_tf}_high": np.linspace(105, 115, n),
        f"{base_tf}_low": np.linspace(95, 105, n),
        f"{base_tf}_close": np.linspace(102, 112, n),
        f"{base_tf}_volume": np.linspace(1000, 2000, n),
    }
    for src_tf, names in target_tfs.items():
        # flat-step: constant within each src_tf candle
        floored = idx.floor(f"{src_tf}min")
        codes = pd.Categorical(floored).codes.astype(float)
        for name in names:
            data[f"{src_tf}_{name}"] = 100.0 + codes
    return pd.DataFrame(data, index=idx)


def _viewer(df, tf, mock=True):
    v = DataViewer(FullData(df), tf=tf)
    if mock:
        r = MagicMock()
        fig = MagicMock()
        fig._subplot_rows = {"price": 1, "volume": 2}
        r.create_figure.return_value = fig
        v.renderer = r
    return v


def _price_lines(mock_renderer):
    """List of (label, color, dash, width) for draw_line calls on the price row."""
    out = []
    for c in mock_renderer.draw_line.call_args_list:
        if c[0][1] != "price":
            continue
        out.append((
            c.kwargs.get("label"),
            c.kwargs.get("color"),
            c.kwargs.get("dash"),
            c.kwargs.get("width"),
        ))
    return out


def _labels(mock_renderer):
    return [lbl for (lbl, _, _, _) in _price_lines(mock_renderer)]


# ---------------------------------------------------------------------------
# available_overlays / default_overlays
# ---------------------------------------------------------------------------

class TestOverlayGroupExposure:
    def test_available_includes_present_target_tf_groups(self):
        df = _make_df({15: TARGET_FIELDS, 60: TARGET_FIELDS})
        ov = _viewer(df, tf=1).available_overlays()
        assert "tgtsl_15m" in ov
        assert "tgtsl_60m" in ov

    def test_available_excludes_absent_target_tf_groups(self):
        df = _make_df({15: TARGET_FIELDS})  # no 60/240/1440 cols
        ov = _viewer(df, tf=1).available_overlays()
        assert "tgtsl_15m" in ov
        for g in ("tgtsl_60m", "tgtsl_240m", "tgtsl_1440m"):
            assert g not in ov

    def test_default_overlays_15m_on_others_off(self):
        df = _make_df({tf: TARGET_FIELDS for tf in HIGHER_TFS})
        d = _viewer(df, tf=1).default_overlays()
        assert "tgtsl_15m" in d
        for g in ("tgtsl_60m", "tgtsl_240m", "tgtsl_1440m"):
            assert g not in d


# ---------------------------------------------------------------------------
# draw behaviour on sub-15 charts
# ---------------------------------------------------------------------------

class TestSub15Draw:
    def test_tf1_draw_all_suffixed_labels(self):
        df = _make_df({15: TARGET_FIELDS, 60: TARGET_FIELDS})
        v = _viewer(df, tf=1)
        v.build_window_figure("2024-01-01", 1)   # overlays=None -> draw all
        labels = _labels(v.renderer)
        for name in TARGET_FIELDS:
            assert f"{name}·15m" in labels
            assert f"{name}·60m" in labels

    def test_semantic_colors_preserved(self):
        df = _make_df({15: TARGET_FIELDS})
        v = _viewer(df, tf=1)
        v.build_window_figure("2024-01-01", 1)
        colors = {lbl: col for (lbl, col, _, _) in _price_lines(v.renderer)}
        assert colors["tgt_long·15m"] == "darkgreen"
        assert colors["sl_long·15m"] == "darkred"
        assert colors["tgt_short·15m"] == "mediumseagreen"
        assert colors["sl_short·15m"] == "indianred"

    def test_toggle_filters_to_selected_group(self):
        df = _make_df({15: TARGET_FIELDS, 60: TARGET_FIELDS})
        v = _viewer(df, tf=1)
        v.build_window_figure("2024-01-01", 1, overlays=["tgtsl_15m"])
        labels = _labels(v.renderer)
        assert "tgt_long·15m" in labels
        assert "tgt_long·60m" not in labels

    def test_empty_overlays_draws_no_higher_tf(self):
        df = _make_df({15: TARGET_FIELDS})
        v = _viewer(df, tf=1)
        v.build_window_figure("2024-01-01", 1, overlays=[])
        labels = _labels(v.renderer)
        assert not any("·15m" in (lbl or "") for lbl in labels)

    def test_skip_if_absent_per_field(self):
        df = _make_df({15: ["tgt_long"]})  # only one field present
        v = _viewer(df, tf=1)
        v.build_window_figure("2024-01-01", 1)
        labels = _labels(v.renderer)
        assert "tgt_long·15m" in labels
        for name in ("sl_long", "tgt_short", "sl_short"):
            assert f"{name}·15m" not in labels

    def test_style_by_source_tf(self):
        df = _make_df({15: TARGET_FIELDS, 60: TARGET_FIELDS})
        v = _viewer(df, tf=1)
        v.build_window_figure("2024-01-01", 1)
        style = {lbl: (dash, width) for (lbl, _, dash, width) in _price_lines(v.renderer)}
        assert style["tgt_long·15m"] == (None, 2.0)     # nearest: solid, bold
        assert style["tgt_long·60m"] == ("dash", 1.6)   # farther: dashed, thinner


# ---------------------------------------------------------------------------
# gate: tf >= 15 unchanged
# ---------------------------------------------------------------------------

class TestGate:
    def test_tf15_draws_no_suffixed_labels(self):
        # 15m OHLC + native 15m targets + a 60m target set present in the df.
        df = _make_df({15: TARGET_FIELDS, 60: TARGET_FIELDS}, base_tf=15)
        v = _viewer(df, tf=15)
        v.build_window_figure("2024-01-01", 1)
        labels = _labels(v.renderer)
        # native bare label still drawn
        assert "tgt_long" in labels
        # higher-TF block gated off at tf >= 15
        for name in TARGET_FIELDS:
            assert f"{name}·15m" not in labels
            assert f"{name}·60m" not in labels


# ---------------------------------------------------------------------------
# ChartRenderer.draw_line back-compat + new kwargs
# ---------------------------------------------------------------------------

class TestDrawLine:
    def test_defaults_no_dash_width(self):
        r = ChartRenderer()
        fig = r.create_figure(["price"])
        r.draw_line(fig, "price", [1, 2], [1.0, 2.0], label="x", color="red")
        line = fig.data[-1].line
        assert line.color == "red"
        assert line.dash is None
        assert line.width is None

    def test_dash_width_applied(self):
        r = ChartRenderer()
        fig = r.create_figure(["price"])
        r.draw_line(fig, "price", [1, 2], [1.0, 2.0], label="x",
                    color="red", dash="dash", width=1.6)
        line = fig.data[-1].line
        assert line.dash == "dash"
        assert line.width == 1.6


# ---------------------------------------------------------------------------
# integration: real ChartRenderer through build_window_figure(tf=1)
# ---------------------------------------------------------------------------

class TestIntegrationRealRenderer:
    def test_tf1_produces_higher_tf_traces(self):
        df = _make_df({15: TARGET_FIELDS, 60: TARGET_FIELDS})
        v = _viewer(df, tf=1, mock=False)   # real ChartRenderer
        fig = v.build_window_figure("2024-01-01", 1)
        names = [t.name for t in fig.data]
        for name in TARGET_FIELDS:
            assert f"{name}·15m" in names
            assert f"{name}·60m" in names
        by_name = {t.name: t for t in fig.data}
        assert by_name["tgt_long·15m"].line.dash is None
        assert by_name["tgt_long·15m"].line.width == 2.0
        assert by_name["tgt_long·60m"].line.dash == "dash"
        assert by_name["tgt_long·60m"].line.width == 1.6
