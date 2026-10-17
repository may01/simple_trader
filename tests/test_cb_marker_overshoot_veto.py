"""Tests for the cb_f_overshoot marker-filter toggle (phase 19, G7).

Replaces the sidecar veto ({tf}_cbover_{side} from df_with_cb_overshoot.pkl,
``data.join_cb_overshoot``, ``DataViewer._MARKER_VETO``): the overshoot pass flag
is now a main-flow column ``{tf}_cb_f_overshoot_{side}`` and the fourth
``cb_f_*`` toggle, on by default, marker-only (the zone line is not blanked).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import data
from frontend.data_viewer import DataViewer, FullData

N = 10
INZONE = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
EXTDONE = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0], dtype=bool)
TREND = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
MCX = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
OVER_LONG = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
OVER_SHORT = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
CB_ALL = ["cb_long", "cb_short"]


def _make_df(with_overshoot: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N, freq="1min")
    d = {
        "1_low": np.full(N, 95.0), "1_high": np.full(N, 105.0),
        "1_close": np.full(N, 100.0), "15_close": np.full(N, 100.0),
        "15_cb_inzone_long": INZONE, "15_cb_inzone_short": INZONE,
        "15_cb_f_extdone_long": EXTDONE, "15_cb_f_extdone_short": EXTDONE,
        "15_cb_f_trend_long": TREND, "15_cb_f_trend_short": TREND,
        "15_cb_f_mcx_long": MCX, "15_cb_f_mcx_short": MCX,
        "15_cb_zone_long": np.full(N, 100.0),
    }
    if with_overshoot:
        d["15_cb_f_overshoot_long"] = OVER_LONG
        d["15_cb_f_overshoot_short"] = OVER_SHORT
    return pd.DataFrame(d, index=idx)


def _times(df, label, overlays):
    v = DataViewer(FullData(df), tf=1)
    v.renderer = MagicMock()
    v._draw_zone_markers(MagicMock(), df, overlays)
    calls = [c for c in v.renderer.draw_marker.call_args_list
             if c.kwargs.get("label") == label]
    return list(calls[0][0][1]) if calls else []


def _idx(df, mask):
    return list(df.index[mask])


def _line(df, overlays):
    v = DataViewer(FullData(df), tf=15)
    v.renderer = MagicMock()
    v._draw_price_overlays(MagicMock(), df, 15, overlays)
    for c in v.renderer.draw_line.call_args_list:
        if c.kwargs.get("label") == "cb_zone_long":
            return np.array(c[0][3], dtype=float)
    raise AssertionError("no cb_zone_long line")


def test_default_masks_overshoot_and_extdone():
    df = _make_df()
    assert _times(df, "cbzone_long_15m", None) == _idx(df, INZONE & EXTDONE & OVER_LONG)


def test_overshoot_unchecked_draws_overshoot_rows():
    df = _make_df()
    assert _times(df, "cbzone_long_15m", CB_ALL + ["cb_f_extdone"]) == _idx(df, INZONE & EXTDONE)
    assert _times(df, "cbzone_long_15m", CB_ALL) == _idx(df, INZONE)


def test_overshoot_ands_with_trend_and_mcx():
    df = _make_df()
    ov = CB_ALL + ["cb_f_trend", "cb_f_mcx", "cb_f_overshoot"]
    assert _times(df, "cbzone_long_15m", ov) == _idx(df, INZONE & TREND & MCX & OVER_LONG)


def test_overshoot_column_absent_is_noop():
    df = _make_df(with_overshoot=False)
    assert _times(df, "cbzone_long_15m", CB_ALL + ["cb_f_overshoot"]) == _idx(df, INZONE)
    assert _times(df, "cbzone_long_15m", None) == _idx(df, INZONE & EXTDONE)


def test_side_specific():
    df = _make_df()
    ov = CB_ALL + ["cb_f_overshoot"]
    assert _times(df, "cbzone_long_15m", ov) == _idx(df, INZONE & OVER_LONG)
    assert _times(df, "cbzone_short_15m", ov) == _idx(df, INZONE & OVER_SHORT)


def test_zone_line_not_blanked_by_overshoot():
    df = _make_df()
    assert not np.isnan(_line(df, ["cb_long", "cb_f_overshoot"])).any()
    # extdone still blanks; adding overshoot changes nothing on the line
    both = _line(df, ["cb_long", "cb_f_extdone", "cb_f_overshoot"])
    assert np.array_equal(np.isnan(both), ~EXTDONE)


def test_toggle_listed_only_when_column_exists_and_on_by_default():
    v = DataViewer(FullData(_make_df()), tf=1)
    assert "cb_f_overshoot" in v.available_overlays()
    assert "cb_f_overshoot" in v.default_overlays()
    assert "cb_f_overshoot" not in DataViewer._DEFAULT_OVERLAYS_OFF
    assert DataViewer._CB_MARKER_FILTERS_DEFAULT == ("cb_f_extdone", "cb_f_overshoot")
    v2 = DataViewer(FullData(_make_df(with_overshoot=False)), tf=1)
    assert "cb_f_overshoot" not in v2.available_overlays()


def test_toggle_adds_no_trace():
    df = _make_df()
    v = DataViewer(FullData(df), tf=1)
    v.renderer = MagicMock()
    v._draw_zone_markers(MagicMock(), df, ["cb_f_overshoot"])
    v._draw_price_overlays(MagicMock(), df, 15, ["cb_f_overshoot"])
    assert not v.renderer.draw_marker.called
    assert not v.renderer.draw_line.called


def test_sidecar_veto_removed():
    assert not hasattr(DataViewer, "_MARKER_VETO")
    assert not hasattr(data, "join_cb_overshoot")
    df = _make_df()
    df["15_cbover_long"] = True            # the old sidecar column has no effect
    assert _times(df, "cbzone_long_15m", CB_ALL) == _idx(df, INZONE)
