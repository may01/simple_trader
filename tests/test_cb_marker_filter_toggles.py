"""Tests for the three CB marker-filter toggles (cb_f_extdone / cb_f_trend / cb_f_mcx)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from frontend.data_viewer import DataViewer, FullData

N = 10
INZONE = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
EXTDONE = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0], dtype=bool)
TREND = np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
MCX = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
TREND_SHORT = ~TREND
CB_ALL = ["cb_long", "cb_short"]


def _make_df(with_mcx: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=N, freq="1min")
    d = {
        "1_low": np.full(N, 95.0), "1_high": np.full(N, 105.0),
        "1_close": np.full(N, 100.0), "15_close": np.full(N, 100.0),
        "15_cb_inzone_long": INZONE, "15_cb_inzone_short": INZONE,
        "15_cb_f_extdone_long": EXTDONE, "15_cb_f_extdone_short": EXTDONE,
        "15_cb_f_trend_long": TREND, "15_cb_f_trend_short": TREND_SHORT,
        "15_cb_zone_long": np.full(N, 100.0),
    }
    if with_mcx:
        d["15_cb_f_mcx_long"] = MCX
        d["15_cb_f_mcx_short"] = MCX
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


def test_default_applies_extdone_only():
    df = _make_df()
    assert _times(df, "cbzone_long_15m", None) == _idx(df, INZONE & EXTDONE)


def test_extdone_unchecked_draws_all_inzone():
    df = _make_df()
    assert _times(df, "cbzone_long_15m", CB_ALL) == _idx(df, INZONE)


def test_trend_and_mcx_and_together():
    df = _make_df()
    ov = CB_ALL + ["cb_f_extdone", "cb_f_trend", "cb_f_mcx"]
    assert _times(df, "cbzone_long_15m", ov) == _idx(df, INZONE & EXTDONE & TREND & MCX)


def test_filter_column_absent_is_noop():
    df = _make_df(with_mcx=False)
    ov = CB_ALL + ["cb_f_mcx"]
    assert _times(df, "cbzone_long_15m", ov) == _idx(df, INZONE)


def test_side_specific():
    df = _make_df()
    ov = CB_ALL + ["cb_f_trend"]
    assert _times(df, "cbzone_long_15m", ov) == _idx(df, INZONE & TREND)
    assert _times(df, "cbzone_short_15m", ov) == _idx(df, INZONE & TREND_SHORT)


def test_other_families_not_filtered():
    df = _make_df()
    df["15_cbx_inzone_long"] = INZONE
    df["15_cbnc_inzone_long"] = INZONE
    ov = ["cbnc_long", "cb_f_extdone", "cb_f_trend", "cb_f_mcx"]
    assert _times(df, "cbx_long_15m", ov) == _idx(df, INZONE)
    assert _times(df, "cbzone_nc_long_15m", ov) == _idx(df, INZONE)


def test_zone_line_blanked_only_with_extdone_toggle():
    df = _make_df()

    def line(overlays):
        v = DataViewer(FullData(df), tf=15)
        v.renderer = MagicMock()
        v._draw_price_overlays(MagicMock(), df.rename(columns=lambda c: c), 15, overlays)
        for c in v.renderer.draw_line.call_args_list:
            if c.kwargs.get("label") == "cb_zone_long":
                return np.array(c[0][3], dtype=float)
        raise AssertionError("no cb_zone_long line")

    on = line(["cb_long", "cb_f_extdone"])
    assert np.array_equal(np.isnan(on), ~EXTDONE)
    off = line(["cb_long"])
    assert not np.isnan(off).any()
    assert np.array_equal(np.isnan(line(None)), ~EXTDONE)


def test_toggles_add_no_trace():
    df = _make_df()
    v = DataViewer(FullData(df), tf=1)
    v.renderer = MagicMock()
    ov = ["cb_f_extdone", "cb_f_trend", "cb_f_mcx"]
    v._draw_zone_markers(MagicMock(), df, ov)
    v._draw_price_overlays(MagicMock(), df, 15, ov)
    assert not v.renderer.draw_marker.called
    assert not v.renderer.draw_line.called


def test_defaults_off_set():
    assert {"cb_f_trend", "cb_f_mcx"} <= DataViewer._DEFAULT_OVERLAYS_OFF
    assert "cb_f_extdone" not in DataViewer._DEFAULT_OVERLAYS_OFF


def test_toggles_listed_when_flag_columns_exist():
    df = _make_df()
    v = DataViewer(FullData(df), tf=1)
    av = v.available_overlays()
    assert {"cb_f_extdone", "cb_f_trend", "cb_f_mcx"} <= set(av)
    d = v.default_overlays()
    assert "cb_f_extdone" in d and "cb_f_trend" not in d


def test_overshoot_veto_still_applies():
    df = _make_df()
    over = np.zeros(N, dtype=bool)
    over[3] = True
    df["15_cbover_long"] = over
    ov = CB_ALL + ["cb_f_extdone", "cb_f_trend", "cb_f_mcx"]
    assert _times(df, "cbzone_long_15m", ov) == _idx(df, INZONE & EXTDONE & TREND & MCX & ~over)
    assert _times(df, "cbzone_long_15m", CB_ALL) == _idx(df, INZONE & ~over)
