import numpy as np
import pandas as pd

from experiments.swing_sinusoid.causal_zones import causal_apexes, cci_zone, build_filter_frame


def test_causal_apexes_on_clean_sine():
    t = np.arange(200)
    y = 100 * np.sin(2 * np.pi * t / 20)
    ap = causal_apexes(y, prom=50.0)
    if ap[0][0] == 0:          # series-start artifact: the first row counts as the running low
        ap = ap[1:]
    rows = [a for a, _, _ in ap]
    signs = [s for _, s, _ in ap]
    assert np.all(np.diff(signs) != 0)
    # true extrema at 5, 15, 25, ...; confirmation strictly after the apex
    assert all(r % 10 == 5 for r in rows)
    assert all(j > i for i, _, j in ap)
    assert ap[0][0] == 5 and ap[0][1] == 1
    assert len(ap) >= 17


def test_zone_is_causal_and_bounded():
    t = np.arange(200)
    y = 100 * np.sin(2 * np.pi * t / 20)
    z, ap = cci_zone(y, prom=50.0, const=10.0, confirm=3, window=3)
    for i, s, j in ap:
        a = max(i + 3, j)
        assert z[a] == (-1 if s > 0 else 1)
        assert z[i] in (0, -z[a]) or i < a   # nothing of this apex's zone before its confirmation


def test_build_filter_frame_shapes():
    n = 15 * 240 * 4
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(index=idx)
    for tf in (15, 60, 240):
        df[f"{tf}_is_closed"] = (np.arange(n) % tf == tf - 1)
        df[f"{tf}_cci_14"] = 200 * np.sin(np.arange(n) / (tf * 3)) + rng.normal(0, 20, n)
        df[f"{tf}_move_class_sym0"] = rng.integers(-3, 4, n)
    out = build_filter_frame(df, cci5_limit=None)   # zone rule alone (the CCI gate has its own test)
    assert len(out) == n
    for tf in (15, 60, 240):
        assert out[f"{tf}_swing_zone"].between(-3, 3).all()
        assert out[f"{tf}_cb_f_swing_long"].dtype == bool
    # when every zone is 0 both sides are allowed
    allzero = (out["15_swing_zone"] == 0) & (out["60_swing_zone"] == 0) & (out["240_swing_zone"] == 0)
    if allzero.any():
        assert out.loc[allzero, "15_cb_f_swing_long"].all() and out.loc[allzero, "15_cb_f_swing_short"].all()


def test_cci5_gate():
    n = 15 * 240 * 2
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(1)
    df = pd.DataFrame(index=idx)
    for tf in (15, 60, 240):
        df[f"{tf}_is_closed"] = (np.arange(n) % tf == tf - 1)
        df[f"{tf}_cci_14"] = 200 * np.sin(np.arange(n) / (tf * 3))
        df[f"{tf}_move_class_sym0"] = 0
    df["5_cci_14"] = np.where(np.arange(n) % 2 == 0, 0.0, 200.0)   # every odd minute is out of range
    df["15_cci_14"] = np.where(np.arange(n) % 4 == 0, -300.0, 0.0)  # every 4th minute out of range
    out = build_filter_frame(df, cci5_limit=140.0)
    bad = (np.arange(n) % 2 == 1) | (np.arange(n) % 4 == 0)
    for tf in (15, 60, 240):
        assert not out.loc[bad, f"{tf}_cb_f_swing_long"].any()
        assert not out.loc[bad, f"{tf}_cb_f_swing_short"].any()
        assert out.loc[~bad, f"{tf}_cb_f_swing_long"].any()
    assert out["cci5_ok"].mean() == 0.5 and out["cci15_ok"].mean() == 0.75
    assert out["cci_gate_ok"].mean() == 0.25
    nogate = build_filter_frame(df, cci5_limit=None)
    assert "cci_gate_ok" not in nogate.columns
