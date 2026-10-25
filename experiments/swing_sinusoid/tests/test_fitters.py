import numpy as np
import pytest

from experiments.swing_sinusoid import fitters

P = 20.0


def _window(W=48, period=P, phase=0.3, amp=1.0, trend=0.0, noise=0.0, seed=1):
    t = np.arange(W, dtype=float)
    rng = np.random.default_rng(seed)
    return amp * np.sin(2 * np.pi * t / period + phase) + trend * t + noise * rng.normal(size=W)


def _true_next_apex(W, period, phase):
    omega = 2 * np.pi / period
    t_end = W - 1
    n = np.floor((omega * t_end + phase - np.pi / 2) / np.pi)
    for nn in (n, n + 1, n + 2):
        t = (np.pi / 2 + nn * np.pi - phase) / omega
        if t > t_end + 1e-9:
            return t - t_end, (1 if (nn % 2 == 0) else -1)
    raise AssertionError


@pytest.mark.parametrize("method,tol_dt", [("nls_sine", 0.6), ("ar2", 0.6), ("zero_cross", 1.0)])
def test_fitted_methods_recover_clean_sine(method, tol_dt):
    y = _window()
    p = fitters.predict(method, y)
    dt_true, sign_true = _true_next_apex(48, P, 0.3)
    assert p.ok, p
    assert abs(p.period - P) / P < 0.03
    assert abs(p.dt - dt_true) < tol_dt, (p.dt, dt_true)
    assert p.sign == sign_true
    assert abs(p.val - sign_true * 1.0) < 0.15


def test_hilbert_recovers_clean_sine_within_a_candle():
    y = _window()
    p = fitters.predict("hilbert", y)
    dt_true, sign_true = _true_next_apex(48, P, 0.3)
    assert p.ok
    assert abs(p.period - P) / P < 0.15   # edge-corrupted estimator, documented weakness
    assert abs(p.dt - dt_true) < 2.0
    assert p.sign == sign_true


def test_nls_handles_trend_and_noise():
    y = _window(trend=0.02, noise=0.1)
    p = fitters.predict("nls_sine", y)
    dt_true, sign_true = _true_next_apex(48, P, 0.3)
    assert p.ok and abs(p.period - P) / P < 0.08 and abs(p.dt - dt_true) < 1.5
    assert abs(p.slope - 0.02) < 0.01


@pytest.mark.parametrize("method", ["mirror", "mean_half_swing"])
def test_baselines_on_clean_sine(method):
    y = _window(phase=0.0)  # highs at t=5,25,45 ; lows at 15,35 ; W=48 → next low at 55 → dt 8
    p = fitters.predict(method, y)
    assert p.ok
    assert p.sign == -1
    assert abs(p.dt - 8) <= 1.0
    assert abs(p.val - (-1.0)) < 0.05


def test_median_period_baseline_uses_given_period():
    y = _window(phase=0.0)
    p = fitters.predict("median_period", y, period=P)
    assert p.ok and p.sign == -1 and abs(p.dt - 8) <= 1.0
    assert p.period == P


def test_every_method_fails_cleanly_on_nan_or_short():
    for m in fitters.METHODS:
        kw = {"period": P} if m == "median_period" else {}
        assert not fitters.predict(m, np.full(48, np.nan), **kw).ok
        assert not fitters.predict(m, np.ones(5), **kw).ok


def test_ar2_fails_on_pure_trend():
    assert not fitters.ar2(np.arange(48, dtype=float) + 1e-9 * np.random.default_rng(0).normal(size=48)).ok


def test_quad_baseline_vertex_ahead():
    # parabola peaking 2 candles after the window end
    t = np.arange(48, dtype=float)
    y = -(t - 49.0) ** 2
    p = fitters.predict("quad", y)
    assert p.ok and p.sign == 1 and abs(p.dt - 2.0) < 1e-9 and abs(p.val - 0.0) < 1e-9
    # vertex behind the window → no prediction
    assert not fitters.predict("quad", -(t - 20.0) ** 2).ok
