"""Sinusoid fitters + naive baselines over a trailing window.

Every function takes the trailing window `y` (oldest → newest, the last element
is the current closed candle, time index 0..W-1) and returns a `Prediction` for
the NEXT apex of the series: `dt` candles ahead of the window end (> 0), the
predicted level `val`, the apex `sign` (+1 high / -1 low), plus fit diagnostics.
Nothing here may look past the window.

Model (fitted methods): y(t) = c + b·t + A·sin(ω t + φ).  Apexes of the
oscillatory part sit where ω t + φ = π/2 + nπ; the trend term is added back to
the predicted level only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert as _hilbert

TWO_PI = 2.0 * np.pi


@dataclass
class Prediction:
    method: str
    ok: bool
    dt: float = np.nan        # candles from the window end to the next apex
    val: float = np.nan       # predicted level at the apex
    sign: int = 0             # +1 high, -1 low
    amp: float = np.nan       # fitted / observed swing amplitude
    period: float = np.nan    # fitted period (candles)
    phase: float = np.nan     # phase of the sine at the window end, in [0, 2π)
    resid_std: float = np.nan # std of (y - model) over the window, NaN for baselines
    slope: float = np.nan     # linear trend per candle (fitted methods)

    def as_dict(self) -> dict:
        return asdict(self)


METHODS = ("nls_sine", "ar2", "hilbert", "zero_cross", "mirror", "mean_half_swing", "median_period", "quad")
FITTED = ("nls_sine", "ar2", "hilbert", "zero_cross")
BASELINES = ("mirror", "mean_half_swing", "median_period", "quad")


def _fail(method: str) -> Prediction:
    return Prediction(method=method, ok=False)


def _detrend(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    t = np.arange(len(y), dtype=float)
    b, c = np.polyfit(t, y, 1)
    return y - (c + b * t), float(b), float(c)


def _project(omega: float, phi: float, amp: float, c: float, b: float, t_end: int,
             method: str, resid_std: float) -> Prediction:
    """Next apex strictly after t_end for A·sin(ω t + φ) + c + b t."""
    if not np.isfinite(omega) or omega <= 0 or not np.isfinite(phi) or amp <= 0:
        return _fail(method)
    # candidate extrema: ω t + φ = π/2 + nπ  →  t = (π/2 + nπ − φ)/ω
    n0 = np.floor((omega * t_end + phi - np.pi / 2) / np.pi)
    for n in (n0, n0 + 1, n0 + 2):
        t_apex = (np.pi / 2 + n * np.pi - phi) / omega
        if t_apex > t_end + 1e-9:
            break
    else:
        return _fail(method)
    sign = 1 if np.cos(n * np.pi) > 0 else -1  # sin(π/2 + nπ) = (-1)^n
    val = c + b * t_apex + sign * amp
    period = TWO_PI / omega
    phase_end = float(np.mod(omega * t_end + phi, TWO_PI))
    return Prediction(method, True, float(t_apex - t_end), float(val), int(sign), float(amp),
                      float(period), phase_end, float(resid_std), float(b))


def _dominant_period_fft(d: np.ndarray, min_period: float = 4.0) -> float:
    n = len(d)
    spec = np.abs(np.fft.rfft(d * np.hanning(n)))
    freqs = np.fft.rfftfreq(n)
    spec[0] = 0.0
    spec[freqs > 1.0 / min_period] = 0.0
    k = int(np.argmax(spec))
    if freqs[k] <= 0:
        return float(n)
    return float(1.0 / freqs[k])


def _sin_cos_ls(d: np.ndarray, omega: float) -> tuple[float, float]:
    """Given ω, least-squares A, φ of A·sin(ωt + φ) on detrended d."""
    t = np.arange(len(d), dtype=float)
    X = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coef, *_ = np.linalg.lstsq(X, d, rcond=None)
    a, b = coef  # a sin + b cos = A sin(ωt + φ), A = hypot, φ = atan2(b, a)
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


# ----------------------------------------------------------------- fitted methods

def nls_sine(y: np.ndarray, min_period: float = 4.0) -> Prediction:
    """Non-linear least squares A·sin(ωt+φ)+c+bt.

    Initialised from the dominant FFT bin (ω) and the linear sin/cos regression
    at that ω (A, φ), then one unbounded Levenberg-Marquardt refinement; the
    refined period is clamped to [min_period, 3W] afterwards.
    """
    y = np.asarray(y, dtype=float)
    W = len(y)
    if W < 8 or np.isnan(y).any():
        return _fail("nls_sine")
    d, b0, c0 = _detrend(y)
    p0 = _dominant_period_fft(d, min_period)
    omega0 = TWO_PI / p0
    amp0, phi0 = _sin_cos_ls(d, omega0)
    t = np.arange(W, dtype=float)

    def model(tt, A, omega, phi, c, b):
        return A * np.sin(omega * tt + phi) + c + b * tt

    try:
        popt, _ = curve_fit(model, t, y, p0=[max(amp0, 1e-9), omega0, phi0, c0, b0],
                            method="lm", maxfev=200)
    except (RuntimeError, ValueError):
        popt = np.array([amp0, omega0, phi0, c0, b0])
    A, omega, phi, c, b = popt
    if A < 0:
        A, phi = -A, phi + np.pi
    omega = abs(omega)
    if not (TWO_PI / (3.0 * W) <= omega <= TWO_PI / min_period):
        return _fail("nls_sine")
    sse = float(np.sum((y - model(t, A, omega, phi, c, b)) ** 2))
    return _project(omega, np.mod(phi, TWO_PI), abs(A), c, b, W - 1, "nls_sine", np.sqrt(sse / W))


def ar2(y: np.ndarray, min_period: float = 4.0) -> Prediction:
    """AR(2) on the detrended window; complex roots give period, LS gives A, φ."""
    y = np.asarray(y, dtype=float)
    W = len(y)
    if W < 8 or np.isnan(y).any():
        return _fail("ar2")
    d, b, c = _detrend(y)
    X = np.column_stack([d[1:-1], d[:-2]])
    (a1, a2), *_ = np.linalg.lstsq(X, d[2:], rcond=None)
    disc = a1 * a1 + 4.0 * a2
    if disc >= 0 or a2 >= 0:
        return _fail("ar2")  # real roots: no oscillation in the window
    omega = float(np.arccos(a1 / (2.0 * np.sqrt(-a2))))
    if omega <= 0 or omega > TWO_PI / min_period:  # period below the floor = noise
        return _fail("ar2")
    amp, phi = _sin_cos_ls(d, omega)
    t = np.arange(W, dtype=float)
    resid = d - amp * np.sin(omega * t + phi)
    return _project(omega, np.mod(phi, TWO_PI), amp, c, b, W - 1, "ar2", float(np.std(resid)))


def hilbert(y: np.ndarray, edge: int | None = None, min_period: float = 4.0) -> Prediction:
    """Analytic-signal instantaneous phase/frequency of the detrended window.

    The Hilbert transform is corrupted near the window edges, so the frequency
    is the median phase step over the interior (edge samples on both sides
    dropped) and the end-point phase is extrapolated from `edge` samples before
    the end at that frequency. A known-weak candidate kept for comparison.
    """
    y = np.asarray(y, dtype=float)
    W = len(y)
    if W < 8 or np.isnan(y).any():
        return _fail("hilbert")
    d, b, c = _detrend(y)
    z = _hilbert(d)
    theta = np.unwrap(np.angle(z))          # d ≈ |z| cos(theta)
    edge = edge if edge is not None else max(W // 8, 2)
    dtheta = np.diff(theta)[edge:W - 1 - edge]
    if len(dtheta) < 3:
        return _fail("hilbert")
    omega = float(np.median(dtheta))
    if not np.isfinite(omega) or omega <= 0 or omega > TWO_PI / min_period:
        return _fail("hilbert")
    anchor = W - 1 - edge
    theta_end = theta[anchor] + omega * edge
    phase_end = float(np.mod(theta_end + np.pi / 2, TWO_PI))  # cos θ = sin(θ + π/2)
    phi = phase_end - omega * (W - 1)
    amp = float(np.median(np.abs(z[edge:W - edge])))
    t = np.arange(W, dtype=float)
    resid = d - amp * np.sin(omega * t + phi)
    return _project(omega, phi, amp, c, b, W - 1, "hilbert", float(np.std(resid)))


def zero_cross(y: np.ndarray, min_period: float = 4.0) -> Prediction:
    """Period from (interpolated) zero crossings of the detrended window, phase from the last one."""
    y = np.asarray(y, dtype=float)
    W = len(y)
    if W < 8 or np.isnan(y).any():
        return _fail("zero_cross")
    d, b, c = _detrend(y)
    s = np.sign(d)
    s[s == 0] = 1
    cross = np.where(np.diff(s) != 0)[0]  # crossing between cross[i] and cross[i]+1
    if len(cross) < 2:
        return _fail("zero_cross")
    denom = d[cross] - d[cross + 1]
    frac = np.where(denom != 0, d[cross] / np.where(denom != 0, denom, 1.0), 0.5)
    t_cross = cross + frac
    period = 2.0 * float(np.mean(np.diff(t_cross)))
    if period < min_period:
        return _fail("zero_cross")
    omega = TWO_PI / period
    i = cross[-1]
    rising = d[i + 1] > d[i]
    # sine phase at the crossing: 0 if rising, π if falling
    phi = (0.0 if rising else np.pi) - omega * t_cross[-1]
    seg = d[i + 1:]
    amp = float(np.max(np.abs(seg))) if len(seg) else float(np.std(d) * np.sqrt(2))
    # use the mean half-swing amplitude of the window when the current leg is young
    amp = max(amp, float(np.mean(np.abs(d))) * np.pi / 2)
    t = np.arange(W, dtype=float)
    resid = d - amp * np.sin(omega * t + phi)
    return _project(omega, np.mod(phi, TWO_PI), amp, c, b, W - 1, "zero_cross", float(np.std(resid)))


# ------------------------------------------------------------------- baselines

def _window_extrema(y: np.ndarray, min_prom: float) -> tuple[np.ndarray, np.ndarray]:
    """Alternating 3-point extrema inside the window with |swing| ≥ min_prom."""
    idx, sgn = [], []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            s = 1
        elif y[i] < y[i - 1] and y[i] <= y[i + 1]:
            s = -1
        else:
            continue
        if sgn and sgn[-1] == s:
            if (s > 0 and y[i] > y[idx[-1]]) or (s < 0 and y[i] < y[idx[-1]]):
                idx[-1] = i
            continue
        if sgn and abs(y[i] - y[idx[-1]]) < min_prom:
            continue
        idx.append(i)
        sgn.append(s)
    return np.asarray(idx, int), np.asarray(sgn, int)


def _baseline(method: str, y: np.ndarray, half_len: float | None) -> Prediction:
    y = np.asarray(y, dtype=float)
    W = len(y)
    if W < 8 or np.isnan(y).any():
        return _fail(method)
    idx, sgn = _window_extrema(y, min_prom=0.25 * float(np.std(y)))
    if len(idx) < 2:
        return _fail(method)
    last, prev = idx[-1], idx[-2]
    if half_len is None:
        half_len = float(last - prev)
    t_apex = last + half_len
    if t_apex <= W - 1:
        t_apex = (W - 1) + max(half_len - (W - 1 - last), 0.5)  # overdue: expect it imminently
    sign = int(-sgn[-1])
    val = float(y[prev])  # mirror the previous same-sign apex level
    amp = abs(float(y[last] - y[prev])) / 2.0
    return Prediction(method, True, float(t_apex - (W - 1)), val, sign, amp, 2.0 * half_len,
                      np.nan, np.nan, np.nan)


def mirror(y: np.ndarray, min_period: float = 4.0) -> Prediction:
    """Next apex = last apex + the last half-swing length; level = previous same-sign apex."""
    return _baseline("mirror", y, None)


def mean_half_swing(y: np.ndarray, min_period: float = 4.0) -> Prediction:
    """Next apex = last apex + mean half-swing length inside the window."""
    y = np.asarray(y, dtype=float)
    if len(y) < 8 or np.isnan(y).any():
        return _fail("mean_half_swing")
    idx, _ = _window_extrema(y, min_prom=0.25 * float(np.std(y)))
    if len(idx) < 2:
        return _fail("mean_half_swing")
    return _baseline("mean_half_swing", y, float(np.mean(np.diff(idx))))


def median_period(y: np.ndarray, period: float, min_period: float = 4.0) -> Prediction:
    """Next apex = last apex + P/2 with a FIXED period (train median per series/tf)."""
    return _baseline("median_period", y, period / 2.0)


def quad(y: np.ndarray, n_fit: int = 5, min_period: float = 4.0) -> Prediction:
    """No-sinusoid timing null: quadratic through the last n_fit points, apex at its vertex.

    Vertex behind the window end or further than 2·n_fit ahead → fail. Sign from
    the curvature, level from the vertex value.
    """
    y = np.asarray(y, dtype=float)
    W = len(y)
    if W < max(8, n_fit) or np.isnan(y).any():
        return _fail("quad")
    seg = y[-n_fit:]
    t = np.arange(n_fit, dtype=float)
    a, b, c = np.polyfit(t, seg, 2)
    if a == 0:
        return _fail("quad")
    t_v = -b / (2.0 * a)
    dt = t_v - (n_fit - 1)
    if dt <= 0 or dt > 2.0 * n_fit:
        return _fail("quad")
    sign = -1 if a > 0 else 1   # a > 0 → minimum ahead
    val = a * t_v * t_v + b * t_v + c
    amp = abs(float(val - seg[-1]))
    return Prediction("quad", True, float(dt), float(val), int(sign), amp, np.nan, np.nan, np.nan, np.nan)


def predict(method: str, y: np.ndarray, **kw) -> Prediction:
    fn = {
        "nls_sine": nls_sine, "ar2": ar2, "hilbert": hilbert, "zero_cross": zero_cross,
        "mirror": mirror, "mean_half_swing": mean_half_swing, "median_period": median_period,
        "quad": quad,
    }[method]
    return fn(y, **kw)
