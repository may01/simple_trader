"""Look-ahead gate: a prediction at row t must not change when rows > t change."""
import numpy as np

from experiments.swing_sinusoid import sweep


def test_work_ignores_future_rows():
    rng = np.random.default_rng(0)
    y = np.sin(np.arange(300) / 3.0) + 0.1 * rng.normal(size=300)
    sweep._init(y.copy(), 18.0)
    a = sweep._work((np.array([100, 150]), [16, 32], list(sweep.fitters.METHODS)))
    y2 = y.copy()
    y2[151:] = rng.normal(size=len(y2) - 151) * 50  # wreck the future
    sweep._init(y2, 18.0)
    b = sweep._work((np.array([100, 150]), [16, 32], list(sweep.fitters.METHODS)))
    np.testing.assert_array_equal(a, b)
