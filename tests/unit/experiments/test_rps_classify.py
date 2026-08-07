import numpy as np
import pytest

from experiments.rsi_params_selection.classify import apply_cuts, fit_cuts


def _x():
    return np.random.default_rng(1).normal(0.5, 2.0, 100_000)  # off-center on purpose


def test_quantile_5_balanced_populations():
    x = _x()
    cuts = fit_cuts(x, "quantile", 5)
    assert len(cuts) == 4 and np.all(np.diff(cuts) > 0)
    cls = apply_cuts(x, cuts)
    share = np.bincount(cls + 2, minlength=5) / len(x)
    assert np.allclose(share, [0.10, 0.20, 0.40, 0.20, 0.10], atol=0.01)


def test_quantile_7_extra_tail_share():
    x = _x()
    cls = apply_cuts(x, fit_cuts(x, "quantile", 7))
    share = np.bincount(cls + 3, minlength=7) / len(x)
    assert abs(share[0] - 0.02) < 0.005 and abs(share[6] - 0.02) < 0.005


def test_sym0_centered_on_zero_not_mean():
    x = _x()  # mean 0.5 — sym0 must ignore it
    cuts = fit_cuts(x, "sym0", 5)
    s = x.std()
    assert np.allclose(cuts, [-1.0 * s, -0.3 * s, 0.3 * s, 1.0 * s], rtol=1e-6)


def test_zscore_centered_on_mean():
    x = _x()
    cuts = fit_cuts(x, "zscore", 7)
    mu, s = x.mean(), x.std()
    want = [mu - 2 * s, mu - 1 * s, mu - 0.5 * s,
            mu + 0.5 * s, mu + 1 * s, mu + 2 * s]
    assert np.allclose(cuts, want, rtol=1e-6)


def test_apply_cuts_centered_ids_and_boundaries():
    cuts = np.array([-1.0, -0.3, 0.3, 1.0])
    cls = apply_cuts(np.array([-5.0, -0.5, 0.0, 0.5, 5.0]), cuts)
    assert cls.tolist() == [-2, -1, 0, 1, 2]


def test_apply_cuts_rejects_nan():
    with pytest.raises(ValueError):
        apply_cuts(np.array([0.0, np.nan]), np.array([-1.0, -0.3, 0.3, 1.0]))


def test_fit_cuts_unknown_technique():
    with pytest.raises(ValueError):
        fit_cuts(np.zeros(10), "tree", 5)
