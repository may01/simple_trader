import numpy as np
import pandas as pd

from experiments.cb_class_bounds import fit as F


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, F.NF))
    c = rng.integers(-1, 2, n)
    y = X[:, 0] * (1 + c) + 0.1 * rng.normal(size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return X, y, c, idx


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, _, _ = _data()
    p = F.ridge_fit(X, y, 10.0)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
    assert np.allclose(F.ridge_predict(p, X), pipe.predict(X), atol=1e-10)


def test_class_model_fallback_equals_uni_and_cells_beat_uni():
    X, y, c, _ = _data()
    uni = F.ridge_fit(X, y, 1.0)
    m = F.fit_class_model(X, y, c, uni, min_n=10_000, alpha=1.0, values=(-1, 0, 1))
    assert all(p["fallback"] for p in m["cells"].values())
    assert np.allclose(F.predict_class_model(m, X, c), F.ridge_predict(uni, X))
    m2 = F.fit_class_model(X, y, c, uni, min_n=100, alpha=1.0, values=(-1, 0, 1))
    mae_u = np.mean(np.abs(y - F.ridge_predict(uni, X)))
    mae_c = np.mean(np.abs(y - F.predict_class_model(m2, X, c)))
    assert mae_c < 0.5 * mae_u


def test_intercept_model_shifts_only():
    X, y, c, _ = _data()
    uni = F.ridge_fit(X, y, 1.0); uni["alpha"] = 1.0
    m = F.fit_intercept_model(X, y, c, uni, 50, (-1, 0, 1))
    for p in m["cells"].values():
        assert np.array_equal(p["coef"], uni["coef"])


def test_block_shuffle_preserves_multiset_and_moves_blocks():
    X, y, c, idx = _data(n=96 * 10)
    s = F.block_shuffle(c, idx, 1)
    assert len(s) == len(c) and sorted(s.tolist()) == sorted(c.tolist())
    assert not np.array_equal(s, c)
    # a whole day keeps its internal order
    day0 = c[:96]
    found = any(np.array_equal(s[i:i + 96], day0) for i in range(0, len(s), 96))
    assert found
