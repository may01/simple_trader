"""Tests for the frozen candle-bound models (numpy predict, JSON loader)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from indicators.candle_bounds import models as cbm
from indicators.candle_bounds.models import BoundModel, load_models
from indicators.candle_bounds.params import FEATURES, SIDES, TFS


def _model(**kw) -> BoundModel:
    base = dict(
        tf=15, side="high",
        mean=np.arange(15, dtype=float) * 0.5,
        scale=np.linspace(1.0, 3.0, 15),
        coef=np.linspace(-0.7, 0.7, 15),
        intercept=0.25, band_pct=0.3, n_train=100,
    )
    base.update(kw)
    return BoundModel(**base)


def _payload(features=None, drop=None) -> dict:
    models = {}
    for tf in TFS:
        for side in SIDES:
            key = f"{tf}_{side}"
            if key == drop:
                continue
            models[key] = {
                "mean": [0.0] * 15, "scale": [1.0] * 15, "coef": [0.1] * 15,
                "intercept": 0.0, "band_pct": 0.5, "n_train": 10,
            }
    return {"pair": "tmp", "train": "2y_az",
            "features": list(features or FEATURES), "models": models}


@pytest.fixture(autouse=True)
def _clear_cache():
    cbm.load_models.cache_clear()
    yield
    cbm.load_models.cache_clear()


def test_predict_matches_manual_formula():
    m = _model()
    X = np.array([np.linspace(-1.0, 2.0, 15), np.arange(15, dtype=float) ** 1.5])
    expected = []
    for row in X:
        acc = m.intercept
        for j in range(15):
            acc += (row[j] - m.mean[j]) / m.scale[j] * m.coef[j]
        expected.append(acc)
    got = m.predict_pct(X)
    assert got.shape == (2,)
    assert np.max(np.abs(got - np.array(expected))) < 1e-12


def test_predict_nan_row_is_nan():
    m = _model()
    X = np.ones((3, 15))
    X[1, 4] = np.nan
    got = m.predict_pct(X)
    assert np.isnan(got[1])
    assert np.isfinite(got[0]) and np.isfinite(got[2])


def test_load_models_missing_file_returns_none():
    assert load_models("no_such_pair") is None


def test_load_models_link_usdt_complete():
    ms = load_models("link_usdt")
    assert ms is not None
    assert set(ms) == {(tf, side) for tf in (15, 60, 240) for side in ("high", "low")}
    for (tf, side), m in ms.items():
        assert (m.tf, m.side) == (tf, side)
        for arr in (m.mean, m.scale, m.coef):
            assert arr.shape == (15,)
        assert np.all(m.scale > 0)
        assert m.band_pct > 0


# Fixed input row (FEATURES order) and the shipped link_usdt predictions for it,
# computed once from models_link_usdt.json: a silent re-export fails here.
_GOLDEN_ROW = [55.0, 0.5, -1.2, 0.01, 0.002, -0.005, 0.3, 0.004, 0.35, 0.28, 0.7,
               0.1, 0.2, -0.1, 0.05]
_GOLDEN_PCT = {
    (15, "high"): -0.03221512457721662,
    (15, "low"): 0.002562535944662415,
    (60, "high"): 0.0012994801267756182,
    (60, "low"): -0.08198509163616606,
    (240, "high"): 0.10219413196430793,
    (240, "low"): -0.5293082086890107,
}


@pytest.mark.parametrize("key", sorted(_GOLDEN_PCT))
def test_shipped_link_usdt_golden_prediction(key):
    m = load_models("link_usdt")[key]
    got = float(m.predict_pct(np.array([_GOLDEN_ROW]))[0])
    assert got == pytest.approx(_GOLDEN_PCT[key], rel=0, abs=1e-9)


def test_load_models_feature_mismatch_raises(tmp_path, monkeypatch):
    feats = list(FEATURES)
    feats[0], feats[1] = feats[1], feats[0]
    p = tmp_path / "models_tmp.json"
    p.write_text(json.dumps(_payload(features=feats)))
    monkeypatch.setattr(cbm, "models_path", lambda pair: str(p))
    with pytest.raises(ValueError):
        load_models("tmp")


def test_load_models_missing_key_raises(tmp_path, monkeypatch):
    p = tmp_path / "models_tmp.json"
    p.write_text(json.dumps(_payload(drop="240_low")))
    monkeypatch.setattr(cbm, "models_path", lambda pair: str(p))
    with pytest.raises(ValueError):
        load_models("tmp")


def test_load_models_bad_array_length_raises(tmp_path, monkeypatch):
    payload = _payload()
    payload["models"]["60_high"]["coef"] = [0.1] * 14
    p = tmp_path / "models_tmp.json"
    p.write_text(json.dumps(payload))
    monkeypatch.setattr(cbm, "models_path", lambda pair: str(p))
    with pytest.raises(ValueError):
        load_models("tmp")


def test_band_pct_matches_stored_meta():
    ms = load_models("link_usdt")
    assert ms[(15, "high")].band_pct == pytest.approx(0.30898300437354337, rel=1e-3)
    assert ms[(15, "low")].band_pct == pytest.approx(0.382271504309902, rel=1e-3)
    assert ms[(60, "high")].band_pct == pytest.approx(0.6307165030404358, rel=1e-3)
    assert ms[(60, "low")].band_pct == pytest.approx(0.7659954524397203, rel=1e-3)
