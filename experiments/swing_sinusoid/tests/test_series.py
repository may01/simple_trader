import numpy as np
import pandas as pd
import talib

from experiments.swing_sinusoid import config
from experiments.swing_sinusoid.series import derive, extract_raw


def _wide(n_min=600, tf=15, with_labels=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_min, freq="min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df[f"{tf}_is_closed"] = (np.arange(n_min) % tf == tf - 1)
    for short in config.RAW_COLS:
        df[f"{tf}_{short}"] = rng.normal(size=n_min)
    if with_labels:
        for wide in config.label_cols(tf).values():
            df[wide] = rng.integers(0, 2, size=n_min)
    return df


def test_extract_raw_keeps_only_closed_rows_and_renames():
    df = _wide()
    raw = extract_raw(df, 15)
    assert len(raw) == 600 // 15
    assert set(config.RAW_COLS) <= set(raw.columns)
    assert set(config.label_cols(15)) <= set(raw.columns)
    assert raw.index.name == "ts"
    # values line up with the closed rows
    closed = df[df["15_is_closed"]]
    np.testing.assert_allclose(raw["close"].to_numpy(), closed["15_close"].to_numpy())


def test_extract_raw_without_labels_has_no_label_columns():
    raw = extract_raw(_wide(with_labels=False), 15)
    assert not (set(config.label_cols(15)) & set(raw.columns))


def test_derive_matches_production_recipes():
    raw = extract_raw(_wide(n_min=15 * 200), 15)
    out = derive(raw)
    for s in config.SERIES:
        assert s in out.columns
    rsi = raw["rsi_14"].to_numpy(dtype=float)
    np.testing.assert_allclose(out["rsi_ma8"].to_numpy(), talib.EMA(rsi, timeperiod=8), equal_nan=True)
    np.testing.assert_allclose(out["cci_ma6"].to_numpy(), raw["cci_14"].rolling(6).mean().to_numpy(), equal_nan=True)
    np.testing.assert_allclose(out["close_m_ema25"].to_numpy(), (raw["close"] - raw["ema_25"]).to_numpy())
    # derive never mutates its input
    assert "rsi_ma8" not in raw.columns
