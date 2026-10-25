import numpy as np
import pandas as pd

from experiments.cb_class_bounds import build_dataset as B
from experiments.cb_class_bounds import classes as C


def _wide(n_min=16 * 60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_min, freq="min", tz="UTC")
    px = 10 + np.cumsum(rng.normal(0, 0.01, n_min))
    df = pd.DataFrame(index=idx)
    df["1_high"], df["1_low"], df["1_close"] = px + 0.01, px - 0.01, px
    for tf in (15, 60, 240):
        cid = np.arange(n_min) // tf
        g = pd.Series(px).groupby(cid)
        df[f"{tf}_open"] = g.transform("first").to_numpy()
        df[f"{tf}_high"] = g.cummax().to_numpy() + 0.01
        df[f"{tf}_low"] = g.cummin().to_numpy() - 0.01
        df[f"{tf}_close"] = px
        df[f"{tf}_is_closed"] = (np.arange(n_min) % tf) == tf - 1
        df[f"{tf}_rsi_14"] = 50 + np.cumsum(rng.normal(0, 0.5, n_min))
        df[f"{tf}_rsi_ma8"] = pd.Series(df[f"{tf}_rsi_14"]).rolling(8, min_periods=1).mean().to_numpy()
        df[f"{tf}_macd_12_26_9"] = rng.normal(0, 0.01, n_min)
        df[f"{tf}_macd_hist_12_26_9"] = rng.normal(0, 0.01, n_min)
        df[f"{tf}_ema_25"] = pd.Series(px).ewm(span=25).mean().to_numpy()
        df[f"{tf}_cci_14"] = np.cumsum(rng.normal(0, 20, n_min))
        df[f"{tf}_move_class"] = 0
    return df


def test_extract_and_forming_alignment():
    df = _wide()
    raws = {tf: B.extract_part(df, tf) for tf in (15, 60, 240)}
    assert len(raws[15]) == 64 and len(raws[60]) == 16 and len(raws[240]) == 4
    zst = {str(tf): (50.0, 5.0) for tf in (15, 60, 240)}
    fc = B.forming_classes(raws[15], 15, raws[60], 60, zst)
    lc = fc["lc_60"]
    # at a 60-min close minute the last closed-before is the previous 60 candle
    assert lc[3] == -1 and lc[4] == 0 and lc[7] == 0 and lc[8] == 1
    # forming class at a 60-close minute equals the closed class of that 60 candle
    own60 = B.own_classes(raws[60], 60, zst)
    closes15_at_60 = np.arange(3, 64, 4)         # 15-min rows that are also 60-min closes
    assert np.array_equal(fc["cls_mc_60"][closes15_at_60][1:], own60["cls_mc"][1:])
    assert np.array_equal(fc["cls_ec_60"][closes15_at_60][1:], own60["cls_ec"][1:])
    assert np.array_equal(fc["cls_zc_60"][closes15_at_60], own60["cls_zc"])


def test_forming_cci_zone_equals_closed_at_close_minutes():
    df = _wide(n_min=240 * 60, seed=3)
    raws = {tf: B.extract_part(df, tf) for tf in (15, 60, 240)}
    zst = {str(tf): (50.0, 5.0) for tf in (15, 60, 240)}
    fc = B.forming_classes(raws[15], 15, raws[60], 60, zst)
    own60 = B.own_classes(raws[60], 60, zst)
    at = np.arange(3, len(raws[15]), 4)
    assert np.array_equal(fc["cls_cz_60"][at][8:], own60["cls_cz"][8:])
