import numpy as np
import pandas as pd
import pytest

from indicators.library.classification import MOVE_CLASS_CUTS
from scripts.move_class7_materialize import recompute_move_class


def _df(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="min")
    df = pd.DataFrame(index=idx)
    for tf in (15, 60, 240):
        df[f"{tf}_rsi_ma8_diff"] = np.linspace(-4, 4, n)
        df[f"{tf}_move_class"] = 0  # legacy values, any dtype
    return df


def test_recomputes_seven_classes_and_reports_changes():
    df = _df()
    changed = recompute_move_class(df)
    assert set(changed) == {"15_move_class", "60_move_class", "240_move_class"}
    for tf in (15, 60, 240):
        cls = df[f"{tf}_move_class"]
        x = df[f"{tf}_rsi_ma8_diff"]
        assert cls.min() == -3 and cls.max() == 3
        assert (cls[x < MOVE_CLASS_CUTS[tf][0]] == -3).all()
        assert (cls[x > MOVE_CLASS_CUTS[tf][-1]] == 3).all()


def test_idempotent_second_run_is_noop():
    df = _df()
    recompute_move_class(df)
    assert recompute_move_class(df) == []


def test_drops_interim_sym7_columns():
    df = _df()
    df["15_move_class_sym7"] = 0
    changed = recompute_move_class(df)
    assert "-15_move_class_sym7" in changed
    assert "15_move_class_sym7" not in df.columns


def test_nan_diff_maps_to_neutral_and_1440_untouched():
    df = _df()
    df.loc[df.index[:5], "15_rsi_ma8_diff"] = np.nan
    df["1440_move_class"] = -2  # legacy 1440 column: left as-is (no fit)
    recompute_move_class(df)
    assert (df["15_move_class"].iloc[:5] == 0).all()
    assert (df["1440_move_class"] == -2).all()


def test_field_and_materializer_agree():
    pytest.importorskip("plotly")  # LiveDataPoint chain stays lightweight; guard uniformly
    from data import LiveDataPoint
    from indicators import MoveClassField

    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    diffs = np.linspace(-4, 4, n)
    frame = pd.DataFrame({
        "15_close": 20.0, "15_rsi_ma8": 50.0,
        "15_rsi_ma8_diff": diffs, "15_is_closed": True,
    }, index=idx)
    via_field = MoveClassField().compute(LiveDataPoint({15: frame}), 15)

    wide = pd.DataFrame({"15_rsi_ma8_diff": diffs, "15_move_class": 0}, index=idx)
    recompute_move_class(wide)
    assert list(via_field) == list(wide["15_move_class"])
