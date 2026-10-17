# tests/unit/data_layer/test_data_preparer_cb.py
# Candle-bounds (CB) stage wired into DataPreparer (phase 18, task 06).
# Heavy indicator passes are stubbed (as in test_data_preparer_chunked.py);
# the wide frame is a real cb_fixtures.make_wide frame so the CB stage runs on
# genuine inputs. Model availability is controlled by monkeypatching
# indicators.candle_bounds.stage.load_models (independent of PAIR).

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

MAIN_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from indicators.candle_bounds import stage  # noqa: E402
from indicators.candle_bounds.stage import expected_columns  # noqa: E402
from tests.unit.data_layer.cb_fixtures import make_models, make_wide  # noqa: E402
from training.data_preparer import DataPreparer  # noqa: E402

DAY = 86_400_000
N_DAYS = 5


@pytest.fixture(scope="module")
def full_wide() -> pd.DataFrame:
    """Real wide frame + CB inputs over N_DAYS from the epoch (chunk-test clock)."""
    return make_wide(n_minutes=N_DAYS * 1440, seed=3, start="1970-01-01 00:00")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PAIR", "link_usdt")
    monkeypatch.setattr(stage, "_WARNED_PAIRS", set())


def _with_models(monkeypatch):
    models = make_models()
    monkeypatch.setattr(stage, "load_models", lambda pair: models)


def _without_models(monkeypatch):
    monkeypatch.setattr(stage, "load_models", lambda pair: None)


class _DummyAttrs:
    def save(self, path):
        with open(path, "w") as f:
            f.write("attrs")


def _raw_df(wide: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame({"c": range(len(wide))}, index=wide.index.copy())
    raw.index.name = "open_time"
    return raw


def _install_stubs(dp: DataPreparer, wide: pd.DataFrame) -> None:
    """Stub the heavy per-row passes; the base frame is the real CB-input frame."""
    raw = _raw_df(wide)
    dp.num_workers = 1
    dp._load_raw_data = lambda path: raw
    dp._build_base_dataframe = lambda r: wide.loc[r.index].copy()

    def fake_base(df, start_ts=None):
        sel = df.index if start_ts is None else df.index[df.index >= start_ts]
        df["base"] = float("nan")
        df.loc[sel, "base"] = [t.value for t in sel]

    def fake_class(df, start_ts=None):
        sel = df.index if start_ts is None else df.index[df.index >= start_ts]
        df["cls"] = float("nan")
        df.loc[sel, "cls"] = [t.value % 7 for t in sel]

    dp._compute_base_indicators = fake_base
    dp._compute_base_attributes = lambda df: None
    dp._global_base_stats = lambda paths: None
    dp._compute_class_indicators = fake_class
    dp._compute_profit_labels = lambda df: df.__setitem__(
        "label", [t.value % 3 for t in df.index]
    )
    dp._data_attributes_container = lambda df: _DummyAttrs()


def _make_dp(tmp_path, name: str, wide: pd.DataFrame) -> DataPreparer:
    d = tmp_path / name
    d.mkdir()
    dp = DataPreparer(
        "configs/indicators_config.yaml",
        str(d / "df_with_indicators.pkl"),
        str(d / "data_attributes.pkl"),
    )
    _install_stubs(dp, wide)
    return dp


def _prepare(tmp_path, name, wide) -> pd.DataFrame:
    dp = _make_dp(tmp_path, name, wide)
    dp.prepare("raw.pkl", data_start_ms=1 * DAY)
    return pd.read_pickle(dp.output_path)


# ---------------------------------------------------------------------------


def test_prepare_writes_cb_columns(tmp_path, monkeypatch, full_wide):
    _with_models(monkeypatch)
    out = _prepare(tmp_path, "ds", full_wide)
    missing = [c for c in expected_columns() if c not in out.columns]
    assert missing == []
    assert out.columns.is_unique
    assert out["15_cb_high"].notna().any()


def test_prepare_cb_dtypes(tmp_path, monkeypatch, full_wide):
    _with_models(monkeypatch)
    out = _prepare(tmp_path, "ds", full_wide)
    for col in ("15_cb_inzone_long", "15_extdone_high", "15_cb_f_trend_long"):
        assert out[col].dtype == bool, (col, out[col].dtype)
    assert out["15_cb_high"].dtype == "float64"


def test_prepare_existing_columns_unchanged(tmp_path, monkeypatch, full_wide):
    _without_models(monkeypatch)
    plain = _prepare(tmp_path, "plain", full_wide)
    _with_models(monkeypatch)
    with_cb = _prepare(tmp_path, "cb", full_wide)
    pd.testing.assert_frame_equal(plain, with_cb[plain.columns])


def test_prepare_no_model_no_cb_columns(tmp_path, monkeypatch, full_wide):
    _without_models(monkeypatch)
    out = _prepare(tmp_path, "ds", full_wide)
    assert [c for c in expected_columns() if c in out.columns] == []
    assert {"base", "cls", "label"}.issubset(out.columns)


def test_prepare_missing_input_fails(tmp_path, monkeypatch, full_wide):
    _with_models(monkeypatch)
    real = stage.required_columns
    monkeypatch.setattr(stage, "required_columns", lambda: real() + ["15_does_not_exist"])
    dp = _make_dp(tmp_path, "ds", full_wide)
    with pytest.raises(ValueError, match="15_does_not_exist"):
        dp.prepare("raw.pkl", data_start_ms=1 * DAY)


def test_chunked_equals_single_pass_cb(tmp_path, monkeypatch, full_wide):
    """G7: CB columns of the chunked path equal the single-pass ones."""
    _with_models(monkeypatch)

    monkeypatch.setenv("CHUNK_MIN_ROWS", "999999999")
    single_dp = _make_dp(tmp_path, "single", full_wide)
    single_dp.prepare_chunked("raw.pkl", 1 * DAY, N_DAYS * DAY)
    single = pd.read_pickle(single_dp.output_path)

    monkeypatch.setenv("CHUNK_MIN_ROWS", "0")
    monkeypatch.setenv("CHUNK_SPAN_DAYS", "1")
    multi_dp = _make_dp(tmp_path, "multi", full_wide)
    multi_dp.prepare_chunked("raw.pkl", 1 * DAY, N_DAYS * DAY)
    chunked = pd.read_pickle(multi_dp.output_path)

    cols = expected_columns()
    assert single.index.equals(chunked.index)
    assert single.index[0] == pd.Timestamp(1 * DAY, unit="ms", tz="UTC")
    pd.testing.assert_frame_equal(
        single[cols], chunked[cols], check_exact=False, rtol=1e-12
    )
    # non-vacuous: the compared frame carries real bounds
    assert single["15_cb_high"].notna().sum() > 0
    assert single["60_cbnc_low"].notna().sum() > 0


def test_rerun_idempotent(monkeypatch, full_wide):
    _with_models(monkeypatch)
    dp = DataPreparer("configs/indicators_config.yaml", "unused.pkl", "unused.pkl")
    frame = full_wide.iloc[: 2 * 1440]
    once = dp._compute_cb_stage(frame)
    twice = dp._compute_cb_stage(once)
    assert once.columns.is_unique and twice.columns.is_unique
    pd.testing.assert_frame_equal(once, twice)
