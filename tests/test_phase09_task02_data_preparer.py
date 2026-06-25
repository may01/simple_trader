"""Tests for DataPreparer (phase 09 task 02)."""

import numpy as np
import pandas as pd
import pytest

from training.data_preparer import DataPreparer


@pytest.fixture
def preparer(tmp_path):
    return DataPreparer(
        "configs/",
        str(tmp_path / "out.pkl"),
        str(tmp_path / "attrs.pkl"),
    )


class TestLoadRawData:
    def test_accepts_open_time_as_index(self, preparer, tmp_path):
        """Phase-02 spec: open_time is the index of graber_data.pkl."""
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        idx.name = "open_time"
        raw = pd.DataFrame(
            {
                "o": [1.0, 2.0, 3.0],
                "h": [1.5, 2.5, 3.5],
                "l": [0.5, 1.5, 2.5],
                "c": [1.2, 2.2, 3.2],
                "v": [10.0, 20.0, 30.0],
                "taker_base_vol": [5.0, 10.0, 15.0],
            },
            index=idx,
        )
        path = str(tmp_path / "graber_data.pkl")
        raw.to_pickle(path)

        df = preparer._load_raw_data(path)

        assert df.index.name == "open_time"
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
        assert df["open"].tolist() == [1.0, 2.0, 3.0]


class TestRunIndicatorPass:
    """Indicator results must land in the wide DataFrame itself."""

    @pytest.fixture
    def wide_df(self):
        n = 130
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        return pd.DataFrame(
            {
                "1_volume": rng.random(n) * 100,
                "1_buy_volume": rng.random(n) * 50,
                "1_close": np.linspace(10, 11, n),
                "1_is_closed": [True] * n,
            },
            index=idx,
        )

    def test_writes_columns_into_wide_df(self, preparer, wide_df):
        preparer._run_indicator_pass(wide_df, groups=["volume"], tfs=[1])

        assert "1_vol_ma_20" in wide_df.columns, "indicator results must land in the wide df"
        assert "1_vol_sell_ma_20" in wide_df.columns, "dependent fields must see prior results"

    def test_values_match_rolling_window_per_row(self, preparer, wide_df):
        preparer._run_indicator_pass(wide_df, groups=["volume"], tfs=[1])

        # tf=1: every row closed; value at last row = mean of last 20 volumes
        expected_last = wide_df["1_volume"].iloc[-20:].mean()
        assert wide_df["1_vol_ma_20"].iloc[-1] == pytest.approx(expected_last)

        # dependent field consistent with its inputs at the same row
        assert wide_df["1_vol_sell_ma_20"].iloc[-1] == pytest.approx(
            wide_df["1_vol_ma_20"].iloc[-1] - wide_df["1_vol_buy_ma_20"].iloc[-1]
        )

    def test_partial_candle_rows_use_closed_candles_plus_partial(self, preparer):
        """Design decision: non-closed rows get a real calculated value —
        rolling over prior CLOSED candles plus the current partial candle."""
        n = 600
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
        raw = pd.DataFrame(
            {
                "open": np.linspace(10, 11, n),
                "high": np.linspace(10, 11, n) + 0.1,
                "low": np.linspace(10, 11, n) - 0.1,
                "close": np.linspace(10, 11, n),
                "volume": rng.random(n) * 100,
                "taker_base_vol": rng.random(n) * 50,
            },
            index=idx,
        )
        raw.index.name = "open_time"
        wide_df = preparer._build_base_dataframe(raw)

        preparer._run_indicator_pass(wide_df, groups=["volume"], tfs=[5])

        # Pick a partial (non-boundary) row late enough for the 20-candle window
        pos = n - 3
        assert not wide_df["5_is_closed"].iloc[pos]

        closed_before = wide_df[wide_df["5_is_closed"]].loc[: wide_df.index[pos]]
        last_19_closed = closed_before["5_volume"].iloc[-19:]
        partial = wide_df["5_volume"].iloc[pos]
        expected = (last_19_closed.sum() + partial) / 20

        assert wide_df["5_vol_ma_20"].iloc[pos] == pytest.approx(expected)


class TestConfigClassDependencySync:
    """yaml depends_on must match the field classes' declared dependencies —
    topological sort uses only the yaml, so drift breaks compute order."""

    def test_yaml_depends_on_matches_class_dependencies(self):
        from config_loader import load_indicators_config
        from indicators import _FIELD_REGISTRY

        mismatches = []
        for cfg in load_indicators_config():
            factory = _FIELD_REGISTRY.get(cfg.name)
            if factory is None:
                continue  # placeholder field — no class to compare
            field = factory(cfg)
            if set(cfg.depends_on) != set(field.dependencies):
                mismatches.append(
                    f"{cfg.name}: yaml={sorted(cfg.depends_on)} "
                    f"class={sorted(field.dependencies)}"
                )
        assert not mismatches, "yaml/class dependency drift:\n" + "\n".join(mismatches)


class TestRegistryNameInvariant:
    """Registry key must equal the instantiated field's derived name — the
    produced column is {tf}_{name}, and Indicators looks fields up by yaml
    name, so any drift silently swaps a real field for a placeholder."""

    def test_registry_keys_match_field_names(self):
        from config_loader import load_indicators_config
        from indicators import _FIELD_REGISTRY

        cfg_by_name = {cfg.name: cfg for cfg in load_indicators_config()}
        mismatches = []
        for key, factory in _FIELD_REGISTRY.items():
            cfg = cfg_by_name.get(key)
            if cfg is None:
                continue  # registry entry not enabled in yaml
            field = factory(cfg)
            if field.name != key:
                mismatches.append(f"{key} → field.name={field.name!r}")
        assert not mismatches, "registry key / field name drift:\n" + "\n".join(mismatches)
