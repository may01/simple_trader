# tests/unit/data_layer/test_data_preparer.py
# Tests for training/data_preparer.py — DataPreparer class.
# All heavy dependencies (Indicators, DataAttributes, _build_wide_df) are mocked.

from __future__ import annotations

import importlib
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — add main/ to sys.path so we can import training.data_preparer
# ---------------------------------------------------------------------------
MAIN_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))


# ---------------------------------------------------------------------------
# Minimal helpers to build a fake raw DataFrame
# ---------------------------------------------------------------------------

def _make_raw_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal graber_data.pkl-style DataFrame with required columns."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": idx,
            "o": [100.0] * n,
            "h": [101.0] * n,
            "l": [99.0] * n,
            "c": [100.5] * n,
            "v": [1000.0] * n,
        },
        index=idx,
    )


def _make_wide_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal wide DataFrame (as returned by _build_wide_df)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(index=idx)
    for tf in [1, 5, 15, 60, 240, 1440]:
        df[f"{tf}_close"] = 100.5
        df[f"{tf}_open"] = 100.0
        df[f"{tf}_high"] = 101.0
        df[f"{tf}_low"] = 99.0
        df[f"{tf}_volume"] = 1000.0
        df[f"{tf}_buy_volume"] = 500.0
        df[f"{tf}_is_closed"] = True
        df[f"{tf}_open_index"] = idx.floor(f"{tf}min")
    return df


# ---------------------------------------------------------------------------
# Helper to import DataPreparer with heavy deps mocked
# The module uses lazy imports so we patch at source-module level.
# ---------------------------------------------------------------------------

def _import_data_preparer_with_mocks(mock_wide_df=None, mock_indicators=None, mock_da_cls=None):
    """Import DataPreparer with data._build_wide_df, indicators.Indicators,
    and indicators.DataAttributes replaced by mocks.

    Because training.data_preparer uses lazy imports inside methods, we must
    patch the source modules (data, indicators) rather than the training module's
    namespace.
    """
    if mock_wide_df is None:
        mock_wide_df = _make_wide_df()
    if mock_indicators is None:
        mock_indicators = MagicMock()
    if mock_da_cls is None:
        mock_da_cls = MagicMock()

    # Force re-import so patches take effect
    if "training.data_preparer" in sys.modules:
        del sys.modules["training.data_preparer"]

    import training.data_preparer as dp_mod
    return dp_mod


# ===========================================================================
# 1. _load_raw_data — validates required columns
# ===========================================================================

class TestLoadRawData:
    """Tests for DataPreparer._load_raw_data."""

    def _make_preparer(self):
        dp_mod = _import_data_preparer_with_mocks()
        return dp_mod.DataPreparer(
            config_path="configs/indicators_config.yaml",
            output_path="/tmp/test_dp_out.pkl",
            attributes_output_path="/tmp/test_dp_attrs.pkl",
            nn_output_path="/tmp/nn_not_present.pkl",
        )

    def test_load_raw_data_raises_on_missing_required_columns(self, tmp_path):
        """_load_raw_data raises ValueError when a required column is absent."""
        bad_df = pd.DataFrame({"open_time": [1], "o": [1.0]})  # missing h, l, c, v
        pkl_path = str(tmp_path / "graber_data.pkl")
        bad_df.to_pickle(pkl_path)

        dp = self._make_preparer()
        with pytest.raises((ValueError, KeyError)):
            dp._load_raw_data(pkl_path)

    def test_load_raw_data_sets_open_time_as_index(self, tmp_path):
        """_load_raw_data sets open_time column as the DataFrame index."""
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        raw = pd.DataFrame(
            {
                "open_time": idx,
                "o": [1.0, 2.0, 3.0],
                "h": [1.1, 2.1, 3.1],
                "l": [0.9, 1.9, 2.9],
                "c": [1.0, 2.0, 3.0],
                "v": [10.0, 20.0, 30.0],
            }
        )
        pkl_path = str(tmp_path / "graber_data.pkl")
        raw.to_pickle(pkl_path)

        dp = self._make_preparer()
        result = dp._load_raw_data(pkl_path)

        # open_time values should now be in the index
        assert list(result.index) == list(idx)


# ===========================================================================
# 2. _merge_nn_output — no-op when file absent
# ===========================================================================

class TestMergeNnOutput:
    """Tests for DataPreparer._merge_nn_output."""

    def _make_preparer(self, nn_output_path: str):
        dp_mod = _import_data_preparer_with_mocks()
        return dp_mod.DataPreparer(
            config_path="configs/indicators_config.yaml",
            output_path="/tmp/test_dp_out.pkl",
            attributes_output_path="/tmp/test_dp_attrs.pkl",
            nn_output_path=nn_output_path,
        )

    def test_merge_nn_output_noop_when_absent(self, tmp_path):
        """_merge_nn_output does nothing when nn_output_path does not exist."""
        absent_path = str(tmp_path / "does_not_exist.pkl")
        dp = self._make_preparer(absent_path)

        wide_df = _make_wide_df()
        cols_before = set(wide_df.columns)
        dp._merge_nn_output(wide_df)

        assert set(wide_df.columns) == cols_before

    def test_merge_nn_output_merges_columns_when_present(self, tmp_path):
        """_merge_nn_output left-joins columns from df_with_nn.pkl onto df."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")

        # nn_df has same index plus extra column
        nn_df = pd.DataFrame({"nn_pred": [0.1, 0.2, 0.3, 0.4, 0.5]}, index=idx)
        nn_path = str(tmp_path / "df_with_nn.pkl")
        nn_df.to_pickle(nn_path)

        dp = self._make_preparer(nn_path)

        wide_df = _make_wide_df(5)
        dp._merge_nn_output(wide_df)

        assert "nn_pred" in wide_df.columns


# ===========================================================================
# 3. prepare() pipeline — all compute steps called in order
# ===========================================================================

class TestPreparePipeline:
    """Tests that prepare() calls sub-steps in the correct order."""

    def test_prepare_calls_steps_in_order(self, tmp_path):
        """prepare() must call the 7 pipeline steps sequentially."""
        call_order = []

        raw_pkl = str(tmp_path / "graber_data.pkl")
        _make_raw_df().to_pickle(raw_pkl)

        output_path = str(tmp_path / "df_with_indicators.pkl")
        attrs_path = str(tmp_path / "data_attributes.pkl")
        nn_path = str(tmp_path / "df_with_nn.pkl")  # absent on purpose

        mock_wide_df = _make_wide_df()
        mock_data_attrs = MagicMock()
        mock_data_attrs.compute = MagicMock()
        mock_data_attrs.compute_nn_stats = MagicMock()
        mock_data_attrs.save = MagicMock()

        dp_mod = _import_data_preparer_with_mocks()
        dp = dp_mod.DataPreparer(
            config_path="configs/indicators_config.yaml",
            output_path=output_path,
            attributes_output_path=attrs_path,
            nn_output_path=nn_path,
        )

        # Patch instance methods to track order
        dp._load_raw_data = lambda path: (call_order.append("load_raw"), pd.DataFrame())[1]
        dp._build_base_dataframe = lambda raw: (call_order.append("build_base"), mock_wide_df)[1]
        dp._compute_base_indicators = lambda df: call_order.append("base_indicators")
        dp._compute_base_attributes = lambda df: call_order.append("base_attributes")
        dp._compute_class_indicators = lambda df: call_order.append("class_indicators")
        dp._merge_nn_output = lambda df: call_order.append("merge_nn")
        dp._compute_nn_attributes = lambda df: (call_order.append("nn_attributes"), mock_data_attrs)[1]

        dp.prepare(raw_pkl)

        expected = [
            "load_raw",
            "build_base",
            "base_indicators",
            "base_attributes",
            "class_indicators",
            "merge_nn",
            "nn_attributes",
        ]
        assert call_order == expected, f"Got: {call_order}"


# ===========================================================================
# 4. Atomic save — no .tmp file remains after prepare()
# ===========================================================================

class TestAtomicSave:
    """Verifies that prepare() saves df atomically (no .tmp leftover)."""

    def _run_prepare_with_mocks(self, tmp_path):
        """Run prepare() with all heavy deps mocked, return output_path."""
        raw_pkl = str(tmp_path / "graber_data.pkl")
        _make_raw_df().to_pickle(raw_pkl)

        output_path = str(tmp_path / "df_with_indicators.pkl")
        attrs_path = str(tmp_path / "data_attributes.pkl")
        nn_path = str(tmp_path / "df_with_nn.pkl")

        mock_wide_df = _make_wide_df()
        mock_data_attrs = MagicMock()
        mock_data_attrs.compute = MagicMock()
        mock_data_attrs.compute_nn_stats = MagicMock()
        mock_data_attrs.save = MagicMock()

        dp_mod = _import_data_preparer_with_mocks()
        dp = dp_mod.DataPreparer(
            config_path="configs/indicators_config.yaml",
            output_path=output_path,
            attributes_output_path=attrs_path,
            nn_output_path=nn_path,
        )
        # Bypass heavy steps, keep only I/O logic
        dp._load_raw_data = lambda path: pd.DataFrame({"x": [1]})
        dp._build_base_dataframe = lambda raw: mock_wide_df
        dp._compute_base_indicators = lambda df: None
        dp._compute_base_attributes = lambda df: None
        dp._compute_class_indicators = lambda df: None
        dp._merge_nn_output = lambda df: None
        dp._compute_nn_attributes = lambda df: mock_data_attrs

        dp.prepare(raw_pkl)
        return output_path

    def test_no_tmp_file_after_prepare(self, tmp_path):
        """After prepare() completes, no .tmp file should exist for output_path."""
        output_path = self._run_prepare_with_mocks(tmp_path)

        assert os.path.exists(output_path), "df_with_indicators.pkl was not created"
        assert not os.path.exists(output_path + ".tmp"), ".tmp file was not cleaned up"

    def test_saved_output_is_plain_dataframe(self, tmp_path):
        """df_with_indicators.pkl must be a plain DataFrame (never a tuple)."""
        output_path = self._run_prepare_with_mocks(tmp_path)

        result = pd.read_pickle(output_path)
        assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"


# ===========================================================================
# 5. _compute_class_indicators — only TFs [15, 60, 240, 1440]
# ===========================================================================

class TestComputeClassIndicators:
    """_compute_class_indicators must skip tf=1 and tf=5."""

    def test_class_indicators_only_runs_on_allowed_tfs(self):
        """Verify compute_group is called only for TFs [15, 60, 240, 1440].

        Because indicators.py imports talib at module level (unavailable here),
        we inject mock modules for 'talib', 'indicators', and 'data' into
        sys.modules before any import of training.data_preparer occurs.
        """
        # Build mock modules at sys.modules level to prevent talib import
        mock_talib = MagicMock()
        mock_indicators_mod = MagicMock()
        mock_ind = MagicMock()
        mock_indicators_mod.Indicators = mock_ind
        mock_indicators_mod.DataAttributes = MagicMock()
        mock_indicators_mod.build_indicator_input = MagicMock()

        mock_data_mod = MagicMock()
        mock_wdp = MagicMock()
        mock_data_mod.WideDataPoint = mock_wdp
        mock_data_mod._build_wide_df = MagicMock(return_value=_make_wide_df())

        modules_to_inject = {
            "talib": mock_talib,
        }
        # Only inject indicators/data if not already loaded with real talib
        if "indicators" not in sys.modules:
            modules_to_inject["indicators"] = mock_indicators_mod
        if "data" not in sys.modules:
            modules_to_inject["data"] = mock_data_mod

        # Always clean training.data_preparer so it picks up our mocks
        if "training.data_preparer" in sys.modules:
            del sys.modules["training.data_preparer"]

        saved = {}
        for key, val in modules_to_inject.items():
            saved[key] = sys.modules.get(key)
            sys.modules[key] = val

        try:
            import training.data_preparer as dp_mod

            dp = dp_mod.DataPreparer(
                config_path="configs/indicators_config.yaml",
                output_path="/tmp/test_dp_out.pkl",
                attributes_output_path="/tmp/test_dp_attrs.pkl",
                nn_output_path="/tmp/nn_not_present.pkl",
            )

            wide_df = _make_wide_df()

            # Now inject mocks for what _run_indicator_pass lazy-imports
            saved2 = {}
            saved2["indicators"] = sys.modules.get("indicators")
            saved2["data"] = sys.modules.get("data")
            sys.modules["indicators"] = mock_indicators_mod
            sys.modules["data"] = mock_data_mod
            mock_wdp.return_value = MagicMock()

            try:
                dp._compute_class_indicators(wide_df)
            finally:
                for k, v in saved2.items():
                    if v is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = v

            calls = mock_ind.compute_group.call_args_list
        finally:
            for key, val in saved.items():
                if val is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = val
            sys.modules.pop("training.data_preparer", None)

        called_tfs = [c[0][1] for c in calls]

        assert 1 not in called_tfs, "tf=1 should be skipped for class indicators"
        assert 5 not in called_tfs, "tf=5 should be skipped for class indicators"
        for tf in [15, 60, 240, 1440]:
            assert tf in called_tfs, f"tf={tf} should be processed for class indicators"

    def test_class_tfs_constant(self):
        """CLASS_TFS exported constant should be [15, 60, 240, 1440]."""
        dp_mod = _import_data_preparer_with_mocks()
        assert dp_mod.CLASS_TFS == [15, 60, 240, 1440]
