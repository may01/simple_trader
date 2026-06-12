"""Tests for parallel _run_indicator_pass (NUM_WORKERS support)."""

import numpy as np
import pandas as pd
import pytest

from training.data_preparer import DataPreparer, _split_index_chunks


def _make_preparer(tmp_path):
    return DataPreparer(
        "configs/",
        str(tmp_path / "out.pkl"),
        str(tmp_path / "attrs.pkl"),
        nn_output_path=str(tmp_path / "nn.pkl"),
    )


def _make_wide_df(n=130):
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


class TestSplitIndexChunks:
    def test_splits_into_contiguous_ordered_chunks(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        chunks = _split_index_chunks(idx, 3)

        assert 1 <= len(chunks) <= 3
        # concatenation of chunks reproduces the index exactly, in order
        recombined = chunks[0]
        for chunk in chunks[1:]:
            recombined = recombined.append(chunk)
        assert recombined.equals(idx)

    def test_more_chunks_than_rows_caps_at_rows(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        chunks = _split_index_chunks(idx, 8)

        assert len(chunks) <= 2
        assert sum(len(c) for c in chunks) == 2

    def test_single_chunk_returns_whole_index(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        chunks = _split_index_chunks(idx, 1)

        assert len(chunks) == 1
        assert chunks[0].equals(idx)


class TestParallelIndicatorPass:
    def test_parallel_results_match_serial(self, tmp_path, monkeypatch):
        df_serial = _make_wide_df()
        df_parallel = df_serial.copy()

        monkeypatch.setenv("NUM_WORKERS", "1")
        _make_preparer(tmp_path)._run_indicator_pass(
            df_serial, groups=["volume"], tfs=[1]
        )

        monkeypatch.setenv("NUM_WORKERS", "3")
        _make_preparer(tmp_path)._run_indicator_pass(
            df_parallel, groups=["volume"], tfs=[1]
        )

        pd.testing.assert_frame_equal(df_serial, df_parallel)

    def test_parallel_respects_start_ts_warmup(self, tmp_path, monkeypatch):
        df = _make_wide_df()
        start_ts = df.index[40]

        monkeypatch.setenv("NUM_WORKERS", "3")
        _make_preparer(tmp_path)._run_indicator_pass(
            df, groups=["volume"], tfs=[1], start_ts=start_ts
        )

        # warmup rows get no values of their own
        assert df["1_vol_ma_20"].iloc[:40].isna().all()
        # computed region matches serial single-worker run
        df_serial = _make_wide_df()
        monkeypatch.setenv("NUM_WORKERS", "1")
        _make_preparer(tmp_path)._run_indicator_pass(
            df_serial, groups=["volume"], tfs=[1], start_ts=start_ts
        )
        pd.testing.assert_frame_equal(df, df_serial)


class TestNumWorkersEnv:
    def test_num_workers_env_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NUM_WORKERS", "9")
        monkeypatch.setenv("AVAIABLE_THREADS", "2")
        assert _make_preparer(tmp_path).num_workers == 9

    def test_falls_back_to_avaiable_threads(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NUM_WORKERS", raising=False)
        monkeypatch.setenv("AVAIABLE_THREADS", "7")
        assert _make_preparer(tmp_path).num_workers == 7

    def test_default_when_neither_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NUM_WORKERS", raising=False)
        monkeypatch.delenv("AVAIABLE_THREADS", raising=False)
        assert _make_preparer(tmp_path).num_workers == 4
