# tests/unit/data_layer/test_data_preparer_chunked.py
# Tests for the chunked, resumable data-preparation path (phase 16).
# Heavy per-row compute methods are stubbed with deterministic per-timestamp
# functions so single-pass vs chunked output can be compared for equivalence
# without running real indicator math (that parity is the Docker job, task 08).

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

MAIN_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from config_loader import chunk_config  # noqa: E402
from training.data_preparer import (  # noqa: E402
    DataPreparer,
    _ChunkProgress,
    _chunk_boundaries,
    _fmt_duration,
    _window_row_count,
)

DAY = 86_400_000
MIN = 60_000


# ---------------------------------------------------------------------------
# Pure helpers: config, boundaries, row count
# ---------------------------------------------------------------------------

class TestChunkConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("CHUNK_SPAN_DAYS", raising=False)
        monkeypatch.delenv("CHUNK_MIN_ROWS", raising=False)
        assert chunk_config() == (30, 200000)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SPAN_DAYS", "3")
        monkeypatch.setenv("CHUNK_MIN_ROWS", "0")
        assert chunk_config() == (3, 0)


class TestBoundaries:
    def test_window_row_count(self):
        assert _window_row_count(0, 14 * DAY) == 14 * 1440
        assert _window_row_count(5 * DAY, 5 * DAY) == 0
        assert _window_row_count(10 * DAY, 0) == 0

    def test_split_14d_span3_last_clamped(self):
        assert _chunk_boundaries(0, 14 * DAY, 3) == [
            (0, 3 * DAY), (3 * DAY, 6 * DAY), (6 * DAY, 9 * DAY),
            (9 * DAY, 12 * DAY), (12 * DAY, 14 * DAY),
        ]

    def test_single_when_below_span(self):
        assert _chunk_boundaries(0, 14 * DAY, 30) == [(0, 14 * DAY)]

    def test_empty_window(self):
        assert _chunk_boundaries(5 * DAY, 5 * DAY, 3) == []
        assert _chunk_boundaries(5 * DAY, 1 * DAY, 3) == []

    def test_exact_multiple(self):
        assert _chunk_boundaries(0, 6 * DAY, 3) == [(0, 3 * DAY), (3 * DAY, 6 * DAY)]


# ---------------------------------------------------------------------------
# Progress + ETA logger
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self, seconds):
        self._s = list(seconds)

    def __call__(self):
        return self._s.pop(0)


class TestChunkProgress:
    def test_eta_linear_midway(self):
        clock = _Clock([0.0, 100.0])  # start, then tick
        p = _ChunkProgress("pass1 base-ind", total=49, now=clock)
        line = p.tick(done=4, rows=4000)
        assert "portion 4/49" in line
        assert "8%" in line
        assert "elapsed 1m40s" in line
        assert "ETA 18m45s" in line
        assert "rows=4000" in line

    def test_first_tick_no_div_by_zero(self):
        clock = _Clock([0.0, 0.0])
        p = _ChunkProgress("pass2 class-ind", total=10, now=clock)
        assert "ETA —" in p.tick(done=0, rows=0)

    def test_zero_total_returns_empty(self):
        clock = _Clock([0.0, 1.0])
        p = _ChunkProgress("x", total=0, now=clock)
        assert p.tick(0, 0) == ""

    def test_fmt_duration_units(self):
        assert _fmt_duration(45) == "45s"
        assert _fmt_duration(100) == "1m40s"
        assert _fmt_duration(3780) == "1h03m"


# ---------------------------------------------------------------------------
# Part-file + manifest path helpers (derive from output_path — no env needed)
# ---------------------------------------------------------------------------

class TestPartPaths:
    def _dp(self):
        return DataPreparer(
            "configs/indicators_config.yaml",
            "/ds/df_with_indicators.pkl",
            "/ds/data_attributes.pkl",
        )

    def test_final_part_zero_padded(self):
        dp = self._dp()
        assert dp._final_part_path(7) == "/ds/df_with_indicators.part_07.pkl"
        assert dp._final_part_path(0) == "/ds/df_with_indicators.part_00.pkl"

    def test_base_part_beside_output(self):
        assert self._dp()._base_part_path(7) == "/ds/df_base.part_07.pkl"

    def test_manifest_beside_output(self):
        assert self._dp()._chunk_manifest_path() == "/ds/chunk_manifest.json"

    def test_config_hash_stable_and_sensitive(self):
        dp = self._dp()
        h1 = dp._chunk_config_hash(0, 14 * DAY, 3)
        assert h1 == dp._chunk_config_hash(0, 14 * DAY, 3)
        assert h1 != dp._chunk_config_hash(0, 14 * DAY, 30)


# ---------------------------------------------------------------------------
# Stubbed orchestration: deterministic per-timestamp compute
# ---------------------------------------------------------------------------

class _DummyAttrs:
    def save(self, path):
        with open(path, "w") as f:
            f.write("attrs")


def _raw_df(days: int) -> pd.DataFrame:
    n = days * 1440
    idx = pd.date_range(pd.Timestamp(0, unit="ms", tz="UTC"), periods=n, freq="1min")
    idx.name = "open_time"
    return pd.DataFrame({"c": range(n)}, index=idx)


def _install_stubs(dp, raw_df):
    """Wire deterministic, per-timestamp stubs for every heavy step."""
    dp.num_workers = 1
    dp._load_raw_data = lambda path: raw_df
    dp._build_base_dataframe = lambda raw: pd.DataFrame(index=raw.index)

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
    dp._merge_nn_output = lambda df: None
    dp._compute_nn_attributes = lambda df: _DummyAttrs()


def _make_dp(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return DataPreparer(
        "configs/indicators_config.yaml",
        str(d / "df_with_indicators.pkl"),
        str(d / "data_attributes.pkl"),
    )


class TestPass1BasePortion:
    def test_owned_window_only(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHUNK_SPAN_DAYS", "1")
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        path = dp._pass1_base_portion(raw, 1, (1 * DAY, 2 * DAY))
        part = pd.read_pickle(path)
        assert part.index.min() == pd.Timestamp(1 * DAY, unit="ms", tz="UTC")
        assert part.index.max() < pd.Timestamp(2 * DAY, unit="ms", tz="UTC")
        assert not part["base"].isna().any()  # owned rows all computed

    def test_skip_if_exists(self, tmp_path):
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        path = dp._pass1_base_portion(raw, 1, (1 * DAY, 2 * DAY))
        mtime = os.path.getmtime(path)
        # corrupt the stub so a recompute would change the file; skip must win
        dp._compute_base_indicators = lambda df, start_ts=None: df.__setitem__("base", -1)
        assert dp._pass1_base_portion(raw, 1, (1 * DAY, 2 * DAY)) == path
        assert os.path.getmtime(path) == mtime


class TestPass2ClassPortion:
    def test_portion0_no_prev_and_skip(self, tmp_path):
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        dp._pass1_base_portion(raw, 0, (0, 1 * DAY))
        path = dp._pass2_class_portion(0, (0, 1 * DAY), prev_base_path=None)
        part = pd.read_pickle(path)
        assert "cls" in part.columns and not part["cls"].isna().any()
        mtime = os.path.getmtime(path)
        assert dp._pass2_class_portion(0, (0, 1 * DAY), None) == path
        assert os.path.getmtime(path) == mtime

    def test_owned_window_drops_lead(self, tmp_path):
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        dp._pass1_base_portion(raw, 0, (0, 1 * DAY))
        dp._pass1_base_portion(raw, 1, (1 * DAY, 2 * DAY))
        path = dp._pass2_class_portion(1, (1 * DAY, 2 * DAY), dp._base_part_path(0))
        part = pd.read_pickle(path)
        # lead rows from portion 0 are dropped — owned window starts at 1d
        assert part.index.min() == pd.Timestamp(1 * DAY, unit="ms", tz="UTC")


class TestMergeParts:
    def _seed_finals(self, dp, raw):
        _install_stubs(dp, raw)
        bases = [dp._pass1_base_portion(raw, i, (i * DAY, (i + 1) * DAY)) for i in range(2)]
        finals = [
            dp._pass2_class_portion(i, (i * DAY, (i + 1) * DAY),
                                    dp._base_part_path(i - 1) if i else None)
            for i in range(2)
        ]
        return finals, bases

    def test_concat_labels_and_cleanup(self, tmp_path):
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(2)
        finals, bases = self._seed_finals(dp, raw)
        dp._merge_parts(finals, bases)
        out = pd.read_pickle(dp.output_path)
        assert out.index.is_monotonic_increasing
        assert {"base", "cls", "label"}.issubset(out.columns)
        assert os.path.exists(dp.attributes_output_path)
        assert all(not os.path.exists(p) for p in finals + bases)  # cleaned

    def test_keep_parts_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHUNK_KEEP_PARTS", "1")
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(2)
        finals, bases = self._seed_finals(dp, raw)
        dp._merge_parts(finals, bases)
        assert all(os.path.exists(p) for p in finals)  # retained


class TestPrepareChunkedOrchestration:
    def test_below_threshold_calls_prepare(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHUNK_MIN_ROWS", "999999999")
        monkeypatch.setenv("CHUNK_SPAN_DAYS", "1")
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        called = {}
        dp.prepare = lambda p, s: called.setdefault("hit", (p, s))
        dp.prepare_chunked("raw.pkl", 0, 5 * DAY)
        assert called["hit"] == ("raw.pkl", 0)
        assert not list(Path(os.path.dirname(dp.output_path)).glob("*.part_*.pkl"))

    def test_stale_manifest_purges_parts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHUNK_MIN_ROWS", "0")
        monkeypatch.setenv("CHUNK_SPAN_DAYS", "1")
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        # seed a stale part + mismatching manifest
        stale = Path(dp._base_part_path(0))
        stale.write_text("stale")
        Path(dp._chunk_manifest_path()).write_text(json.dumps({"config_hash": "OLD"}))
        dp.prepare_chunked("raw.pkl", 0, 5 * DAY)
        assert os.path.exists(dp.output_path)
        stored = json.load(open(dp._chunk_manifest_path()))["config_hash"]
        assert stored == dp._chunk_config_hash(0, 5 * DAY, 1)

    def test_chunked_equals_single(self, tmp_path, monkeypatch):
        raw = _raw_df(5)

        # single-pass baseline
        monkeypatch.setenv("CHUNK_MIN_ROWS", "999999999")
        single = _make_dp(tmp_path, "single")
        _install_stubs(single, raw)
        single.prepare_chunked("raw.pkl", 0, 5 * DAY)
        single_df = pd.read_pickle(single.output_path)

        # forced multi-chunk
        monkeypatch.setenv("CHUNK_MIN_ROWS", "0")
        monkeypatch.setenv("CHUNK_SPAN_DAYS", "1")
        multi = _make_dp(tmp_path, "multi")
        _install_stubs(multi, raw)
        multi.prepare_chunked("raw.pkl", 0, 5 * DAY)
        multi_df = pd.read_pickle(multi.output_path)

        pd.testing.assert_frame_equal(
            single_df.sort_index(), multi_df.sort_index(),
            check_like=True, check_exact=False,
        )

    def test_resume_skips_completed_portions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHUNK_MIN_ROWS", "0")
        monkeypatch.setenv("CHUNK_SPAN_DAYS", "1")
        monkeypatch.setenv("CHUNK_KEEP_PARTS", "1")  # keep parts to inspect resume
        dp = _make_dp(tmp_path, "ds")
        raw = _raw_df(5)
        _install_stubs(dp, raw)
        dp.prepare_chunked("raw.pkl", 0, 5 * DAY)

        finals = sorted(Path(os.path.dirname(dp.output_path)).glob("*.part_*.pkl"))
        base_finals = [p for p in finals if "df_with_indicators.part" in p.name]
        mtimes = {p: os.path.getmtime(p) for p in base_finals}
        victim = base_finals[2]
        victim.unlink()
        os.remove(dp.output_path)

        # spy: pass1 must not recompute surviving portions
        dp.prepare_chunked("raw.pkl", 0, 5 * DAY)
        for p in base_finals:
            if p == victim:
                assert os.path.exists(p)  # recomputed
            else:
                assert os.path.getmtime(p) == mtimes[p]  # untouched
        assert os.path.exists(dp.output_path)
