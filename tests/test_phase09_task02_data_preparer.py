"""Tests for DataPreparer (phase 09 task 02)."""

import pandas as pd
import pytest

from training.data_preparer import DataPreparer


@pytest.fixture
def preparer(tmp_path):
    return DataPreparer(
        "configs/",
        str(tmp_path / "out.pkl"),
        str(tmp_path / "attrs.pkl"),
        nn_output_path=str(tmp_path / "nn.pkl"),
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
