import numpy as np
import pandas as pd

from experiments.rsi_params_selection import config
from experiments.rsi_params_selection.run_experiment import run_grid


def _fake_tf_frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"diff{w}": rng.normal(0, 1, n) for w in config.WINDOWS}
    for kind in ("plain", "strict"):
        for hn in (1, 2):
            data[f"{kind}_n{hn}_long"] = (rng.random(n) < 0.3).astype(float)
            data[f"{kind}_n{hn}_short"] = (rng.random(n) < 0.3).astype(float)
    return pd.DataFrame(data)


def test_run_grid_full_cell_coverage_and_oos_uses_train_cuts():
    train = {tf: _fake_tf_frame(seed=tf) for tf in config.TFS}
    oos = {tf: _fake_tf_frame(n=800, seed=tf + 100) for tf in config.TFS}
    res = run_grid(train, oos)
    assert len(res["cells"]) == 54
    keys = {(c["window"], c["tf"], c["technique"], c["n_classes"])
            for c in res["cells"]}
    assert len(keys) == 54
    for c in res["cells"]:
        assert len(c["cuts"]) == c["n_classes"] - 1
        assert sum(c["train"]["population"].values()) == c["train"]["n_rows"]
        assert sum(c["oos"]["population"].values()) == c["oos"]["n_rows"]
    # oos scored with train cuts: same cell, same cuts list is the contract
    cell = res["cells"][0]
    assert cell["train"]["n_rows"] == 6000 and cell["oos"]["n_rows"] == 800


def test_run_grid_counts_nan_feature_drops():
    train = {tf: _fake_tf_frame(seed=tf) for tf in config.TFS}
    train[15].loc[train[15].index[:100], "diff8"] = np.nan
    oos = {tf: _fake_tf_frame(n=800, seed=tf + 100) for tf in config.TFS}
    res = run_grid(train, oos)
    assert res["meta"]["dropped_nan_feature"]["train"]["15"]["8"] == 100
    cell = next(c for c in res["cells"]
                if c["tf"] == 15 and c["window"] == 8)
    assert cell["train"]["n_rows"] == 5900
