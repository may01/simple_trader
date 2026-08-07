import numpy as np
import pandas as pd

from experiments.rsi_params_selection import config
from experiments.rsi_params_selection.report import kind_mi, render
from experiments.rsi_params_selection.run_experiment import run_grid


def _result():
    rng = np.random.default_rng(7)

    def frame(n):
        d = {f"diff{w}": rng.normal(0, 1, n) for w in config.WINDOWS}
        for kind in ("plain", "strict"):
            for hn in (1, 2):
                d[f"{kind}_n{hn}_long"] = (rng.random(n) < 0.3).astype(float)
                d[f"{kind}_n{hn}_short"] = (rng.random(n) < 0.3).astype(float)
        return pd.DataFrame(d)

    train = {tf: frame(4000) for tf in config.TFS}
    oos = {tf: frame(700) for tf in config.TFS}
    return run_grid(train, oos)


def test_kind_mi_averages_four_columns():
    res = _result()
    cell = res["cells"][0]
    got = kind_mi(cell, "strict", "oos")
    cols = [f"strict_n{n}_{s}" for n in (1, 2) for s in ("long", "short")]
    want = sum(cell["oos"]["labels"][c]["mi"] for c in cols) / 4
    assert got == want


def test_render_contains_required_sections():
    md = render(_result())
    for heading in ("## Ranking — strict", "## Ranking — non-strict",
                    "## 5 vs 7 classes", "## Technique comparison",
                    "## Flip test (7-class)", "## Row counts & drops"):
        assert heading in md, heading
    assert "| window | tf | technique |" in md
