"""The notebook drivers use the production CB kernel instead of their own copies.

``notebooks/candle_bounds_nc/cbnc.py`` and ``notebooks/zone_profitability/
run_extdone.py`` keep their experiment-only parts (sklearn fitting, oos2m
ladder, sticky state) but take the shared maths from
``indicators.candle_bounds``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from indicators.candle_bounds import kernel, params
from tests.unit.data_layer.cb_fixtures import make_wide

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NB_CBNC = os.path.join(ROOT, "notebooks", "candle_bounds_nc")
NB_ZP = os.path.join(ROOT, "notebooks", "zone_profitability")
for _p in (NB_CBNC, NB_ZP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cbnc  # noqa: E402
import run_extdone  # noqa: E402


@pytest.fixture(scope="module")
def wide() -> pd.DataFrame:
    return make_wide(6000)


def test_cbnc_closed_frame_equals_kernel_plus_target(wide):
    for tf in params.TFS:
        for side in params.SIDES:
            got = cbnc.closed_frame(wide, tf, side)
            ref = kernel.closed_frame(wide, tf, side)
            pd.testing.assert_frame_equal(got.drop(columns=["target"]), ref, check_exact=True)
            pd.testing.assert_series_equal(got["target"], ref["d"].shift(-1), check_names=False,
                                           check_exact=True)
            assert list(got.columns) == list(ref.columns) + ["target"]


def test_cbnc_forming_features_is_kernel(wide):
    assert cbnc.forming_features is kernel.forming_features
    for tf in params.TFS:
        for side in params.SIDES:
            pd.testing.assert_frame_equal(cbnc.forming_features(wide, tf, side),
                                          kernel.forming_features(wide, tf, side),
                                          check_exact=True)


def test_cbnc_to_rows_is_kernel():
    assert cbnc._to_rows is kernel.to_rows


def test_cbnc_constants_are_params():
    assert cbnc.FEATURES is params.FEATURES
    assert cbnc.TFS == params.TFS
    assert cbnc.SIDES == params.SIDES
    assert cbnc.ZONE_FRAC == params.ZONE_FRAC
    assert cbnc.TGT_SHRINK_PCT == params.TGT_SHRINK_PCT


def test_run_extdone_d_of_progress_delegates_to_kernel(monkeypatch):
    calls = []

    def _spy(progress, ladder):
        calls.append(ladder)
        return np.zeros(len(progress))

    monkeypatch.setattr(kernel, "ladder_threshold", _spy)
    run_extdone.d_of_progress(np.array([0.5]), [(0.25, 4.0)])
    assert calls == [((0.25, 4.0),)]


def test_run_extdone_import_has_no_side_effects():
    """Importing the driver must not create its output dir (main() does)."""
    assert run_extdone.OUT
    src = open(run_extdone.__file__).read()
    head = src.split("def main")[0]
    assert "os.makedirs" not in head


def test_run_extdone_d_of_progress_matches_kernel():
    prog = np.linspace(0.0, 1.0, 241)
    for tf, ladder in run_extdone.LADDER.items():
        got = run_extdone.d_of_progress(prog, ladder)
        ref = kernel.ladder_threshold(prog, tuple(tuple(s) for s in ladder))
        np.testing.assert_array_equal(got, ref)
        assert got.shape == prog.shape
    # oos ladder is unchanged (the file documents the experiment as run)
    assert run_extdone.LADDER == {
        15: [(0.25, 4.0), (0.50, 2.5), (0.75, 1.0)],
        60: [(0.25, 8.0), (0.50, 5.0), (0.75, 3.0)],
        240: [(0.50, 12.0), (0.75, 8.0)],
    }


def test_notebook_modules_import_without_sklearn():
    """Fresh interpreter, sklearn imports blocked: both drivers still import.

    cwd = repo root: the indicators package reads configs/ relative to cwd."""
    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'sklearn' or name.startswith('sklearn.'):\n"
        "            raise ImportError('sklearn blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path[:0] = [{NB_CBNC!r}, {NB_ZP!r}]\n"
        "import cbnc, run_extdone\n"
        "assert 'sklearn' not in sys.modules\n"
        "print('ok')\n"
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")
