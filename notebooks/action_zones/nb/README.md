# action_zones notebooks (Task 9)

Five thin, papermill-parametrized notebooks that import `azlib` (Tasks
1-8, 168 tests) and orchestrate it end to end. No business logic lives
here — every formula/model/estimator is in `azlib`; these notebooks only
load data, call `azlib`, plot, and persist reports/artifacts.

| Notebook | Wraps | Parameters |
|---|---|---|
| `10_action_space.ipynb` | `azlib.space.price_levels`/`label_coeff` | `tf`, `direction` |
| `40_regression_classification.ipynb` | `azlib.models` regression+classification, all 1D/2D/3D groups (`azlib.indicators`) | `tf`, `direction` |
| `60_rr_grid.ipynb` | `azlib.rr.reach_prob_estimator`/`rr_grid`, `azlib.zones.select_levels_safe` | `tf`, `direction`, `fee` |
| `80_zones.ipynb` | `azlib.validate.run_train` + `run_oos` (applied back onto TRAIN) — infer/fuse/Y-sweep/mark zones, write zoned dataset + `ResultsFile` | `tf`, `direction` |
| `90_validate.ipynb` | `azlib.validate.run_train` + `run_oos` (against a separate `oos_env`) + `metrics` | `tf`, `direction`, `oos_env` |

`nb/nbutil.py` is the one shared piece of glue (not azlib): an
artifact-directory convention and a tiny env-file-snapshot helper. See its
own module docstring for why it exists and why it is *not* itself
importable until each notebook's setup cell bootstraps `sys.path` first
(chicken-and-egg — the module lives where it needs `sys.path` support to be
reached).

## Format choice: cleared-output `.ipynb`, not jupytext

Committed as plain `.ipynb` with **every cell's `outputs`/`execution_count`
cleared** (verified by the smoke test's own generation step and re-checked
manually — see task-9-report.md) — chosen over a jupytext-paired `.py`
because this project's Docker image does not install jupytext, and adding
it would be a new dependency to `docker/Dockerfile.experiment` beyond this
task's scope. Cleared outputs mean the committed files diff/review as plain
structured JSON with no output blobs, which was the actual requirement
(jupytext was offered as an alternative way to get the same property, not a
requirement in itself).

## Dataset selection: ambient env, not a notebook parameter

None of these notebooks takes a "which dataset" parameter. The TRAIN
dataset is selected the same way every other `azlib`-consuming Docker
invocation in this project selects one: ambient environment variables
(`ROOT_FOLDER`/`DATA_ROOT`/`DATA_SET_NAME`/`PAIR`/`EXCHANGE_FEE`), set via
`docker compose run --env-file <env>`, which `azlib.loader.load_wide_df()`
(10/40/60) reads directly, and which `80_zones.ipynb`/`90_validate.ipynb`
snapshot into a temp env-FILE (`nbutil.snapshot_env_file`) to satisfy
`azlib.validate.run_train(train_env, tf, direction)`'s own signature (an
env-file PATH, not env vars). `90_validate.ipynb`'s `oos_env` parameter is
the one exception — the OOS dataset is a *second*, independent dataset, so
it genuinely needs its own explicit path, matching
`azlib.validate.run_oos(results_path, oos_env)`'s own signature.

Artifacts (charts/reports/zoned datasets/`ResultsFile`) always write to
`{dataset_folder()}action_zones/{direction}/{tf}/<subdir>/` on the **mounted
volume** (`helpers.dataset_folder()`, env-derived — `ROOT_FOLDER=long`
always resolves under `/trader_data_long`) — never the worktree. `<subdir>`
is `space` / `reg_clf` / `rr` / `results` / `validate`, one per notebook
above (`80_zones.ipynb` and `azlib.validate._results_dir` deliberately
share the exact same `results/` path — see that notebook's own markdown
cell).

## Run mode 1 — memory-safe smoke (automated gate)

`../tests/test_layer9_notebooks.py` (`pytest -k layer9`) runs all 5
notebooks against a **tiny synthetic wide df** (`synthetic_wide_df`, the
same ~750-row fixture every other layer's tests already use — see
`tests/conftest.py`'s Appendix A docstring), written to a dedicated
`azsmoke`/`azsmoke_oos` dataset dir **on the mounted volume**
(`/trader_data_long/train/azsmoke_link_usdt/`, never the worktree) and
cleaned up afterward. **This never touches the real ~6 GB
`df_with_indicators.pkl`.**

```bash
docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm experiment pytest notebooks/action_zones/tests -k layer9 -v
```

Or, to reproduce a single notebook's smoke run by hand (same shape the test
itself drives, via a real `papermill` subprocess — see that test module for
how the ambient env is set):

```bash
docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm -e ROOT_FOLDER=long -e DATA_ROOT=train -e DATA_SET_NAME=azsmoke \
  -e PAIR=link_usdt -e EXCHANGE_FEE=0.001 experiment \
  papermill notebooks/action_zones/nb/10_action_space.ipynb \
  /tmp/10_smoke.ipynb -p tf 15 -p direction long
```

(You would need to have written a matching tiny `df_with_indicators.pkl` to
`/trader_data_long/train/azsmoke_link_usdt/` first — the pytest fixture
does this for you; this hand form is just to show the shape of the
command.)

## Run mode 2 — full 2-year run (heavy, user-triggered, NOT part of the automated gate)

**Memory caveat:** the real 2-year `df_with_indicators.pkl` is ~6 GB and
the 2y pipeline is OOM-sensitive on a 15 GB host (project history). Do not
run this inside the automated test gate. If you hit memory pressure,
subset columns before calling into `azlib` (labels are already chunked
internally — see `indicators/labels.py`), and prefer running one `(tf,
direction)` at a time rather than looping all 6 combinations in one process.

```bash
docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  build experiment   # once, or after docker/Dockerfile.experiment changes

# 10/40/60: dataset via ambient env (--env-file), tf/direction via -p
docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm --env-file configs/nn_train_dataset_2y.env experiment \
  papermill notebooks/action_zones/nb/10_action_space.ipynb \
  /trader_data_long/train/2y_link_usdt/action_zones/out/10_action_space.ipynb \
  -p tf 15 -p direction long

docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm --env-file configs/nn_train_dataset_2y.env experiment \
  papermill notebooks/action_zones/nb/40_regression_classification.ipynb \
  /trader_data_long/train/2y_link_usdt/action_zones/out/40_regression_classification.ipynb \
  -p tf 15 -p direction long

docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm --env-file configs/nn_train_dataset_2y.env experiment \
  papermill notebooks/action_zones/nb/60_rr_grid.ipynb \
  /trader_data_long/train/2y_link_usdt/action_zones/out/60_rr_grid.ipynb \
  -p tf 15 -p direction long -p fee 0.001

# 80: human validation checkpoint (design spec §10) -- review 40's + 80's
# own charts/reports before treating the written ResultsFile as locked.
docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm --env-file configs/nn_train_dataset_2y.env experiment \
  papermill notebooks/action_zones/nb/80_zones.ipynb \
  /trader_data_long/train/2y_link_usdt/action_zones/out/80_zones.ipynb \
  -p tf 15 -p direction long

# 90: TRAIN dataset still via --env-file; OOS dataset via -p oos_env
# (task-9-brief.md's own documented command, verbatim):
docker compose -f docker-compose.yml -f docker-compose.experiment.yml \
  run --rm --env-file configs/nn_train_dataset_2y.env experiment \
  papermill notebooks/action_zones/nb/90_validate.ipynb \
  /trader_data_long/train/2y_link_usdt/action_zones/out/90_validate.ipynb \
  -p oos_env configs/oos2m_dataset.env
```

Repeat with `-p direction short` and/or `-p tf 60` / `-p tf 240` for the
other 5 of the 6 `(tf, direction)` combinations design spec §"Restrictions"
calls for.

## papermill / warning-policy notes

- The `azlib` unit suite (`tests/test_layer{1..8}_*.py`, 168 tests) runs
  pristine under `-W error` (`pytest notebooks/action_zones/tests -q -W
  error`, unchanged by this task).
- `test_layer9_notebooks.py` drives each notebook as a **separate
  `papermill` subprocess**, not the in-process API — a subprocess's own
  Python warnings filter is independent of the parent pytest process's, so
  the outer `-W error` flag has no effect on (and is not defeated by)
  notebook-internal warnings either way. In practice none of these
  notebooks trigger any warnings: `azlib.models` already forces the `Agg`
  Matplotlib backend at import time (no "no display" warning), and every
  `azlib` function reused here already carries its own `-W error`-clean
  guarantee from Tasks 1-8's own test suites.
- Running the full combined suite (`pytest notebooks/action_zones/tests -q
  -W error`, 173 tests) passes clean — verified during this task's own
  development (see task-9-report.md).
