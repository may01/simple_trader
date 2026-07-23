"""tdlib/models.py -- Task 4: classifiers (Layer 4, part 2).

Fits the two candidate classifiers this experiment compares (logistic
regression as the simple/interpretable baseline, gradient-boosted trees as
the higher-capacity comparison), evaluates them with decile-precision/lift
metrics tuned for a trading signal (not just plain accuracy), ranks
features by gbc permutation importance, and persists a fitted bundle to
disk in a form that survives a save/load round trip byte-for-byte.

Four things live here:

1. ``fit_classifiers(X_tr, y_tr)`` -- fit ONE ``FreezeStats`` (Task 3) on
   the train split, transform once, fit both ``LogisticRegression`` and
   ``GradientBoostingClassifier`` on that SAME transformed matrix.
   Hyperparameters are azlib.models' own precedent (zone-selection-
   experiment worktree's ``notebooks/action_zones/azlib/models.py``
   ``_GB_PARAMS``), copied verbatim per task-4-brief.md.

2. ``eval_classifier(bundle, name, X, y)`` -- roc_auc/accuracy plus
   decile-based precision and lift: "if you only acted on the model's top
   decile of predictions, what fraction would actually be long (vs the
   unconditional base rate)?" -- ``lift_long``/``lift_short`` answer that
   question directly, which plain accuracy/AUC do not.

3. ``importance_table(bundle, X, y, n_repeats=5)`` -- permutation
   importance of the gbc, the model-agnostic "how much does shuffling this
   one column hurt roc_auc" ranking.

4. ``save_bundle``/``load_bundle`` -- a bundle spans 4 files (freeze.json +
   2 joblib models + feature_order.json). The 4th file is NOT cosmetic:
   ``FreezeStats.to_json`` alphabetizes its keys (see its own docstring),
   so ``freeze.json`` alone loses the ORIGINAL fit column order --
   ``feature_order.json`` records that order explicitly so ``load_bundle``
   can restore it. This matters because it is not merely aesthetic: as of
   sklearn 1.9, an estimator fit on a DataFrame records
   ``feature_names_in_``, and calling ``predict_proba`` with a DataFrame
   whose columns are present but in a DIFFERENT order raises
   ``ValueError`` rather than silently reordering (verified empirically,
   see task-4-report.md) -- so a reload that skipped restoring the
   original order would leave the loaded bundle unable to score anything.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from tdlib.features import FreezeStats

# Gradient-boosting hyperparams: azlib.models._GB_PARAMS precedent, task-4-
# brief's exact numbers. Logistic: brief's exact max_iter/random_state (C
# left at sklearn's own default, 1.0 -- the brief names only these two).
_GBC_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}
_LOGISTIC_PARAMS = {"max_iter": 1000, "random_state": 42}

# Single source of truth for save_bundle/load_bundle's 4 filenames, so the
# two functions can never silently drift apart on what a bundle directory
# is supposed to contain.
_BUNDLE_FILES = {
    "freeze": "freeze.json",
    "logistic": "logistic.joblib",
    "gbc": "gbc.joblib",
    "feature_order": "feature_order.json",
}


def fit_classifiers(X_tr: pd.DataFrame, y_tr: pd.Series) -> dict:
    """Fit a bundle: ``{"freeze": FreezeStats, "logistic": LogisticRegression,
    "gbc": GradientBoostingClassifier}``.

    ``FreezeStats`` is fit on ``X_tr`` and transformed exactly ONCE; both
    classifiers are fit on that SAME resulting matrix ``Z`` -- never two
    independently-rescaled matrices (``eval_classifier``/
    ``importance_table`` both assume every model in a bundle shares one
    consistent transform).
    """
    freeze = FreezeStats().fit(X_tr)
    Z = freeze.transform(X_tr)

    logistic = LogisticRegression(**_LOGISTIC_PARAMS)
    logistic.fit(Z, y_tr)

    gbc = GradientBoostingClassifier(**_GBC_PARAMS)
    gbc.fit(Z, y_tr)

    return {"freeze": freeze, "logistic": logistic, "gbc": gbc}


def eval_classifier(bundle: dict, name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate ``bundle[name]`` (``"logistic"`` or ``"gbc"``) on ``(X, y)``.

    ``X`` is transformed through ``bundle["freeze"]`` -- this function only
    ever calls ``.transform()``, NEVER ``.fit()`` -- so scoring an OOS/
    holdout split always reuses the TRAIN split's own median/scale, never
    the holdout's own distribution (``FreezeStats``'s module docstring
    explains why that matters: an OOS frame median-imputing against its own
    distribution would leak that frame's own statistics into what is
    supposed to be an honest out-of-sample evaluation).

    Metrics: ``roc_auc``, ``acc`` (0.5 threshold), ``base_rate`` (mean y),
    ``n``, ``prec_top_decile``/``prec_bottom_decile`` (fraction y==1 among
    the top ``ceil(n/10)`` rows by predicted probability / fraction y==0
    among the bottom ``ceil(n/10)``), ``lift_long``/``lift_short`` (those
    two precisions minus their respective unconditional base rates -- "how
    much better than random does acting on the model's extremes do").

    Ranking ties: ONE stable ascending sort of the predicted probabilities
    (``np.argsort(p, kind="stable")``) supplies BOTH ends -- the first
    ``k`` positions are the bottom decile, the last ``k`` are the top decile
    -- so tied probabilities keep their original row order and top/bottom
    can never disagree with each other about where a tied row landed.

    ``n`` is always the real row count. Every OTHER value is
    ``float("nan")`` when ``y`` is single-class (roc_auc/lift/precision are
    all genuinely undefined without both classes present) -- deliberately
    NOT lumped in with that NaN-out, since the row count itself is still
    real, useful information about what was (attempted to be) scored.
    """
    n = int(len(y))
    if y.nunique(dropna=True) < 2:
        return {
            "roc_auc": float("nan"), "acc": float("nan"), "base_rate": float("nan"), "n": n,
            "prec_top_decile": float("nan"), "prec_bottom_decile": float("nan"),
            "lift_long": float("nan"), "lift_short": float("nan"),
        }

    Z = bundle["freeze"].transform(X)
    model = bundle[name]
    p = model.predict_proba(Z)[:, 1]

    y_arr = y.to_numpy()
    base_rate = float(y_arr.mean())
    roc_auc = float(roc_auc_score(y_arr, p))
    acc = float(accuracy_score(y_arr, (p >= 0.5).astype(int)))

    k = math.ceil(n / 10)
    order = np.argsort(p, kind="stable")  # ascending; ties keep original row order
    bottom_idx = order[:k]
    top_idx = order[-k:]

    prec_top_decile = float((y_arr[top_idx] == 1).mean())
    prec_bottom_decile = float((y_arr[bottom_idx] == 0).mean())

    return {
        "roc_auc": roc_auc,
        "acc": acc,
        "base_rate": base_rate,
        "n": n,
        "prec_top_decile": prec_top_decile,
        "prec_bottom_decile": prec_bottom_decile,
        "lift_long": prec_top_decile - base_rate,
        "lift_short": prec_bottom_decile - (1.0 - base_rate),
    }


def importance_table(bundle: dict, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance of the ``gbc`` model, on ``X`` transformed
    through ``bundle["freeze"]`` (same never-refit contract as
    ``eval_classifier``).

    ``random_state=42`` is fixed (not caller-configurable): two calls with
    the same ``(bundle, X, y, n_repeats)`` return byte-identical
    ``imp_mean``/``imp_std`` columns.

    Returns exactly the 3 columns ``feature``/``imp_mean``/``imp_std``
    (``feature`` a normal column here, NOT the index -- unlike
    ``univariate_screen``'s result), sorted by ``imp_mean`` descending.
    """
    Z = bundle["freeze"].transform(X)
    result = permutation_importance(bundle["gbc"], Z, y, n_repeats=n_repeats, random_state=42, scoring="roc_auc")
    table = pd.DataFrame({
        "feature": Z.columns,
        "imp_mean": result.importances_mean,
        "imp_std": result.importances_std,
    })
    return table.sort_values("imp_mean", ascending=False).reset_index(drop=True)


def save_bundle(bundle: dict, dir_path) -> None:
    """Persist a ``fit_classifiers`` bundle across 4 files in ``dir_path``
    (created, parents included, if it doesn't already exist) -- see the
    module docstring for why all 4 (including ``feature_order.json``) are
    required for a faithful round trip.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    freeze: FreezeStats = bundle["freeze"]
    freeze.to_json(str(dir_path / _BUNDLE_FILES["freeze"]))

    joblib.dump(bundle["logistic"], dir_path / _BUNDLE_FILES["logistic"])
    joblib.dump(bundle["gbc"], dir_path / _BUNDLE_FILES["gbc"])

    feature_order = list(freeze.stats.keys())
    with open(dir_path / _BUNDLE_FILES["feature_order"], "w") as f:
        json.dump(feature_order, f)


def load_bundle(dir_path) -> dict:
    """Exact inverse of ``save_bundle``: returns ``{"freeze", "logistic",
    "gbc"}``, matching ``fit_classifiers``'s own return shape.

    ``bundle["freeze"].stats`` is rebuilt in ``feature_order.json``'s order
    -- NOT ``freeze.json``'s own alphabetized order (see the module
    docstring: this is what makes ``bundle["freeze"].transform(X)`` hand
    the two loaded sklearn estimators a matrix with the SAME column order
    they were originally fit on).

    Every one of the 4 expected files is checked for existence up front;
    if any are missing, raises ``FileNotFoundError`` naming EVERY missing
    one at once (not just the first) -- mirrors ``tdlib.config.slim_columns``'s
    own "name every missing column, not just the first" convention.
    """
    dir_path = Path(dir_path)

    missing = [fname for fname in _BUNDLE_FILES.values() if not (dir_path / fname).exists()]
    if missing:
        raise FileNotFoundError(f"load_bundle: missing file(s) in {dir_path}: {missing!r}")

    freeze = FreezeStats.from_json(str(dir_path / _BUNDLE_FILES["freeze"]))
    with open(dir_path / _BUNDLE_FILES["feature_order"]) as f:
        feature_order = json.load(f)
    freeze.stats = {col: freeze.stats[col] for col in feature_order}

    logistic = joblib.load(dir_path / _BUNDLE_FILES["logistic"])
    gbc = joblib.load(dir_path / _BUNDLE_FILES["gbc"])

    return {"freeze": freeze, "logistic": logistic, "gbc": gbc}
