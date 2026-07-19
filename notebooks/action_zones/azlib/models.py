"""azlib/models.py — Task 4: regression + classification (Layer 4).

Implements design spec §9
(external/docs/superpowers/specs/2026-07-18-zone-selection-design.md):
indicator-attribute -> ``label_coeff`` regression AND classification,
run for every SAME-SPACE 1D/2D/3D attribute group of a single indicator
(RSI+RSI, MACD+MACD, MA+MA — never cross-indicator), per ``groups`` below.

Two model families, each with a small fixed set of allowed ``kind``s
(YAGNI — the brief's exact list, nothing more):

- Regression (§9.1): ``fit_regression``/``predict_reg``, ``kind`` in
  ``{"linear", "poly2", "gbr"}``. Predicts ``label_coeff`` mean & std.
- Classification (§9.2): ``fit_classification``/``predict_clf``, ``kind``
  in ``{"logistic", "gbc"}``. Two classes, label=1 the target
  (profitable/action) class; the error of interest is the FALSE-POSITIVE
  rate (label=0 mistaken for label=1), per §9.2.1.

Regression std choice (brief's "Constraints / notes", explicitly a
document-your-choice decision): std is the fit's IN-SAMPLE residual std
(``np.std(y - model.predict(X), ddof=1)``), a single scalar per fitted
model, broadcast to every point by ``predict_reg`` — for ALL THREE kinds,
including ``gbr``. The brief explicitly allows this as an alternative to
quantile regressors / binned residual std for ``gbr``
("...OR a simple global residual std — document your choice"); chosen here
for implementation simplicity and so every ``kind`` shares one
uncertainty-estimation code path (a single ``resid_std`` metric feeds
``predict_reg`` identically regardless of ``kind``). Known limitation: for
``gbr`` specifically, in-sample residual std likely UNDERESTIMATES true
predictive uncertainty relative to linear/poly2, since gradient boosting
with enough estimators/depth can overfit small training sets far more than
a 1-2 term linear model can — flagged in task-4-report.md, not fixed here
(the two harder alternatives are both legitimate future upgrades, not
required by this task).

``save_result``/``load_result`` split each ``RegResult``/``ClfResult``
across two files sharing one path stem: a JSON file (``kind``, ``params``,
``metrics``, plus which dataclass it is) for the human-reproducible report
side, and a sibling ``.joblib`` file for the fitted sklearn estimator
itself (not JSON-serializable in general — e.g. GBR/GBC's underlying
decision trees). ``path`` is the JSON path; the joblib sibling is derived
by swapping a trailing ``.json`` for ``.joblib`` (or appending ``.joblib``
if ``path`` has no ``.json`` suffix) — both functions derive it the same
way, so callers only ever pass the one ``path``.

``groups(indicator, dim)`` enumerates same-space attribute combinations of
size ``dim`` from ``azlib.indicators.ATTRS[indicator]`` via
``itertools.combinations`` — order-preserving, and naturally empty when
``dim`` exceeds how many attrs that indicator has (e.g. ``ma`` only has 2:
``groups("ma", 3) == []``, no special-casing needed).

``plot_1d``/``plot_2d`` (§9's "Charts: 1D and 2D save a chart with the
fitted function/boundary and label=1/label=0 points overlaid. 3D is not
plotted"): the interface's ``y`` argument is ``label_coeff`` itself (a
continuous value in ``[0, 1]``, not a separate binary label array) — the
0/1 "label" split used for point coloring is derived here as ``y >= 0.5``
(coeff >= 0.5 sits closer to the profitable target level than the stop
level; see ``azlib.space.coeff``/``label_coeff``). This module's
``plot_1d``/``plot_2d`` work with EITHER a ``RegResult`` or a
``ClfResult`` passed as ``res``: for a ``RegResult`` the overlaid curve is
the predicted mean (``predict_reg``'s first element); for a ``ClfResult``
it is ``P(label=1)`` (``predict_clf``). A non-interactive Matplotlib
backend (``Agg``) is forced at import time so chart code never touches a
display and never raises the "no display" warning ``-W error`` would turn
into a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations

import joblib
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

from azlib.indicators import ATTRS  # noqa: E402

# --- interface constants -----------------------------------------------------

REG_KINDS = ("linear", "poly2", "gbr")
CLF_KINDS = ("logistic", "gbc")

# Fixed knobs shared by both gradient-boosting kinds — kept as one constant so
# fit_regression("gbr", ...) and fit_classification("gbc", ...) can never
# silently drift apart. random_state pins tree structure so save_result/
# load_result round-trips (and any two calls given the same data) are
# reproducible, per this module's "reproducible" contract.
_GB_PARAMS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}


# --- result dataclasses --------------------------------------------------------


@dataclass
class RegResult:
    """One fitted regression model + its reproducibility/quality record.

    ``kind``/``params``/``metrics`` are the brief's interface fields
    (``metrics``: ``{"r2", "rmse", "resid_std"}``). ``model`` is this
    module's own addition — the fitted sklearn estimator ``predict_reg``
    calls into — excluded from the dataclass-generated ``__eq__``/``repr``
    (``compare=False``/``repr=False``) since sklearn estimators have no
    meaningful value equality and a full repr would dump every tree.
    """

    kind: str
    params: dict
    metrics: dict
    model: object = field(default=None, compare=False, repr=False)


@dataclass
class ClfResult:
    """One fitted classification model + its reproducibility/quality record.

    ``kind``/``params``/``metrics`` are the brief's interface fields
    (``metrics``: ``{"false_pos_rate", "accuracy", "auc"}``). ``model`` —
    see ``RegResult``'s docstring, identical rationale.
    """

    kind: str
    params: dict
    metrics: dict
    model: object = field(default=None, compare=False, repr=False)


# --- regression ---------------------------------------------------------------


def fit_regression(X: np.ndarray, y: np.ndarray, kind: str) -> RegResult:
    """Fit one regression model, ``kind`` in ``{"linear", "poly2", "gbr"}``.

    ``params`` records the hyperparameters the model was constructed with
    (not fitted weights — those live inside ``model``, persisted separately
    by ``save_result``). ``metrics["resid_std"]`` is the IN-SAMPLE residual
    std of ``y - model.predict(X)`` (module docstring's std choice) — the
    value ``predict_reg`` broadcasts back out as its ``std`` return.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if kind == "linear":
        params = {"fit_intercept": True}
        model = LinearRegression(**params)
        model.fit(X, y)
    elif kind == "poly2":
        params = {"degree": 2, "include_bias": False, "fit_intercept": True}
        model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(fit_intercept=True),
        )
        model.fit(X, y)
    elif kind == "gbr":
        params = dict(_GB_PARAMS)
        model = GradientBoostingRegressor(**params)
        model.fit(X, y)
    else:
        raise ValueError(f"unknown regression kind {kind!r}, must be one of {REG_KINDS}")

    y_pred = np.asarray(model.predict(X), dtype=float)
    resid = y - y_pred
    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(np.mean(resid**2)))
    resid_std = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0

    metrics = {"r2": r2, "rmse": rmse, "resid_std": resid_std}
    return RegResult(kind=kind, params=params, metrics=metrics, model=model)


def predict_reg(res: RegResult, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) from a fitted ``RegResult``.

    ``mean`` = ``res.model.predict(X)``. ``std`` = ``res.metrics
    ["resid_std"]`` broadcast to every point (module docstring's std
    choice) — the SAME scalar for every row of ``X``, regardless of
    ``kind``.
    """
    X = np.asarray(X, dtype=float)
    mean = np.asarray(res.model.predict(X), dtype=float)
    std = np.full(mean.shape, res.metrics["resid_std"], dtype=float)
    return mean, std


# --- classification -------------------------------------------------------------


def fit_classification(X: np.ndarray, label: np.ndarray, kind: str) -> ClfResult:
    """Fit one classifier, ``kind`` in ``{"logistic", "gbc"}``.

    ``label`` is two-class, label=1 the target (profitable/action) class.
    ``metrics["false_pos_rate"]`` = FP / (FP + TN), i.e. the fraction of
    actual label=0 rows the fitted model mistakes for label=1 (§9.2.1's
    "error of interest") — 0.0 when there are no actual negatives to
    misclassify (guarded, no divide-by-zero warning). ``metrics["auc"]``
    falls back to ``float("nan")`` if ``label`` turns out single-class
    (``roc_auc_score`` is undefined there) rather than raising.
    """
    X = np.asarray(X, dtype=float)
    label = np.asarray(label)

    if kind == "logistic":
        params = {"max_iter": 1000, "C": 1.0}
        model = LogisticRegression(**params)
        model.fit(X, label)
    elif kind == "gbc":
        params = dict(_GB_PARAMS)
        model = GradientBoostingClassifier(**params)
        model.fit(X, label)
    else:
        raise ValueError(f"unknown classification kind {kind!r}, must be one of {CLF_KINDS}")

    pred = model.predict(X)
    proba = np.asarray(model.predict_proba(X), dtype=float)[:, 1]

    fp = int(np.sum((label == 0) & (pred == 1)))
    tn = int(np.sum((label == 0) & (pred == 0)))
    denom = fp + tn
    false_pos_rate = float(fp) / denom if denom > 0 else 0.0
    accuracy = float(accuracy_score(label, pred))
    try:
        auc = float(roc_auc_score(label, proba))
    except ValueError:
        auc = float("nan")

    metrics = {"false_pos_rate": false_pos_rate, "accuracy": accuracy, "auc": auc}
    return ClfResult(kind=kind, params=params, metrics=metrics, model=model)


def predict_clf(res: ClfResult, X: np.ndarray) -> np.ndarray:
    """``P(label=1)`` in ``[0, 1]`` from a fitted ``ClfResult``."""
    X = np.asarray(X, dtype=float)
    return np.asarray(res.model.predict_proba(X), dtype=float)[:, 1]


# --- save_result / load_result -------------------------------------------------


def _joblib_path(path: str) -> str:
    """Derive the sibling ``.joblib`` model path from a JSON ``path``.

    Swaps a trailing ``.json`` for ``.joblib``; appends ``.joblib`` if
    ``path`` has no ``.json`` suffix. ``save_result``/``load_result`` both
    call this on the SAME ``path`` argument, so they always agree on where
    the model half lives without the caller juggling two paths.
    """
    if path.endswith(".json"):
        return path[: -len(".json")] + ".joblib"
    return path + ".joblib"


def save_result(res, path: str) -> None:
    """Save a ``RegResult``/``ClfResult`` reproducibly: JSON + joblib.

    ``path`` gets ``{"class", "kind", "params", "metrics"}`` as JSON
    (``"class"`` is ``"reg"``/``"clf"``, so ``load_result`` knows which
    dataclass to reconstruct — the interface's ``load_result`` takes only a
    path, no separate type argument). ``res.model`` is written to the
    sibling ``.joblib`` path (see ``_joblib_path``) via ``joblib.dump``,
    the standard way to persist a fitted sklearn estimator exactly
    (including e.g. GBR/GBC's underlying trees, which JSON cannot represent).
    """
    class_name = "reg" if isinstance(res, RegResult) else "clf"
    payload = {
        "class": class_name,
        "kind": res.kind,
        "params": res.params,
        "metrics": res.metrics,
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    joblib.dump(res.model, _joblib_path(path))


def load_result(path: str):
    """Load a ``RegResult``/``ClfResult`` saved by ``save_result``.

    Round-trips to a dataclass whose ``predict_reg``/``predict_clf``
    output is IDENTICAL to the original (joblib preserves the fitted
    sklearn estimator exactly; ``kind``/``params``/``metrics`` come back
    from JSON unchanged).
    """
    with open(path) as f:
        payload = json.load(f)
    model = joblib.load(_joblib_path(path))
    cls = RegResult if payload["class"] == "reg" else ClfResult
    return cls(kind=payload["kind"], params=payload["params"], metrics=payload["metrics"], model=model)


# --- groups ----------------------------------------------------------------------


def groups(indicator: str, dim: int) -> list[tuple[str, ...]]:
    """Same-space attribute combinations of size ``dim`` for ``indicator``.

    ``itertools.combinations(ATTRS[indicator], dim)`` — order-preserving
    (matches ``ATTRS[indicator]``'s own attr order), naturally empty when
    ``dim`` exceeds how many attrs that indicator has (``ma`` has only 2:
    ``groups("ma", 3) == []``, no special-casing needed; ``combinations``
    already returns nothing for ``r > len(iterable)``). Never mixes
    attributes across indicators (§9's "no cross-indicator mixing") —
    always drawn from exactly one indicator's own ``ATTRS`` entry.
    """
    if indicator not in ATTRS:
        raise ValueError(f"unknown indicator {indicator!r}, must be one of {tuple(ATTRS)}")
    return list(combinations(ATTRS[indicator], dim))


# --- plot_1d / plot_2d ----------------------------------------------------------


def _predict_curve(res, X: np.ndarray) -> np.ndarray:
    """Curve to overlay: predicted mean for a ``RegResult``, P(label=1) for
    a ``ClfResult`` — see module docstring's plot semantics."""
    if isinstance(res, RegResult):
        mean, _std = predict_reg(res, X)
        return mean
    return predict_clf(res, X)


def plot_1d(x: np.ndarray, y: np.ndarray, res, out_path: str) -> None:
    """1D chart: fitted curve + label=0/1 points, saved as a PNG.

    ``y`` is ``label_coeff`` (continuous, ``[0, 1]``) — point color splits
    on ``y >= 0.5`` (module docstring's label-derivation choice), the
    overlaid curve is ``res``'s prediction (``_predict_curve``) evaluated
    on a fine grid across ``x``'s range.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    is_label1 = y >= 0.5

    fig, ax = plt.subplots()
    ax.scatter(x[~is_label1], y[~is_label1], s=12, c="tab:red", alpha=0.6, label="label=0")
    ax.scatter(x[is_label1], y[is_label1], s=12, c="tab:green", alpha=0.6, label="label=1")

    grid = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    curve = _predict_curve(res, grid)
    ax.plot(grid.ravel(), curve, color="black", linewidth=1.5, label=f"{res.kind} fit")

    ax.set_xlabel("x")
    ax.set_ylabel("label_coeff" if isinstance(res, RegResult) else "P(label=1)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def plot_2d(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, res, out_path: str) -> None:
    """2D chart: fitted surface/boundary + label=0/1 points, saved as a PNG.

    Same ``y >= 0.5`` label split as ``plot_1d``. The fitted surface is
    ``res``'s prediction (``_predict_curve``) over a ``(x1, x2)`` grid,
    drawn with ``pcolormesh`` (not ``contourf``: a near-constant surface —
    e.g. a very flat regression fit — has no well-defined contour levels
    and would raise a Matplotlib warning; ``pcolormesh`` has no such
    degenerate case).
    """
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    is_label1 = y >= 0.5

    grid1 = np.linspace(x1.min(), x1.max(), 80)
    grid2 = np.linspace(x2.min(), x2.max(), 80)
    gg1, gg2 = np.meshgrid(grid1, grid2)
    grid_points = np.column_stack([gg1.ravel(), gg2.ravel()])
    surface = _predict_curve(res, grid_points).reshape(gg1.shape)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(gg1, gg2, surface, shading="auto", cmap="viridis", alpha=0.6)
    fig.colorbar(mesh, ax=ax)

    ax.scatter(
        x1[~is_label1], x2[~is_label1], s=12, c="tab:red", alpha=0.8, edgecolors="none", label="label=0"
    )
    ax.scatter(
        x1[is_label1], x2[is_label1], s=12, c="tab:green", alpha=0.8, edgecolors="none", label="label=1"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)
