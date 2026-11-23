"""Shared LOYO-CV sequential forward feature selection, used by every city/target's
notebook 06 (Bogotá PM2.5/O3, Medellín PM2.5/O3, and future cities).

Parallelized across candidates within each SFS step: evaluating candidate A's LOYO R2 is
completely independent of candidate B's, so this is embarrassingly parallel. The earlier
per-notebook inline (sequential, single-core) version was taking 1-2+ hours per run and
directly caused a real failure (two such runs contending for one core each timed out
silently) -- this cuts that down by roughly the number of workers used.
"""
from __future__ import annotations

import os

# Must be set before joblib spawns worker processes: each worker's own BLAS backend
# would otherwise try to multithread internally too, oversubscribing the machine's cores
# on top of joblib's own process-level parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def loyo_r2(data: pd.DataFrame, features: list[str], target: str, year_col: str = "year") -> float:
    """Leave-one-year-out cross-validated R2 for a small MLP on the given feature set."""
    sub = data.dropna(subset=features + [target])
    years = sorted(sub[year_col].unique())
    if len(sub) < 50 or len(years) < 3:
        return np.nan
    preds = np.full(len(sub), np.nan)
    y = sub[target].to_numpy()
    X = sub[features].to_numpy()
    for test_year in years:
        test_mask = (sub[year_col] == test_year).to_numpy()
        train_mask = ~test_mask
        if test_mask.sum() < 10 or train_mask.sum() < 50:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        Xtr, Xte = scaler.transform(X[train_mask]), scaler.transform(X[test_mask])
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            early_stopping=True, n_iter_no_change=10, max_iter=500, random_state=0,
        )
        model.fit(Xtr, y[train_mask])
        pred_pos = np.where(test_mask)[0]
        preds[pred_pos] = model.predict(Xte)
    valid = ~np.isnan(preds)
    if valid.sum() < 50:
        return np.nan
    return r2_score(y[valid], preds[valid])


def loyo_predictions(
    data: pd.DataFrame, features: list[str], target: str, year_col: str = "year"
) -> tuple[np.ndarray, np.ndarray]:
    """Like ``loyo_r2`` but returns the (y_true, y_pred) out-of-fold arrays themselves,
    so Pearson r/RMSE/NMB can be computed directly from real predictions rather than
    derived from R2 (e.g. via sqrt) -- matches the project's own established practice of
    verifying R2-vs-r empirically instead of assuming the textbook relationship holds.
    """
    sub = data.dropna(subset=features + [target])
    years = sorted(sub[year_col].unique())
    preds = np.full(len(sub), np.nan)
    y = sub[target].to_numpy()
    X = sub[features].to_numpy()
    for test_year in years:
        test_mask = (sub[year_col] == test_year).to_numpy()
        train_mask = ~test_mask
        if test_mask.sum() < 10 or train_mask.sum() < 50:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        Xtr, Xte = scaler.transform(X[train_mask]), scaler.transform(X[test_mask])
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            early_stopping=True, n_iter_no_change=10, max_iter=500, random_state=0,
        )
        model.fit(Xtr, y[train_mask])
        pred_pos = np.where(test_mask)[0]
        preds[pred_pos] = model.predict(Xte)
    valid = ~np.isnan(preds)
    return y[valid], preds[valid]


def nn_in_sample_model(
    data: pd.DataFrame, features: list[str], target: str
) -> tuple[StandardScaler, MLPRegressor, np.ndarray, np.ndarray]:
    """Fits the same architecture as loyo_r2/loyo_predictions on the SAME (full) data --
    a goodness-of-fit ceiling, not a generalization estimate. Returns the fitted
    (scaler, model) alongside (y_true, y_pred) so the model itself can be reused (e.g.
    for a response-surface diagnostic plot), not just its predictions.
    """
    sub = data.dropna(subset=features + [target])
    y = sub[target].to_numpy()
    X = sub[features].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        early_stopping=True, n_iter_no_change=10, max_iter=500, random_state=0,
    )
    model.fit(Xs, y)
    return scaler, model, y, model.predict(Xs)


def nn_in_sample_predictions(
    data: pd.DataFrame, features: list[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Same architecture as loyo_r2/loyo_predictions, but fit and evaluated on the SAME
    (full) data -- a goodness-of-fit ceiling, not a generalization estimate. Report this
    *alongside* loyo_predictions, never instead of it: the gap between the two is itself
    informative (a wide gap flags overfitting -- more capacity fitting noise rather than
    learning anything more generalizable), and PySR/pySINDy's own R2 is in-sample too, so
    this is the fair comparison point for them rather than the LOYO number.
    """
    _, _, y, y_pred = nn_in_sample_model(data, features, target)
    return y, y_pred


def sequential_forward_selection(
    data: pd.DataFrame,
    candidates: list[str],
    target: str,
    max_features: int = 12,
    min_gain: float = 0.002,
    year_col: str = "year",
    n_jobs: int = 8,
    verbose: bool = True,
) -> tuple[list[str], list[float]]:
    """Greedy forward selection: at each step, evaluate every remaining candidate's LOYO
    R2 (in parallel across ``n_jobs`` worker processes) and keep whichever helps most,
    stopping once no candidate improves R2 by at least ``min_gain``.
    """
    selected: list[str] = []
    scores: list[float] = []
    remaining = list(candidates)
    best_r2 = -np.inf
    while remaining and len(selected) < max_features:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(loyo_r2)(data, selected + [feat], target, year_col) for feat in remaining
        )
        step_scores = dict(zip(remaining, results))
        best_feat = max(
            step_scores, key=lambda f: (step_scores[f] if not np.isnan(step_scores[f]) else -np.inf)
        )
        best_feat_r2 = step_scores[best_feat]
        if np.isnan(best_feat_r2) or best_feat_r2 - best_r2 < min_gain:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_r2 = best_feat_r2
        scores.append(best_r2)
        if verbose:
            print(f"  + {best_feat:35s} -> across-year R2 = {best_r2:.3f}")
    return selected, scores
