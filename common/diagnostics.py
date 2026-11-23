"""Diagnostic plots mirroring Pechacova/Casallas et al. (2026, "Maximum updraft velocity
beyond CAPE", the Bety paper this project's whole methodology follows) Figs. 4/6/7 --
not just an aggregate R2/RMSE, but *where in feature space* a discovered equation
matches or diverges from the real data, and a stratified look at error by pollution
level (a la Bety's Fig. 4c "observed values" panel, extended with an explicit
low/middle/high error breakdown that paper doesn't need but a health-relevant pollutant
target does).
"""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# User's preferred palette (a custom LinearSegmentedColormap, matching the
# white -> blue -> purple -> orange -> red density-plot style they use elsewhere) --
# the default for every diagnostic plot in this module, both value- and density-colored.
USER_CMAP = LinearSegmentedColormap.from_list(
    "user_pref",
    ["#ffffff", "#3b4cc0", "#6a2c91", "#c2298a", "#e8611c", "#d62728"],
)


def stratified_error_table(
    y_true: np.ndarray, y_pred: np.ndarray, unit: str = ""
) -> pd.DataFrame:
    """RMSE/Mean Bias/NMB within low (<=p10), middle (p40-p60) and high (>=p90) bands
    of the OBSERVED value -- reveals asymmetric error (e.g. underprediction of the
    worst days) that a single aggregate RMSE/R2 hides entirely.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p10, p40, p60, p90 = np.percentile(y_true, [10, 40, 60, 90])
    bins = {
        "low (<=p10)": y_true <= p10,
        "middle (p40-p60)": (y_true >= p40) & (y_true <= p60),
        "high (>=p90)": y_true >= p90,
    }
    rows = []
    for name, mask in bins.items():
        yt, yp = y_true[mask], y_pred[mask]
        mb = float(np.mean(yp - yt))
        rows.append({
            "band": name,
            "n": int(mask.sum()),
            "obs_mean": float(np.mean(yt)),
            "pred_mean": float(np.mean(yp)),
            "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "mb": mb,
            "nmb_pct": 100.0 * mb / np.mean(yt),
        })
    df = pd.DataFrame(rows).set_index("band")
    print(f"Stratified error by observed-value band{f' ({unit})' if unit else ''}:")
    print(df.round(2).to_string())
    return df


def sindy_term_contributions(
    coef_model, lib_X: np.ndarray, lib_names: list[str], y_true: np.ndarray, title: str = ""
) -> tuple[pd.DataFrame, "plt.Figure"]:
    """Per-term decomposition of a fitted pySINDy (STLSQ) model: since the polynomial
    library form is already a literal sum of coefficient*term, each term's contribution
    to the prediction can be isolated directly (unlike PySR's nested equations or the NN,
    which don't decompose this cleanly). Ranks terms by mean |contribution| to the
    predicted value and reports each term's own correlation with the true target --
    the two can disagree (a term can have a large swing but track the target poorly, or
    vice versa; a term's fitted-model coefficient sign can also differ from its own raw
    correlation sign when terms are collinear -- a real multicollinearity effect, not a
    bug, and worth flagging rather than silently accepting the coefficient's sign as the
    physical answer).
    """
    coefs = coef_model.coef_.ravel()
    active = [(n, c, lib_X[:, i]) for i, (n, c) in enumerate(zip(lib_names, coefs)) if abs(c) > 1e-10]
    rows = []
    for n, c, vals in active:
        contrib = c * vals
        rows.append({
            "term": n, "coef": c,
            "mean_abs_contribution": float(np.mean(np.abs(contrib))),
            "std_contribution": float(np.std(contrib)),
            "corr_with_target": float(np.corrcoef(contrib, y_true)[0, 1]) if np.std(contrib) > 0 else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False)
    print(f"Per-term contribution breakdown ({len(active)} active terms):")
    print(summary.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(active))))
    ordered = summary.sort_values("mean_abs_contribution")
    ax.barh(ordered["term"], ordered["mean_abs_contribution"])
    ax.set_xlabel("Mean |contribution| to prediction")
    ax.set_title(title or "Per-term contribution magnitude")
    fig.tight_layout()
    return summary, fig


def pysr_term_contributions(
    equation_str: str, features: list[str], data: pd.DataFrame, y_true: np.ndarray, title: str = "",
    ax: "plt.Axes | None" = None, color=None,
) -> tuple[pd.DataFrame, "plt.Figure"]:
    """Per-term decomposition of a PySR-discovered equation, splitting it into its
    TOP-LEVEL additive terms (via sympy's Add.make_args, which handles arbitrary nesting
    correctly rather than a naive string split on '+'). Complements
    ``sindy_term_contributions``: pySINDy's polynomial library can structurally miss a
    real but nonlinearly-related predictor (e.g. a heavily right-skewed/zero-inflated
    fire-hotspot count, whose useful relationship is via sqrt(sqrt(x)) or similar, not a
    plain linear/quadratic term -- it gets thresholded away at every sparsity level
    tested, not because it doesn't matter, but because the library can't represent it
    well enough to beat the L0 penalty). A PySR equation search that already found and
    kept a sqrt/log-transformed version of that term won't have this blind spot, so this
    function should be used whenever a discovered PySR equation contains a
    physically-important term (e.g. a fire-transport signal) that pySINDy's decomposition
    drops entirely -- report both rather than silently deferring to whichever method is
    sparser.
    """
    import sympy

    local_dict = {f: sympy.Symbol(f) for f in features}
    expr = sympy.sympify(equation_str, locals=local_dict)
    top_level_terms = sympy.Add.make_args(expr)

    rows = []
    term_values = {}
    for term_expr in top_level_terms:
        free_syms = [str(s) for s in term_expr.free_symbols]
        term_name = str(term_expr) if free_syms else f"const={float(term_expr):.4g}"
        if free_syms:
            fn = sympy.lambdify([local_dict[f] for f in features if f in free_syms], term_expr, "numpy")
            vals = fn(*[data[f].to_numpy() for f in features if f in free_syms])
            vals = np.broadcast_to(np.asarray(vals, dtype=float), (len(data),))
        else:
            vals = np.full(len(data), float(term_expr))
        term_values[term_name] = vals
        rows.append({
            "term": term_name,
            "mean_abs_contribution": float(np.mean(np.abs(vals))),
            "std_contribution": float(np.std(vals)),
            "corr_with_target": float(np.corrcoef(vals, y_true)[0, 1]) if np.std(vals) > 0 else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False)
    print(f"PySR equation per-term contribution ({len(top_level_terms)} top-level terms):")
    print(summary.round(3).to_string(index=False))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(top_level_terms))))
    else:
        fig = ax.figure
    ordered = summary.sort_values("mean_abs_contribution")
    ax.barh(ordered["term"], ordered["mean_abs_contribution"], color=color)
    ax.set_xlabel("Mean |contribution| to prediction")
    ax.set_title(title or "PySR equation: per-term contribution magnitude")
    if own_fig:
        fig.tight_layout()
    return summary, fig


def check_features_used(equation_str: str, features: list[str]) -> list[str]:
    """Which selected features are actually present anywhere in a PySR equation string
    (at any nesting depth, via sympy's free_symbols -- not just a naive substring search,
    which would also be fine here since feature names don't collide, but this is the
    robust way). Prints and returns the features that got selected by SFS but then
    dropped entirely from the final discovered equation -- worth knowing explicitly
    rather than assuming "selected" implies "used in the reported equation".
    """
    import sympy

    local_dict = {f: sympy.Symbol(f) for f in features}
    expr = sympy.sympify(equation_str, locals=local_dict)
    used = {str(s) for s in expr.free_symbols}
    missing = [f for f in features if f not in used]
    if missing:
        print(f"Selected but NOT present in this equation: {missing}")
    else:
        print("All selected features appear somewhere in this equation.")
    return missing


def predicted_vs_observed(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    unit: str = "",
    title: str = "",
    cmap = USER_CMAP,
):
    """Mirrors Abramian et al. (2025, npj Climate and Atmospheric Science) Fig. 3A/B:
    a density-colored one-to-one scatter of predicted vs. observed, one panel per model,
    with the 1:1 line drawn explicitly. Complements ``response_surface_comparison``
    (which shows bias as a function of feature space) with the more fundamental view --
    bias in prediction space itself, regardless of which feature is responsible. Point
    color = local density (via a Gaussian KDE), so a cloud of overlapping points at
    "typical" values doesn't visually drown out the sparser, more policy-relevant tail.

    ``predictions`` maps a model name to (y_true, y_pred) -- e.g.
    {"NN (LOYO out-of-fold)": (y_true_loyo, y_pred_loyo), "PySR (in-sample)": (y, pysr_pred)}.
    """
    from scipy.stats import gaussian_kde

    n_panels = len(predictions)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.2), squeeze=False)
    axes = axes[0]

    all_vals = np.concatenate([np.concatenate([yt, yp]) for yt, yp in predictions.values()])
    lo, hi = np.percentile(all_vals, [0.5, 99.5])

    for ax, (name, (y_true, y_pred)) in zip(axes, predictions.items()):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        xy = np.vstack([y_true, y_pred])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        sc = ax.scatter(y_true[order], y_pred[order], c=density[order], s=10, cmap=cmap)
        ax.plot([lo, hi], [lo, hi], "k-", lw=1, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        r2 = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        ax.set_title(f"{name}\nR2={r2:.3f}")
        ax.set_xlabel(f"Observed{f' ({unit})' if unit else ''}")
        ax.set_ylabel(f"Predicted{f' ({unit})' if unit else ''}")
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(sc, ax=ax, label="density")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def response_surface_comparison(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    predict_fns: dict[str, Callable[[np.ndarray], np.ndarray]],
    feature_x: str | None = None,
    feature_y: str | None = None,
    grid_n: int = 60,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    title: str = "",
    cmap = USER_CMAP,
    label_map: dict[str, str] | None = None,
    axes: "list[plt.Axes] | None" = None,
):
    """Mirrors Bety et al. Figs. 4/6: predicted-vs-observed comparison over the plane of
    the two leading (first-selected, i.e. most independently informative) features, with
    every other feature held at its sample median -- the standard partial-dependence
    convention. One panel per entry in ``predict_fns`` (e.g. {"PySR": ..., "NN": ...}),
    plus a final "Observed" hexbin panel colored by the true target.

    ``predict_fns`` values take a (N, len(features)) array in the same column order as
    ``features`` and return an (N,) prediction array -- e.g.
    ``lambda Xg: sr_model.predict(Xg, index=best_idx)`` or
    ``lambda Xg: nn_model.predict(scaler.transform(Xg))``.
    """
    feature_x = feature_x or features[0]
    feature_y = feature_y or features[1]
    sub = data.dropna(subset=features + [target])
    lm = label_map or {}
    disp_x, disp_y, disp_t = lm.get(feature_x, feature_x), lm.get(feature_y, feature_y), lm.get(target, target)

    x_lo, x_hi = sub[feature_x].quantile(quantile_range[0]), sub[feature_x].quantile(quantile_range[1])
    y_lo, y_hi = sub[feature_y].quantile(quantile_range[0]), sub[feature_y].quantile(quantile_range[1])
    xx, yy = np.meshgrid(np.linspace(x_lo, x_hi, grid_n), np.linspace(y_lo, y_hi, grid_n))

    medians = sub[features].median()
    grid_df = pd.DataFrame(np.tile(medians.to_numpy(), (grid_n * grid_n, 1)), columns=features)
    grid_df[feature_x] = xx.ravel()
    grid_df[feature_y] = yy.ravel()
    Xg = grid_df[features].to_numpy()

    n_panels = len(predict_fns) + 1
    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5), squeeze=False)
        axes = axes[0]
    else:
        fig = axes[0].figure

    # When plotting into externally-provided axes (a combined, space-constrained figure), each panel
    # repeating its own y-axis label collides with the previous panel's colorbar, and the default
    # full-height colorbar eats too much of an already-narrow panel -- so only the first panel keeps
    # a y-label/y-ticks, and every colorbar shrinks. Standalone use (own_fig, generous 5.5in panels)
    # is unaffected.
    # 2026-09-28: the embedded branch used to omit `label` entirely (to save space in a tight combined
    # figure), which left every colorbar in the embedded case (e.g. Fig. S12) unlabelled -- readers had
    # no way to tell what quantity/unit the shading represented. Keep the label in both cases; the caller
    # can still shrink its font afterward via the returned colorbar's `.ax.yaxis.label`.
    cbar_kwargs = dict(label=disp_t) if own_fig else dict(label=disp_t, shrink=0.85, aspect=16, pad=0.03)
    vmin, vmax = sub[target].quantile([0.02, 0.98])
    for i, (ax, (name, fn)) in enumerate(zip(axes[:-1], predict_fns.items())):
        zz = np.asarray(fn(Xg)).reshape(grid_n, grid_n)
        cf = ax.contourf(xx, yy, zz, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name} (predicted)")
        ax.set_xlabel(disp_x)
        if own_fig or i == 0:
            ax.set_ylabel(disp_y)
        else:
            ax.set_yticklabels([])
        fig.colorbar(cf, ax=ax, **cbar_kwargs)

    ax = axes[-1]
    hb = ax.hexbin(
        sub[feature_x], sub[feature_y], C=sub[target], gridsize=30,
        cmap=cmap, vmin=vmin, vmax=vmax, reduce_C_function=np.mean,
        extent=(x_lo, x_hi, y_lo, y_hi),
    )
    ax.set_title("Observed")
    ax.set_xlabel(disp_x)
    if own_fig:
        ax.set_ylabel(disp_y)
    else:
        ax.set_yticklabels([])
    fig.colorbar(hb, ax=ax, **cbar_kwargs)

    if title and own_fig:
        fig.suptitle(title)
    if own_fig:
        fig.tight_layout()
    return fig
