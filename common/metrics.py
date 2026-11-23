"""Validation-metric reporting, matching the convention from the group's own papers
(Tendencies_PM2p5.pdf uses IOA/Rho/RMSE/MB/FAC2) and the project's own "Follow-up
metrics" precedent: Pearson r, RMSE, and Normalized Mean Bias alongside R2, not R2 alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2, Pearson r, RMSE, Mean Bias, and Normalized Mean Bias (% of observed mean)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mb = float(np.mean(y_pred - y_true))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mb": mb,
        "nmb_pct": float(100.0 * mb / np.mean(y_true)),
    }


def hit_far_table(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds_pct: tuple = (75, 90, 95)
) -> pd.DataFrame:
    """Categorical forecast-verification metrics (standard in meteorology/air-quality
    forecast evaluation), treating "exceeds the Nth percentile of observed values" as
    the event of interest -- e.g. does the equation actually flag the days that turn out
    to be the worst air-quality days, not just get the average day roughly right.

    - Hit rate (POD, probability of detection) = TP / (TP + FN): of the actual event
      days, what fraction did the model also flag as an event?
    - FAR (false alarm ratio) = FP / (TP + FP): of the days the model flagged as an
      event, what fraction were NOT actually events?
    - CSI (critical success index) = TP / (TP + FN + FP): a combined score, penalizing
      both misses and false alarms, less sensitive to the base rate than accuracy.

    Threshold is defined on the OBSERVED distribution and applied to both true and
    predicted values (i.e. "predicted exceeds the same percentile cutoff"), so this is
    threshold-consistent regardless of any systematic bias in the predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for pct in thresholds_pct:
        thresh = np.percentile(y_true, pct)
        obs_event = y_true >= thresh
        pred_event = y_pred >= thresh
        tp = int(np.sum(obs_event & pred_event))
        fn = int(np.sum(obs_event & ~pred_event))
        fp = int(np.sum(~obs_event & pred_event))
        tn = int(np.sum(~obs_event & ~pred_event))
        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        far = fp / (tp + fp) if (tp + fp) > 0 else float("nan")
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else float("nan")
        rows.append({
            "threshold_pct": pct, "threshold_value": thresh,
            "n_events": tp + fn, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "hit_rate": hit_rate, "far": far, "csi": csi,
        })
    df = pd.DataFrame(rows).set_index("threshold_pct")
    print("Categorical (HIT/FAR/CSI) verification, event = exceeds observed Nth percentile:")
    print(df.round(3).to_string())
    return df


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str, unit: str = "") -> dict:
    m = regression_metrics(y_true, y_pred)
    unit_suffix = f" {unit}" if unit else ""
    print(
        f"{label}: R2={m['r2']:.3f}  r={m['pearson_r']:.3f}  "
        f"RMSE={m['rmse']:.2f}{unit_suffix}  MB={m['mb']:+.2f}{unit_suffix}  "
        f"NMB={m['nmb_pct']:+.1f}%"
    )
    return m
