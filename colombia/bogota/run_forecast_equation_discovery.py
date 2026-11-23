"""PySR equation discovery on the lagged forecast pipeline's selected features, for both
Bogota pollutants -- using the **no_self_lag_no_calendar** pass (2026-09-18, user
correction: calendar terms like day_of_week/month_sin are "not interpretable" and the
target's own lag "co-varies very very strongly" with itself by construction, so neither
belongs in an equation meant to be physically read). This is the fully interpretable
forecast equation: every term is a genuinely different physical/chemical lagged
variable, nothing that's just a calendar shortcut or the target predicting itself.

Same PySR config as the explanatory-pipeline notebooks (07_equation_discovery.ipynb /
07_o3_equation_discovery.ipynb) for direct comparability.

Run: python Scripts/colombia/bogota/run_forecast_equation_discovery.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import r2_score

PROJECT_ROOT = Path("/path/to/project")
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))
from common.build_forecast_pool import build_forecast_pool  # noqa: E402

DATA_DIR = PROJECT_ROOT / "Data" / "processed" / "colombia" / "bogota"

RUNS = [
    {
        "target": "pm25",
        "master": DATA_DIR / "bogota_master_dataset.parquet",
        "selection": DATA_DIR / "bogota_forecast_feature_selection.json",
        "out_csv": DATA_DIR / "bogota_pm25_forecast_interpretable_pysr_pareto.csv",
    },
    {
        "target": "o3",
        "master": DATA_DIR / "bogota_o3_master_dataset.parquet",
        "selection": DATA_DIR / "bogota_o3_forecast_feature_selection.json",
        "out_csv": DATA_DIR / "bogota_o3_forecast_interpretable_pysr_pareto.csv",
    },
]

for run in RUNS:
    print(f"\n{'='*70}\nBogota {run['target']} -- forecast-pipeline equation discovery\n{'='*70}")
    with open(run["selection"]) as f:
        selection = json.load(f)
    features = selection["no_self_lag_no_calendar"]["selected"]
    print("Features:", features)

    df = pd.read_parquet(run["master"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    pool = build_forecast_pool(df, targets=[run["target"]])

    data = pool.dropna(subset=features + [run["target"]]).reset_index(drop=True)
    X = data[features].to_numpy()
    y = data[run["target"]].to_numpy()
    print(f"n={len(data)}")

    sr = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "exp"],
        model_selection="best",
        maxsize=25,
        populations=24,
    )
    sr.fit(X, y, variable_names=features)
    print(sr)

    pareto = sr.equations_[["complexity", "loss", "equation"]].copy()
    pareto["r2"] = pareto.apply(lambda row: r2_score(y, sr.predict(X, index=row.name)), axis=1)
    pareto.to_csv(run["out_csv"], index=False)
    print(f"Saved: {run['out_csv']}")
    print("\nFull Pareto front:")
    for _, row in pareto.sort_values("complexity").iterrows():
        print(f"  [{int(row['complexity']):2d}] R2={row['r2']:.3f}  {row['equation']}")
