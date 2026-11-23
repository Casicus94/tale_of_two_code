"""T3 support: Medellin PySR runs needed for the best-equation-first walk.

1. Explanatory O3 refresh. The stored front medellin_o3_pysr_pareto_front.csv is dated
   2026-09-10 15:12, i.e. BEFORE the final O3 master rebuild (2026-09-11 08:51, HCHO
   merge) and the leakage-fix feature selection (2026-09-11 10:12). Re-run PySR on the
   CURRENT selected_no_lag features so the walk rests on an unambiguously current front.
   Output: medellin_o3_pysr_pareto_front_refresh.csv (has an in-sample r2 column).
2. Lagged-only (no self-lag, no calendar) fronts for both Medellin pollutants -- the
   equations MASTER.md flagged as missing (only SFS selections existed).
   Outputs: medellin_{pm25,o3}_forecast_interpretable_pysr_pareto.csv
   NOTE: Medellin O3 forecast R2 is ~0.135 (o3_lag1 is never selected), so its front is
   reported for completeness of the walk only; it is not a forecasting result.
Same PySR configuration as every other run in this project.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts"))
from common.build_forecast_pool import build_forecast_pool  # noqa: E402

D = ROOT / "Data" / "processed" / "colombia" / "medellin"


def run_pysr(X, y, features, out_csv, label):
    print(f"\n=== {label} ===\nfeatures: {features}\nn={len(y)}", flush=True)
    sr = PySRRegressor(
        niterations=200,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "sqrt", "exp"],
        model_selection="best",
        maxsize=25,
        populations=24,
    )
    sr.fit(X, y, variable_names=features)
    p = sr.equations_[["complexity", "loss", "equation"]].copy()
    p["r2"] = p.apply(lambda r: r2_score(y, sr.predict(X, index=r.name)), axis=1)
    p.to_csv(out_csv, index=False)
    for _, r in p.sort_values("complexity").iterrows():
        print(f"  [{int(r.complexity):2d}] R2={r.r2:.3f}  {r.equation}", flush=True)
    print("saved", out_csv, flush=True)


# 1. explanatory O3 refresh --------------------------------------------------
sel = json.load(open(D / "medellin_o3_feature_selection.json"))["selected_no_lag"]
df = pd.read_parquet(D / "medellin_o3_master_dataset.parquet").dropna(subset=["o3"])
data = df.dropna(subset=sel + ["o3"]).reset_index(drop=True)
run_pysr(data[sel].to_numpy(), data["o3"].to_numpy(), sel,
         D / "medellin_o3_pysr_pareto_front_refresh.csv", "Medellin O3 explanatory (refresh)")

# 2. lagged-only fronts --------------------------------------------------------
for target, master, sel_file in [
    ("pm25", "medellin_master_dataset.parquet", "medellin_forecast_feature_selection.json"),
    ("o3", "medellin_o3_master_dataset.parquet", "medellin_o3_forecast_feature_selection.json"),
]:
    feats = json.load(open(D / sel_file))["no_self_lag_no_calendar"]["selected"]
    m = pd.read_parquet(D / master)
    m["date"] = pd.to_datetime(m["date"])
    m["year"] = m["date"].dt.year
    pool = build_forecast_pool(m, targets=[target])
    dd = pool.dropna(subset=feats + [target]).reset_index(drop=True)
    run_pysr(dd[feats].to_numpy(), dd[target].to_numpy(), feats,
             D / f"medellin_{target}_forecast_interpretable_pysr_pareto.csv",
             f"Medellin {target} lagged-only (no self-lag, no calendar)")
print("ALL DONE", flush=True)
