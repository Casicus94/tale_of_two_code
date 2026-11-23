"""A9 support: constrained PySR (3 seeds) on the PREVIOUS-DAY versions of the lead equations' variables.
PM2.5: co_lag1, wind_speed_era5_lag1, hotspots_sector_N_lag1, hotspots_sector_E_lag3 (= east-sector count two days before yesterday... i.e. yesterday's 'E lag 2').
O3: no_lag1, no2_lag1, radiation_mean_lag1, pm25_lag1, hotspots_sector_E_lag3.
Same PySR settings and units as every constrained run. Outputs bogota_{pm25,o3}_pysr_lagged_lead_pareto_front_seed{n}.csv"""
import sys, pandas as pd, numpy as np
from pathlib import Path
from pysr import PySRRegressor
R = Path("/path/to/project"); B = R / "Data/processed/colombia/bogota"
RUNS = {"pm25": ("bogota_master_dataset.parquet", {"co_lag1": "kg/m^3", "wind_speed_era5_lag1": "m/s", "hotspots_sector_N_lag1": "", "hotspots_sector_E_lag3": ""}),
        "o3": ("bogota_o3_master_dataset.parquet", {"no_lag1": "kg/m^3", "no2_lag1": "kg/m^3", "radiation_mean_lag1": "W/m^2", "pm25_lag1": "kg/m^3", "hotspots_sector_E_lag3": ""})}
for tgt, (mp, um) in RUNS.items():
    d = pd.read_parquet(B / mp); d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date").set_index("date").asfreq("D")
    for c in um:                                   # previous-day versions that the master does not carry are built by shifting one day
        if c not in d.columns: d[c] = d[c[:-5]].shift(int(c[-1]))
    d = d.dropna(subset=list(um) + [tgt]).reset_index(drop=True); feats = list(um)
    for seed in (0, 1, 2):
        sr = PySRRegressor(niterations=200, binary_operators=["+", "-", "*", "/"], unary_operators=["log", "sqrt", "exp"], model_selection="best", maxsize=25, populations=24, random_state=seed)
        sr.fit(d[feats].to_numpy(), d[tgt].to_numpy(), variable_names=feats, X_units=[um[f] for f in feats], y_units="kg/m^3")
        f = sr.equations_[["complexity", "loss", "equation"]].copy(); f["seed"] = seed
        f.to_csv(B / f"bogota_{tgt}_pysr_lagged_lead_pareto_front_seed{seed}.csv", index=False); print(tgt, seed, "done", len(d), flush=True)
