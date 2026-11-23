"""A11 support: does choosing the equation form on ALL years inflate the leave-one-year-out R2?
For each held-out year, the constrained PySR search (same settings, units and candidate pool as the lead equations) is re-run on the OTHER years only;
the equation is then chosen from that fold's own Pareto front by a rule that uses no held-out information, and scored on the held-out year with
the constants the search returned (no refit, no look at the held-out year). Two rules: (a) the complexity of the lead equation (13 PM2.5, 15 O3), or the
nearest lower one on that fold's front; (b) a knee rule: the smallest complexity whose in-sample R2 is within 0.01 of the best on the front.
Output: Paper/figures/data/nested_pysr_check_{pol}_seed{n}.csv (one row per held-out year and rule)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from pysr import PySRRegressor
R = Path("/path/to/project"); B = R / "Data/processed/colombia/bogota"; OUT = R / "Paper/figures/data"
POOLS = {"pm25": ("bogota_master_dataset.parquet", 13,
                  {"w700": "Pa/s", "co": "kg/m^3", "wind_speed_era5": "m/s", "blh_mean": "m", "sensible_heat_flux": "W/m^2", "sensible_heat_flux_lag1": "W/m^2",
                   "hotspots_sector_NE_lag2": "", "hotspots_sector_N": "", "hotspots_sector_E_lag2": ""}),
         "o3": ("bogota_o3_master_dataset.parquet", 15,
                {"no": "kg/m^3", "pm25": "kg/m^3", "no2": "kg/m^3", "rh_2m": "", "w700": "Pa/s", "wind_speed_era5": "m/s", "hotspots_sector_E_lag2": "", "radiation_mean": "W/m^2", "temp_2m": "K"})}
pols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["pm25", "o3"]; seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
for pol in pols:
    mp, lead_c, um = POOLS[pol]; feats = list(um)
    d = pd.read_parquet(B / mp); d["date"] = pd.to_datetime(d["date"]); d["year"] = d.date.dt.year
    d = d.dropna(subset=feats + [pol]).reset_index(drop=True)
    if pol == "o3": d = d.dropna(subset=["photochemical_potential"]).reset_index(drop=True) if "photochemical_potential" in d else d
    for seed in seeds:
        rows = []
        for yr in sorted(d.year.unique()):
            tr, te = d[d.year != yr], d[d.year == yr]
            sr = PySRRegressor(niterations=200, binary_operators=["+", "-", "*", "/"], unary_operators=["log", "sqrt", "exp"], model_selection="best", maxsize=25, populations=24, random_state=seed, verbosity=0, progress=False)
            sr.fit(tr[feats].to_numpy(), tr[pol].to_numpy(), variable_names=feats, X_units=[um[f] for f in feats], y_units="kg/m^3")
            eq = sr.equations_.reset_index(drop=True); r2in = 1 - eq["loss"] / np.var(tr[pol].to_numpy())
            ia = eq.index[eq["complexity"] <= lead_c][-1]; ib = eq.index[r2in >= r2in.max() - 0.01][0]
            for rule, idx in (("fixed complexity", ia), ("knee (within 0.01 of the best in-sample R2)", ib)):
                pred = sr.predict(te[feats].to_numpy(), index=int(idx)); y = te[pol].to_numpy()
                rows.append(dict(year=int(yr), rule=rule, complexity=int(eq.loc[idx, "complexity"]), n=len(te), sse=float(((y - pred) ** 2).sum()), sst=float(((y - y.mean()) ** 2).sum()),
                                 r2_year=float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), equation=eq.loc[idx, "equation"]))
            print(pol, seed, yr, "done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"nested_pysr_check_{pol}_seed{seed}.csv", index=False)
