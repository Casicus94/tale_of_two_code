"""Ingest the new Antioquia SISAIRE batch (2026-09-23): CO, SO2, NO (network, for comparison
with the existing Hourmean-derived "no"), NOx (as reported, distinct from the derived no+no2
"nox"), PST (total suspended particulates), R_GLOBAL (station radiation), TEMPR_AIR_10CM
(near-surface, 10 cm air temperature -- distinct from the existing 10 m "temp_10m"), and
pLIQUIDA (station precipitation).

Same convention as 00_antioquia_network.ipynb (same Aburrá Valley whitelist, physical range
checks, Chauvenet outlier flagging, >=75% station-year completeness threshold, network daily
mean of qualifying station-years) -- the whitelist and ranges are now shared via common/qc.py
so this doesn't drift from the original notebook's logic. Output merges into the master
datasets alongside (not instead of) the existing `medellin_antioquia_network_daily.parquet`,
so the original 7 variables and their provenance are untouched.

Run: python Scripts/colombia/medellin/ingest_new_antioquia_vars.py
Output: Data/processed/colombia/medellin/medellin_antioquia_network_new_daily.parquet
        Data/processed/colombia/medellin/medellin_antioquia_<var>_completeness_by_station_year.csv
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

import pandas as pd
from functools import reduce

from common.sisaire_io import load_sisaire_file
from common.qc import (
    apply_physical_range_checks, flag_chauvenet_outliers,
    completeness_by_station_year, select_stations_years, PHYSICAL_RANGES, is_aburra_station,
)

RAW_DIR = PROJECT_ROOT / "Data" / "COLOMBIA"
OUT_DIR = PROJECT_ROOT / "Data" / "processed" / "colombia" / "medellin"

# tidy_name -> raw filename. Not city-prefixed (unlike the ANTIOQUIA_* batch), so paths are
# given directly.
FILES = {
    "co": "CO_reporte_sisaire.csv",
    "so2": "SO2_reporte_sisaire.csv",
    "no_sisaire": "NO_reporte_sisaire.csv",              # distinct from the existing Hourmean "no"
    "nox_reported": "NOx_reporte_sisaire.csv",            # distinct from the derived "nox" = no + no2
    "pst": "PST_reporte_sisaire.csv",
    "radiation_station": "R_GLOBAL_reporte_sisaire.csv",
    "temp_10cm": "TEMPR_AIR_10CM_reporte_sisaire.csv",
    "precip_station": "pLIQUIDA_reporte_sisaire.csv",
}
RANGE_KEY = {  # tidy_name -> key in PHYSICAL_RANGES (only "no_sisaire"/"nox_reported" need remapping)
    "co": "co", "so2": "so2", "no_sisaire": "no", "nox_reported": "nox", "pst": "pst",
    "radiation_station": "radiation_station", "temp_10cm": "temp_10cm", "precip_station": "precip_station",
}

raw = {}
for tidy_name, fname in FILES.items():
    d = load_sisaire_file(RAW_DIR / fname, tidy_name)
    n_before = d["station"].nunique()
    d = d[d["station"].map(is_aburra_station)].reset_index(drop=True)
    n_after = d["station"].nunique()
    raw[tidy_name] = d
    dmin, dmax = (d["date"].min(), d["date"].max()) if len(d) else (None, None)
    print(f"{tidy_name}: {n_before} -> {n_after} stations after Aburra filter  {d.shape}  {dmin} to {dmax}")

network_series, completeness_tables = {}, {}
for tidy_name, d in raw.items():
    if d.empty:
        print(f"{tidy_name}: no Aburra-area rows, skipping"); continue
    lo_hi = PHYSICAL_RANGES[RANGE_KEY[tidy_name]]
    clean = apply_physical_range_checks(d, {tidy_name: lo_hi})
    clean = flag_chauvenet_outliers(clean, [tidy_name], group_col="station")

    completeness = completeness_by_station_year(clean, tidy_name, date_col="date", station_col="station")
    qualifying = select_stations_years(completeness, threshold=0.75)
    completeness_tables[tidy_name] = completeness

    clean["year"] = clean["date"].dt.year
    qualifying_keys = set(zip(qualifying["station"], qualifying["year"]))
    clean["is_qualifying"] = clean.apply(lambda r: (r["station"], r["year"]) in qualifying_keys, axis=1)
    qc_rows = clean[clean["is_qualifying"]]

    network = qc_rows.groupby("date")[tidy_name].mean().reset_index()
    network_series[tidy_name] = network
    print(f"{tidy_name}: {len(qualifying)}/{len(completeness)} station-years qualify (>=75%) -- "
          f"network series {network.shape}, {network['date'].min() if len(network) else None} to "
          f"{network['date'].max() if len(network) else None}")

network_daily = reduce(lambda a, b: a.merge(b, on="date", how="outer"), network_series.values())
network_daily = network_daily.sort_values("date").reset_index(drop=True)
print("\nCombined:", network_daily.shape)

network_daily["year"] = network_daily["date"].dt.year
for col in network_series:
    cov = network_daily.groupby("year")[col].apply(lambda s: s.notna().mean())
    nonzero_years = cov[cov > 0]
    if len(nonzero_years):
        print(f"--- {col}: non-null years {nonzero_years.index.min()}-{nonzero_years.index.max()}, "
              f"mean coverage in those years {nonzero_years.mean():.2f} ---")
    else:
        print(f"--- {col}: no non-null years ---")
network_daily = network_daily.drop(columns=["year"])

out_path = OUT_DIR / "medellin_antioquia_network_new_daily.parquet"
network_daily.to_parquet(out_path, index=False)
print("\nSaved:", out_path, network_daily.shape)
for tidy_name, comp in completeness_tables.items():
    comp.to_csv(OUT_DIR / f"medellin_antioquia_{tidy_name}_completeness_by_station_year.csv", index=False)
print("Saved completeness tables for", list(completeness_tables.keys()))
