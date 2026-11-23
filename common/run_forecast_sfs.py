"""B8: run the lag-only forecast-pipeline SFS for all four (city, pollutant) targets,
using the exact same sequential_forward_selection machinery as the explanatory
pipeline's notebook 06 runs, but on a candidate pool restricted to lag-1/2/3 and
calendar features only (build_forecast_pool.py) -- no same-day predictors at all, so
the resulting equation can genuinely be run the morning before day-t's own weather is
known. See MASTER.md, "Co-author discussion (2026-09-18)", item 1 / checklist item B8.

Run (long-running -- 4 independent SFS searches, each up to 12 rounds of LOYO-CV MLP
fits over the full candidate pool, parallelized n_jobs=8 per round):

    python Scripts/common/run_forecast_sfs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

from common.build_forecast_pool import build_forecast_pool  # noqa: E402
from common.feature_selection import sequential_forward_selection  # noqa: E402

MIN_COVERAGE = 0.70

# Medellin-only: raw/intermediate calibration-source columns, redundant with (and in
# some cases literal ingredients of) the final calibrated column -- same leakage-
# adjacent bases excluded in Scripts/colombia/medellin/05_feature_selection.ipynb and
# 05_o3_feature_selection.ipynb's RAW_SUBCOMPONENT_EXCLUDE. Missed in the first version
# of this script (2026-09-18): the Medellin O3 forecast run had already selected
# `pm25_hourmean_lag1`, and the completed Medellin PM2.5 forecast run had selected
# `wind_speed_antioquia_lag3` and `wind_speed_hybrid_station_era5_lag1` -- all three on
# this list. Bogota has no such hybrid-calibration sources, so it doesn't need this.
MEDELLIN_RAW_SUBCOMPONENT_BASES = [
    "pm25_hourmean", "pm25_antioquia", "pm10_hourmean", "pm10_antioquia",
    "no2_hourmean", "no2_antioquia",
    "wind_speed_antioquia", "wind_speed_hybrid_station_era5", "wind_speed_era5", "wind_speed_station",
    "wind_dir_antioquia", "wind_dir_hybrid_station_era5", "wind_dir_era5", "wind_dir_station",
    "temp_2m_era5", "temp_2m_station", "temp_2m_hybrid_station_era5",
    "rh_2m_era5", "rh_2m_station", "rh_2m_hybrid_station_era5",
]

RUNS = [
    {
        "city": "bogota",
        "target": "pm25",
        "master": PROJECT_ROOT / "Data/processed/colombia/bogota/bogota_master_dataset.parquet",
        "out": PROJECT_ROOT / "Data/processed/colombia/bogota/bogota_forecast_feature_selection.json",
        "causal_exclude_prefixes": ["pm10", "o3"],
        "raw_subcomponent_bases": [],
    },
    {
        "city": "bogota",
        "target": "o3",
        "master": PROJECT_ROOT / "Data/processed/colombia/bogota/bogota_o3_master_dataset.parquet",
        "out": PROJECT_ROOT / "Data/processed/colombia/bogota/bogota_o3_forecast_feature_selection.json",
        "causal_exclude_prefixes": ["pm10"],
        "raw_subcomponent_bases": [],
    },
    {
        "city": "medellin",
        "target": "pm25",
        "master": PROJECT_ROOT / "Data/processed/colombia/medellin/medellin_master_dataset.parquet",
        "out": PROJECT_ROOT / "Data/processed/colombia/medellin/medellin_forecast_feature_selection.json",
        "causal_exclude_prefixes": ["pm10", "o3"],
        "raw_subcomponent_bases": MEDELLIN_RAW_SUBCOMPONENT_BASES,
    },
    {
        "city": "medellin",
        "target": "o3",
        "master": PROJECT_ROOT / "Data/processed/colombia/medellin/medellin_o3_master_dataset.parquet",
        "out": PROJECT_ROOT / "Data/processed/colombia/medellin/medellin_o3_forecast_feature_selection.json",
        "causal_exclude_prefixes": ["pm10"],
        "raw_subcomponent_bases": MEDELLIN_RAW_SUBCOMPONENT_BASES,
    },
]


# Calendar columns are legitimate forecast predictors in principle (known in advance,
# no leakage) but are not physically interpretable on their own -- user-raised concern
# (2026-09-18): day_of_week/is_weekend/month_sin/month_cos are very likely standing in
# for a real physical driver (fire-season timing, traffic patterns) rather than being
# mechanisms themselves. This mirrors the exact ablation already done for the main
# explanatory pipeline (06_feature_selection.ipynb: "SFS excluding ... raw calendar
# terms (forcing physical substitutes)", which found fire-hotspot features step in with
# ~no cost once the calendar shortcut is removed). Applying the same discipline here:
# run the forecast SFS twice per target, with and without calendar columns, so the
# no-calendar run can show whether fire (or another physical lagged variable)
# substitutes for the calendar terms found in the first pass.
CALENDAR_COLS = {"day_of_week", "doy_sin", "doy_cos", "month_sin", "month_cos", "is_weekend"}


def run_one_pass(pool, candidates, target, label):
    print(f"  --- pass: {label} ({len(candidates)} candidates) ---", flush=True)
    selected, scores = sequential_forward_selection(pool, candidates, target=target, n_jobs=8)
    print(f"  [{label}] Selected: {selected}")
    print(f"  [{label}] Forecast-pipeline LOYO R2 = {scores[-1] if scores else None}", flush=True)
    return {"selected": selected, "scores": scores, "final_loyo_r2": scores[-1] if scores else None}


def main() -> None:
    import sys as _sys
    only_cities = set(_sys.argv[1:]) or None
    for run in RUNS:
        if only_cities and run["city"] not in only_cities:
            continue
        print(f"\n{'='*70}\n{run['city']} / {run['target']} forecast pipeline\n{'='*70}", flush=True)
        if not run["master"].exists():
            print(f"  Master dataset not found at {run['master']}, skipping")
            continue
        df = pd.read_parquet(run["master"])
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        pool = build_forecast_pool(df, targets=[run["target"]])

        exclude = {"date", run["target"], "year", "season"}
        for prefix in run["causal_exclude_prefixes"]:
            exclude |= {c for c in pool.columns if c == prefix or c.startswith(prefix + "_lag")}
        for base in run["raw_subcomponent_bases"]:
            exclude |= {c for c in pool.columns if c == base or c.startswith(base + "_lag")}
        all_candidates = [c for c in pool.columns if c not in exclude]

        coverage = pool[all_candidates].notna().mean()
        candidates_with_cal = coverage[coverage >= MIN_COVERAGE].index.tolist()
        candidates_no_cal = [c for c in candidates_with_cal if c not in CALENDAR_COLS]
        dropped = sorted(set(all_candidates) - set(candidates_with_cal))
        print(f"  {len(candidates_with_cal)}/{len(all_candidates)} candidates meet >={MIN_COVERAGE:.0%} coverage "
              f"({len(dropped)} dropped)", flush=True)

        with_cal = run_one_pass(pool, candidates_with_cal, run["target"], "with calendar")
        no_cal = run_one_pass(pool, candidates_no_cal, run["target"], "no calendar (forcing physical substitutes)")

        result = {
            "city": run["city"],
            "target": run["target"],
            "n_candidates": len(candidates_with_cal),
            "with_calendar": with_cal,
            "no_calendar": no_cal,
        }
        with open(run["out"], "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {run['out']}", flush=True)


if __name__ == "__main__":
    main()
