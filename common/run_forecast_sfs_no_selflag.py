"""B8, third ablation (2026-09-18, user-raised): "we don't want to use the same
variable as a lag" -- pm25_lag1/o3_lag1 co-varies very strongly with its own target by
construction, exactly the same "own variable, strongly co-varying" concern already
raised about the literature comparison, now applied to our own forecast pipeline.

Adds a THIRD pass per target on top of the two already saved by run_forecast_sfs.py
(with_calendar, no_calendar): **no_self_lag_no_calendar** -- excludes both the
target's own lag-1/2/3 (pm25_lag* when target=pm25, o3_lag* when target=o3) AND the
calendar columns, so the resulting equation uses only OTHER lagged physical variables,
no persistence shortcut and no calendar shortcut. Loads each target's existing JSON
(from run_forecast_sfs.py) and adds a 'no_self_lag_no_calendar' key rather than
overwriting the with_calendar/no_calendar results already there.

Run once run_forecast_sfs.py has finished (all four *_forecast_feature_selection.json
files present):

    python Scripts/common/run_forecast_sfs_no_selflag.py
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
from common.run_forecast_sfs import (  # noqa: E402
    CALENDAR_COLS,
    MEDELLIN_RAW_SUBCOMPONENT_BASES,
    MIN_COVERAGE,
    RUNS,
)


def main() -> None:
    import sys as _sys
    only_cities = set(_sys.argv[1:]) or None
    for run in RUNS:
        if only_cities and run["city"] not in only_cities:
            continue
        print(f"\n{'='*70}\n{run['city']} / {run['target']} -- no-self-lag, no-calendar pass\n{'='*70}", flush=True)
        if not run["out"].exists():
            print(f"  {run['out']} not found -- run run_forecast_sfs.py first, skipping")
            continue
        with open(run["out"]) as f:
            existing = json.load(f)

        df = pd.read_parquet(run["master"])
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        pool = build_forecast_pool(df, targets=[run["target"]])

        exclude = {"date", run["target"], "year", "season"}
        for prefix in run["causal_exclude_prefixes"]:
            exclude |= {c for c in pool.columns if c == prefix or c.startswith(prefix + "_lag")}
        for base in run["raw_subcomponent_bases"]:
            exclude |= {c for c in pool.columns if c == base or c.startswith(base + "_lag")}
        # New this pass: also exclude the target's own lag -- e.g. pm25_lag1/2/3 when
        # forecasting pm25, o3_lag1/2/3 when forecasting o3.
        exclude |= {c for c in pool.columns if c == run["target"] or c.startswith(run["target"] + "_lag")}
        all_candidates = [c for c in pool.columns if c not in exclude]

        coverage = pool[all_candidates].notna().mean()
        candidates = coverage[coverage >= MIN_COVERAGE].index.tolist()
        candidates = [c for c in candidates if c not in CALENDAR_COLS]
        print(f"  {len(candidates)} candidates (self-lag and calendar both excluded)", flush=True)

        selected, scores = sequential_forward_selection(pool, candidates, target=run["target"], n_jobs=8)
        print(f"  Selected: {selected}")
        print(f"  Forecast-pipeline LOYO R2 = {scores[-1] if scores else None}", flush=True)

        existing["no_self_lag_no_calendar"] = {
            "selected": selected,
            "scores": scores,
            "final_loyo_r2": scores[-1] if scores else None,
        }
        with open(run["out"], "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Updated: {run['out']}", flush=True)


if __name__ == "__main__":
    main()
