"""B8: build the lag-only ("forecast") candidate pool for a master dataset -- every
same-day/contemporaneous predictor removed, keeping only lag-1/2/3 versions (computed
fresh for any column that doesn't already have them) plus calendar features that are
known in advance (day-of-week, day-of-year, month) regardless of "today"'s conditions.

This is the predictor pool for the parallel "predict day t from day t-1 or earlier"
pipeline proposed in MASTER.md ("Co-author discussion (2026-09-18)", item 1 / B8),
distinct from the existing "explanatory" pipeline that mixes same-day and lagged terms.
"""
from __future__ import annotations

import pandas as pd

CALENDAR_COLS = ["day_of_week", "doy_sin", "doy_cos", "month_sin", "month_cos", "is_weekend"]

# Columns that are never legitimate predictors of anything (identifiers/labels), not
# pollutant targets themselves -- those are handled by the caller via `targets`.
NON_FEATURE_COLS = {"date", "year", "season"}


def build_forecast_pool(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Returns a copy of `df` with every same-day predictor column dropped (kept:
    calendar columns, date/year, and all *_lag1/2/3 columns -- existing ones as-is, plus
    freshly computed lag1/2/3 for any same-day numeric column that didn't already have
    them). `targets` (e.g. ['pm25'] or ['o3']) are kept as same-day columns since they
    are the thing being forecast, not a predictor.
    """
    df = df.sort_values("date").reset_index(drop=True)
    same_day_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c not in CALENDAR_COLS
        and c not in targets
        and not c.endswith(("_lag1", "_lag2", "_lag3"))
    ]

    existing_lag_bases = {
        c[: -len("_lag1")] for c in df.columns if c.endswith("_lag1")
    }

    added = []
    for col in same_day_cols:
        if col in existing_lag_bases:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        added.append(col)
    print(f"Added fresh lag1/2/3 for {len(added)} columns that didn't already have them: {added}")

    keep = (
        list(NON_FEATURE_COLS)
        + CALENDAR_COLS
        + targets
        + [c for c in df.columns if c.endswith(("_lag1", "_lag2", "_lag3"))]
    )
    keep = [c for c in keep if c in df.columns]
    return df[keep]
