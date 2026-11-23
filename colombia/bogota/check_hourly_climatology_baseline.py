"""Third fairness check on the literature comparison (2026-09-18, user-raised): is
Cifuentes et al. (2021)'s hourly R2=0.76-0.91 even the same kind of number as this
project's daily-mean LOYO-CV R2?

User's objection: an R2 that high for hourly ozone/PM2.5 "seems crazy" given this
project's own daily-resolution equations plateau around R2=0.54-0.64. Checked directly
against this project's own hourly Bogota data (same source used for the diurnal
lag-phase test elsewhere in this project).

First pass (informal, NOT trusted -- see below) let a "climatology" (hour-of-day mean)
predictor be fit and evaluated on the SAME 30-day window, which is in-sample and
therefore invalid -- it let the baseline peek at the exact data being scored. Recomputed
properly here as a real train/test split, training the hour-of-day climatology on
Oct-Dec 2019 only, and evaluating on Jan 2020 -- Cifuentes' own stated study period and
test window (Table 3, "O3/PM2.5 on Test Set 1, Jan. 2020, n=621").

Result: hour-of-day alone (ZERO real weather, zero pollutant data) already gets
R2=0.115 (O3) / R2=0.122 (PM2.5) on this exact test window in Bogota. Not "most" of
Cifuentes' reported skill as the sloppier first check suggested, but non-trivial, and
it demonstrates the deeper point directly: hourly readings within a day are strongly
autocorrelated, which mechanically inflates R2 for a time-resolved series independent
of genuine predictive skill -- a different statistical object from a daily-mean R2,
where within-day autocorrelation has already been averaged away. Combined with the
same-species (Gulati/PM10) and black-box (ANN/SVR) issues already fixed, this is a
third, independent reason not to plot Cifuentes' hourly numbers on the same axis as
this project's daily LOYO-CV numbers.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
HOURMEAN_DIR = REPO_ROOT / "Data" / "COLOMBIA" / "Hourmean"


def climatology_forward_r2(path, col, train_start, test_start, test_days=30):
    df = pd.read_excel(path, usecols=["Dates", col])
    df = df.rename(columns={"Dates": "time", col: "y"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=["y"])
    df["hour"] = df["time"].dt.hour
    train_start, test_start = pd.Timestamp(train_start), pd.Timestamp(test_start)
    test_end = test_start + pd.Timedelta(days=test_days)
    train = df[(df["time"] >= train_start) & (df["time"] < test_start)]
    test = df[(df["time"] >= test_start) & (df["time"] < test_end)]
    hourly_clim = train.groupby("hour")["y"].mean()
    pred = test["hour"].map(hourly_clim)
    return r2_score(test["y"], pred), len(test), len(train)


def main():
    r2_o3, n_test, n_train = climatology_forward_r2(
        HOURMEAN_DIR / "O3_hourmean.xlsx", "Bogota", "2019-10-01", "2020-01-01"
    )
    print(f"O3: train Oct-Dec 2019 (n={n_train}), test Jan 2020 (n={n_test}): "
          f"hour-of-day-only R2 = {r2_o3:.3f}")

    r2_pm25, n_test, n_train = climatology_forward_r2(
        HOURMEAN_DIR / "PM25_hourmean.xlsx", "Bogota", "2019-10-01", "2020-01-01"
    )
    print(f"PM2.5: train Oct-Dec 2019 (n={n_train}), test Jan 2020 (n={n_test}): "
          f"hour-of-day-only R2 = {r2_pm25:.3f}")


if __name__ == "__main__":
    main()
