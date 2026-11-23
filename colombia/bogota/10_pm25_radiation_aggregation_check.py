"""Check whether the PM2.5*radiation daily-mean interaction term is a real signal or
partly a daily-averaging artifact.

Motivation (2026-09-11): every PySR/partial-correlation test of the `pm25*radiation`
interaction so far has used DAILY-MEAN pm25 times DAILY-MEAN radiation
(`pm25_daily_mean * radiation_daily_mean`). But that is not the same quantity as the
true daily average of the hourly product,
`mean_over_hours(pm25_hourly * radiation_hourly)` -- the difference between the two is
exactly the within-day covariance between PM2.5 and radiation (mean(X*Y) =
mean(X)*mean(Y) + cov(X,Y) within the day). If PM2.5 and radiation have systematically
shaped diurnal cycles (e.g. PM2.5 peaking in a morning traffic/inversion period before
radiation ramps up), the daily-mean PRODUCT used everywhere so far can carry structure
from that within-day relationship that has nothing to do with any real causal
PM2.5-O3 interaction -- it would just be baked in by how the daily feature was built.

This would help explain why the interaction is such a robust, persistent signal
(PySR keeps finding it, in both cities) while every specific causal mechanism proposed
for it (shading, heterogeneous chemistry, renoxification, PBLH-mediation, ...) has
failed a real distinguishing test: some of it may not be a causal interaction at all.

This needs the real hourly source data + xarray, which the Cowork sandbox doesn't have
network access to install (`pip install xarray` fails there) -- run this locally in the
project's own environment instead:

    python Scripts/colombia/bogota/10_pm25_radiation_aggregation_check.py

Inputs used (both already present in the repo, no new download needed):
    Data/COLOMBIA/Hourmean/PM25_hourmean.xlsx        (hourly PM2.5, "Bogota" column)
    Data/external/era5/bogota_era5_ssrd_*.nc          (hourly ERA5 solar radiation,
                                                        already downloaded for the
                                                        existing daily radiation_mean
                                                        feature -- see external_data.py)
    Data/processed/colombia/bogota/bogota_o3_master_dataset.parquet  (for `o3`, and for
                                                        the existing naive `pm25` /
                                                        `radiation_mean` daily columns
                                                        to compare against)

Output: prints a comparison of
    - raw corr(naive_daily_product, o3)   [pm25_daily_mean * radiation_daily_mean --
      the quantity every PySR/partial-correlation test so far has actually used]
    - raw corr(true_daily_product, o3)    [mean_over_hours(pm25_hourly*radiation_hourly)
      -- the quantity that actually corresponds to "PM2.5 and radiation interacting
      within the same hour," which is what a causal shading/chemistry story would need]
    - how much the two differ from each other (mean absolute % difference, and their
      own correlation with each other) -- a large gap with a good "true" correlation
      would mean the causal interaction is real and the naive daily version is a
      reasonable (if imperfect) proxy for it; a gap where the TRUE product correlates
      much more weakly with O3 than the naive one does would support the
      daily-averaging-artifact reading instead.
Also writes Data/external/pm25_radiation_hourly_check.csv with the daily comparison
table, in case a closer look at specific days is useful.

Note: hourly PM2.5's own missingness (SISAIRE hourly data has real gaps -- worse than
the daily aggregates, which were likely built to tolerate some missing hours per day)
means the usable N here will probably be noticeably smaller than the ~3285-4017 rows
used in the daily-level tests; that's expected, not a bug, and should be reported
alongside the correlation numbers rather than treated as equivalent evidence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[3]
PM25_HOURLY_XLSX = REPO_ROOT / "Data" / "COLOMBIA" / "Hourmean" / "PM25_hourmean.xlsx"
ERA5_SSRD_GLOB = str(REPO_ROOT / "Data" / "external" / "era5" / "bogota_era5_ssrd_*.nc")
MASTER_PARQUET = (
    REPO_ROOT / "Data" / "processed" / "colombia" / "bogota" / "bogota_o3_master_dataset.parquet"
)
OUTPUT_CSV = REPO_ROOT / "Data" / "external" / "pm25_radiation_hourly_check.csv"

BOGOTA_UTC_OFFSET_HOURS = -5  # COT, fixed year-round (no DST) -- same as external_data.py


def load_hourly_pm25() -> pd.Series:
    """Hourly PM2.5 for Bogota, indexed by local time (SISAIRE hourly data is recorded
    in local time already, same convention as the daily SISAIRE files this pipeline
    already uses elsewhere -- unlike ERA5, which is UTC and needs the shift below)."""
    df = pd.read_excel(PM25_HOURLY_XLSX, usecols=["Dates", "Bogota"])
    df = df.rename(columns={"Dates": "time", "Bogota": "pm25_hourly"})
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")["pm25_hourly"]


def load_hourly_radiation() -> pd.Series:
    """Hourly ERA5 surface solar radiation downwards for the Bogota box, spatially
    averaged, shifted from UTC to local time (matches external_data.py's
    _shift_to_local_time convention exactly, so this is directly comparable to the
    radiation_mean feature already in the master dataset)."""
    import glob

    paths = sorted(glob.glob(ERA5_SSRD_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matched {ERA5_SSRD_GLOB}")
    datasets = [xr.open_dataset(p) for p in paths]
    for i, d in enumerate(datasets):
        if "valid_time" in d.dims:
            datasets[i] = d.rename({"valid_time": "time"})
    ds = xr.concat(sorted(datasets, key=lambda d: d.time.values[0]), dim="time")
    # ssrd is J/m^2 accumulated over the hour; external_data.py's daily radiation_mean
    # divides by 3600 to get W/m^2 -- apply the same conversion here so this hourly
    # series is in the same units as the existing daily feature (matters for the
    # magnitude/percent-difference diagnostic below; the correlation numbers
    # themselves are scale-invariant and unaffected either way).
    ssrd = ds.mean(dim=["latitude", "longitude"])["ssrd"] / 3600.0
    ssrd = ssrd.assign_coords(time=ssrd["time"] + np.timedelta64(BOGOTA_UTC_OFFSET_HOURS, "h"))
    s = ssrd.to_series()
    s.name = "radiation_hourly"
    return s


def main() -> None:
    print("Loading hourly PM2.5...")
    pm25_h = load_hourly_pm25()
    print(f"  {pm25_h.notna().sum()} non-null hourly readings, "
          f"{pm25_h.index.min()} to {pm25_h.index.max()}")

    print("Loading hourly ERA5 radiation...")
    rad_h = load_hourly_radiation()
    print(f"  {rad_h.notna().sum()} non-null hourly readings, "
          f"{rad_h.index.min()} to {rad_h.index.max()}")

    hourly = pd.DataFrame({"pm25_hourly": pm25_h, "radiation_hourly": rad_h}).dropna()
    hourly["product_hourly"] = hourly["pm25_hourly"] * hourly["radiation_hourly"]
    print(f"  {len(hourly)} hours with both PM2.5 and radiation present")

    daily_true = hourly.resample("1D").agg(
        true_pm25_mean=("pm25_hourly", "mean"),
        true_radiation_mean=("radiation_hourly", "mean"),
        true_daily_product=("product_hourly", "mean"),
        n_hours=("pm25_hourly", "count"),
    )
    # Require a reasonably complete day (>=18 of 24 hours) so a day built from 2 lucky
    # hours doesn't masquerade as a full daily mean.
    daily_true = daily_true[daily_true["n_hours"] >= 18].drop(columns="n_hours")
    daily_true.index.name = "date"
    daily_true = daily_true.reset_index()
    daily_true["date"] = pd.to_datetime(daily_true["date"])

    print("Loading master dataset for o3 and the existing naive daily pm25/radiation...")
    master = pd.read_parquet(MASTER_PARQUET)
    master["date"] = pd.to_datetime(master["date"])
    master["naive_daily_product"] = master["pm25"] * master["radiation_mean"]

    merged = master.merge(daily_true, on="date", how="inner")
    print(f"Matched {len(merged)} days with both the true hourly-derived product and "
          f"the master dataset's naive daily product.")

    d = merged.dropna(subset=["naive_daily_product", "true_daily_product", "o3"])
    print(f"\nUsable N for the comparison: {len(d)}")

    r_naive = np.corrcoef(d["naive_daily_product"], d["o3"])[0, 1]
    r_true = np.corrcoef(d["true_daily_product"], d["o3"])[0, 1]
    r_between = np.corrcoef(d["naive_daily_product"], d["true_daily_product"])[0, 1]
    pct_diff = (
        (d["true_daily_product"] - d["naive_daily_product"]).abs()
        / d["naive_daily_product"].abs().replace(0, np.nan)
    ).median() * 100

    print(f"\ncorr(naive_daily_product, o3)  = {r_naive:.4f}   "
          f"[pm25_daily_mean * radiation_daily_mean -- what every prior test used]")
    print(f"corr(true_daily_product,  o3)  = {r_true:.4f}   "
          f"[mean_over_hours(pm25_hourly * radiation_hourly) -- the real thing]")
    print(f"corr(naive_daily_product, true_daily_product) = {r_between:.4f}")
    print(f"median |true - naive| / |naive| = {pct_diff:.1f}%")
    print(
        "\nReading guide: if r_true is close to r_naive (and their correlation with "
        "each other is high), the naive daily product is a reasonable stand-in for the "
        "real within-day interaction -- the causal-mechanism search so far has been "
        "testing approximately the right quantity, it just hasn't found the mechanism "
        "yet. If r_true is notably WEAKER than r_naive, that supports the "
        "daily-averaging-artifact reading: part of what's been driving PySR to keep "
        "finding this interaction is an artifact of building the feature from daily "
        "means rather than a real same-hour PM2.5-radiation effect on ozone."
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
