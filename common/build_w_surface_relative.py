"""Build the surface-relative vertical-velocity feature (w_srel) once the multi-level
download from download_surface_relative_w.py has landed, and re-test PM2.5's
relationship with it against the existing fixed-700-hPa w700 result.

Definition: w_srel at each grid point/hour = vertical velocity interpolated (linearly
in pressure) to the level (local ERA5 surface pressure - 313 hPa) -- the same
depth-below-surface that "700 hPa" represents for a sea-level city (1013.25 - 700 =
313.25 hPa, rounded to 313). See MASTER.md, "HIGH PRIORITY ... check whether w700 even
means what it's assumed to mean at Bogota's altitude", sub-item 3, for the full
motivation.

Run after Scripts/common/download_surface_relative_w.py has finished (check its log /
that all bogota_era5_w{400,450,500,550}_*.nc files exist for both cities):

    python Scripts/common/build_w_surface_relative.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

SURFACE_OFFSET_HPA = 313.0  # 1013.25 - 700, rounded -- see module docstring
LEVELS = [400, 450, 500, 550]
BOGOTA_UTC_OFFSET_HOURS = -5

CITY_CONFIG = {
    "bogota": {
        "era5_dir": PROJECT_ROOT / "Data" / "external" / "era5",
        "sp_glob": "bogota_era5_sp_lts_*.nc",
        "master_parquet": PROJECT_ROOT / "Data" / "processed" / "colombia" / "bogota" / "bogota_master_dataset.parquet",
        "pm25_col": "pm25",
        "utc_offset_hours": -5,
    },
    "medellin": {
        "era5_dir": PROJECT_ROOT / "Data" / "external" / "era5_medellin",
        "sp_glob": "bogota_era5_sp_lts_*.nc",
        "master_parquet": PROJECT_ROOT / "Data" / "processed" / "colombia" / "medellin" / "medellin_master_dataset.parquet",
        "pm25_col": "pm25",
        "utc_offset_hours": -5,
    },
}


def _open_concat(paths: list[Path]) -> xr.Dataset:
    datasets = [xr.open_dataset(p) for p in paths]
    for i, d in enumerate(datasets):
        if "valid_time" in d.dims:
            datasets[i] = d.rename({"valid_time": "time"})
    return xr.concat(sorted(datasets, key=lambda d: d.time.values[0]), dim="time")


def build_w_srel_daily(era5_dir: Path, utc_offset_hours: int) -> pd.DataFrame:
    sp_paths = sorted(era5_dir.glob("bogota_era5_sp_lts_*.nc"))
    if not sp_paths:
        raise FileNotFoundError(f"No surface-pressure files found in {era5_dir}")
    sp = _open_concat(sp_paths).mean(dim=["latitude", "longitude"])["sp"] / 100.0  # Pa -> hPa

    w_by_level = {}
    for level in LEVELS:
        paths = sorted(era5_dir.glob(f"bogota_era5_w{level}_*.nc"))
        if not paths:
            raise FileNotFoundError(f"No w{level} files found in {era5_dir} -- download not finished yet?")
        w_by_level[level] = _open_concat(paths).mean(dim=["latitude", "longitude"])["w"]

    # sp/w_by_level are already box-mean 1-D (time) series at this point (not a full
    # grid), so a plain per-timestep numpy interpolation is simplest.
    w_arr = np.stack([w_by_level[lv].values for lv in LEVELS], axis=1)  # (time, level)
    sp_vals = sp.values
    target_vals = np.clip(sp_vals - SURFACE_OFFSET_HPA, min(LEVELS), max(LEVELS))
    w_srel_vals = np.array([
        np.interp(t, LEVELS, w_arr[i, :]) for i, t in enumerate(target_vals)
    ])

    time = sp["time"].values
    s = pd.Series(w_srel_vals, index=pd.DatetimeIndex(time), name="w_srel")
    s.index = s.index + pd.Timedelta(hours=utc_offset_hours)
    daily = s.resample("1D").mean()
    df = daily.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    for city, cfg in CITY_CONFIG.items():
        print(f"=== {city} ===")
        try:
            w_srel_daily = build_w_srel_daily(cfg["era5_dir"], cfg["utc_offset_hours"])
        except FileNotFoundError as e:
            print(f"  Skipping {city}: {e}")
            continue
        print(f"  {w_srel_daily.shape[0]} daily rows, w_srel describe:")
        print(w_srel_daily["w_srel"].describe())

        out_path = cfg["era5_dir"].parent.parent / "processed" / "colombia" / city / f"{city}_w_srel_daily.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w_srel_daily.to_parquet(out_path, index=False)
        print(f"  Wrote {out_path}")

        if not cfg["master_parquet"].exists():
            print(f"  Master dataset not found at {cfg['master_parquet']}, skipping correlation check")
            continue
        master = pd.read_parquet(cfg["master_parquet"])
        master["date"] = pd.to_datetime(master["date"])
        merged = master.merge(w_srel_daily, on="date", how="inner")
        d = merged.dropna(subset=[cfg["pm25_col"], "w_srel"])
        if "w700" in d.columns:
            d2 = d.dropna(subset=["w700"])
            print(f"  n={len(d2)} corr(w700, pm25)   = {np.corrcoef(d2['w700'], d2[cfg['pm25_col']])[0,1]:.3f}")
            print(f"  n={len(d2)} corr(w_srel, pm25) = {np.corrcoef(d2['w_srel'], d2[cfg['pm25_col']])[0,1]:.3f}")
        else:
            print(f"  n={len(d)} corr(w_srel, pm25) = {np.corrcoef(d['w_srel'], d[cfg['pm25_col']])[0,1]:.3f}")


if __name__ == "__main__":
    main()
