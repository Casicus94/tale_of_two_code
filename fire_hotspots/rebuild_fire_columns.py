"""Rebuild the fire-hotspot columns of the four master datasets from a given hotspot grid.

Why (2026-10-20): the old gridded product (`Hotspots_count_2013-2023_South_America_combined.nc`) was found to be
spatially wrong in most years (daily totals agree with FIRMS, but where the fires sit does not; see
`Eq_discovery_countries/MASTER_countries.md`). A FIRMS regeneration (VIIRS S-NPP 375 m, same 0.75 deg grid,
2013 -> 2026-10-19) exists at `Data/external/firms/Hotspots_count_2013-2026-10-19_South_America_FIRMS.nc`.

What it replaces, exactly as the original notebooks built it (Bogota 04/05, Medellin 02/04):
  1. daily counts in the 3 named source regions and in the 8 compass sectors of the city (`hotspots_<region>`, `hotspots_sector_<X>`),
  2. wind-transport variants  = count * max(0, cos(wind_dir - bearing))  (`..._transport`),
  3. lags 1-3 of every one of those columns.
Nothing else in a master dataset is touched.

Usage:  python rebuild_fire_columns.py validate          # old grid -> must reproduce the OLD master columns exactly
        python rebuild_fire_columns.py apply  <new.nc>   # replace the fire columns, keeping a backup of every file touched
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/path/to/project")
sys.path.insert(0, str(ROOT / "Scripts"))
from common.external_data import (load_hotspots_daily_by_region, load_hotspots_daily_by_sector, FIRE_SOURCE_REGIONS,
                                  bearing_from_bogota, BOGOTA_LAT, BOGOTA_LON, MEDELLIN_LAT, MEDELLIN_LON)

DATA = ROOT / "Data" / "processed" / "colombia"
OLD_NC = Path("/path/to/fire_data/Hotspots_count_2013-2023_South_America_combined.nc")
SECTOR_BEARINGS = {n: i * 45.0 for i, n in enumerate(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}
MASTERS = {"bogota": ["bogota_master_dataset.parquet", "bogota_o3_master_dataset.parquet"],
           "medellin": ["medellin_master_dataset.parquet", "medellin_o3_master_dataset.parquet"]}
LAGS = (1, 2, 3)


def daily_series(nc: Path, city: str) -> pd.DataFrame:
    reg = load_hotspots_daily_by_region(nc, FIRE_SOURCE_REGIONS)
    kw = dict(lat=MEDELLIN_LAT, lon=MEDELLIN_LON) if city == "medellin" else {}
    sec = load_hotspots_daily_by_sector(nc, **kw)
    return reg.merge(sec, on="date", how="outer")


def fire_block(nc: Path, city: str, dates: pd.Series, wind_dir: pd.Series) -> pd.DataFrame:
    """All 88 fire columns for the rows of a master dataset (one row per day, sorted by date)."""
    base = daily_series(nc, city)
    df = pd.DataFrame({"date": pd.to_datetime(dates).to_numpy(), "wind_dir": wind_dir.to_numpy()}).merge(base, on="date", how="left")
    region_cols = [c for c in base.columns if c.startswith("hotspots_") and not c.startswith("hotspots_sector_")]
    sector_cols = [c for c in base.columns if c.startswith("hotspots_sector_")]
    for rc in region_cols:
        name = rc.replace("hotspots_", "")
        b = (bearing_from_bogota(FIRE_SOURCE_REGIONS[name]["lat"], FIRE_SOURCE_REGIONS[name]["lon"], origin_lat=MEDELLIN_LAT, origin_lon=MEDELLIN_LON)
             if city == "medellin" else bearing_from_bogota(FIRE_SOURCE_REGIONS[name]["lat"], FIRE_SOURCE_REGIONS[name]["lon"]))
        df[f"{rc}_transport"] = df[rc] * np.cos(np.deg2rad(df["wind_dir"] - b)).clip(lower=0)
    for sc in sector_cols:
        df[f"{sc}_transport"] = df[sc] * np.cos(np.deg2rad(df["wind_dir"] - SECTOR_BEARINGS[sc.replace("hotspots_sector_", "")])).clip(lower=0)
    fire_cols = [c for c in df.columns if c.startswith("hotspots_")]
    out = df.set_index("date")
    for c in list(fire_cols):
        for k in LAGS: out[f"{c}_lag{k}"] = out[c].shift(k)
    out = out.drop(columns="wind_dir").reset_index()
    return out


def fire_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("hotspots_")]


def validate() -> bool:
    ok_all = True
    for city, files in MASTERS.items():
        for f in files:
            m = pd.read_parquet(DATA / city / f); m["date"] = pd.to_datetime(m["date"])
            new = fire_block(OLD_NC, city, m["date"], m["wind_dir"])
            cols = fire_columns(m)
            miss = set(cols) - set(new.columns); extra = set(new.columns) - set(cols) - {"date"}
            a = m[cols].to_numpy(float); b = new.set_index(new["date"].values)[cols].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            same_nan = (np.isfinite(a) == np.isfinite(b)).mean()
            maxdiff = np.nanmax(np.abs(a[both] - b[both])) if both.any() else np.nan
            ok = (not miss) and (not extra) and maxdiff < 1e-6 and same_nan > 0.999
            ok_all &= ok
            print(f"{city:9s} {f:32s} cols {len(cols)} (missing {len(miss)}, extra {len(extra)}) | max |old master - rebuilt from old grid| = {maxdiff:.2e} | same NaN pattern {same_nan:.4f} -> {'OK' if ok else 'MISMATCH'}")
    return ok_all


def apply(new_nc: Path) -> None:
    bk = DATA / "_oldfire_backup"; bk.mkdir(exist_ok=True)
    for city, files in MASTERS.items():
        for f in files:
            src = DATA / city / f
            if not (bk / f).exists(): shutil.copy2(src, bk / f)
            m = pd.read_parquet(src); m["date"] = pd.to_datetime(m["date"])
            new = fire_block(new_nc, city, m["date"], m["wind_dir"])
            cols = fire_columns(m)
            assert set(cols) == set(c for c in new.columns if c.startswith("hotspots_")), "column set differs"
            m2 = m.copy()
            for c in cols: m2[c] = new[c].to_numpy()
            m2.to_parquet(src)
            print(f"{city:9s} {f:32s} fire columns replaced ({len(cols)}); old NaN share {m[cols].isna().mean().mean():.3f} -> {m2[cols].isna().mean().mean():.3f}")
        # the daily (unlagged) series files the 04/02 notebooks wrote
        hp = DATA / city / f"{city}_hotspots_daily.parquet"
        if hp.exists() and not (bk / hp.name).exists(): shutil.copy2(hp, bk / hp.name)
        daily_series(new_nc, city).to_parquet(hp, index=False); print(f"{city:9s} {hp.name} rewritten from the new grid")


if __name__ == "__main__":
    if sys.argv[1] == "validate": sys.exit(0 if validate() else 1)
    if sys.argv[1] == "apply": apply(Path(sys.argv[2]))
