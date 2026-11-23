"""Helpers for the two external datasets used to explain PM2.5 controls:

- ERA5 (via ``cdsapi``) for boundary-layer height, wind, temperature, radiation, precip.
- The group's existing daily fire-hotspot-count NetCDF (``Fires_Amazon/Fire_Amazon``),
  reused here rather than re-downloading from FIRMS.

City coordinates are taken from the group's own published work rather than guessed:
Bogotá is at lat 4.5, lon -74.05, 2550 m a.s.l. (Casallas et al. 2022, STOTEN 852:158504).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BOGOTA_LAT, BOGOTA_LON = 4.5, -74.05
BOGOTA_UTC_OFFSET_HOURS = -5  # COT, fixed year-round (no DST) -- Casallas et al. papers.


def _shift_to_local_time(ds: xr.Dataset) -> xr.Dataset:
    """Relabel an hourly ERA5 dataset's UTC time coordinate as Bogotá local time.

    ERA5 timestamps are UTC; a naive ``resample(time="1D")`` therefore buckets by UTC
    calendar days, offset 5 hours from Bogotá's actual local day. That's enough to
    misalign daily aggregates with the SISAIRE stations' local-day PM2.5 values, and it
    would badly corrupt anything computed over a specific local time-of-day window (e.g.
    an early-morning inversion signal). Bogotá has no DST, so this is a fixed shift.
    """
    return ds.assign_coords(time=ds["time"] + np.timedelta64(BOGOTA_UTC_OFFSET_HOURS, "h"))

# [North, West, South, East] box around Bogotá, wide enough to cover the urban area
# and give ERA5's ~31km-native grid a couple of surrounding points.
BOGOTA_ERA5_AREA = [4.9, -74.35, 4.35, -73.85]

# Medellín (Valle de Aburrá metro area) -- same fixed UTC-5 offset as Bogotá, all of
# Colombia is one timezone. Center ~6.244N, -75.581W; box widened north/south to cover
# the whole Aburrá Valley urban corridor (Barbosa/Girardota down to Caldas/La Estrella).
MEDELLIN_LAT, MEDELLIN_LON = 6.244, -75.581
MEDELLIN_ERA5_AREA = [6.65, -75.85, 6.0, -75.35]

# Only variables Bogotá's SISAIRE station network cannot provide: BLH has no ground
# instrument (needs a radiosonde or reanalysis/model), and neither radiation nor
# precipitation appear in any local Data/COLOMBIA file. Wind, temperature, RH and
# pressure are already covered by station gauges (see sisaire_io.BOGOTA_VARIABLES) and
# are NOT re-pulled from ERA5 here -- the gauges are the better source for those.
ERA5_VARIABLES = [
    "boundary_layer_height",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]


# CDS's per-request "cost limit" rejects even 3 variables x 1 year x hourly for this
# area (confirmed empirically), but 1 variable x 1 year x hourly succeeds. So requests
# are chunked all the way down to (variable, year) pairs.
_VARIABLE_SHORT_NAMES = {
    "boundary_layer_height": "blh",
    "surface_solar_radiation_downwards": "ssrd",
    "total_precipitation": "tp",
    # Pulled specifically for Lower Tropospheric Stability (LTS), not as a general
    # temperature predictor (station 2m temperature remains the better source for that
    # -- see the module docstring). LTS needs a physically self-consistent surface
    # potential temperature, i.e. computed from ERA5's own 2m temperature and surface
    # pressure at the same grid point as the 700 hPa temperature, not mixed with a
    # station reading.
    "2m_temperature": "t2m_lts",
    "surface_pressure": "sp_lts",
    # v6: total column water vapour (distinct from surface RH -- integrated atmospheric
    # moisture, relevant to haze/secondary-aerosol potential) and 10m wind components
    # (pulled specifically to compute a within-day turbulence/TKE-like proxy from
    # hourly variance -- station wind is daily-mean only, so only ERA5 can give us
    # sub-daily variability; NOT meant to replace station wind_speed as the daily-mean
    # predictor, which stays the better source for that).
    "total_column_water_vapour": "tcwv",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    # Thermal driver of daytime boundary-layer growth/mixing -- distinct from the
    # TKE proxy above, which only captures mechanical (wind-driven) turbulence.
    "surface_sensible_heat_flux": "sshf",
}


def download_era5_hourly_variable_year(
    variable: str,
    year: int,
    target: Path | str,
    area: list[float] = BOGOTA_ERA5_AREA,
) -> Path:
    """Submit one CDS request for a single hourly ERA5 variable, one year, over `area`."""
    import cdsapi

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [variable],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
        },
        str(target),
    )
    return target


def download_era5_hourly(
    start_year: int,
    end_year: int,
    target_dir: Path | str,
    area: list[float] = BOGOTA_ERA5_AREA,
    variables: list[str] = ERA5_VARIABLES,
    batch_size: int = 5,
) -> list[Path]:
    """Download hourly ERA5 single-level fields for each (variable, year) pair.

    Requires a working ``~/.cdsapirc``. Missing (variable, year) requests are submitted
    in batches of `batch_size` and downloaded before the next batch is submitted, so
    they queue/process concurrently on the CDS rather than fully one-at-a-time
    (~30 min/request serially would be many hours for a full multi-year pull) while
    staying under CDS's "number of queued requests for this dataset is temporarily
    limited" cap (submitting all requests at once was rejected outright). Already-
    downloaded files are skipped, so a partial/interrupted run can be safely re-invoked.
    """
    import cdsapi

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    to_submit = []  # (variable, year, path)
    all_paths = []
    for variable in variables:
        short = _VARIABLE_SHORT_NAMES.get(variable, variable)
        for year in range(start_year, end_year + 1):
            path = target_dir / f"bogota_era5_{short}_{year}.nc"
            all_paths.append(path)
            if path.exists():
                print(f"[download_era5_hourly] {short} {year} already downloaded, skipping")
            else:
                to_submit.append((variable, year, path))

    for batch_start in range(0, len(to_submit), batch_size):
        batch = to_submit[batch_start : batch_start + batch_size]
        remotes = []
        for variable, year, path in batch:
            short = _VARIABLE_SHORT_NAMES.get(variable, variable)
            print(f"[download_era5_hourly] submitting {short} {year} ...")
            # cdsapi's underlying cads_api_client no longer exposes `.client.submit(name,
            # request)` (confirmed 2026-09-18 -- `AttributeError: 'ApiClient' object has
            # no attribute 'submit'`, a library-version change from whatever cdsapi
            # version this batched-concurrent-submit pattern was originally written
            # against). The equivalent non-blocking submit is
            # `client.client.collection(name).submit(**request)`.
            remote = client.client.collection("reanalysis-era5-single-levels").submit(
                product_type="reanalysis",
                format="netcdf",
                variable=[variable],
                year=[str(year)],
                month=[f"{m:02d}" for m in range(1, 13)],
                day=[f"{d:02d}" for d in range(1, 32)],
                time=[f"{h:02d}:00" for h in range(24)],
                area=area,
            )
            remotes.append((remote, path, short, year))

        for remote, path, short, year in remotes:
            print(f"[download_era5_hourly] downloading {short} {year} ...")
            remote.download(str(path))

    return all_paths


_PRESSURE_VARIABLE_SHORT_NAMES = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
}


def download_era5_pressure_level_hourly(
    start_year: int,
    end_year: int,
    target_dir: Path | str,
    pressure_level: str = "700",
    variable: str = "temperature",
    area: list[float] = BOGOTA_ERA5_AREA,
    batch_size: int = 5,
) -> list[Path]:
    """Download one hourly ERA5 pressure-level variable at `pressure_level` hPa.

    Separate collection from ``reanalysis-era5-single-levels`` (needs a `pressure_level`
    key). Originally added for temperature (Lower Tropospheric Stability -- see
    ``compute_lts_daily``); also usable for wind components (`u_component_of_wind` /
    `v_component_of_wind`) to get the steering-level transport flow, which can differ
    from surface wind (shaped by local terrain/urban effects) -- a fallback candidate if
    LTS/stagnation features don't capture the signal well enough. Same batching
    rationale as ``download_era5_hourly``.
    """
    import cdsapi

    short = _PRESSURE_VARIABLE_SHORT_NAMES.get(variable, variable)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    to_submit, all_paths = [], []
    for year in range(start_year, end_year + 1):
        path = target_dir / f"bogota_era5_{short}{pressure_level}_{year}.nc"
        all_paths.append(path)
        if path.exists():
            print(f"[download_era5_pressure_level_hourly] {short}{pressure_level} {year} already downloaded, skipping")
        else:
            to_submit.append((year, path))

    for batch_start in range(0, len(to_submit), batch_size):
        batch = to_submit[batch_start : batch_start + batch_size]
        remotes = []
        for year, path in batch:
            print(f"[download_era5_pressure_level_hourly] submitting {short}{pressure_level} {year} ...")
            # See the matching fix/comment in download_era5_hourly above -- same
            # cads_api_client version change, same fix.
            remote = client.client.collection("reanalysis-era5-pressure-levels").submit(
                product_type="reanalysis",
                format="netcdf",
                variable=[variable],
                pressure_level=[pressure_level],
                year=[str(year)],
                month=[f"{m:02d}" for m in range(1, 13)],
                day=[f"{d:02d}" for d in range(1, 32)],
                time=[f"{h:02d}:00" for h in range(24)],
                area=area,
            )
            remotes.append((remote, path, year))
        for remote, path, year in remotes:
            print(f"[download_era5_pressure_level_hourly] downloading {short}{pressure_level} {year} ...")
            remote.download(str(path))

    return all_paths


def compute_lts_daily(
    t2m_paths: list[Path],
    sp_paths: list[Path],
    t700_paths: list[Path],
) -> pd.DataFrame:
    """Daily Lower Tropospheric Stability: LTS = theta_700 - theta_surface.

    theta = T * (1000 / P_hPa)^0.286 (Poisson's equation). Surface potential temperature
    uses ERA5's own 2m temperature and surface pressure at the same grid point as the
    700 hPa temperature (self-consistent within one reanalysis, not mixed with station
    data). Higher LTS = a stronger inversion / more stable, stagnant lower atmosphere
    that traps surface pollution -- the mechanism the group's own papers (e.g.
    Lagrangian_manuscript.pdf) cite as important via a similar Estimated Inversion
    Strength (EIS) diagnostic; LTS is a simpler, standard proxy for the same physics.
    """
    def _open_concat(paths):
        datasets = [xr.open_dataset(p) for p in paths]
        for i, d in enumerate(datasets):
            if "valid_time" in d.dims:
                datasets[i] = d.rename({"valid_time": "time"})
        return xr.concat(sorted(datasets, key=lambda d: d.time.values[0]), dim="time")

    t2m = _open_concat(t2m_paths).mean(dim=["latitude", "longitude"])["t2m"]
    sp = _open_concat(sp_paths).mean(dim=["latitude", "longitude"])["sp"] / 100.0  # Pa -> hPa
    t700 = _open_concat(t700_paths).mean(dim=["latitude", "longitude"])["t"]

    theta_surface = t2m * (1000.0 / sp) ** 0.286
    theta_700 = t700 * (1000.0 / 700.0) ** 0.286
    theta_diff = theta_700 - theta_surface
    theta_diff = theta_diff.assign_coords(time=theta_diff["time"] + np.timedelta64(BOGOTA_UTC_OFFSET_HOURS, "h"))

    lts = theta_diff.resample(time="1D").mean()
    # Local 05:00-08:00: LTS computed from the pre-dawn/early-morning state specifically,
    # since a 24h mean dilutes the actual overnight inversion with the well-mixed
    # daytime boundary layer (the mechanism doesn't operate uniformly across the day).
    morning = theta_diff.sel(time=theta_diff["time"].dt.hour.isin([5, 6, 7]))
    lts_morning = morning.resample(time="1D").mean()

    df = lts.to_dataframe(name="lts").reset_index().rename(columns={"time": "date"})
    df_morning = lts_morning.to_dataframe(name="lts_morning").reset_index().rename(columns={"time": "date"})
    df = df.merge(df_morning, on="date", how="outer")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "lts", "lts_morning"]]


def load_era5_daily(nc_path: Path | str | list[Path]) -> pd.DataFrame:
    """Load hourly ERA5 NetCDF file(s), spatially average over the box, aggregate to daily.

    ``nc_path`` may be a single file or a list of per-(variable, year) files (see
    ``download_era5_hourly``); files are opened individually and merged (outer join on
    time), since different files may cover disjoint variables and/or years.

    Returns one row per day with mean/min/max of boundary-layer height, mean solar
    radiation and daily-total precipitation, plus (if the corresponding files are
    passed) tcwv/sensible-heat-flux/TKE-proxy and general-purpose wind speed/direction
    (``*_era5`` suffix), temperature, RH and pressure. For Bogotá, wind/temperature/RH/
    pressure are deliberately NOT pulled from ERA5 -- the SISAIRE station gauges are the
    better source and are already in the cleaned network dataset (notebook 02) -- but a
    city without comparably good local gauge coverage (e.g. Medellín) can request the
    `_era5` variants here and rename them after merging.
    """
    def _open(p):
        d = xr.open_dataset(p)
        # The CDS's newer backend names the time dimension "valid_time" (and adds an
        # "expver"/"number" coordinate) instead of the classic "time".
        if "valid_time" in d.dims:
            d = d.rename({"valid_time": "time"})
        return d.drop_vars([v for v in ("expver", "number") if v in d.coords], errors="ignore")

    if isinstance(nc_path, (list, tuple)):
        datasets = [_open(p) for p in nc_path]
        # IMPORTANT: xr.merge(..., compat="override") does NOT concatenate datasets
        # that share a variable name but cover disjoint time ranges (e.g. one file per
        # year) -- it silently keeps only the FIRST dataset's values for that variable
        # and NaNs out the rest (confirmed empirically: merging 12 yearly BLH files this
        # way left only year 1 populated). The fix is to xr.concat same-variable
        # datasets along time first, then xr.merge the resulting (disjoint-variable)
        # per-variable datasets together.
        by_variable: dict[str, list[xr.Dataset]] = {}
        for d in datasets:
            for var in d.data_vars:
                by_variable.setdefault(var, []).append(d)
        per_variable_ds = [
            xr.concat(sorted(dsets, key=lambda d: d.time.values[0]), dim="time")
            for dsets in by_variable.values()
        ]
        ds = xr.merge(per_variable_ds, join="outer")
    else:
        ds = _open(nc_path)
    # Spatial mean over the small Bogotá box collapses lat/lon to a single city series.
    ds = ds.mean(dim=[d for d in ("latitude", "longitude") if d in ds.dims])
    ds = _shift_to_local_time(ds)  # UTC -> Bogotá local day boundaries before resampling

    daily = {}
    if "blh" in ds:
        g = ds["blh"].resample(time="1D")
        daily["blh_mean"] = g.mean()
        daily["blh_min"] = g.min()
        daily["blh_max"] = g.max()
        # Local 05:00-08:00: the pre-dawn/early-rush-hour inversion state that a 24h
        # mean or single hourly min washes out -- Bogotá's documented AM PM2.5 peak
        # (Tendencies_PM2p5.pdf) occurs at 8-9h local, right as this window ends.
        morning = ds["blh"].sel(time=ds["time"].dt.hour.isin([5, 6, 7]))
        daily["blh_morning_mean"] = morning.resample(time="1D").mean()
    if "ssrd" in ds:
        # J/m2 accumulated per hour -> daily mean W/m2
        daily["radiation_mean"] = ds["ssrd"].resample(time="1D").mean() / 3600.0
    if "tp" in ds:
        # m accumulated per hour -> daily total mm
        daily["precip_total_mm"] = ds["tp"].resample(time="1D").sum() * 1000.0
    if "tcwv" in ds:
        daily["tcwv"] = ds["tcwv"].resample(time="1D").mean()
    if "sshf" in ds:
        # J/m2 accumulated per hour -> daily mean W/m2 (ERA5 convention: positive = flux
        # into the atmosphere, i.e. daytime heating that drives convective mixing).
        daily["sensible_heat_flux"] = ds["sshf"].resample(time="1D").mean() / 3600.0
    if "u10" in ds and "v10" in ds:
        # A turbulent-kinetic-energy-like proxy: half the within-day variance of the
        # horizontal wind components. Needs ERA5's hourly resolution specifically --
        # station wind is daily-mean only, so this sub-daily variability isn't available
        # from any local gauge. NOT a replacement for station wind_speed as the
        # daily-mean wind predictor (that stays station-sourced).
        # NOTE: ds.resample(time="1D") (whole-dataset) does not support __getitem__ by
        # variable name in this xarray version -- `in` checks work but indexing raises
        # KeyError. Resample each DataArray individually instead.
        daily["tke_proxy"] = 0.5 * (ds["u10"].resample(time="1D").var()
                                     + ds["v10"].resample(time="1D").var())
        # General-purpose wind speed/direction from ERA5 -- named with an "_era5" suffix
        # so they never collide with a city's own station-sourced wind_speed/wind_dir
        # (Bogotá keeps using its station gauges; a city with no good local wind gauge,
        # e.g. Medellín, can rename these to wind_speed/wind_dir after merging).
        hourly_speed = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        daily["wind_speed_era5"] = hourly_speed.resample(time="1D").mean()
        u_mean = ds["u10"].resample(time="1D").mean()
        v_mean = ds["v10"].resample(time="1D").mean()
        # Meteorological "wind FROM" bearing (0=N, 90=E, ...) from the daily-mean vector
        # components -- matches the "wind blows FROM this bearing" convention used
        # elsewhere (e.g. bearing_from_bogota / the hotspot wind-transport alignment).
        daily["wind_dir_era5"] = (270 - np.degrees(np.arctan2(v_mean, u_mean))) % 360
    if "t2m" in ds:
        daily["temp_2m_era5"] = ds["t2m"].resample(time="1D").mean() - 273.15  # K -> degC
    if "d2m" in ds and "t2m" in ds:
        # Relative humidity from 2m temperature/dewpoint via the Magnus-Tetens formula
        # (both in degC); no local RH gauge to prefer for a city without good coverage.
        t_c = ds["t2m"] - 273.15
        td_c = ds["d2m"] - 273.15
        es = np.exp(17.625 * t_c / (243.04 + t_c))
        e = np.exp(17.625 * td_c / (243.04 + td_c))
        rh = 100.0 * e / es
        daily["rh_2m_era5"] = rh.resample(time="1D").mean()
    if "sp" in ds:
        daily["pressure_era5"] = ds["sp"].resample(time="1D").mean() / 100.0  # Pa -> hPa

    df = xr.Dataset(daily).to_dataframe().reset_index().rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_vertical_velocity_daily(w700_paths: list[Path]) -> pd.DataFrame:
    """Daily mean ERA5 vertical velocity (omega) at 700 hPa.

    Negative omega = ascent/instability (mixing, convection); positive = subsidence
    (a stable, capping, pollution-trapping regime) -- a genuinely different physical
    axis from LTS (temperature-based inversion strength) or horizontal wind.
    """
    datasets = [xr.open_dataset(p) for p in w700_paths]
    for i, d in enumerate(datasets):
        if "valid_time" in d.dims:
            datasets[i] = d.rename({"valid_time": "time"})
    ds = xr.concat(sorted(datasets, key=lambda d: d.time.values[0]), dim="time")
    ds = ds.mean(dim=["latitude", "longitude"])
    ds = _shift_to_local_time(ds)

    daily = ds["w"].resample(time="1D").mean()
    df = daily.to_dataframe(name="w700").reset_index().rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "w700"]]


def bearing_from_bogota(lat: float, lon: float, origin_lat: float = BOGOTA_LAT, origin_lon: float = BOGOTA_LON) -> float:
    """Initial compass bearing (0-360, 0=N) from Bogotá to (lat, lon)."""
    p1, p2 = np.radians(origin_lat), np.radians(lat)
    dlambda = np.radians(lon - origin_lon)
    x = np.sin(dlambda) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlambda)
    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


# Named fire/LRT source regions relevant to Bogotá's PM2.5, per Casallas et al. 2024
# (STOTEN 906:167350, "Air pollution analysis in Northwestern South America: A new
# Lagrangian framework"). That paper tracks actual pollution plumes in gridded CAMS/
# WRF-Chem fields; we don't have those fields, so this uses the fire-hotspot data we do
# have (Fires_Amazon product) as a proxy.
#
# REVISED (v7, user-flagged bug): the original version of this dict also had "cvb",
# "eastern_plains" and "north_colombia" as separate named circles, each our own
# unsourced approximation. Checking the paper's supplementary Fig. SM33 (mean hotspot
# count by season, actual data, not our proxy) shows this was a real mistake -- DJF/MAM
# hotspot activity in NWSA is ONE large, continuous band sweeping from the Venezuela
# border down through the Llanos (roughly 4-12N, 76-68W), not three separable clusters.
# Three overlapping circles placed inside that single band were highly collinear by
# construction (verified: cvb/eastern_plains centers were only 283km apart against a
# combined radius of 700km -- each center sat inside the other's circle), which is why
# the LOYO SFS only ever selected "cvb" and never "eastern_plains" or "caqueta": not
# because the others don't matter, but because cvb/eastern_plains were barely
# distinguishable, and caqueta is a real but objectively much smaller/lower-magnitude
# source in raw hotspot counts (confirmed in SM33: DJF contours reach 200 in the
# northern band vs ~20-40 near Caquetá).
#
# Fix: keep the three point sources with exact paper-quoted coordinates (caqueta,
# magdalena_valley, nevados -- geographically distinct, low mutual overlap), and
# replace the invented northern-band circles with **8 compass-direction sectors from
# Bogotá** (see load_hotspots_daily_by_sector) -- non-overlapping by construction, and
# actually the paper's own method for relating source direction to Bogotá plume
# arrivals (their Fig. 6f is a season x 8-direction matrix), rather than us guessing
# named-region boundaries the paper doesn't give.
FIRE_SOURCE_REGIONS = {
    # DJF source for Bogotá plumes (paper, exact coords given in text). Radius reduced
    # from 300 to 220km to cut overlap with neighboring regions (was overlapping
    # nevados and eastern_plains at 300km).
    "caqueta": {"lat": 1.03, "lon": -73.9, "radius_km": 220.0, "source": "paper"},
    # JJA/SON source for Bogotá plumes (paper, exact coords given in text). A narrow
    # N-S river valley, not really a 300km-radius disk -- reduced to 180km.
    "magdalena_valley": {"lat": 7.06, "lon": -73.85, "radius_km": 180.0, "source": "paper"},
    # JJA/SON source, volcanic + wildfire-prone (paper, exact coords given in text). A
    # specific volcanic massif -- reduced from 150 to 100km.
    "nevados": {"lat": 4.6, "lon": -75.36, "radius_km": 100.0, "source": "paper"},
}


def _haversine_km(lat0: float, lon0: float, lat2d: xr.DataArray, lon2d: xr.DataArray) -> xr.DataArray:
    r = 6371.0
    p1, p2 = np.radians(lat0), np.radians(lat2d)
    dphi = np.radians(lat2d - lat0)
    dlambda = np.radians(lon2d - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_hotspots_daily_by_region(
    nc_path: Path | str,
    regions: dict = FIRE_SOURCE_REGIONS,
) -> pd.DataFrame:
    """Daily fire-hotspot count within each named source region (see FIRE_SOURCE_REGIONS).

    Returns one column per region (``hotspots_<region>``), NOT a single flat-radius
    total -- so that which region matters in which season can be learned rather than
    assumed. Uses a gridded ``Hotspots_count_*.nc`` product (0.75deg, South America). Since 2026-10-20 this
    is the FIRMS regeneration ``Data/external/firms/Hotspots_count_2013-2026-10-19_South_America_FIRMS.nc``; the
    earlier 2013-2023 product was spatially wrong in most years (see Eq_discovery_countries/MASTER_countries.md).
    """
    ds = xr.open_dataset(nc_path)
    lat2d, lon2d = xr.broadcast(ds["latitude"], ds["longitude"])

    out = None
    for name, cfg in regions.items():
        dist_km = _haversine_km(cfg["lat"], cfg["lon"], lat2d, lon2d)
        mask = dist_km <= cfg["radius_km"]
        daily_count = ds["Hotspots"].where(mask).sum(dim=["latitude", "longitude"])
        col = f"hotspots_{name}"
        region_df = daily_count.to_dataframe(name=col).reset_index().rename(columns={"time": "date"})
        region_df = region_df[["date", col]]
        out = region_df if out is None else out.merge(region_df, on="date", how="outer")

    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def load_hotspots_daily_by_sector(
    nc_path: Path | str,
    lat: float = BOGOTA_LAT,
    lon: float = BOGOTA_LON,
    n_sectors: int = 8,
    max_radius_km: float = 1500.0,
) -> pd.DataFrame:
    """Daily fire-hotspot count in each compass-direction sector from (lat, lon).

    Mirrors the paper's own method for relating fire source direction to Bogotá plume
    arrivals (their Fig. 6f is a season x 8-direction matrix), rather than us guessing
    named-region boundaries -- and sectors are non-overlapping by construction, unlike
    the circles this replaced (see FIRE_SOURCE_REGIONS docstring for why that mattered).
    ``max_radius_km`` follows Fig. 6i's onset-distance histogram (events mostly within
    ~100-800 km of the city, with a tail out to ~1400 km) with some margin.

    Returns one column per sector (``hotspots_sector_N``, ``_NE``, ``_E``, ...).
    """
    ds = xr.open_dataset(nc_path)
    lat2d, lon2d = xr.broadcast(ds["latitude"], ds["longitude"])
    dist_km = _haversine_km(lat, lon, lat2d, lon2d)
    p1, p2 = np.radians(lat), np.radians(lat2d)
    dlambda = np.radians(lon2d - lon)
    x = np.sin(dlambda) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlambda)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360

    sector_width = 360.0 / n_sectors
    sector_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][:n_sectors]
    within_radius = dist_km <= max_radius_km

    out = None
    for i, name in enumerate(sector_names):
        center = i * sector_width
        # Angular distance from sector center, wrapped to [-180, 180].
        angle_diff = (bearing - center + 180) % 360 - 180
        mask = within_radius & (np.abs(angle_diff) <= sector_width / 2)
        daily_count = ds["Hotspots"].where(mask).sum(dim=["latitude", "longitude"])
        col = f"hotspots_sector_{name}"
        sector_df = daily_count.to_dataframe(name=col).reset_index().rename(columns={"time": "date"})
        sector_df = sector_df[["date", col]]
        out = sector_df if out is None else out.merge(sector_df, on="date", how="outer")

    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)
