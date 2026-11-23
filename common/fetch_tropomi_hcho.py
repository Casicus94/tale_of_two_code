"""Pull TROPOMI (Sentinel-5P) tropospheric HCHO and/or NO2 as daily precursor proxies.

Motivation: Casallas et al. (2023, AQAH) used TROPOMI NO2 and HCHO ("formaldehyde
as a VOC representative") as ozone-precursor inputs. This project's O3 equation
currently has no VOC term at all -- PM2.5 shows up in the discovered equation with
a large unexplained residual association with O3 after testing every mechanism this
project could find in the literature (radiative shading, PBLH-mediated, heterogeneous
N2O5/HOx-ROx chemistry, PMC/VOC-oxidation, particulate-nitrate renoxification -- all
null, partial, or decisively ruled out; see MASTER.md, "Mechanism tests actually run"
and the later 2026-09-11 TODO sections). HCHO (a real VOC proxy) was added and also
came back null/unselected. The one still-untested lead: ground-station NO2 (used
throughout this pipeline) is a single-point measurement of a spatially heterogeneous
pollutant -- TROPOMI's satellite NO2 column captures a wider/regional signal and may
be a more complete measure of the same traffic/regional-precursor loading PM2.5 also
reflects, worth ruling out before concluding PM2.5's residual signal is unexplained.
Covers both Bogota and Medellin, both variables (2026-09-11 addition -- originally
HCHO/Bogota-only, generalized here to add NO2 without duplicating any of the already-
debugged Earth Engine plumbing).

Run this on a machine with normal internet access (Google's OAuth/API endpoints
are blocked by the org network policy inside the Cowork cloud sandbox and its
linked-device shell -- confirmed 2026-09-11). One-time setup:

    pip install earthengine-api pandas

Then run (defaults to both cities and both variables):

    python Scripts/common/fetch_tropomi_hcho.py
    python Scripts/common/fetch_tropomi_hcho.py --city bogota --variable no2
    python Scripts/common/fetch_tropomi_hcho.py --city medellin --variable hcho
    python Scripts/common/fetch_tropomi_hcho.py --variable no2   # both cities, NO2 only

First run opens a browser for Google sign-in (via Earth Engine's own relay auth
flow at code.earthengine.google.com/client-auth -- NOT the deprecated oob flow) and
will ask you to pick or create a Google Cloud project (any free project works; the
Earth Engine API just needs to be enabled on it, which the sign-in flow offers to
do automatically). Credentials are cached afterward under .ee_credentials/credentials
(inside this project, not the usual ~/.config/earthengine -- see the note by
_project_credentials_path() below) -- subsequent runs won't prompt again, and the
already-authenticated HCHO run's cached credentials/project id are reused as-is for
NO2, no re-auth needed. That credentials file holds a refresh token; keep it out of
git (add .ee_credentials/ to .gitignore if the project is version-controlled).

Output: Data/external/tropomi_hcho/<city>_hcho_daily.csv and/or
        Data/external/tropomi_hcho/<city>_no2_daily.csv
with columns date, <band>, n_obs, where <band> is the region-mean tropospheric column
density (mol/m^2) over that city's ERA5 box (same boxes used for the ERA5 pulls in
external_data.py -- BOGOTA_ERA5_AREA / MEDELLIN_ERA5_AREA), and n_obs is the number of
valid (QA-filtered) satellite pixels that went into that day's mean, so low-confidence
days can be filtered or weighted downstream. Both collections start in the S5P
satellite era (HCHO: 2018-04-30; NO2: 2018-04-30 as well, same platform/launch) --
neither will cover either dataset's full historical span (Bogota back to 2014,
Medellin's own start date), only the satellite era.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import ee.oauth
import pandas as pd

# ~/.config/earthengine can end up owned by root or otherwise unwritable on some
# machines (e.g. left behind by an earlier `sudo pip install` or a Docker tool) --
# rather than asking for sudo just to fix that directory, keep Earth Engine's
# credentials inside the project instead. get_credentials_path() is called fresh
# each time (not cached at import), so patching it here is enough.
_CREDENTIALS_DIR = Path(__file__).resolve().parents[2] / ".ee_credentials"


def _project_credentials_path() -> str:
    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    return str(_CREDENTIALS_DIR / "credentials")


ee.oauth.get_credentials_path = _project_credentials_path

# Same boxes as external_data.BOGOTA_ERA5_AREA / MEDELLIN_ERA5_AREA: [North, West,
# South, East].
CITY_AREAS = {
    "bogota": [4.9, -74.35, 4.35, -73.85],
    "medellin": [6.65, -75.85, 6.0, -75.35],
}

# Per-variable Earth Engine config. Both HCHO and NO2 turned out to have no qa_value
# band in this actual collection's live band list (confirmed 2026-09-18 by running the
# NO2 fetch: EE raised "Band pattern 'qa_value' did not match any bands. Available
# bands: [NO2_column_number_density, tropospheric_NO2_column_number_density,
# stratospheric_NO2_column_number_density, NO2_slant_column_number_density,
# tropopause_pressure, absorbing_aerosol_index, cloud_fraction, sensor_altitude,
# sensor_azimuth_angle, sensor_zenith_angle, solar_azimuth_angle,
# solar_zenith_angle]") -- the original assumption (documented pre-2026-09-18) that
# NO2 carries a standard qa_value band was wrong for this live product/API version.
# Both variables use the same cloud_fraction<=0.3 clear-sky threshold instead.
VARIABLE_CONFIG = {
    "hcho": {
        "collection": "COPERNICUS/S5P/OFFL/L3_HCHO",
        "band": "tropospheric_HCHO_column_number_density",
        "qa_band": "cloud_fraction",
        "qa_keep": lambda img: img.select("cloud_fraction").lte(0.3),
        "output_column": "hcho_trop_col_mol_m2",
        "start_date": "2018-05-01",
    },
    "no2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
        "qa_band": "cloud_fraction",
        "qa_keep": lambda img: img.select("cloud_fraction").lte(0.3),
        "output_column": "no2_trop_col_mol_m2",
        "start_date": "2018-05-01",
    },
}

END_DATE = "2026-01-01"  # exclusive upper bound; adjust as new data becomes available

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "Data" / "external" / "tropomi_hcho"


def _region_for(city: str) -> ee.Geometry:
    """Built lazily, after ee.Initialize() -- constructing ee.Geometry at import time
    (before the client is initialized) raises "Earth Engine client library not
    initialized" even though nothing here actually needs the network yet."""
    area = CITY_AREAS[city]
    return ee.Geometry.Rectangle([area[1], area[2], area[3], area[0]])  # [W, S, E, N]


def _daily_mean_for_date(date: ee.Date, region: ee.Geometry, variable: str) -> ee.Feature:
    """Region-mean, QA-filtered column density for one UTC day, as a null-safe Feature.

    Cloud/orbit gaps mean some days have zero matching images. Reducing an empty
    ImageCollection's .mean()/.count() has no CRS to reduce over and raises
    "Unable to find a crs" -- so the empty case is short-circuited to a
    null-valued Feature via ee.Algorithms.If before reduceRegion ever runs.
    """
    cfg = VARIABLE_CONFIG[variable]
    band, qa_band, output_column = cfg["band"], cfg["qa_band"], cfg["output_column"]

    day_imgs = (
        ee.ImageCollection(cfg["collection"])
        .filterDate(date, date.advance(1, "day"))
        .filterBounds(region)
        .select([band, qa_band])
    )
    masked = day_imgs.map(lambda img: img.updateMask(cfg["qa_keep"](img)))

    def _reduce() -> ee.Dictionary:
        # Two separate reduceRegion calls instead of Reducer.combine(): combine's
        # output band-naming (outputPrefix + collision rules between the two
        # reducers) turned out not to produce the "n_obs_sum" key this code
        # expected -- reduceRegion raised "Dictionary does not contain key:
        # 'n_obs_sum'". Doing it as two plain reductions is unambiguous.
        mean_img = masked.select(band).mean()
        count_img = masked.select(band).count().rename("n_obs")
        mean_stats = mean_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5566,  # native ~5.5km resolution (same for HCHO and NO2 L3 products)
            maxPixels=1e9,
            bestEffort=True,
        )
        count_stats = count_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=5566,
            maxPixels=1e9,
            bestEffort=True,
        )
        return ee.Dictionary(
            {
                band: mean_stats.get(band),
                "n_obs_sum": count_stats.get("n_obs"),
            }
        )

    empty_stats = ee.Dictionary({band: None, "n_obs_sum": 0})
    stats = ee.Dictionary(
        ee.Algorithms.If(day_imgs.size().gt(0), _reduce(), empty_stats)
    )
    return ee.Feature(
        None,
        {
            "date": date.format("YYYY-MM-dd"),
            output_column: stats.get(band),
            "n_obs": stats.get("n_obs_sum"),
        },
    )


def fetch(city: str, variable: str, end_date: str = END_DATE) -> pd.DataFrame:
    cfg = VARIABLE_CONFIG[variable]
    start_date = cfg["start_date"]
    region = _region_for(city)
    n_days = ee.Date(end_date).difference(ee.Date(start_date), "day").round()
    dates = ee.List.sequence(0, n_days.subtract(1)).map(
        lambda offset: ee.Date(start_date).advance(offset, "day")
    )
    features = ee.FeatureCollection(
        dates.map(lambda d: _daily_mean_for_date(ee.Date(d), region, variable))
    )

    # getInfo() on the whole FeatureCollection at once -- for a ~7-8 year daily series
    # (~2700 features) this is a single moderate-sized request, well under Earth
    # Engine's interactive-request limits. If it ever times out, chunk by year instead.
    info = features.getInfo()
    rows = [f["properties"] for f in info["features"]]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


_PROJECT_ID_PATH = _CREDENTIALS_DIR / "project_id.txt"


def _get_or_ask_project_id() -> str:
    """Recent earthengine-api versions require an explicit Cloud project for
    ee.Initialize() -- it's no longer inferred from the OAuth credentials alone.
    Any Google Cloud project works as long as the Earth Engine API is enabled on
    it (see https://console.cloud.google.com/earth-engine -- it offers to enable
    the API and even create a free project for you on first visit). Cached in
    .ee_credentials/project_id.txt so this only has to be entered once."""
    if _PROJECT_ID_PATH.exists():
        cached = _PROJECT_ID_PATH.read_text().strip()
        if cached:
            return cached
    project_id = input(
        "Enter your Google Cloud project ID for Earth Engine "
        "(create one free at https://console.cloud.google.com/earth-engine if needed): "
    ).strip()
    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    _PROJECT_ID_PATH.write_text(project_id + "\n")
    return project_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--city",
        choices=["bogota", "medellin", "both"],
        default="both",
        help="Which city to fetch (default: both).",
    )
    parser.add_argument(
        "--variable",
        choices=["hcho", "no2", "both"],
        default="both",
        help="Which TROPOMI product to fetch (default: both).",
    )
    args = parser.parse_args()
    cities = list(CITY_AREAS) if args.city == "both" else [args.city]
    variables = list(VARIABLE_CONFIG) if args.variable == "both" else [args.variable]

    project_id = _get_or_ask_project_id()
    try:
        ee.Initialize(project=project_id)
    except Exception:
        print("Not authenticated yet -- launching Earth Engine sign-in flow...", file=sys.stderr)
        ee.Authenticate()
        ee.Initialize(project=project_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variable in variables:
        cfg = VARIABLE_CONFIG[variable]
        for city in cities:
            print(f"Fetching TROPOMI {variable.upper()} for {city} box {CITY_AREAS[city]}, "
                  f"{cfg['start_date']} to {END_DATE}...")
            df = fetch(city, variable)
            out_col = cfg["output_column"]
            print(f"  {city}/{variable}: {len(df)} daily rows, "
                  f"{df[out_col].notna().sum()} with data "
                  f"({df[out_col].isna().sum()} cloud/QA-gap days).")

            out_path = OUTPUT_DIR / f"{city}_{variable}_daily.csv"
            df.to_csv(out_path, index=False)
            print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
