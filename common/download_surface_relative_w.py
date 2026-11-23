"""Download ERA5 vertical velocity at multiple pressure levels, for both cities, so a
surface-relative (rather than fixed-700-hPa) vertical-velocity feature can be built.

Motivation (see MASTER.md, "HIGH PRIORITY, new (user-raised, 2026-09-10): check whether
`w700` even means what it's assumed to mean at Bogota's altitude", sub-item 3): Bogota's
own surface pressure (729-767 hPa) sits only 29-67 hPa above the fixed 700 hPa level, so
"w700" there is nowhere near as deep into the free troposphere as "700 hPa" is for a
sea-level city -- for a sea-level city (standard surface pressure ~1013.25 hPa), 700 hPa
sits ~313 hPa below the surface. This script downloads vertical velocity at four
pressure levels (400, 450, 500, 550 hPa) for both cities so a *surface-relative* level,
defined as (local ERA5 surface pressure - 313 hPa) at each grid point and hour --
reproducing the same depth-below-surface as the sea-level convention -- can be built by
linear interpolation between the two bracketing downloaded levels. This lands:
  - Bogota (surface 729-767 hPa): target range 416-454 hPa -> needs 400/450 (500 kept
    as a safety margin above the top of the range).
  - Medellin (surface 772-837 hPa): target range 459-524 hPa -> needs 450/500/550.
Four levels (400, 450, 500, 550) cover both cities' full target ranges with margin.

Run directly (long-running -- CDS queue times can be substantial; safe to run with
nohup and check back later, same pattern as the v6 Medellin ERA5 pull):

    python Scripts/common/download_surface_relative_w.py

Output: Data/external/era5/bogota_era5_w{400,450,500,550}_{year}.nc (2014-2025) and
        Data/external/era5_medellin/bogota_era5_w{400,450,500,550}_{year}.nc (2015-2022)
        -- same "bogota_era5_..." filename prefix convention already used for
        Medellin's own w700/t700/sp_lts files (kept for consistency; the files live in
        the city-specific directory, so there's no collision).
Analysis (computing w_srel and re-testing the PM2.5 relationship) is a separate,
follow-up notebook step once these land.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

from common.external_data import (  # noqa: E402
    BOGOTA_ERA5_AREA,
    MEDELLIN_ERA5_AREA,
    download_era5_pressure_level_hourly,
)

LEVELS = ["400", "450", "500", "550"]

CITY_CONFIG = {
    "bogota": {
        "start_year": 2014,
        "end_year": 2025,
        "area": BOGOTA_ERA5_AREA,
        "target_dir": PROJECT_ROOT / "Data" / "external" / "era5",
    },
    "medellin": {
        "start_year": 2015,
        "end_year": 2022,
        "area": MEDELLIN_ERA5_AREA,
        "target_dir": PROJECT_ROOT / "Data" / "external" / "era5_medellin",
    },
}


def main() -> None:
    for city, cfg in CITY_CONFIG.items():
        for level in LEVELS:
            print(f"=== {city} / {level} hPa / {cfg['start_year']}-{cfg['end_year']} ===", flush=True)
            paths = download_era5_pressure_level_hourly(
                cfg["start_year"],
                cfg["end_year"],
                cfg["target_dir"],
                pressure_level=level,
                variable="vertical_velocity",
                area=cfg["area"],
            )
            print(f"  {len(paths)} files ready in {cfg['target_dir']}", flush=True)
    print("All surface-relative-w downloads complete.", flush=True)


if __name__ == "__main__":
    main()
