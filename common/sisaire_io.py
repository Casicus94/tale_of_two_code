"""Loaders for SISAIRE 'reporte_sisaire' daily-mean CSV exports.

Each file (e.g. ``BG_PM25_reporte_sisaire.csv``) holds one variable for one city/department,
in long format: ``"Estacion","Fecha inicial","Fecha final","<value column>"``, one row per
station-day. ``Fecha final`` is unused (always blank in the daily exports seen so far).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Bogotá (BG_*) variable-code -> (raw column name in the CSV, tidy name used downstream).
BOGOTA_VARIABLES = {
    "PM25": ("PM2.5", "pm25"),
    "PM10": ("PM10", "pm10"),
    "NO2": ("NO2", "no2"),
    "O3": ("O3", "o3"),
    "VV": ("VViento", "wind_speed"),
    "DV": ("DViento", "wind_dir"),
    "T2": ("TAire2", "temp_2m"),
    "T10": ("TAire10", "temp_10m"),
    "HAire2": ("HAire2", "rh_2m"),
    "HAire10": ("HAire10", "rh_10m"),
    "P": ("P", "pressure"),
}


def normalize_station_name(name: str) -> str:
    """Strip stray quote characters/whitespace left over from doubled CSV quoting."""
    return re.sub(r'"+', "", str(name)).strip()


def load_sisaire_file(path: Path | str, tidy_name: str) -> pd.DataFrame:
    """Load one ``*_reporte_sisaire.csv`` file into a tidy (station, date, value) frame."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    value_col = df.columns[-1]
    out = pd.DataFrame(
        {
            "station": df["Estacion"].map(normalize_station_name),
            "date": pd.to_datetime(df["Fecha inicial"], errors="coerce"),
            tidy_name: pd.to_numeric(df[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date"])
    # A handful of duplicate station-day rows exist in the raw export; average them.
    out = out.groupby(["station", "date"], as_index=False)[tidy_name].mean()
    return out


def load_city_dataset(
    data_dir: Path | str,
    city_prefix: str,
    variables: dict[str, tuple[str, str]] = BOGOTA_VARIABLES,
) -> pd.DataFrame:
    """Load and merge every ``<city_prefix>_<code>_reporte_sisaire.csv`` file for a city.

    Returns one wide table indexed by (station, date) with one column per tidy variable
    name (e.g. ``pm25``, ``wind_speed``, ...). Missing variable files are skipped with a
    printed warning rather than raising, since coverage differs city to city.
    """
    data_dir = Path(data_dir)
    merged: pd.DataFrame | None = None
    for code, (_raw_col, tidy_name) in variables.items():
        path = data_dir / f"{city_prefix}_{code}_reporte_sisaire.csv"
        if not path.exists():
            print(f"[load_city_dataset] missing file, skipping: {path.name}")
            continue
        var_df = load_sisaire_file(path, tidy_name)
        merged = var_df if merged is None else merged.merge(
            var_df, on=["station", "date"], how="outer"
        )
    if merged is None:
        raise FileNotFoundError(f"No SISAIRE files found for prefix '{city_prefix}' in {data_dir}")
    return merged.sort_values(["station", "date"]).reset_index(drop=True)
