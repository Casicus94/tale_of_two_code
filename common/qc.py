"""Quality-control utilities matching the conventions used in the group's prior papers
(Casallas et al. 2022, "Spatio-temporal analysis of PM2.5 and policies in Northwestern
South America", Sci. Total Environ. 852:158504 — see ``Articles/Articles_mios/Tendencies_PM2p5.pdf``
sec. 2.1.1 for the QC description this module follows):

- Chauvenet's criterion (via Lishu et al., 2015) to flag statistical outliers.
- A minimum temporal-completeness threshold (the paper uses >=75%) to decide which
  station-years are usable for analysis.
- Physical plausibility ranges per variable.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from scipy import special

# Physically plausible ranges for Bogotá/Antioquia surface obs (generous, to catch sensor
# faults and transcription errors, not to trim genuine extreme-pollution days).
PHYSICAL_RANGES = {
    "pm25": (0.0, 1000.0),      # µg/m3
    "pm10": (0.0, 1500.0),      # µg/m3
    "no2": (0.0, 500.0),        # µg/m3
    "o3": (0.0, 500.0),         # µg/m3
    "wind_speed": (0.0, 30.0),  # m/s
    "wind_dir": (0.0, 360.0),   # degrees
    "temp_2m": (-5.0, 35.0),    # °C, Bogotá is at 2550 m a.s.l.
    "temp_10m": (-5.0, 35.0),
    "rh_2m": (0.0, 100.0),      # %
    "rh_10m": (0.0, 100.0),
    "pressure": (500.0, 620.0), # hPa, station-level at Bogotá's altitude
    # Added 2026-09-23 for the new Antioquia (Medellín) CO/SO2/etc. batch -- co/so2 ranges are
    # calibrated against Bogotá's own co_daily/so2_daily processed series (co: mean 0.78, max 1.88
    # in whatever unit that network uses -- probably ppm, see notebook A4_units_audit; so2: mean
    # 11.6, max 44.7 µg/m3), generous margins around both.
    "co": (0.0, 50.0),            # ppm (probable, unverified -- see A4_units_audit)
    "so2": (0.0, 500.0),          # µg/m3
    "no": (0.0, 500.0),           # matches the existing Hourmean "no" range
    "nox": (0.0, 1000.0),         # µg/m3, NO + NO2 scale
    "pst": (0.0, 2000.0),         # µg/m3, total suspended particulates (coarser than PM10)
    "radiation_station": (0.0, 1500.0),   # W/m2, station pyranometer (R_GLOBAL)
    "temp_10cm": (-5.0, 40.0),    # °C, near-surface (10 cm) air temperature
    "precip_station": (0.0, 400.0),       # mm/day, station rain gauge (pLIQUIDA)
}


def apply_physical_range_checks(df: pd.DataFrame, ranges: dict = PHYSICAL_RANGES) -> pd.DataFrame:
    """Set out-of-physical-range values to NaN (in place on a copy) for known columns."""
    df = df.copy()
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            n_bad = int(bad.sum())
            if n_bad:
                print(f"[range check] {col}: {n_bad} values outside [{lo}, {hi}] -> NaN")
            df.loc[bad, col] = np.nan
    return df


def chauvenet_outlier_mask(x: pd.Series) -> pd.Series:
    """Chauvenet's criterion: flag x_i as an outlier if N * P(|Z| > |x_i - mean|/std) < 0.5.

    Applied per station (the group's own practice cross-checks flagged points against
    neighboring stations to distinguish sensor error from a real regional event — that
    manual step is not automated here; flagged points should be spot-checked, not dropped
    blindly, when the count is large for one station).
    """
    x = pd.to_numeric(x, errors="coerce")
    n = int(x.notna().sum())
    if n < 3:
        return pd.Series(False, index=x.index)
    mean, std = x.mean(), x.std(ddof=1)
    if not std or np.isnan(std):
        return pd.Series(False, index=x.index)
    d = (x - mean).abs() / std
    tail_prob = special.erfc(d / np.sqrt(2))  # two-sided
    criterion = n * tail_prob
    return (criterion < 0.5).fillna(False)


def flag_chauvenet_outliers(
    df: pd.DataFrame, value_cols: list[str], group_col: str = "station"
) -> pd.DataFrame:
    """Add ``<col>_outlier`` boolean columns and set the outlier's value to NaN in place."""
    df = df.copy()
    for col in value_cols:
        if col not in df.columns:
            continue
        mask = df.groupby(group_col)[col].transform(
            lambda s: chauvenet_outlier_mask(s)
        )
        df[f"{col}_outlier"] = mask
        n_flagged = int(mask.sum())
        if n_flagged:
            print(f"[Chauvenet] {col}: flagged {n_flagged} outliers")
        df.loc[mask, col] = np.nan
    return df


def completeness_by_station_year(
    df: pd.DataFrame, value_col: str, date_col: str = "date", station_col: str = "station"
) -> pd.DataFrame:
    """Fraction of calendar days in each (station, year) with a non-null value_col."""
    d = df[[station_col, date_col, value_col]].copy()
    d["year"] = d[date_col].dt.year
    valid_days = (
        d.dropna(subset=[value_col])
        .groupby([station_col, "year"])[date_col]
        .nunique()
        .rename("n_valid_days")
    )
    days_in_year = d.groupby("year")[date_col].apply(
        lambda s: pd.Timestamp(s.iloc[0].year, 12, 31).dayofyear
    )
    out = valid_days.reset_index()
    out["days_in_year"] = out["year"].map(days_in_year)
    out["completeness"] = out["n_valid_days"] / out["days_in_year"]
    return out


def select_stations_years(
    completeness_df: pd.DataFrame, threshold: float = 0.75
) -> pd.DataFrame:
    """Rows of the completeness table meeting the Casallas-precedent >=75% threshold."""
    return completeness_df[completeness_df["completeness"] >= threshold].copy()


# Aburrá Valley / Medellín-metro station whitelist (factored out of the original
# 00_antioquia_network.ipynb, 2026-09-23, so the new CO/SO2/etc. batch can reuse it
# identically instead of drifting from a second copy). The raw SISAIRE exports cover the
# WHOLE Antioquia department (and, for some of the newer files, even other departments --
# e.g. a Pereira station shows up in the NO export), not just the Aburrá Valley/Medellín
# metro area; see the original notebook markdown for the bug this caused when unfiltered
# (network-mean pressure ~627 hPa vs. ERA5's ~803 hPa).
ABURRA_MUNICIPALITIES = ['medellin', 'itagui', 'envigado', 'sabaneta', 'estrella', 'caldas',
                          'copacabana', 'girardota', 'barbosa', 'bello']
ABURRA_PREFIX_CODES = ['MED-', 'ITA-', 'ENV-', 'GIR-', 'BEL-', 'SAB-', 'CAL-', 'COP-', 'BAR-']
ABURRA_EXPLICIT_EXTRA = ['UNIVERSIDAD NACIONAL - NÚCLEO EL VOLADOR', 'CORPORACIÓN UNIVERSITARIA LASALLISTA']


def _strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')


def is_aburra_station(name: str) -> bool:
    """True if a SISAIRE station name belongs to the Aburrá Valley / Medellín metro area."""
    norm = _strip_accents(name).lower()
    if any(m in norm for m in ABURRA_MUNICIPALITIES):
        return True
    if any(name.upper().startswith(p) for p in ABURRA_PREFIX_CODES):
        return True
    if name.strip().upper() in ABURRA_EXPLICIT_EXTRA:
        return True
    return False
