"""Helpers shared by the paper figure scripts."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from style import DATA_OUT, GREY, LIGHT, NAVY, O3, PM25, RED  # noqa: E402
from termmap_utils import term_presence  # noqa: E402

# Readable labels for term-presence map rows (review item R6). Anything not listed falls back to its
# raw variable name, so new variables never break a build -- add them here as they come up.
VAR_LABELS = {
    "co": "CO", "co_lag1": "CO (lag 1)", "no": "NO", "no2": "NO$_2$", "nox": "NO$_x$", "no_ratio": "NO ratio", "no_lag1": "NO (lag 1)", "no_lag2": "NO (lag 2)", "no_lag3": "NO (lag 3)",
    "w700_lag1": "w$_{700}$ (lag 1)", "hotspots_sector_E_lag2": "fire E (lag 2)",
    "pm25": "PM$_{2.5}$", "pm25_lag1": "PM$_{2.5}$ (lag 1)", "pm25_lag2": "PM$_{2.5}$ (lag 2)", "o3": "O$_3$",
    "w700": "w$_{700}$", "wind_speed_era5": "wind speed (ERA5)", "wind_speed": "wind speed (station)",
    "tcwv": "TCWV", "tcwv_lag1": "TCWV (lag 1)", "rh_2m": "RH", "rh_10m": "RH (10 m)",
    "temp_2m": "T$_{2m}$", "temp_2m_lag1": "T$_{2m}$ (lag 1)", "temp_2m_lag2": "T$_{2m}$ (lag 2)",
    "blh_mean": "BLH", "blh_max": "BLH (max)", "radiation_mean": "SW$\\downarrow$",
    "photochemical_potential": "NO$_2$ $\\times$ SW$\\downarrow$", "photochemical_potential_lag1": "NO$_2 \\times$ SW$\\downarrow$ (lag 1)",
    "precip_total_mm": "precipitation", "precip_total_mm_lag1": "precipitation (lag 1)",
    "hotspots_sector_E_lag3": "fire E (lag 3)", "hotspots_sector_N_lag2": "fire N (lag 2)", "hotspots_sector_SW_lag2": "fire SW (lag 2)",
    "tke_proxy": "TKE proxy", "ventilation_station": "ventilation (station)", "dry_stagnation_lag1": "dry stagnation (lag 1)",
    "fog_stagnation": "fog stagnation", "fog_stagnation_lag1": "fog stagnation (lag 1)", "day_of_week": "day of week",
}


# 2026-10-20: after the fire-file correction the selected fire variables changed; label every sector/lag combination
for _sec in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
    VAR_LABELS.setdefault(f"hotspots_sector_{_sec}", f"fire {_sec}")
    for _k in (1, 2, 3): VAR_LABELS.setdefault(f"hotspots_sector_{_sec}_lag{_k}", f"fire {_sec} (lag {_k})")
for _b, _l in (("wind_dir", "wind direction"), ("wind_speed_era5", "wind speed"), ("fog_stagnation", "fog stagnation"), ("w700", "w$_{700}$"), ("hotspots_magdalena_valley", "fire Magdalena"),
               ("photochemical_potential", "NO$_2 \\times$ SW$\\downarrow$"), ("blh_min", "BLH (min)"), ("tke_proxy", "TKE proxy"), ("no2_stagnation", "NO$_2$ stagnation"), ("ventilation_min", "ventilation (min)")):
    for _k in (1, 2, 3): VAR_LABELS.setdefault(f"{_b}_lag{_k}", f"{_l} (lag {_k})")
VAR_LABELS.update({"sensible_heat_flux": "sensible heat flux", "sensible_heat_flux_lag1": "sensible heat flux (lag 1)",
                   "fire_eq1": "fire (Eq. 1 fire term)"})


def draw_termmap(ax, front, color, highlight=None, rows=None, min_rows=None, label_size=6.5,
                 max_c=None, mark_rows=None, readable_labels=True, tick_size=7):
    """Term-presence map: rows = variables (ordered by first appearance), x = complexity.
    Filled cell = variable appears in the equation at that complexity. highlight =
    complexity to outline. mark_rows = {variable: color} to emphasise specific rows.
    A hatched grey column marks a complexity for which the front has NO equation at all
    (not "no term present"), so it can't be misread as every variable being absent (R6)."""
    mat = term_presence(front, rows)
    cols = mat.columns.to_numpy()
    xmax = max_c or cols.max()
    missing = sorted(set(range(1, int(xmax) + 1)) - set(cols.tolist()))
    for c in missing:
        ax.add_patch(Rectangle((c - 0.5, -0.5), 1, len(mat), facecolor=LIGHT, edgecolor="white", lw=0.4, hatch="////", zorder=1))
    for i, (v, r) in enumerate(mat.iterrows()):
        c0 = (mark_rows or {}).get(v, color)
        for c, val in zip(cols, r.to_numpy()):
            if val:
                ax.add_patch(Rectangle((c - 0.5, i - 0.5), 1, 1, facecolor=c0, edgecolor="white", lw=0.4, zorder=2))
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_ylim(len(mat) - 0.5, -0.5)
    ax.set_yticks(range(len(mat)))
    names = [VAR_LABELS.get(v, v) if readable_labels else v for v in mat.index]
    lab = ax.set_yticklabels(names, fontsize=label_size)
    for t, v in zip(lab, mat.index):
        if mark_rows and v in mark_rows:
            t.set_fontweight("bold")
            t.set_color(mark_rows[v])
    ax.set_xticks(list(range(1, int(xmax) + 1, 2)))
    ax.tick_params(axis="x", labelsize=tick_size)
    ax.set_xlabel("equation complexity", fontsize=label_size)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if highlight is not None:
        ax.add_patch(Rectangle((highlight - 0.5, -0.5), 1, len(mat), fill=False, edgecolor=NAVY, lw=1.6, zorder=5))
    return mat


def loyo_preds(pol):
    d = pd.read_csv(DATA_OUT / f"loyo_predictions_{pol}.csv", parse_dates=["date"])
    return d


def r2_from_csv():
    return pd.read_csv(DATA_OUT / "front_r2_recomputed.csv")
