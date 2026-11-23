# A tale of two pollutants: code

Code for the paper *A tale of two pollutants: equation discovery untangles ozone titration, particulate transport, and their coupling* (Casallas, González, Mogollón-Sotelo, Lopez-Barrera, Rojas and Pozzer).

The code downloads the data, builds daily datasets for Bogotá and Medellín, selects predictors, discovers equations for PM2.5 and ozone with PySR, and runs every analysis in the paper. The notebooks are saved with their outputs, so you can read the results without running anything.

## Setup

1. Make a project folder and put this repository inside it as `Scripts/`:
   ```
   project/
   ├── Scripts/   <- this repository
   └── Data/      <- created by the download steps
   ```
2. In every file, replace `/path/to/project` with the path of your project folder (one find-and-replace).
3. Install Python 3.11 and the packages: `pip install -r requirements.txt`. PySR installs Julia the first time it runs.
4. Get the free accounts the downloads need:
   - **ERA5**: a Copernicus Climate Data Store account and a `~/.cdsapirc` file.
   - **Fire hotspots**: a NASA FIRMS map key, saved in `.firms_key` in the project folder (or in the `FIRMS_KEY` environment variable).
   - **TROPOMI**: a Google Earth Engine account (only for the satellite checks).
   - **Air quality**: the station data of both cities come from SISAIRE (IDEAM), downloaded from its website into `Data/COLOMBIA/`.

All the data are public; the paper's reference list gives the source of each.

## What each folder does

Run the folders in this order. Inside each folder, run the files by their number.

### `common/`: shared functions and downloads
| File | What it does |
|---|---|
| `sisaire_io.py` | Reads the SISAIRE station files. |
| `qc.py` | Quality control of the station series. |
| `external_data.py` | Downloads ERA5 and reads ERA5 and the fire-hotspot grid. |
| `download_surface_relative_w.py`, `build_w_surface_relative.py` | Download ERA5 vertical velocity and build the surface-relative version. |
| `fetch_tropomi_hcho.py` | Downloads TROPOMI formaldehyde and NO2 columns over each city. |
| `feature_selection.py` | Sequential forward feature selection with a neural network and leave-one-year-out cross-validation. |
| `metrics.py`, `diagnostics.py` | Skill scores and diagnostic plots. |
| `loyo_cv_lead_equations.py` | Leave-one-year-out evaluation of the lead equations, with their constants refitted. |
| `build_forecast_pool.py`, `run_forecast_sfs.py`, `run_forecast_sfs_no_selflag.py` | Build the lagged-only (previous-day) candidate pool and run feature selection on it. |

### `fire_hotspots/`: fire data
| File | What it does |
|---|---|
| `download_firms.py` | Downloads VIIRS active fires from NASA FIRMS and grids them to daily counts on a 0.75° grid over South America. |
| `rebuild_fire_columns.py` | Rebuilds the fire columns (counts by compass sector and source region, and their lags) of the city datasets from that grid. |

### `colombia/bogota/`: Bogotá
| Files | What they do |
|---|---|
| `01`–`02` | Load the SISAIRE station data and quality-control it. |
| `03` | ERA5 variables. |
| `04`, `04b`–`04e` | Fire hotspots, and the network series of NO, ozone (MDA8), CO and SO2. |
| `05`, `05b` | Build the daily dataset and the candidate pool; quick feature screen. |
| `06` | Feature selection for PM2.5, ozone, and PM2.5 with ozone offered. |
| `07` | Equation discovery with PySR: free-form and dimensionally constrained searches, several seeds, with and without temperature, PM2.5 or ozone offered. |
| `08`, `09`, `11`, `11b`, `10_...py` | Mechanism tests: radiative shading, heterogeneous chemistry, temperature, daily averaging. |
| `run_forecast_equation_discovery.py` | Equation discovery on the lagged-only pool. |
| `check_hourly_climatology_baseline.py` | How much of an hourly R2 the diurnal cycle alone gives (for the comparison with published models). |
| `literature_comparison_table.csv` | Published scores used in the comparison figure. |

### `colombia/medellin/`: Medellín
The same steps for Medellín: network data (`00`–`01b`), fires (`02`), ERA5 (`03`), datasets (`04`), feature selection (`05`), equation discovery (`06`–`07`), and the radiative-shading test. `ingest_new_antioquia_vars.py` adds the other Antioquia network variables.

### `paper/`: analyses, tables and figures of the paper
`notebooks/` has one notebook per analysis or figure. Run them in this order:

| Notebook | What it does |
|---|---|
| `A0` | Bogotá station locations (map in Fig. 1). |
| `A1` | Data inventory (Table S1, Fig. 1c). |
| `A2` | Leave-one-year-out skill, persistence baseline and metrics (Tables 1, S4; Fig. S2). |
| `B3` | In-sample R2 of every equation on every Pareto front. |
| `T1_T2` | Partial correlations between PM2.5 and ozone (Fig. 6, Table S5). |
| `T3` | Choosing the Medellín equations from their Pareto fronts (Fig. 7). |
| `A3` | Numbers of the main tables. |
| `A4`–`A10` | Units check, constrained vs free-form equations, PM2.5 equation details, temperature and rain checks, shared driver vs co-production tests, previous-day equations, Medellín leakage check (Tables S2, S6; Texts S1–S6). |
| `fig01`–`fig08`, `S_supplementary_figures_1`–`3` | Main and supplementary figures. |

`style.py`, `fig_common.py` and `termmap_utils.py` hold the plotting settings and helpers. `run_lagged_lead_pysr.py`, `run_medellin_pysr_refresh.py` and `run_nested_pysr_check.py` run the PySR searches that the notebooks `A9`, `T3` and the nested-validation check read.

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the paper.
