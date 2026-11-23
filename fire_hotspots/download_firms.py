"""NASA FIRMS active-fire detections (VIIRS 375 m, S-NPP) for ALL of South America, 2024 -> today, gridded to the SAME daily 0.75-degree grid as the existing
Hotspots_count_2013-2023_South_America_combined.nc (lat -56..14.5, lon -85..-32.5, 95 x 71 cells, variable 'Hotspots' = detections per cell per day).

Key: a free FIRMS MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/map_key/ -> enter e-mail -> key arrives by e-mail) stored in Eq_discovery_countries/.firms_key (chmod 600).
Usage:
  python download_firms.py validate      # downloads 5 days of 2023 and compares with the existing file for several detection filters (which one reproduces the old product?)
  python download_firms.py download      # raw CSV per 5-day window -> Data/external/firms/raw/  (resumable; SP = standard processing for dates > 100 days old, else NRT)
  python download_firms.py grid [FILTER] # builds Data/external/firms/Hotspots_count_2024-<end>_South_America.nc and the merged 2013-<end> file
API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/<KEY>/<SOURCE>/<west,south,east,north>/<days 1-5>/<YYYY-MM-DD>
"""
import glob, io, os, sys, time, datetime as dt, urllib.request
import numpy as np, pandas as pd, xarray as xr

ROOT = "/path/to/project/"
OLD = "/path/to/fire_data/Hotspots_count_2013-2023_South_America_combined.nc"
RAW = ROOT + "Data/external/firms/raw/"; os.makedirs(RAW, exist_ok=True)
AREA = "-86,-57,-32,15"                   # a little larger than the grid, edge cells are cut at gridding
KEY = open(ROOT + ".firms_key").read().strip() if os.path.exists(ROOT + ".firms_key") else os.environ.get("FIRMS_KEY", "")
LAT0, LON0, STEP, NLAT, NLON = -56.0, -85.0, 0.75, 95, 71

def fetch(start, days, source):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{source}/{AREA}/{days}/{start:%Y-%m-%d}"
    for a in range(6):
        try:
            txt = urllib.request.urlopen(url, timeout=180).read().decode()
            if txt.startswith("latitude"): return txt
            if "Invalid" in txt or "exceed" in txt.lower(): print("API said:", txt[:150]); time.sleep(60 * (a + 1)); continue
            return txt
        except Exception as e: print("retry", start, repr(e)[:80]); time.sleep(20 * (a + 1))
    raise RuntimeError(f"failed {start}")

def to_cells(df):
    iy = np.floor((df.latitude.values - LAT0) / STEP + 0.5).astype(int); ix = np.floor((df.longitude.values - LON0) / STEP + 0.5).astype(int)
    ok = (iy >= 0) & (iy < NLAT) & (ix >= 0) & (ix < NLON); return iy[ok], ix[ok], df.acq_date.values[ok]

def grid(df, filt):
    d = df
    if filt == "no_low": d = d[d.confidence.astype(str).str.lower() != "l"]
    elif filt == "nominal_high": d = d[d.confidence.astype(str).str.lower().isin(["n", "h"])]
    elif filt == "high": d = d[d.confidence.astype(str).str.lower() == "h"]
    iy, ix, day = to_cells(d); days = np.array(sorted(set(day))); out = np.zeros((len(days), NLAT, NLON), np.float32)
    di = {x: i for i, x in enumerate(days)}
    for a, b, c in zip(iy, ix, day): out[di[c], a, b] += 1
    return xr.DataArray(out, dims=("time", "latitude", "longitude"), coords={"time": pd.to_datetime(days), "latitude": LAT0 + STEP * np.arange(NLAT), "longitude": LON0 + STEP * np.arange(NLON)}, name="Hotspots")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if not KEY: sys.exit("No FIRMS key: put it in Eq_discovery_countries/.firms_key")
    if mode == "validate":
        start = dt.date(2023, 8, 10); df = pd.read_csv(io.StringIO(fetch(start, 5, "VIIRS_SNPP_SP"))); print(len(df), "detections downloaded (5 days from 2023-08-10); columns:", list(df.columns))
        old = xr.open_dataset(OLD)["Hotspots"].sel(time=slice("2023-08-10", "2023-08-14"))
        print("old file, same 5 days, total per day:", [int(x) for x in old.sum(dim=["latitude", "longitude"]).values])
        for filt in ("all", "no_low", "nominal_high", "high"):
            g = grid(df, filt).reindex(time=old.time, fill_value=0); tot = [int(x) for x in g.sum(dim=["latitude", "longitude"]).values]
            r = np.corrcoef(g.values.ravel(), old.values.ravel())[0, 1]; print(f"  filter {filt:13s}: totals/day {tot}  cell-level correlation with the old file {r:.4f}  max abs diff {np.abs(g.values - old.values).max():.0f}")
    elif mode == "download":
        # Whole record 2013-01-01 -> yesterday, 5-day windows. Each window is aggregated immediately to the 0.75-degree daily grid (sparse day/iy/ix/count in a small .npz);
        # raw detections are not kept (a year is ~2.5 M rows). Resumable. All detections are counted (the daily totals then match the old product to 0.1 %).
        os.makedirs(RAW, exist_ok=True); end = dt.date.today() - dt.timedelta(days=1); base = dt.date(2013, 1, 1); d = base; n = 0
        if len(sys.argv) > 2:                                  # optional start / stop date: windows stay aligned to the 5-day grid anchored at 2013-01-01 (no overlaps between workers)
            k = -(-(dt.date.fromisoformat(sys.argv[2]) - base).days // 5); d = base + dt.timedelta(days=5 * k)
        if len(sys.argv) > 3: end = min(end, dt.date.fromisoformat(sys.argv[3]))
        while d <= end:
            days = min(5, (end - d).days + 1); src = "VIIRS_SNPP_SP" if (dt.date.today() - (d + dt.timedelta(days=days))).days > 100 else "VIIRS_SNPP_NRT"
            fn = f"{RAW}{d:%Y%m%d}_{days}d_{src}.npz"
            if not os.path.exists(fn):
                txt = fetch(d, days, src)
                if txt.startswith("latitude"):
                    df = pd.read_csv(io.StringIO(txt), usecols=["latitude", "longitude", "acq_date"]); iy, ix, day = to_cells(df)
                    u, c = np.unique(np.stack([pd.factorize(day)[0], iy, ix], 1), axis=0, return_counts=True); lab = pd.factorize(day)[1]
                    np.savez_compressed(fn, day=np.array([lab[i] for i in u[:, 0]]).astype(str), iy=u[:, 1], ix=u[:, 2], n=c)
                else: np.savez_compressed(fn, day=np.array([], str), iy=np.array([], int), ix=np.array([], int), n=np.array([], int))
                n += 1; time.sleep(1.5)
                if n % 25 == 0: print("downloaded up to", d, f"({n} windows this run)", flush=True)
            d += dt.timedelta(days=days)
        print("done; windows downloaded this run:", n)
    elif mode == "grid":
        fs = sorted(glob.glob(RAW + "*.npz")); rows = [np.load(f) for f in fs]
        day = np.concatenate([r["day"] for r in rows]); iy = np.concatenate([r["iy"] for r in rows]); ix = np.concatenate([r["ix"] for r in rows]); n = np.concatenate([r["n"] for r in rows])
        t0, t1 = pd.Timestamp("2013-01-01"), pd.Timestamp(str(max(day))); times = pd.date_range(t0, t1); out = np.zeros((len(times), NLAT, NLON), np.float32)
        ti = (pd.to_datetime(day) - t0).days.values; np.add.at(out, (ti, iy, ix), n)
        da = xr.DataArray(out, dims=("time", "latitude", "longitude"), coords={"time": times, "latitude": (LAT0 + STEP * np.arange(NLAT)).astype("float32"), "longitude": (LON0 + STEP * np.arange(NLON)).astype("float32")}, name="Hotspots")
        end = str(t1)[:10]; da.to_dataset().to_netcdf(ROOT + f"Data/external/firms/Hotspots_count_2013-{end}_South_America_FIRMS.nc")
        t = da.sum(dim=["latitude", "longitude"]).to_series(); print("annual totals (new, consistent gridding):", t.groupby(t.index.year).sum().astype(int).to_dict())
