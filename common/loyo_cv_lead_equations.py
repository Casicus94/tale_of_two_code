"""B2 / C1: proper LOYO-CV R2 for the two chosen lead equations (refit numeric
constants on n-1 years via nonlinear least squares, evaluate on the held-out year,
average across years) -- distinct from the PySR front's in-sample R2 on all the data
at once, which is what was quoted so far."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

PROJECT_ROOT = "/path/to/project"


def loyo_cv_r2(df, year_col, feature_cols, target_col, model_fn, p0):
    years = sorted(df[year_col].unique())
    fold_r2 = {}
    fold_n = {}
    for held_out in years:
        train = df[df[year_col] != held_out]
        test = df[df[year_col] == held_out]
        X_train = tuple(train[c].to_numpy() for c in feature_cols)
        y_train = train[target_col].to_numpy()
        X_test = tuple(test[c].to_numpy() for c in feature_cols)
        y_test = test[target_col].to_numpy()
        try:
            popt, _ = curve_fit(model_fn, X_train, y_train, p0=p0, maxfev=20000)
        except RuntimeError as e:
            print(f"  year {held_out}: curve_fit failed ({e}), using p0 as-is")
            popt = p0
        y_pred = model_fn(X_test, *popt)
        fold_r2[held_out] = r2_score(y_test, y_pred)
        fold_n[held_out] = len(y_test)
    return fold_r2, fold_n


# ---------------------------------------------------------------------------
# PM2.5 lead equation (complexity 18, dimensionally-constrained front):
# pm25 = [co + (a - w700)*b/(wind_speed_era5*tcwv_lag1)] * [c*sqrt(hotspots_sector_E_lag3) + d]
# fitted constants (full-data PySR fit): a=0.297, b=16.20, c=0.0617, d=15.79
# ---------------------------------------------------------------------------
def pm25_model(X, a, b, c, d):
    co, w700, wind_speed_era5, tcwv_lag1, hotspots_sector_E_lag3 = X
    return (co + (a - w700) * b / (wind_speed_era5 * tcwv_lag1)) * (
        c * np.sqrt(np.clip(hotspots_sector_E_lag3, 0, None)) + d
    )


pm25_features = ["co", "w700", "wind_speed_era5", "tcwv_lag1", "hotspots_sector_E_lag3"]
pm25_p0 = [0.297, 16.20, 0.0617, 15.79]

pm25_df = pd.read_parquet(f"{PROJECT_ROOT}/Data/processed/colombia/bogota/bogota_master_dataset.parquet")
pm25_df["date"] = pd.to_datetime(pm25_df["date"])
pm25_df["year"] = pm25_df["date"].dt.year
pm25_d = pm25_df.dropna(subset=pm25_features + ["pm25", "year"]).reset_index(drop=True)

print("=== PM2.5 lead equation (Bogota, complexity 18) ===")
print(f"n = {len(pm25_d)}, years = {sorted(pm25_d['year'].unique())}")

full_pred = pm25_model(tuple(pm25_d[c].to_numpy() for c in pm25_features), *pm25_p0)
print(f"In-sample R2 with the quoted PySR constants (sanity check, should be near the "
      f"front's reported R2 for this row): {r2_score(pm25_d['pm25'], full_pred):.4f}")

pm25_fold_r2, pm25_fold_n = loyo_cv_r2(pm25_d, "year", pm25_features, "pm25", pm25_model, pm25_p0)
for y in sorted(pm25_fold_r2):
    print(f"  held-out {y}: n={pm25_fold_n[y]:4d}  R2={pm25_fold_r2[y]:.4f}")
pm25_loyo_r2 = np.mean(list(pm25_fold_r2.values()))
pm25_loyo_r2_weighted = sum(pm25_fold_r2[y] * pm25_fold_n[y] for y in pm25_fold_r2) / sum(pm25_fold_n.values())
print(f"PM2.5 LOYO-CV R2 (unweighted mean across years) = {pm25_loyo_r2:.4f}")
print(f"PM2.5 LOYO-CV R2 (n-weighted)                    = {pm25_loyo_r2_weighted:.4f}")
print()

# ---------------------------------------------------------------------------
# O3 lead equation (complexity 16, falsification-test front):
# o3 = temp_2m/((no/pm25 + e) + w700) + sqrt((photochemical_potential/f)/rh_2m)
# fitted constants (full-data PySR fit): e=0.6341454, f=0.48264104
# ---------------------------------------------------------------------------
def o3_model(X, e, f):
    temp_2m, no, pm25, w700, photochemical_potential, rh_2m = X
    term1 = temp_2m / ((no / pm25 + e) + w700)
    term2 = np.sqrt(np.clip((photochemical_potential / f) / rh_2m, 0, None))
    return term1 + term2


o3_features = ["temp_2m", "no", "pm25", "w700", "photochemical_potential", "rh_2m"]
o3_p0 = [0.6341454, 0.48264104]

o3_df = pd.read_parquet(f"{PROJECT_ROOT}/Data/processed/colombia/bogota/bogota_o3_master_dataset.parquet")
o3_df["date"] = pd.to_datetime(o3_df["date"])
o3_df["year"] = o3_df["date"].dt.year
o3_d = o3_df.dropna(subset=o3_features + ["o3", "year"]).reset_index(drop=True)

print("=== O3 lead equation (Bogota, complexity 16) ===")
print(f"n = {len(o3_d)}, years = {sorted(o3_d['year'].unique())}")

full_pred = o3_model(tuple(o3_d[c].to_numpy() for c in o3_features), *o3_p0)
print(f"In-sample R2 with the quoted PySR constants (sanity check): "
      f"{r2_score(o3_d['o3'], full_pred):.4f}")

o3_fold_r2, o3_fold_n = loyo_cv_r2(o3_d, "year", o3_features, "o3", o3_model, o3_p0)
for y in sorted(o3_fold_r2):
    print(f"  held-out {y}: n={o3_fold_n[y]:4d}  R2={o3_fold_r2[y]:.4f}")
o3_loyo_r2 = np.mean(list(o3_fold_r2.values()))
o3_loyo_r2_weighted = sum(o3_fold_r2[y] * o3_fold_n[y] for y in o3_fold_r2) / sum(o3_fold_n.values())
print(f"O3 LOYO-CV R2 (unweighted mean across years) = {o3_loyo_r2:.4f}")
print(f"O3 LOYO-CV R2 (n-weighted)                    = {o3_loyo_r2_weighted:.4f}")
