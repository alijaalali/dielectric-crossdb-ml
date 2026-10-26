# Splits the near-metallic (band gap < 0.1 eV) excess-transfer-error finding into
# electronic vs. ionic dielectric components. Reproduces train_crossdb.py's protocol
# exactly and checks the result against metrics.json before using it for anything new.
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

PROJ = "/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project"
DATA = f"{PROJ}/data"
OUT = f"{PROJ}/models/crossdb"

mp = pd.read_parquet(f"{DATA}/mp_dielectric_features_dedup.parquet")
jv = pd.read_parquet(f"{DATA}/jarvis_dielectric_features.parquet")
feat_cols = [c for c in mp.columns if c.startswith("MagpieData")]
col_mean = mp[feat_cols].mean(numeric_only=True)

def X(df):
    return df[feat_cols].fillna(col_mean).to_numpy(dtype=float)

ref = json.load(open(f"{OUT}/metrics.json"))

results = {}
for target, eps in [("e_electronic", 0.0), ("e_ionic", 0.01)]:
    mp_t = mp.dropna(subset=[target]).reset_index(drop=True)
    jv_t = jv.dropna(subset=[target]).reset_index(drop=True)
    Xmp = X(mp_t)
    ymp = np.log10(mp_t[target].to_numpy(dtype=float) + eps)
    Xjv = X(jv_t)
    yjv = np.log10(jv_t[target].to_numpy(dtype=float) + eps)

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          n_jobs=8, random_state=42)
    model.fit(Xmp, ymp)
    pred = model.predict(Xjv)

    r2 = r2_score(yjv, pred)
    mae = mean_absolute_error(yjv, pred)
    ref_r2 = ref[target]["mp_to_jarvis"]["r2_log10"]
    ref_mae = ref[target]["mp_to_jarvis"]["mae_log10"]
    print(f"{target}: recomputed R2={r2:.6f} MAE={mae:.6f} | "
          f"saved R2={ref_r2:.6f} MAE={ref_mae:.6f} | "
          f"match={abs(r2-ref_r2)<1e-6 and abs(mae-ref_mae)<1e-6}")
    assert abs(r2 - ref_r2) < 1e-6 and abs(mae - ref_mae) < 1e-6, \
        f"MISMATCH for {target}"

    abs_err = np.abs(pred - yjv)
    bg = jv_t["optb88vdw_bandgap"].astype(float).to_numpy()
    near_metal = bg < 0.1

    n_nm = int(near_metal.sum())
    n_non = int((~near_metal).sum())
    mae_nm = float(abs_err[near_metal].mean())
    mae_non = float(abs_err[~near_metal].mean())
    frac_rows_nm = float(near_metal.mean() * 100)
    frac_err_nm = float(abs_err[near_metal].sum() / abs_err.sum() * 100)
    r2_nm = float(r2_score(yjv[near_metal], pred[near_metal])) if n_nm > 1 else None
    r2_non = float(r2_score(yjv[~near_metal], pred[~near_metal]))

    results[target] = {
        "overall_r2": float(r2), "overall_mae": float(mae),
        "n_near_metal": n_nm, "n_non_metal": n_non,
        "mae_near_metal": mae_nm, "mae_non_metal": mae_non,
        "r2_near_metal": r2_nm, "r2_non_metal": r2_non,
        "frac_rows_near_metal_pct": frac_rows_nm,
        "frac_abs_error_near_metal_pct": frac_err_nm,
    }
    print(f"  near-metal (n={n_nm}, {frac_rows_nm:.1f}% of rows): "
          f"MAE={mae_nm:.3f} R2={r2_nm}")
    print(f"  non-metal  (n={n_non}): MAE={mae_non:.3f} R2={r2_non:.3f}")
    print(f"  near-metal share of total abs error: {frac_err_nm:.1f}%")

ref_results = json.load(open(f"{OUT}/revision_analysis_results.json"))
results["e_total_reference"] = ref_results.get("near_metallic_split", ref_results.get("nearmetal", {}))

with open(f"{OUT}/nearmetal_components.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved models/crossdb/nearmetal_components.json")
