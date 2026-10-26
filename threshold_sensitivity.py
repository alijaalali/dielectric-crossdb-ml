# Sensitivity of the near-metallic transfer-penalty paired-bootstrap result to the
# choice of band-gap threshold defining "near-metallic" (0.05, 0.1, 0.2, 0.5 eV),
# on the identical held-out JARVIS-DFT test split and external predictions.
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

PROJ = "/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project"
DATA = f"{PROJ}/data"
OUT = f"{PROJ}/models/crossdb"
RNG = 42
THRESHOLDS = [0.05, 0.1, 0.2, 0.5]

jv = pd.read_parquet(f"{DATA}/jarvis_dielectric_features.parquet")
mp = pd.read_parquet(f"{DATA}/mp_dielectric_features_dedup.parquet")
feat_cols = [c for c in mp.columns if c.startswith("MagpieData")]
col_mean = mp[feat_cols].mean(numeric_only=True)

def X(df):
    return df[feat_cols].fillna(col_mean).to_numpy(dtype=float)

jv_t = jv.dropna(subset=["e_total"]).reset_index(drop=True)
Xjv = X(jv_t)
yjv = np.log10(jv_t["e_total"].to_numpy(dtype=float))

Xtr_j, Xte_j, ytr_j, yte_j, idx_tr_j, idx_te_j = train_test_split(
    Xjv, yjv, np.arange(len(yjv)), test_size=0.2, random_state=RNG)

m_jint = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=1.0, n_jobs=8, random_state=RNG)
m_jint.fit(Xtr_j, ytr_j)
pred_internal = m_jint.predict(Xte_j)

ref = json.load(open(f"{OUT}/metrics.json"))["e_total"]["jarvis_internal"]
mae_check = mean_absolute_error(yte_j, pred_internal)
r2_check = r2_score(yte_j, pred_internal)
print(f"JARVIS-internal reproduction check: MAE={mae_check:.6f} (saved {ref['mae_log10']:.6f}), "
      f"R2={r2_check:.6f} (saved {ref['r2_log10']:.6f})")
assert abs(mae_check - ref["mae_log10"]) < 1e-6 and abs(r2_check - ref["r2_log10"]) < 1e-6
print("Matches exactly.\n")

ext = pd.read_csv(f"{OUT}/predictions_mp_to_jarvis_etotal.csv")
jv_test_jids = jv_t.iloc[idx_te_j]["jid"].to_numpy()
ext_idx = ext.set_index("jid")
pred_external_on_held_out = ext_idx.loc[jv_test_jids, "pred_log10"].to_numpy()
true_external_on_held_out = ext_idx.loc[jv_test_jids, "true_log10"].to_numpy()
bg_held_out = ext_idx.loc[jv_test_jids, "optb88vdw_bandgap"].astype(float).to_numpy()
assert np.allclose(true_external_on_held_out, yte_j, atol=1e-6), "target mismatch on held-out set"

abs_err_internal = np.abs(pred_internal - yte_j)
abs_err_external = np.abs(pred_external_on_held_out - yte_j)

N_BOOT = 10000

def paired_bootstrap_diff(err_int, err_ext, mask, n_boot=N_BOOT, seed=0):
    idx = np.where(mask)[0]
    rr = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        samp = rr.choice(idx, size=len(idx), replace=True)
        diffs[b] = err_ext[samp].mean() - err_int[samp].mean()
    return diffs

results = {}
for i, thr in enumerate(THRESHOLDS):
    near_metal = bg_held_out < thr
    n_nm = int(near_metal.sum())
    n_non = int((~near_metal).sum())
    if n_nm < 5:
        print(f"threshold={thr} eV: only {n_nm} near-metallic compounds in held-out set, skipping bootstrap")
        continue
    int_nm = float(abs_err_internal[near_metal].mean())
    ext_nm = float(abs_err_external[near_metal].mean())
    int_non = float(abs_err_internal[~near_metal].mean())
    ext_non = float(abs_err_external[~near_metal].mean())
    diffs_nm = paired_bootstrap_diff(abs_err_internal, abs_err_external, near_metal, seed=10 + i)
    diffs_non = paired_bootstrap_diff(abs_err_internal, abs_err_external, ~near_metal, seed=20 + i)
    ci_nm = (float(np.percentile(diffs_nm, 2.5)), float(np.percentile(diffs_nm, 97.5)))
    ci_non = (float(np.percentile(diffs_non, 2.5)), float(np.percentile(diffs_non, 97.5)))
    print(f"threshold={thr} eV: n_near_metal={n_nm}, n_other={n_non}")
    print(f"  near-metal:  internal MAE={int_nm:.3f}, external MAE={ext_nm:.3f}, "
          f"diff={ext_nm-int_nm:+.3f}, 95% CI={ci_nm}")
    print(f"  other:       internal MAE={int_non:.3f}, external MAE={ext_non:.3f}, "
          f"diff={ext_non-int_non:+.3f}, 95% CI={ci_non}")
    results[str(thr)] = {
        "threshold_eV": thr, "n_near_metal": n_nm, "n_other": n_non,
        "internal_mae_near_metal": int_nm, "external_mae_near_metal": ext_nm,
        "diff_near_metal": ext_nm - int_nm, "ci95_near_metal": ci_nm,
        "internal_mae_other": int_non, "external_mae_other": ext_non,
        "diff_other": ext_non - int_non, "ci95_other": ci_non,
    }

with open(f"{OUT}/threshold_sensitivity.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}/threshold_sensitivity.json")
