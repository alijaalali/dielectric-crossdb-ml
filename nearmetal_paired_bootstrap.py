# Paired bootstrap comparison of the JARVIS-internal model and the MP-trained external
# model on the exact same held-out JARVIS test compounds, split by near-metallic status.
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

PROJ = "/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project"
DATA = f"{PROJ}/data"
OUT = f"{PROJ}/models/crossdb"
RNG = 42

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
from sklearn.metrics import r2_score
mae_check = mean_absolute_error(yte_j, pred_internal)
r2_check = r2_score(yte_j, pred_internal)
print(f"JARVIS-internal reproduction check: MAE={mae_check:.6f} (saved {ref['mae_log10']:.6f}), "
      f"R2={r2_check:.6f} (saved {ref['r2_log10']:.6f})")
assert abs(mae_check - ref["mae_log10"]) < 1e-6 and abs(r2_check - ref["r2_log10"]) < 1e-6
print("Matches exactly.")

ext = pd.read_csv(f"{OUT}/predictions_mp_to_jarvis_etotal.csv")
jv_test_jids = jv_t.iloc[idx_te_j]["jid"].to_numpy()
ext_idx = ext.set_index("jid")
pred_external_on_held_out = ext_idx.loc[jv_test_jids, "pred_log10"].to_numpy()
true_external_on_held_out = ext_idx.loc[jv_test_jids, "true_log10"].to_numpy()
bg_held_out = ext_idx.loc[jv_test_jids, "optb88vdw_bandgap"].astype(float).to_numpy()

assert np.allclose(true_external_on_held_out, yte_j, atol=1e-6), "target mismatch on held-out set"

abs_err_internal = np.abs(pred_internal - yte_j)
abs_err_external = np.abs(pred_external_on_held_out - yte_j)
near_metal = bg_held_out < 0.1
n_nm = int(near_metal.sum())
n_non = int((~near_metal).sum())
print(f"\nHeld-out JARVIS test set: n={len(yte_j)}, near-metal n={n_nm}, non-metal n={n_non}")
print(f"Internal MAE: near-metal={abs_err_internal[near_metal].mean():.4f}, "
      f"non-metal={abs_err_internal[~near_metal].mean():.4f}")
print(f"External MAE: near-metal={abs_err_external[near_metal].mean():.4f}, "
      f"non-metal={abs_err_external[~near_metal].mean():.4f}")

rng = np.random.RandomState(0)
N_BOOT = 10000

def paired_bootstrap_diff(err_int, err_ext, mask, n_boot=N_BOOT, seed=0):
    idx = np.where(mask)[0]
    rr = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        samp = rr.choice(idx, size=len(idx), replace=True)
        diffs[b] = err_ext[samp].mean() - err_int[samp].mean()
    return diffs

diffs_nm = paired_bootstrap_diff(abs_err_internal, abs_err_external, near_metal, seed=1)
diffs_non = paired_bootstrap_diff(abs_err_internal, abs_err_external, ~near_metal, seed=2)

ci_nm = (float(np.percentile(diffs_nm, 2.5)), float(np.percentile(diffs_nm, 97.5)))
ci_non = (float(np.percentile(diffs_non, 2.5)), float(np.percentile(diffs_non, 97.5)))
mean_diff_nm = float(diffs_nm.mean())
mean_diff_non = float(diffs_non.mean())

print(f"\nPaired bootstrap (external MAE - internal MAE), {N_BOOT} resamples, held-out set only:")
print(f"  Near-metal:  mean diff = {mean_diff_nm:.4f}, 95% CI = {ci_nm}")
print(f"  Non-metal:   mean diff = {mean_diff_non:.4f}, 95% CI = {ci_non}")

results = {
    "n_held_out": int(len(yte_j)), "n_near_metal": n_nm, "n_non_metal": n_non,
    "internal_mae_near_metal": float(abs_err_internal[near_metal].mean()),
    "internal_mae_non_metal": float(abs_err_internal[~near_metal].mean()),
    "external_mae_near_metal": float(abs_err_external[near_metal].mean()),
    "external_mae_non_metal": float(abs_err_external[~near_metal].mean()),
    "paired_bootstrap_diff_near_metal": {"mean": mean_diff_nm, "ci95": ci_nm},
    "paired_bootstrap_diff_non_metal": {"mean": mean_diff_non, "ci95": ci_non},
    "n_boot": N_BOOT,
}
with open(f"{OUT}/nearmetal_paired_bootstrap.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved models/crossdb/nearmetal_paired_bootstrap.json")
