# Reruns the paired-bootstrap near-metallic internal-vs-external comparison, splitting
# the near-metallic subset by TB-mbj gap metadata into confirmed-metallic vs.
# reclassified-narrow-gap-semiconductor, to test whether the transfer penalty is driven
# only by GGA band-gap misclassification or also present in confirmed non-metals.
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
assert abs(mae_check - ref["mae_log10"]) < 1e-6 and abs(r2_check - ref["r2_log10"]) < 1e-6
print("JARVIS-internal reproduction check: matches exactly.\n")

ext = pd.read_csv(f"{OUT}/predictions_mp_to_jarvis_etotal.csv")
jv_test_jids = jv_t.iloc[idx_te_j]["jid"].to_numpy()
ext_idx = ext.set_index("jid")
pred_external_on_held_out = ext_idx.loc[jv_test_jids, "pred_log10"].to_numpy()
true_external_on_held_out = ext_idx.loc[jv_test_jids, "true_log10"].to_numpy()
bg_held_out = ext_idx.loc[jv_test_jids, "optb88vdw_bandgap"].astype(float).to_numpy()
assert np.allclose(true_external_on_held_out, yte_j, atol=1e-6)

abs_err_internal = np.abs(pred_internal - yte_j)
abs_err_external = np.abs(pred_external_on_held_out - yte_j)
near_metal = bg_held_out < 0.1

meta = pd.read_csv(f"{OUT}/nearmetal_metadata_detail.csv")
meta_nm = meta[meta["set"] == "near_metallic"].set_index("jid")

mbj_gap = np.array([
    meta_nm["mbj_bandgap"].get(jid, np.nan) for jid in jv_test_jids
])
confirmed_metallic = near_metal & (np.isnan(mbj_gap) | (mbj_gap < 0.1))
reclassified_semiconductor = near_metal & (~np.isnan(mbj_gap)) & (mbj_gap >= 0.1)
other = ~near_metal

print(f"Held-out near-metallic (n={int(near_metal.sum())}) splits into:")
print(f"  confirmed metallic under both functionals (incl. TB-mbj unavailable): n={int(confirmed_metallic.sum())}")
print(f"  reclassified narrow-gap semiconductor (TB-mbj >= 0.1 eV): n={int(reclassified_semiconductor.sum())}")
print(f"  other (not near-metallic): n={int(other.sum())}\n")

N_BOOT = 10000

def paired_bootstrap_diff(err_int, err_ext, mask, seed=0, n_boot=N_BOOT):
    idx = np.where(mask)[0]
    rr = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        samp = rr.choice(idx, size=len(idx), replace=True)
        diffs[b] = err_ext[samp].mean() - err_int[samp].mean()
    return diffs

results = {}
for label, mask, seed in [
    ("confirmed_metallic", confirmed_metallic, 100),
    ("reclassified_semiconductor", reclassified_semiconductor, 101),
    ("other", other, 102),
]:
    n = int(mask.sum())
    if n < 5:
        print(f"{label}: n={n}, too small for bootstrap, skipping")
        results[label] = {"n": n}
        continue
    int_mae = float(abs_err_internal[mask].mean())
    ext_mae = float(abs_err_external[mask].mean())
    diffs = paired_bootstrap_diff(abs_err_internal, abs_err_external, mask, seed=seed)
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    print(f"{label}: n={n}, internal MAE={int_mae:.3f}, external MAE={ext_mae:.3f}, "
          f"diff={ext_mae-int_mae:+.3f}, 95% CI={ci}")
    results[label] = {"n": n, "internal_mae": int_mae, "external_mae": ext_mae,
                       "diff": ext_mae - int_mae, "ci95": ci}

with open(f"{OUT}/mbj_excluded_bootstrap.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT}/mbj_excluded_bootstrap.json")
