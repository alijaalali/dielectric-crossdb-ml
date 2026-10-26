# Trains one model on an 80% MP split and evaluates that same model on the held-out
# 20% MP test set and on the full JARVIS-DFT set, for a training-size-matched
# internal/external comparison, stratified by band gap and dielectric-constant range.
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJ = "/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project"
DATA = f"{PROJ}/data"
OUT = f"{PROJ}/models/crossdb"
RNG = 42

mp = pd.read_parquet(f"{DATA}/mp_dielectric_features_dedup.parquet").dropna(subset=["e_total"]).reset_index(drop=True)
jv = pd.read_parquet(f"{DATA}/jarvis_dielectric_features.parquet").dropna(subset=["e_total"]).reset_index(drop=True)
feat_cols = [c for c in mp.columns if c.startswith("MagpieData")]
col_mean = mp[feat_cols].mean(numeric_only=True)

def X(df):
    return df[feat_cols].fillna(col_mean).to_numpy(dtype=float)

Xmp, ymp = X(mp), np.log10(mp["e_total"].to_numpy(dtype=float))
Xjv, yjv = X(jv), np.log10(jv["e_total"].to_numpy(dtype=float))
bg_jv = jv["optb88vdw_bandgap"].astype(float).to_numpy()

Xtr, Xte, ytr, yte = train_test_split(Xmp, ymp, test_size=0.2, random_state=RNG)

model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=8, random_state=RNG)
model.fit(Xtr, ytr)

pred_mp_held = model.predict(Xte)
pred_jv = model.predict(Xjv)

r2_mp_held = r2_score(yte, pred_mp_held)
mae_mp_held = mean_absolute_error(yte, pred_mp_held)
r2_jv = r2_score(yjv, pred_jv)
mae_jv = mean_absolute_error(yjv, pred_jv)

ref = json.load(open(f"{OUT}/metrics.json"))["e_total"]["mp_internal"]
print(f"Held-out MP reproduction check: R2={r2_mp_held:.6f} (saved {ref['r2_log10']:.6f}), "
      f"MAE={mae_mp_held:.6f} (saved {ref['mae_log10']:.6f})")
assert abs(r2_mp_held - ref["r2_log10"]) < 1e-6 and abs(mae_mp_held - ref["mae_log10"]) < 1e-6
print("Matches exactly -- this IS the same model used for the MP-internal baseline.")

print(f"\nSame model (trained on 80% MP, n={len(ytr)}), two held-out test sets:")
print(f"  Held-out MP (n={len(yte)}):      R2={r2_mp_held:.3f} MAE={mae_mp_held:.3f}")
print(f"  Full JARVIS-DFT (n={len(yjv)}, never seen in training): R2={r2_jv:.3f} MAE={mae_jv:.3f}")

def stratify_by_bandgap(bg, pred, true):
    bins = [(-0.01, 0.1, "near-metallic (<0.1 eV)"),
            (0.1, 1.0, "narrow-gap (0.1-1 eV)"),
            (1.0, 3.0, "moderate-gap (1-3 eV)"),
            (3.0, 100.0, "wide-gap (>3 eV)")]
    out = []
    for lo, hi, label in bins:
        mask = (bg > lo) & (bg <= hi)
        if mask.sum() < 2:
            continue
        out.append({"bin": label, "n": int(mask.sum()),
                     "mae": float(mean_absolute_error(true[mask], pred[mask])),
                     "r2": float(r2_score(true[mask], pred[mask])) if mask.sum() > 1 else None})
    return out

jv_bandgap_strat = stratify_by_bandgap(bg_jv, pred_jv, yjv)
print("\nJARVIS-DFT transfer, stratified by OptB88vdW band gap:")
for row in jv_bandgap_strat:
    print(f"  {row['bin']:28s} n={row['n']:5d}  MAE={row['mae']:.3f}  R2={row['r2']}")

def stratify_by_target(true, pred):
    edges = np.quantile(true, [0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        mask = (true >= lo) & (true <= hi) if i == 3 else (true >= lo) & (true < hi)
        out.append({"quartile": i + 1, "log10_range": [float(lo), float(hi)],
                     "n": int(mask.sum()),
                     "mae": float(mean_absolute_error(true[mask], pred[mask])),
                     "r2": float(r2_score(true[mask], pred[mask])) if mask.sum() > 1 else None})
    return out

mp_held_target_strat = stratify_by_target(yte, pred_mp_held)
jv_target_strat = stratify_by_target(yjv, pred_jv)
print("\nHeld-out MP, stratified by true log10(e_total) quartile:")
for row in mp_held_target_strat:
    print(f"  Q{row['quartile']} {row['log10_range']}: n={row['n']:4d} MAE={row['mae']:.3f} R2={row['r2']}")
print("\nJARVIS transfer, stratified by true log10(e_total) quartile:")
for row in jv_target_strat:
    print(f"  Q{row['quartile']} {row['log10_range']}: n={row['n']:4d} MAE={row['mae']:.3f} R2={row['r2']}")

results = {
    "n_train": int(len(ytr)),
    "held_out_mp": {"n": int(len(yte)), "r2": float(r2_mp_held), "mae": float(mae_mp_held)},
    "jarvis_transfer": {"n": int(len(yjv)), "r2": float(r2_jv), "mae": float(mae_jv)},
    "jarvis_bandgap_stratified": jv_bandgap_strat,
    "mp_held_target_stratified": mp_held_target_strat,
    "jarvis_target_stratified": jv_target_strat,
}
with open(f"{OUT}/single_model_internal_external.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved models/crossdb/single_model_internal_external.json")
