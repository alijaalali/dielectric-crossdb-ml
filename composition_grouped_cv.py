# Compares the random-split MP-internal baseline to a composition-grouped split
# (GroupKFold by reduced formula) to check for polymorph leakage across train/test.
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJ = "/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project"
DATA = f"{PROJ}/data"
OUT = f"{PROJ}/models/crossdb"
RNG = 42

mp = pd.read_parquet(f"{DATA}/mp_dielectric_features_dedup.parquet").dropna(subset=["e_total"]).reset_index(drop=True)
feat_cols = [c for c in mp.columns if c.startswith("MagpieData")]
col_mean = mp[feat_cols].mean(numeric_only=True)
X = mp[feat_cols].fillna(col_mean).to_numpy(dtype=float)
y = np.log10(mp["e_total"].to_numpy(dtype=float))

def reduced_formula(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return None

mp["reduced_formula"] = mp["formula_pretty"].apply(reduced_formula)
n_unique = mp["reduced_formula"].nunique()
n_dup_rows = len(mp) - n_unique
vc = mp["reduced_formula"].value_counts()
n_polymorphic_formulas = int((vc > 1).sum())
print(f"MP (post-filter): {len(mp)} rows, {n_unique} unique reduced formulas, "
      f"{n_dup_rows} 'extra' rows beyond one-per-formula, "
      f"{n_polymorphic_formulas} formulas with >1 entry (polymorphs/duplicate computations)")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RNG)
m1 = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.7, reg_lambda=1.0, n_jobs=8, random_state=RNG)
m1.fit(Xtr, ytr)
pred1 = m1.predict(Xte)
r2_random = r2_score(yte, pred1)
mae_random = mean_absolute_error(yte, pred1)

ref = json.load(open(f"{OUT}/metrics.json"))["e_total"]["mp_internal"]
print(f"\nRandom-split reproduction: R2={r2_random:.6f} (saved {ref['r2_log10']:.6f}), "
      f"MAE={mae_random:.6f} (saved {ref['mae_log10']:.6f})")
assert abs(r2_random - ref["r2_log10"]) < 1e-6 and abs(mae_random - ref["mae_log10"]) < 1e-6
print("Matches exactly.")

groups = mp["reduced_formula"].to_numpy()
gkf = GroupKFold(n_splits=5)
fold_metrics = []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=8, random_state=RNG)
    m.fit(X[tr_idx], y[tr_idx])
    p = m.predict(X[te_idx])
    fm = {"fold": fold, "n_train": len(tr_idx), "n_test": len(te_idx),
          "mae": float(mean_absolute_error(y[te_idx], p)),
          "rmse": float(np.sqrt(mean_squared_error(y[te_idx], p))),
          "r2": float(r2_score(y[te_idx], p))}
    fold_metrics.append(fm)
    print(f"  GroupKFold fold {fold}: n_train={fm['n_train']} n_test={fm['n_test']} "
          f"MAE={fm['mae']:.4f} R2={fm['r2']:.4f}")

mae_grouped = float(np.mean([f["mae"] for f in fold_metrics]))
r2_grouped = float(np.mean([f["r2"] for f in fold_metrics]))
mae_grouped_std = float(np.std([f["mae"] for f in fold_metrics]))
r2_grouped_std = float(np.std([f["r2"] for f in fold_metrics]))
print(f"\nComposition-grouped 5-fold CV: mean MAE={mae_grouped:.4f} (+-{mae_grouped_std:.4f}), "
      f"mean R2={r2_grouped:.4f} (+-{r2_grouped_std:.4f})")
print(f"Random single split:            MAE={mae_random:.4f}, R2={r2_random:.4f}")

results = {
    "n_rows": len(mp), "n_unique_formulas": n_unique,
    "n_polymorphic_formulas": n_polymorphic_formulas,
    "random_split_mae": mae_random, "random_split_r2": r2_random,
    "grouped_cv_mae_mean": mae_grouped, "grouped_cv_mae_std": mae_grouped_std,
    "grouped_cv_r2_mean": r2_grouped, "grouped_cv_r2_std": r2_grouped_std,
    "grouped_cv_folds": fold_metrics,
}
with open(f"{OUT}/composition_grouped_cv.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved models/crossdb/composition_grouped_cv.json")
