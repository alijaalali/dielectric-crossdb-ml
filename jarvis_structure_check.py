# Tests whether adding structural descriptors (volume, density, space group) to the
# Magpie composition features changes JARVIS-internal accuracy.
import json
import numpy as np
import pandas as pd
from jarvis.db.figshare import data
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

jv = pd.read_parquet("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project/data/jarvis_dielectric_features.parquet")
raw = {d["jid"]: d for d in data("dft_3d")}

vols, dens, nsites = [], [], []
for jid in jv["jid"]:
    r = raw.get(jid)
    if r is None or r.get("atoms") in (None, "na"):
        vols.append(np.nan); dens.append(np.nan); nsites.append(np.nan)
        continue
    at = r["atoms"]
    lat = np.array(at["lattice_mat"])
    vol = abs(np.linalg.det(lat))
    n = len(at["elements"])
    vols.append(vol)
    dens.append(n / vol if vol > 0 else np.nan)
    nsites.append(n)

jv["volume"] = vols
jv["density_atoms_per_vol"] = dens
jv["n_sites"] = nsites
n_missing = jv["volume"].isna().sum()
print(f"missing structural data: {n_missing}/{len(jv)}")

feat_cols = [c for c in jv.columns if c.startswith("MagpieData")]
jv_c = jv.dropna(subset=["e_total", "volume"]).reset_index(drop=True)
y = np.log10(jv_c["e_total"].to_numpy(dtype=float))

X_comp = jv_c[feat_cols].to_numpy(dtype=float)
X_struct = jv_c[feat_cols + ["spg_number", "volume", "density_atoms_per_vol", "n_sites"]].to_numpy(dtype=float)

for name, X in [("composition-only", X_comp), ("composition+structure", X_struct)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name}: n={len(y)} MAE={mean_absolute_error(yte,p):.4f} R2={r2_score(yte,p):.4f}")

out = {"n_total": int(len(jv_c)), "n_missing_structure": int(n_missing)}
with open("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project/models/crossdb/jarvis_structure_test.json", "w") as f:
    json.dump(out, f, indent=2)
