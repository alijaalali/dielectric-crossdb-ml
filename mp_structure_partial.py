# Tests whether matminer's bundled Petousis et al. (2017) DFPT dielectric dataset can
# supply real structural metadata (space group, volume, density) for a subset of MP
# compounds, as a composition-only vs. composition+structure comparison and as
# space-group support for polymorph identification in the composition-overlap set.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"
RNG = 42


def metrics(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else None}


def main():
    our = pd.DataFrame(json.load(open(DATA / "mp_dielectric_raw_dedup.json")))
    our = our.dropna(subset=["e_total", "e_electronic", "e_ionic"])
    our = our[our["e_total"] < 1000].reset_index(drop=True)

    pet = load_dataset("dielectric_constant")
    pet = pet.rename(columns={"material_id": "mid_pet"})

    merged = our.merge(pet[["mid_pet", "space_group", "volume", "nsites", "structure"]],
                        left_on="material_id", right_on="mid_pet", how="inner")
    out = {"n_our_unique": int(our["material_id"].nunique()),
           "n_petousis_bundled": int(pet.shape[0]),
           "n_overlap": int(merged.shape[0])}

    merged["volume_per_atom"] = merged["volume"] / merged["nsites"]
    merged["density_proxy"] = merged["nsites"] / merged["volume"]
    merged["space_group"] = merged["space_group"].astype(int)

    ep = ElementProperty.from_preset("magpie")
    comps, idx = [], []
    for i, f in enumerate(merged["formula_pretty"]):
        try:
            comps.append(Composition(f)); idx.append(i)
        except Exception:
            pass
    feat = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition", ignore_errors=True)
    feat = feat.drop(columns=["composition"]).reset_index(drop=True)
    merged = merged.iloc[idx].reset_index(drop=True)
    merged = pd.concat([merged, feat], axis=1)

    magpie_cols = [c for c in merged.columns if c.startswith("MagpieData")]
    struct_cols = ["space_group", "volume_per_atom", "density_proxy"]

    y = np.log10(merged["e_total"].to_numpy(dtype=float))
    Xcomp = merged[magpie_cols].to_numpy(dtype=float)
    Xcs = merged[magpie_cols + struct_cols].to_numpy(dtype=float)

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=RNG)

    m1 = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=1.0, random_state=RNG)
    m1.fit(Xcomp[idx_tr], y[idx_tr])
    p1 = m1.predict(Xcomp[idx_te])

    m2 = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=1.0, random_state=RNG)
    m2.fit(Xcs[idx_tr], y[idx_tr])
    p2 = m2.predict(Xcs[idx_te])

    out["composition_only"] = metrics(y[idx_te], p1)
    out["composition_plus_structure"] = metrics(y[idx_te], p2)
    out["n_train"] = int(len(idx_tr))
    out["n_test"] = int(len(idx_te))

    jv = pd.read_parquet(DATA / "jarvis_dielectric_features_dedup.parquet"
                          if (DATA / "jarvis_dielectric_features_dedup.parquet").exists()
                          else DATA / "jarvis_dielectric_features.parquet")
    jv["reduced_formula"] = jv["formula"].apply(lambda f: Composition(f).reduced_formula)
    merged["reduced_formula"] = merged["formula_pretty"].apply(lambda f: Composition(f).reduced_formula)
    overlap_formulas = set(merged["reduced_formula"]) & set(jv["reduced_formula"])
    out["n_structure_covered_overlap_formulas"] = len(overlap_formulas)

    with open(OUT / "v3_mp_structure_partial.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
