# Structure-matched comparison for the composition-overlap formulas with known space
# groups on both sides, classifying each as matching polymorph, different polymorph, or
# ambiguous (multiple polymorphs reported), then compares transfer prediction error by label.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.core import Composition

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"


def reduced_formula(f):
    try:
        return Composition(f).reduced_formula
    except Exception:
        return None


def main():
    mp = pd.read_parquet(DATA / "mp_dielectric_features_dedup.parquet")
    jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet")
    mp["reduced_formula"] = mp["formula_pretty"].apply(reduced_formula)
    jv["reduced_formula"] = jv["formula"].apply(reduced_formula)

    pet = load_dataset("dielectric_constant").rename(columns={"material_id": "mid_pet"})
    mp_struct = mp.merge(pet[["mid_pet", "space_group"]], left_on="material_id",
                          right_on="mid_pet", how="inner")

    overlap = set(mp["reduced_formula"].dropna()) & set(jv["reduced_formula"].dropna())
    struct_overlap = set(mp_struct["reduced_formula"].dropna()) & overlap

    rows = []
    for f in sorted(struct_overlap):
        mp_spgs = set(mp_struct.loc[mp_struct["reduced_formula"] == f, "space_group"].astype(int))
        jv_spgs = set(jv.loc[jv["reduced_formula"] == f, "spg_number"].astype(int))
        if len(mp_spgs) == 1 and len(jv_spgs) == 1:
            label = "single_matching" if mp_spgs == jv_spgs else "single_different"
        else:
            label = "ambiguous_multi"
        rows.append({"reduced_formula": f, "mp_spgs": sorted(mp_spgs), "jv_spgs": sorted(jv_spgs),
                     "label": label})

    df = pd.DataFrame(rows)
    counts = df["label"].value_counts().to_dict()

    pred = pd.read_csv(OUT / "predictions_mp_to_jarvis_etotal.csv")
    pred["reduced_formula"] = pred["formula"].apply(reduced_formula)
    pred["abs_err"] = (pred["pred_log10"] - pred["true_log10"]).abs()

    err_by_label = {}
    for label in ["single_matching", "single_different", "ambiguous_multi"]:
        formulas = set(df.loc[df["label"] == label, "reduced_formula"])
        sub = pred[pred["reduced_formula"].isin(formulas)]
        if len(sub):
            err_by_label[label] = {"n": int(len(sub)), "mae": float(sub["abs_err"].mean())}

    novel = pred[~pred["reduced_formula"].isin(overlap)]
    err_by_label["composition_novel_no_overlap"] = {"n": int(len(novel)), "mae": float(novel["abs_err"].mean())}

    out = {
        "n_composition_overlap_formulas": len(overlap),
        "n_structure_covered_formulas": len(struct_overlap),
        "label_counts": counts,
        "mae_by_polymorph_label": err_by_label,
    }
    with open(OUT / "v3_polymorph_matching.json", "w") as fh:
        json.dump(out, fh, indent=2)
    df.to_csv(OUT / "v3_polymorph_matching_detail.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
