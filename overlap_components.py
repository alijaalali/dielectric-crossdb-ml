# Compares MP and JARVIS-DFT's own reported electronic and ionic dielectric constants
# directly for composition-overlapping compounds, to check whether the electronic/ionic
# transfer asymmetry reflects a real physical difference or a database-protocol difference.
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"


def reduced_formula(f):
    try:
        from pymatgen.core import Composition
        return Composition(f).reduced_formula
    except Exception:
        return None


def main():
    mp = pd.read_parquet(DATA / "mp_dielectric_features_dedup.parquet")
    jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet")

    mp["reduced_formula"] = mp["formula_pretty"].apply(reduced_formula)
    jv["reduced_formula"] = jv["formula"].apply(reduced_formula)
    mp_set = set(mp["reduced_formula"].dropna())
    jv_set = set(jv["reduced_formula"].dropna())
    overlap = mp_set & jv_set

    mp_ov = mp[mp["reduced_formula"].isin(overlap)]
    jv_ov = jv[jv["reduced_formula"].isin(overlap)]

    out = {"n_overlap_formulas": len(overlap)}
    for comp in ["e_electronic", "e_ionic"]:
        mp_med = mp_ov.dropna(subset=[comp]).groupby("reduced_formula")[comp].median()
        jv_med = jv_ov.dropna(subset=[comp]).groupby("reduced_formula")[comp].median()
        common = mp_med.index.intersection(jv_med.index)
        paired = pd.DataFrame({"mp": mp_med.loc[common], "jv": jv_med.loc[common]})
        log_ratio = np.log10(paired["jv"] / paired["mp"])
        out[comp] = {
            "n_paired": int(len(paired)),
            "median_log_ratio_jv_over_mp": float(log_ratio.median()),
            "iqr_log_ratio": [float(log_ratio.quantile(.25)), float(log_ratio.quantile(.75))],
            "mae_log_ratio": float(log_ratio.abs().median()),
        }
        paired.to_csv(OUT / f"overlap_paired_compounds_{comp}.csv")
        print(comp, out[comp])

    with open(OUT / "overlap_components.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
