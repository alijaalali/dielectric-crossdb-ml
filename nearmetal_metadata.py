# Investigates JARVIS-DFT compounds with band gap < 0.1 eV using additional metadata
# (TB-mbj gap, magnetic moment, dielectric tensor anisotropy, energy above hull,
# calculation protocol) to check whether these are genuinely metallic or narrow-gap
# semiconductors misclassified by the OptB88vdW functional.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.db.figshare import data as jarvis_data

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def main():
    our = pd.DataFrame(json.load(open(DATA / "jarvis_dielectric_raw.json")))
    our = our.dropna(subset=["e_total", "e_electronic", "e_ionic"]).reset_index(drop=True)
    near_metal_mask = our["optb88vdw_bandgap"].astype(float) < 0.1
    near_metal_ids = set(our.loc[near_metal_mask, "jid"])
    non_metal_ids = set(our.loc[~near_metal_mask, "jid"])
    print(f"near-metal n={len(near_metal_ids)}, non-metal n={len(non_metal_ids)}")

    full = jarvis_data("dft_3d")
    full_by_id = {r["jid"]: r for r in full}

    def extract(jid_set, label):
        rows = []
        for jid in jid_set:
            r = full_by_id.get(jid)
            if r is None:
                continue
            epsx, epsy, epsz = (safe_float(r.get(k)) for k in ("epsx", "epsy", "epsz"))
            eps_vals = [v for v in (epsx, epsy, epsz) if v is not None and v > 0]
            aniso = (max(eps_vals) / min(eps_vals)) if len(eps_vals) >= 2 and min(eps_vals) > 0 else None
            rows.append({
                "jid": jid,
                "optb88vdw_bandgap": safe_float(r.get("optb88vdw_bandgap")),
                "mbj_bandgap": safe_float(r.get("mbj_bandgap")),
                "hse_gap": safe_float(r.get("hse_gap")),
                "magmom_outcar": safe_float(r.get("magmom_outcar")),
                "magmom_oszicar": safe_float(r.get("magmom_oszicar")),
                "ehull": safe_float(r.get("ehull")),
                "func": r.get("func"),
                "encut": safe_float(r.get("encut")),
                "kpoint_length_unit": safe_float(r.get("kpoint_length_unit")),
                "eps_electronic_anisotropy": aniso,
            })
        df = pd.DataFrame(rows)
        df["set"] = label
        return df

    nm = extract(near_metal_ids, "near_metallic")
    non = extract(non_metal_ids, "non_near_metallic")
    both = pd.concat([nm, non], ignore_index=True)
    both.to_csv(OUT / "nearmetal_metadata_detail.csv", index=False)

    out = {"n_near_metal_matched": int(len(nm)), "n_non_metal_matched": int(len(non))}

    nm_mbj = nm.dropna(subset=["mbj_bandgap"])
    out["n_near_metal_with_mbj"] = int(len(nm_mbj))
    out["n_near_metal_mbj_still_below_0.1"] = int((nm_mbj["mbj_bandgap"] < 0.1).sum())
    out["n_near_metal_mbj_above_0.1"] = int((nm_mbj["mbj_bandgap"] >= 0.1).sum())
    out["frac_near_metal_mbj_reclassified_semiconductor_pct"] = (
        float(100 * (nm_mbj["mbj_bandgap"] >= 0.1).mean()) if len(nm_mbj) else None
    )
    out["median_mbj_bandgap_near_metal_subset"] = float(nm_mbj["mbj_bandgap"].median()) if len(nm_mbj) else None

    nm_mag = nm.dropna(subset=["magmom_outcar"])
    non_mag = non.dropna(subset=["magmom_outcar"])
    out["frac_near_metal_magnetic_pct"] = (
        float(100 * (nm_mag["magmom_outcar"].abs() > 0.1).mean()) if len(nm_mag) else None
    )
    out["frac_non_metal_magnetic_pct"] = (
        float(100 * (non_mag["magmom_outcar"].abs() > 0.1).mean()) if len(non_mag) else None
    )

    nm_aniso = nm.dropna(subset=["eps_electronic_anisotropy"])
    non_aniso = non.dropna(subset=["eps_electronic_anisotropy"])
    out["median_eps_anisotropy_near_metal"] = float(nm_aniso["eps_electronic_anisotropy"].median()) if len(nm_aniso) else None
    out["median_eps_anisotropy_non_metal"] = float(non_aniso["eps_electronic_anisotropy"].median()) if len(non_aniso) else None
    out["frac_near_metal_aniso_gt10_pct"] = float(100 * (nm_aniso["eps_electronic_anisotropy"] > 10).mean()) if len(nm_aniso) else None
    out["frac_non_metal_aniso_gt10_pct"] = float(100 * (non_aniso["eps_electronic_anisotropy"] > 10).mean()) if len(non_aniso) else None

    out["median_ehull_near_metal"] = float(nm["ehull"].dropna().median()) if nm["ehull"].notna().any() else None
    out["median_ehull_non_metal"] = float(non["ehull"].dropna().median()) if non["ehull"].notna().any() else None

    out["near_metal_func_value_counts"] = nm["func"].value_counts().to_dict()
    out["non_metal_func_value_counts"] = non["func"].value_counts().to_dict()
    out["median_encut_near_metal"] = float(nm["encut"].dropna().median()) if nm["encut"].notna().any() else None
    out["median_encut_non_metal"] = float(non["encut"].dropna().median()) if non["encut"].notna().any() else None

    with open(OUT / "nearmetal_metadata.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
