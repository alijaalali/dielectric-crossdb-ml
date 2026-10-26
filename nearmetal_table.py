# Builds a table of the highest-error near-metallic JARVIS-DFT compounds (band gap,
# dielectric constants, predicted value, absolute error) using the MP->JARVIS
# predictions and the metadata recovered in nearmetal_metadata.py.
import json
from pathlib import Path

import pandas as pd
import numpy as np

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"


def main():
    pred = pd.read_csv(OUT / "predictions_mp_to_jarvis_etotal.csv")
    raw = pd.DataFrame(json.load(open(DATA / "jarvis_dielectric_raw.json")))
    meta = pd.read_csv(OUT / "nearmetal_metadata_detail.csv")

    df = pred.merge(raw[["jid", "e_total", "e_electronic", "e_ionic"]], on="jid", how="left")
    df = df.merge(meta[["jid", "mbj_bandgap", "magmom_outcar"]], on="jid", how="left")

    near_metal = df[df["optb88vdw_bandgap"] < 0.1].copy()
    near_metal["abs_err_log10"] = (near_metal["pred_log10"] - near_metal["true_log10"]).abs()
    near_metal["pred_e_total"] = 10 ** near_metal["pred_log10"]
    top10 = near_metal.sort_values("abs_err_log10", ascending=False).head(10)

    cols = ["jid", "formula", "optb88vdw_bandgap", "mbj_bandgap", "magmom_outcar",
            "e_total", "e_electronic", "e_ionic", "pred_e_total", "abs_err_log10"]
    top10[cols].to_csv(OUT / "nearmetal_top10_table.csv", index=False)
    print(top10[cols].to_string(index=False))


if __name__ == "__main__":
    main()
