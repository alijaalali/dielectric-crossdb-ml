# Builds Magpie composition feature tables for the MP and JARVIS dielectric datasets.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def magpie_features(formulas):
    ep = ElementProperty.from_preset("magpie")
    comps = []
    valid_idx = []
    for i, f in enumerate(formulas):
        try:
            comps.append(Composition(f))
            valid_idx.append(i)
        except Exception:
            pass
    df = pd.DataFrame({"composition": comps})
    df = ep.featurize_dataframe(df, "composition", ignore_errors=True)
    df = df.drop(columns=["composition"])
    return df, valid_idx


def main():
    t0 = time.time()

    # MP raw data has each material_id duplicated; build from the deduplicated file.
    mp = json.load(open(DATA / "mp_dielectric_raw_dedup.json"))
    mp_df = pd.DataFrame(mp)
    n0 = len(mp_df)
    mp_df = mp_df.dropna(subset=["e_total", "e_electronic", "e_ionic"])
    mp_df = mp_df[mp_df["e_total"] < 1000].reset_index(drop=True)
    log(f"MP: {n0} raw (deduplicated) -> {len(mp_df)} after dropping NaN/outlier (e_total<1000 ceiling)")
    feat, idx = magpie_features(mp_df["formula_pretty"].tolist())
    mp_df = mp_df.iloc[idx].reset_index(drop=True)
    mp_out = pd.concat([mp_df, feat.reset_index(drop=True)], axis=1)
    mp_out.to_parquet(DATA / "mp_dielectric_features_dedup.parquet")
    log(f"MP final table (dedup): {mp_out.shape}")

    # JARVIS
    jv = json.load(open(DATA / "jarvis_dielectric_raw.json"))
    jv_df = pd.DataFrame(jv)
    n0j = len(jv_df)
    jv_df = jv_df.dropna(subset=["e_total", "e_electronic", "e_ionic"])
    log(f"JARVIS: {n0j} raw -> {len(jv_df)} after dropping NaN")
    featj, idxj = magpie_features(jv_df["formula"].tolist())
    jv_df = jv_df.iloc[idxj].reset_index(drop=True)
    jv_out = pd.concat([jv_df, featj.reset_index(drop=True)], axis=1)
    jv_out.to_parquet(DATA / "jarvis_dielectric_features.parquet")
    log(f"JARVIS final table: {jv_out.shape}")

    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
