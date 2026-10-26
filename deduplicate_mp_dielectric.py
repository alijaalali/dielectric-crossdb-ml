# Checks and quantifies the effect of duplicate material_id rows in the raw MP
# dielectric retrieval: reruns the four headline evaluation conditions (MP-internal,
# JARVIS-internal, MP->JARVIS, JARVIS->MP) with the MP side deduplicated for comparison.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"

XGB_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, random_state=42,
)


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    mp = pd.read_parquet(DATA / "mp_dielectric_features.parquet")
    jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet")

    out = {}
    out["n_mp_rows_raw"] = int(len(mp))
    out["n_mp_unique_material_id_raw"] = int(mp["material_id"].nunique())

    # Confirm duplicates are byte-identical (not two different calculations).
    feat_cols = [c for c in mp.columns if c.startswith("Magpie")]
    check_cols = feat_cols + ["e_total", "e_electronic", "e_ionic"]
    n_checked = 0
    n_mismatch = 0
    for mid, grp in mp.groupby("material_id"):
        if len(grp) < 2:
            continue
        n_checked += 1
        a = grp[check_cols].iloc[0].to_numpy(dtype=float)
        b = grp[check_cols].iloc[1].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True):
            n_mismatch += 1
    out["n_duplicate_pairs_checked"] = n_checked
    out["n_duplicate_pairs_with_mismatched_values"] = n_mismatch

    mp_dedup = mp.drop_duplicates(subset="material_id", keep="first").reset_index(drop=True)
    out["n_mp_rows_after_dedup"] = int(len(mp_dedup))

    mp_dup_filt = mp[mp["e_total"] < 1000].reset_index(drop=True)
    mp_dedup_filt = mp_dedup[mp_dedup["e_total"] < 1000].reset_index(drop=True)
    out["n_mp_after_outlier_filter_duplicated"] = int(len(mp_dup_filt))
    out["n_mp_after_outlier_filter_dedup"] = int(len(mp_dedup_filt))

    jv_filt = jv.copy()

    def xy(df, feat_cols, target):
        X = df[feat_cols].to_numpy(dtype=float)
        y = np.log10(df[target].to_numpy(dtype=float))
        return X, y

    results = {}

    for label, mp_src in [("duplicated_original", mp_dup_filt), ("deduplicated", mp_dedup_filt)]:
        X, y = xy(mp_src, feat_cols, "e_total")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[f"mp_internal_{label}"] = {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)), **metrics(y_te, pred),
        }

        if label == "duplicated_original":
            ids = mp_src["material_id"].to_numpy()
            ids_tr, ids_te = train_test_split(ids, test_size=0.2, random_state=42)
            train_id_set = set(ids_tr)
            leak = sum(1 for i in ids_te if i in train_id_set)
            results["mp_internal_duplicated_original"]["n_test_with_duplicate_in_train"] = int(leak)
            results["mp_internal_duplicated_original"]["frac_test_with_duplicate_in_train"] = float(leak / len(ids_te))

    Xj, yj = xy(jv_filt, feat_cols, "e_total")
    for label, mp_src in [("duplicated_original", mp_dup_filt), ("deduplicated", mp_dedup_filt)]:
        Xm, ym = xy(mp_src, feat_cols, "e_total")
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(Xm, ym)
        pred = model.predict(Xj)
        results[f"mp_to_jarvis_{label}"] = {
            "n_train": int(len(Xm)), "n_test": int(len(Xj)), **metrics(yj, pred),
        }

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(Xj, yj)
    for label, mp_src in [("duplicated_original", mp_dup_filt), ("deduplicated", mp_dedup_filt)]:
        Xm, ym = xy(mp_src, feat_cols, "e_total")
        pred = model.predict(Xm)
        results[f"jarvis_to_mp_{label}"] = {
            "n_train": int(len(Xj)), "n_test": int(len(Xm)), **metrics(ym, pred),
        }

    out["results"] = results

    with open(OUT / "dedup_fix.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
