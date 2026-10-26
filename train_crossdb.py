# Core cross-database transfer test for dielectric constant (e_total, e_electronic, e_ionic).
# All targets are modeled in log10 space due to strong right-skew; e_ionic uses log10(e_ionic + 0.01)
# to handle exact zeros. Trains on MP, zero-shot tests on JARVIS, and the reverse direction, with
# MP-internal and JARVIS-internal random splits as in-distribution references.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["e_total", "e_electronic", "e_ionic"]
IONIC_EPS = 0.01


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("MagpieData")]


def to_log_target(df, target):
    if target == "e_ionic":
        return np.log10(df[target].to_numpy(dtype=float) + IONIC_EPS)
    return np.log10(df[target].to_numpy(dtype=float))


def fit_eval(X_train, y_train, X_test, y_test, n_estimators=300):
    model = XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          n_jobs=8, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    m = {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "mae_log10": float(mean_absolute_error(y_test, pred)),
        "rmse_log10": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2_log10": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
    }
    return model, m, pred


def main():
    t0 = time.time()
    mp = pd.read_parquet(DATA / "mp_dielectric_features_dedup.parquet")
    jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet")
    feat_cols = get_feature_cols(mp)
    log(f"Feature columns: {len(feat_cols)}")

    col_mean = mp[feat_cols].mean(numeric_only=True)

    def X(df):
        return df[feat_cols].fillna(col_mean).to_numpy(dtype=float)

    results = {}

    for target in TARGETS:
        log(f"=== Target: log10({target}) ===")
        mp_t = mp.dropna(subset=[target]).reset_index(drop=True)
        jv_t = jv.dropna(subset=[target]).reset_index(drop=True)

        Xmp, ymp = X(mp_t), to_log_target(mp_t, target)
        Xjv, yjv = X(jv_t), to_log_target(jv_t, target)

        Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
            Xmp, ymp, np.arange(len(ymp)), test_size=0.2, random_state=42)
        model_mp, m_internal, pred_internal = fit_eval(Xtr, ytr, Xte, yte)
        log(f"  MP-internal random split: {m_internal}")

        model_full_mp, m_mp2jv, pred_mp2jv = fit_eval(Xmp, ymp, Xjv, yjv)
        log(f"  MP -> JARVIS: {m_mp2jv}")

        model_full_jv, m_jv2mp, pred_jv2mp = fit_eval(Xjv, yjv, Xmp, ymp)
        log(f"  JARVIS -> MP: {m_jv2mp}")

        Xtr_j, Xte_j, ytr_j, yte_j = train_test_split(Xjv, yjv, test_size=0.2, random_state=42)
        _, m_internal_jv, _ = fit_eval(Xtr_j, ytr_j, Xte_j, yte_j)
        log(f"  JARVIS-internal random split: {m_internal_jv}")

        results[target] = {
            "mp_internal": m_internal,
            "jarvis_internal": m_internal_jv,
            "mp_to_jarvis": m_mp2jv,
            "jarvis_to_mp": m_jv2mp,
        }

        if target == "e_total":
            pred_df = jv_t[["jid", "formula", "func", "optb88vdw_bandgap", "spg_number"]].copy()
            pred_df["true_log10"] = yjv
            pred_df["pred_log10"] = pred_mp2jv
            pred_df["residual_log10"] = pred_df["pred_log10"] - pred_df["true_log10"]
            pred_df.to_csv(OUT / "predictions_mp_to_jarvis_etotal.csv", index=False)

            pred_df_internal = mp_t.iloc[idx_te][["material_id", "formula_pretty"]].copy()
            pred_df_internal["true_log10"] = yte
            pred_df_internal["pred_log10"] = pred_internal
            pred_df_internal.to_csv(OUT / "predictions_mp_internal_etotal.csv", index=False)

            pred_df_j2m = mp_t[["material_id", "formula_pretty"]].copy()
            pred_df_j2m["true_log10"] = ymp
            pred_df_j2m["pred_log10"] = pred_jv2mp
            pred_df_j2m.to_csv(OUT / "predictions_jarvis_to_mp_etotal.csv", index=False)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
