# Compares several regressors (XGBoost, RandomForest, SVR, kNN, Gaussian Process,
# Ridge, Lasso, MLP) on the MP -> JARVIS dielectric transfer task.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("MagpieData")]


MODELS = {
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                     n_jobs=8, random_state=42),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, n_jobs=8, random_state=42),
    "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "kNN": lambda: KNeighborsRegressor(n_neighbors=8, weights="distance"),
    "GaussianProcess": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True, random_state=42),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.001, max_iter=10000),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 random_state=42, early_stopping=True),
}
NEEDS_SCALING = {"SVR", "kNN", "Ridge", "Lasso", "MLP", "GaussianProcess"}

GP_CAP = 1500  # GaussianProcess is O(n^3); cap training set for tractability.


def maybe_cap_gp(name, X, y, seed=42):
    if name == "GaussianProcess" and len(y) > GP_CAP:
        rr = np.random.RandomState(seed)
        idx = rr.choice(len(y), GP_CAP, replace=False)
        return X[idx], y[idx]
    return X, y


def main():
    t0 = time.time()
    mp = pd.read_parquet(DATA / "mp_dielectric_features_dedup.parquet").dropna(subset=["e_total"])
    jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet").dropna(subset=["e_total"])
    feat_cols = get_feature_cols(mp)
    col_mean = mp[feat_cols].mean(numeric_only=True)

    Xmp_raw = mp[feat_cols].fillna(col_mean).to_numpy(dtype=float)
    ymp = np.log10(mp["e_total"].to_numpy(dtype=float))
    Xjv_raw = jv[feat_cols].fillna(col_mean).to_numpy(dtype=float)
    yjv = np.log10(jv["e_total"].to_numpy(dtype=float))

    scaler = StandardScaler().fit(Xmp_raw)
    Xmp_scaled = scaler.transform(Xmp_raw)
    Xjv_scaled = scaler.transform(Xjv_raw)

    results = {}
    for name, ctor in MODELS.items():
        model = ctor()
        Xtr = Xmp_scaled if name in NEEDS_SCALING else Xmp_raw
        Xte = Xjv_scaled if name in NEEDS_SCALING else Xjv_raw
        Xtr, ytr = maybe_cap_gp(name, Xtr, ymp)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        m = {
            "mae_log10": float(mean_absolute_error(yjv, pred)),
            "rmse_log10": float(np.sqrt(mean_squared_error(yjv, pred))),
            "r2_log10": float(r2_score(yjv, pred)),
        }
        results[name] = m
        log(f"  {name} (MP->JARVIS): {m}")

    with open(OUT / "benchmark_models.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
