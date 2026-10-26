# Training-size-matched learning curve for MP -> JARVIS transfer, repeated across
# several random seeds per training size (mean +/- std rather than one point estimate).
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"

N_SEEDS = 5
SIZES = [2000, 4000, 4706, 6000, 7320]


def main():
    mp = pd.read_parquet(DATA / "mp_dielectric_features_dedup.parquet").dropna(subset=["e_total"]).reset_index(drop=True)
    jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet").dropna(subset=["e_total"]).reset_index(drop=True)
    feat_cols = [c for c in mp.columns if c.startswith("MagpieData")]

    Xmp = mp[feat_cols].to_numpy(dtype=float)
    ymp = np.log10(mp["e_total"].to_numpy(dtype=float))
    Xjv = jv[feat_cols].to_numpy(dtype=float)
    yjv = np.log10(jv["e_total"].to_numpy(dtype=float))

    results = {}
    for n in SIZES:
        if n > len(ymp):
            continue
        r2s, maes = [], []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(1000 + seed)
            idx = rng.choice(len(ymp), size=n, replace=False) if n < len(ymp) else np.arange(len(ymp))
            model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                  random_state=seed)
            model.fit(Xmp[idx], ymp[idx])
            pred = model.predict(Xjv)
            r2s.append(r2_score(yjv, pred))
            maes.append(mean_absolute_error(yjv, pred))
        results[str(n)] = {
            "n_train": n, "n_seeds": N_SEEDS,
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        }
        print(f"n={n}: R2={np.mean(r2s):.3f}+-{np.std(r2s):.3f}  MAE={np.mean(maes):.3f}+-{np.std(maes):.3f}")

    with open(OUT / "learning_curve_multiseed.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
