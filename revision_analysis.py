# Extended analysis bundle: composition overlap between MP/JARVIS, outlier-filter
# sensitivity, training-size-matched comparison, near-metallic internal-vs-external
# test, electronic/ionic split, 8-model x 4-condition benchmark, bootstrap CIs,
# raw-unit metrics, and bandgap-residual analysis.
import json, time, warnings
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
DATA = PROJ / "data"
OUT = PROJ / "models" / "crossdb"
OUT.mkdir(parents=True, exist_ok=True)
RNG = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fcols(df):
    return [c for c in df.columns if c.startswith("MagpieData")]


results = {}

mp_raw = pd.DataFrame(json.load(open(DATA / "mp_dielectric_raw_dedup.json")))
jv_raw = pd.DataFrame(json.load(open(DATA / "jarvis_dielectric_raw.json")))
mp = pd.read_parquet(DATA / "mp_dielectric_features_dedup.parquet")
jv = pd.read_parquet(DATA / "jarvis_dielectric_features.parquet")
feat_cols = fcols(mp)
col_mean = mp[feat_cols].mean(numeric_only=True)


def Xof(df):
    return df[feat_cols].fillna(col_mean).to_numpy(dtype=float)


def ylog(df, target="e_total", eps=0.0):
    return np.log10(df[target].to_numpy(dtype=float) + eps)


# ---- Composition overlap between MP and JARVIS ----
log("=== Composition overlap ===")


def reduced_formula(f):
    try:
        from pymatgen.core import Composition
        return Composition(f).reduced_formula
    except Exception:
        return None


mp["reduced_formula"] = mp["formula_pretty"].apply(reduced_formula)
jv["reduced_formula"] = jv["formula"].apply(reduced_formula)
mp_set = set(mp["reduced_formula"].dropna())
jv_set = set(jv["reduced_formula"].dropna())
overlap = mp_set & jv_set
log(f"MP unique reduced formulas: {len(mp_set)}, JARVIS: {len(jv_set)}, overlap: {len(overlap)}")

jv["composition_overlap"] = jv["reduced_formula"].isin(overlap)
mp["composition_overlap"] = mp["reduced_formula"].isin(overlap)

mp_med = mp[mp["composition_overlap"]].groupby("reduced_formula")["e_total"].median()
jv_med = jv[jv["composition_overlap"]].groupby("reduced_formula")["e_total"].median()
common_idx = mp_med.index.intersection(jv_med.index)
paired = pd.DataFrame({"mp_e_total": mp_med.loc[common_idx], "jv_e_total": jv_med.loc[common_idx]})
paired["log_ratio"] = np.log10(paired["jv_e_total"] / paired["mp_e_total"])
log(f"Paired overlapping compounds (median e_total compared): n={len(paired)}, "
    f"median log10 ratio JV/MP={paired['log_ratio'].median():.3f}, "
    f"IQR=[{paired['log_ratio'].quantile(.25):.3f},{paired['log_ratio'].quantile(.75):.3f}]")

results["overlap"] = {
    "n_mp_unique": len(mp_set), "n_jarvis_unique": len(jv_set), "n_overlap": len(overlap),
    "n_jarvis_overlap_rows": int(jv["composition_overlap"].sum()),
    "n_jarvis_novel_rows": int((~jv["composition_overlap"]).sum()),
    "n_paired_compounds": len(paired),
    "median_log_ratio_jv_over_mp": float(paired["log_ratio"].median()) if len(paired) else None,
}
paired.to_csv(OUT / "overlap_paired_compounds.csv")

mp_t = mp.dropna(subset=["e_total"]).reset_index(drop=True)
jv_t = jv.dropna(subset=["e_total"]).reset_index(drop=True)
Xmp, ymp = Xof(mp_t), ylog(mp_t)
Xjv, yjv = Xof(jv_t), ylog(jv_t)

model_full = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=RNG)
model_full.fit(Xmp, ymp)
pred_full = model_full.predict(Xjv)
ov_mask = jv_t["composition_overlap"].to_numpy()


def metrics(y, p):
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else None}


results["overlap"]["mp_to_jarvis_overlap_subset"] = metrics(yjv[ov_mask], pred_full[ov_mask])
results["overlap"]["mp_to_jarvis_novel_subset"] = metrics(yjv[~ov_mask], pred_full[~ov_mask])
log(f"  MP->JARVIS on composition-OVERLAP subset: {results['overlap']['mp_to_jarvis_overlap_subset']}")
log(f"  MP->JARVIS on composition-NOVEL subset:   {results['overlap']['mp_to_jarvis_novel_subset']}")

# ---- Outlier-filtering sensitivity ----
log("=== Outlier-filter sensitivity ===")
mp_raw2 = mp_raw.dropna(subset=["e_total", "e_electronic", "e_ionic"]).reset_index(drop=True)
jv_raw2 = jv_raw.dropna(subset=["e_total", "e_electronic", "e_ionic"]).reset_index(drop=True)
jv_max = jv_raw2["e_total"].max()


def build_and_eval(mp_filt, jv_filt, label):
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    ep = ElementProperty.from_preset("magpie")

    def feats(formulas):
        comps, idx = [], []
        for i, f in enumerate(formulas):
            try:
                comps.append(Composition(f)); idx.append(i)
            except Exception:
                pass
        d = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition", ignore_errors=True)
        return d.drop(columns=["composition"]), idx

    fm, im = feats(mp_filt["formula_pretty"].tolist())
    mpf = pd.concat([mp_filt.iloc[im].reset_index(drop=True), fm.reset_index(drop=True)], axis=1)
    fj, ij = feats(jv_filt["formula"].tolist())
    jvf = pd.concat([jv_filt.iloc[ij].reset_index(drop=True), fj.reset_index(drop=True)], axis=1)
    fc = fcols(mpf)
    cm = mpf[fc].mean(numeric_only=True)
    Xm = mpf[fc].fillna(cm).to_numpy(dtype=float)
    ym = np.log10(mpf["e_total"].to_numpy(dtype=float))
    Xj = jvf[fc].fillna(cm).to_numpy(dtype=float)
    yj = np.log10(jvf["e_total"].to_numpy(dtype=float))
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=RNG)
    m.fit(Xm, ym)
    p = m.predict(Xj)
    met = metrics(yj, p)
    met["n_mp_train"] = len(mpf)
    log(f"  [{label}] n_mp={len(mpf)} n_jv={len(jvf)} -> {met}")
    return met


filter_results = {}
filter_results["no_filter"] = build_and_eval(mp_raw2, jv_raw2, "no_filter")
filter_results["current_mp1000_jv_unfiltered"] = {"note": "same as main pipeline", **results["overlap"].get("mp_to_jarvis_overlap_subset", {})}
filter_results["shared_threshold_1000"] = build_and_eval(
    mp_raw2[mp_raw2["e_total"] < 1000], jv_raw2[jv_raw2["e_total"] < 1000], "shared_threshold_1000")
filter_results["shared_threshold_1200"] = build_and_eval(
    mp_raw2[mp_raw2["e_total"] < 1200], jv_raw2[jv_raw2["e_total"] < 1200], "shared_threshold_1200")

with open(OUT / "outlier_filter_sensitivity.json", "w") as f:
    json.dump(filter_results, f, indent=2, default=str)
results["outlier_filter_sensitivity"] = filter_results
log(f"JARVIS raw max e_total = {jv_max:.1f}")

# ---- Training-size-matched comparison + learning curve ----
log("=== Training-size-matched / learning curve ===")
n_jv = len(yjv)
sizes = [n_jv, 2000, 4000, 6000, 8000, 10000, len(ymp)]
sizes = sorted(set(s for s in sizes if s <= len(ymp)))
lc = []
rng = np.random.RandomState(RNG)
for n in sizes:
    idx = rng.choice(len(ymp), size=n, replace=False)
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=RNG)
    m.fit(Xmp[idx], ymp[idx])
    p = m.predict(Xjv)
    met = metrics(yjv, p)
    met["n_train"] = n
    lc.append(met)
    log(f"  n_train={n}: {met}")
results["learning_curve_mp_to_jarvis"] = lc

# ---- Near-metallic internal-vs-external + electronic/ionic split ----
log("=== Near-metallic internal-vs-external, electronic/ionic split ===")
bg = jv_t["optb88vdw_bandgap"].astype(float).to_numpy()
near_metal = bg < 0.1
abs_res_ext = np.abs(pred_full - yjv)

Xtr_j, Xte_j, ytr_j, yte_j, idx_tr_j, idx_te_j = train_test_split(
    Xjv, yjv, np.arange(len(yjv)), test_size=0.2, random_state=RNG)
m_jint = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=RNG)
m_jint.fit(Xtr_j, ytr_j)
pred_jint = m_jint.predict(Xte_j)
abs_res_int = np.abs(pred_jint - yte_j)
near_metal_te = near_metal[idx_te_j]

mae_ext_metal = abs_res_ext[near_metal].mean()
mae_ext_nonmetal = abs_res_ext[~near_metal].mean()
mae_int_metal = abs_res_int[near_metal_te].mean() if near_metal_te.sum() > 0 else None
mae_int_nonmetal = abs_res_int[~near_metal_te].mean()
excess_metal = mae_ext_metal - (mae_int_metal or mae_ext_nonmetal)
excess_nonmetal = mae_ext_nonmetal - mae_int_nonmetal
frac_rows_metal = near_metal.mean() * 100
frac_err_metal = abs_res_ext[near_metal].sum() / abs_res_ext.sum() * 100
excess_err_total = abs_res_ext.sum() - (abs_res_int.mean() * len(abs_res_ext))
excess_err_metal_share = (abs_res_ext[near_metal].sum() - mae_int_metal * near_metal.sum()) / excess_err_total * 100 if excess_err_total > 0 and mae_int_metal else None

results["near_metallic"] = {
    "frac_rows_pct": float(frac_rows_metal), "frac_total_error_pct": float(frac_err_metal),
    "mae_external_metal": float(mae_ext_metal), "mae_external_nonmetal": float(mae_ext_nonmetal),
    "mae_internal_metal": float(mae_int_metal) if mae_int_metal else None,
    "mae_internal_nonmetal": float(mae_int_nonmetal),
    "excess_error_metal_vs_internal": float(excess_metal),
    "excess_error_nonmetal_vs_internal": float(excess_nonmetal),
    "share_of_excess_transfer_error_pct": float(excess_err_metal_share) if excess_err_metal_share else None,
}
log(f"  near-metal: external MAE={mae_ext_metal:.3f} internal MAE={mae_int_metal}, "
    f"non-metal: external MAE={mae_ext_nonmetal:.3f} internal MAE={mae_int_nonmetal:.3f}")

# ---- Extended benchmark: 8 models x 4 conditions, e_total ----
log("=== 8-model x 4-condition benchmark ===")
MODELS = {
    "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                     n_jobs=4, random_state=RNG),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=RNG),
    "SVR": lambda: SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "kNN": lambda: KNeighborsRegressor(n_neighbors=8, weights="distance"),
    "GaussianProcess": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True, random_state=RNG),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.001, max_iter=10000),
    "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 random_state=RNG, early_stopping=True),
}
NEEDS_SCALING = {"SVR", "kNN", "Ridge", "Lasso", "MLP", "GaussianProcess"}

Xtr_mp, Xte_mp, ytr_mp, yte_mp = train_test_split(Xmp, ymp, test_size=0.2, random_state=RNG)
scaler_mp = StandardScaler().fit(Xtr_mp)

GP_CAP = 1500  # GaussianProcess is O(n^3); cap training set size for tractability


def maybe_cap(name, X, y, seed=RNG):
    if name == "GaussianProcess" and len(y) > GP_CAP:
        rr = np.random.RandomState(seed)
        idx = rr.choice(len(y), GP_CAP, replace=False)
        return X[idx], y[idx]
    return X, y


full_bench = {}
for name, ctor in MODELS.items():
    needs = name in NEEDS_SCALING
    m1 = ctor()
    Xtr_c, ytr_c = maybe_cap(name, Xtr_mp, ytr_mp)
    Xtr_s = scaler_mp.transform(Xtr_c) if needs else Xtr_c
    Xte_s = scaler_mp.transform(Xte_mp) if needs else Xte_mp
    m1.fit(Xtr_s, ytr_c)
    p1 = m1.predict(Xte_s)
    mp_internal = metrics(yte_mp, p1)

    m2 = ctor()
    Xtr_j2, Xte_j2, ytr_j2, yte_j2 = train_test_split(Xjv, yjv, test_size=0.2, random_state=RNG)
    scaler_jv = StandardScaler().fit(Xtr_j2)
    Xtr_j2c, ytr_j2c = maybe_cap(name, Xtr_j2, ytr_j2)
    Xtr_j2s = scaler_jv.transform(Xtr_j2c) if needs else Xtr_j2c
    Xte_j2s = scaler_jv.transform(Xte_j2) if needs else Xte_j2
    m2.fit(Xtr_j2s, ytr_j2c)
    p2 = m2.predict(Xte_j2s)
    jv_internal = metrics(yte_j2, p2)

    scaler_full_mp = StandardScaler().fit(Xmp)
    m3 = ctor()
    Xmp_c, ymp_c = maybe_cap(name, Xmp, ymp)
    Xmp_s = scaler_full_mp.transform(Xmp_c) if needs else Xmp_c
    Xjv_s = scaler_full_mp.transform(Xjv) if needs else Xjv
    m3.fit(Xmp_s, ymp_c)
    p3 = m3.predict(Xjv_s)
    mp2jv = metrics(yjv, p3)

    scaler_full_jv = StandardScaler().fit(Xjv)
    m4 = ctor()
    Xjv_c, yjv_c = maybe_cap(name, Xjv, yjv)
    Xjv_s2 = scaler_full_jv.transform(Xjv_c) if needs else Xjv_c
    Xmp_s2 = scaler_full_jv.transform(Xmp) if needs else Xmp
    m4.fit(Xjv_s2, yjv_c)
    p4 = m4.predict(Xmp_s2)
    jv2mp = metrics(ymp, p4)

    full_bench[name] = {"mp_internal": mp_internal, "jarvis_internal": jv_internal,
                         "mp_to_jarvis": mp2jv, "jarvis_to_mp": jv2mp}
    log(f"  {name}: MPint R2={mp_internal['r2']:.3f} JVint R2={jv_internal['r2']:.3f} "
        f"MP->JV R2={mp2jv['r2']:.3f} JV->MP R2={jv2mp['r2']:.3f}")

with open(OUT / "benchmark_4conditions.json", "w") as f:
    json.dump(full_bench, f, indent=2, default=str)
results["benchmark_4conditions_summary"] = {
    k: {c: v["r2"] for c, v in d.items()} for k, d in full_bench.items()
}

best_mae = min(full_bench.items(), key=lambda kv: kv[1]["mp_to_jarvis"]["mae"])
log(f"  Lowest MP->JARVIS MAE: {best_mae[0]} = {best_mae[1]['mp_to_jarvis']['mae']:.4f} "
    f"(XGBoost = {full_bench['XGBoost']['mp_to_jarvis']['mae']:.4f}, "
    f"RandomForest = {full_bench['RandomForest']['mp_to_jarvis']['mae']:.4f})")
results["strongest_model_by_mae_mp2jv"] = best_mae[0]

# ---- Bootstrap 95% CIs (paired, on MP->JARVIS XGBoost predictions) ----
log("=== Bootstrap CIs ===")
n_boot = 2000
boot_rng = np.random.RandomState(RNG)
n = len(yjv)
maes, r2s = [], []
for _ in range(n_boot):
    idx = boot_rng.randint(0, n, n)
    maes.append(mean_absolute_error(yjv[idx], pred_full[idx]))
    r2s.append(r2_score(yjv[idx], pred_full[idx]))
maes, r2s = np.array(maes), np.array(r2s)
results["bootstrap_mp_to_jarvis"] = {
    "mae_ci95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
    "r2_ci95": [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))],
    "mae_mean": float(maes.mean()), "r2_mean": float(r2s.mean()),
}
log(f"  MP->JARVIS bootstrap: MAE 95% CI {results['bootstrap_mp_to_jarvis']['mae_ci95']}, "
    f"R2 95% CI {results['bootstrap_mp_to_jarvis']['r2_ci95']}")

rf_model = RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=RNG)
rf_model.fit(Xmp, ymp)
pred_rf_full = rf_model.predict(Xjv)
diffs = []
for _ in range(n_boot):
    idx = boot_rng.randint(0, n, n)
    mae_xgb = mean_absolute_error(yjv[idx], pred_full[idx])
    mae_rf = mean_absolute_error(yjv[idx], pred_rf_full[idx])
    diffs.append(mae_xgb - mae_rf)
diffs = np.array(diffs)
results["paired_bootstrap_xgb_vs_rf_mae_diff"] = {
    "mean_diff_xgb_minus_rf": float(diffs.mean()),
    "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
}
log(f"  XGBoost-RF paired MAE diff: {diffs.mean():.4f}, 95% CI {results['paired_bootstrap_xgb_vs_rf_mae_diff']['ci95']}")

# ---- Raw-unit metrics (back-transformed) ----
log("=== Raw-unit metrics ===")
true_raw = 10 ** yjv
pred_raw = 10 ** pred_full
ape = np.abs(pred_raw - true_raw) / true_raw * 100
within2x = (np.maximum(pred_raw, true_raw) / np.minimum(pred_raw, true_raw) <= 2.0).mean() * 100
results["raw_unit_metrics_mp_to_jarvis"] = {
    "median_ape_pct": float(np.median(ape)),
    "median_abs_error_raw": float(np.median(np.abs(pred_raw - true_raw))),
    "frac_within_2x_pct": float(within2x),
}
log(f"  median APE={np.median(ape):.1f}%, frac within 2x={within2x:.1f}%")

# ---- Band-gap residual analysis ----
log("=== Band-gap residual analysis ===")
res = pred_full - yjv
results["residual_analysis"] = {
    "median_residual": float(np.median(res)),
    "mad_residual": float(np.median(np.abs(res - np.median(res)))),
    "spearman_residual_vs_bandgap": float(pd.Series(res).corr(pd.Series(bg), method="spearman")),
}
log(f"  median residual={np.median(res):.3f}, MAD={results['residual_analysis']['mad_residual']:.3f}, "
    f"Spearman(residual,bandgap)={results['residual_analysis']['spearman_residual_vs_bandgap']:.3f}")

with open(OUT / "revision_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
log("DONE - saved models/crossdb/revision_analysis_results.json")
