# Figure generation for the dielectric cross-database ML manuscript.
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
MDIR = PROJ / "14th - Dielectric-CrossDB-ML-project"
DATA = MDIR / "data"
FIG = MDIR / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#1a9850"
GRAY = "#666666"
ORANGE = "#e08214"
PURPLE = "#762a83"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


mp = pd.read_parquet(PROJ / "data" / "mp_dielectric_features_dedup.parquet")
jv = pd.read_parquet(PROJ / "data" / "jarvis_dielectric_features.parquet")
metrics = json.load(open(PROJ / "models" / "crossdb" / "metrics.json"))
bench = json.load(open(PROJ / "models" / "crossdb" / "benchmark_models.json"))
pred_mp_internal = pd.read_csv(PROJ / "models" / "crossdb" / "predictions_mp_internal_etotal.csv")
pred_mp2jv = pd.read_csv(PROJ / "models" / "crossdb" / "predictions_mp_to_jarvis_etotal.csv")
pred_jv2mp = pd.read_csv(PROJ / "models" / "crossdb" / "predictions_jarvis_to_mp_etotal.csv")

for d in (DATA,):
    d.mkdir(exist_ok=True, parents=True)
mp.drop(columns=[c for c in mp.columns if c.startswith("MagpieData")]).to_csv(DATA / "mp_dielectric_summary.csv", index=False)
jv.drop(columns=[c for c in jv.columns if c.startswith("MagpieData")]).to_csv(DATA / "jarvis_dielectric_summary.csv", index=False)
json.dump(metrics, open(DATA / "metrics.json", "w"), indent=2)
json.dump(bench, open(DATA / "benchmark_models.json", "w"), indent=2)
pred_mp2jv.to_csv(DATA / "predictions_mp_to_jarvis_etotal.csv", index=False)


def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.bar(["Materials\nProject", "JARVIS-DFT"], [len(mp), len(jv)], color=[BLUE, RED])
    ax.set_ylabel("Number of compounds")
    ax.set_title("(a) Dataset size")
    for i, v in enumerate([len(mp), len(jv)]):
        ax.text(i, v + 150, str(v), ha="center", fontsize=10)

    ax = axes[1]
    bins = np.linspace(0, 3, 40)
    ax.hist(np.log10(mp["e_total"]), bins=bins, alpha=0.6, color=BLUE, label="MP (PBE/PBE+U)", density=True)
    ax.hist(np.log10(jv["e_total"]), bins=bins, alpha=0.6, color=RED, label="JARVIS (OptB88vdW)", density=True)
    ax.set_xlabel(r"$\log_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_ylabel("Probability density")
    ax.set_title("(b) Total dielectric constant")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.scatter(mp["e_electronic"], mp["e_ionic"], s=4, alpha=0.3, color=BLUE, label="MP")
    ax.scatter(jv["e_electronic"], jv["e_ionic"], s=4, alpha=0.3, color=RED, label="JARVIS")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_{\mathrm{electronic}}$")
    ax.set_ylabel(r"$\varepsilon_{\mathrm{ionic}} + 0.01$")
    ax.set_ylim(bottom=0.01)
    ax.set_title("(c) Electronic vs. ionic contribution")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save(fig, "fig1_dataset_overview")


def fig2():
    fig, ax = plt.subplots(figsize=(5.2, 5))
    t, p = pred_mp_internal["true_log10"], pred_mp_internal["pred_log10"]
    ax.scatter(t, p, s=6, alpha=0.3, color=BLUE)
    lims = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    m = metrics["e_total"]["mp_internal"]
    ax.text(0.05, 0.92, f"$R^2$ = {m['r2_log10']:.2f}\nMAE = {m['mae_log10']:.2f}",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlabel(r"True $\log_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_ylabel(r"Predicted $\log_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_title("MP-internal random 80/20 split\n(in-distribution reference)")
    fig.tight_layout()
    save(fig, "fig2_mp_internal_parity")


def fig3():
    targets = ["e_total", "e_electronic", "e_ionic"]
    labels = [r"$\varepsilon_{\mathrm{total}}$", r"$\varepsilon_{\mathrm{electronic}}$", r"$\varepsilon_{\mathrm{ionic}}$"]
    conditions = ["mp_internal", "jarvis_internal", "mp_to_jarvis", "jarvis_to_mp"]
    cond_labels = ["MP-internal", "JARVIS-internal", "MP→JARVIS", "JARVIS→MP"]
    colors = [GRAY, "#aaaaaa", BLUE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(targets))
    width = 0.2

    ax = axes[0]
    for i, (cond, lab, col) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [metrics[t][cond]["r2_log10"] for t in targets]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab, color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$R^2$ (log$_{10}$ scale)")
    ax.set_title("(a) Transfer accuracy by target")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    for i, (cond, lab, col) in enumerate(zip(conditions, cond_labels, colors)):
        vals = [metrics[t][cond]["mae_log10"] for t in targets]
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"MAE (log$_{10}$ scale)")
    ax.set_title("(b) Transfer error by target")

    fig.tight_layout()
    save(fig, "fig3_crossdb_transfer_summary")


def fig4():
    df = pred_mp2jv.copy()
    bg = df["optb88vdw_bandgap"].astype(float)
    abs_res = df["residual_log10"].abs()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    order = np.argsort(bg.to_numpy())
    bg_sorted = bg.to_numpy()[order]
    res_sorted = df["residual_log10"].to_numpy()[order]
    bins = [0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, bg.max() + 1]
    bin_labels = ["<0.1", "0.1-0.5", "0.5-1.0", "1.0-2.0", "2.0-3.0", "3.0-5.0", ">5.0"]
    bin_idx = np.digitize(bg, bins) - 1
    means = [abs_res[bin_idx == i].mean() for i in range(len(bin_labels))]
    counts = [np.sum(bin_idx == i) for i in range(len(bin_labels))]
    ax.bar(bin_labels, means, color=ORANGE)
    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, m + 0.01, f"n={c}", ha="center", fontsize=8)
    ax.axhline(metrics["e_total"]["mp_internal"]["mae_log10"], color="k", ls="--", lw=1,
               label="MP-internal MAE")
    ax.set_xlabel("OptB88vdW band gap bin (eV)")
    ax.set_ylabel(r"Mean $|$residual$|$, log$_{10}$ scale")
    ax.set_title("(a) Transfer error vs. band gap")
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    near_metal = bg < 0.1
    frac_rows = near_metal.mean() * 100
    frac_err = abs_res[near_metal].sum() / abs_res.sum() * 100
    cats = ["Share of\ntest compounds", "Share of total\nabsolute error"]
    vals = [frac_rows, frac_err]
    ax.bar(cats, vals, color=[GRAY, ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 40)
    ax.set_title("(b) Near-metal compounds (gap < 0.1 eV)\ndrive a disproportionate share of error")

    fig.tight_layout()
    save(fig, "fig4_functional_mismatch_bandgap")


def fig5():
    names = list(bench.keys())
    r2 = [bench[n]["r2_log10"] for n in names]
    order = np.argsort(r2)[::-1]
    names = [names[i] for i in order]
    r2 = [r2[i] for i in order]
    colors = [BLUE if n in ("XGBoost", "RandomForest") else GRAY for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names[::-1], r2[::-1], color=colors[::-1])
    ax.axvline(metrics["e_total"]["mp_internal"]["r2_log10"], color="k", ls="--", lw=1,
               label="MP-internal baseline")
    ax.set_xlabel(r"$R^2$, MP $\to$ JARVIS, log$_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_title("Transfer degradation persists across\ncomposition-only models")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_multimodel_benchmark")


def fig6():
    df = pred_mp2jv.copy()
    res = df["residual_log10"]
    corrected = res - res.mean()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(-1.2, 1.2, 50)
    ax.hist(res, bins=bins, alpha=0.6, color=BLUE, label=f"Raw residual\nMAE={res.abs().mean():.3f}")
    ax.hist(corrected, bins=bins, alpha=0.6, color=RED,
            label=f"After removing\nmean offset\nMAE={corrected.abs().mean():.3f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"Residual, $\log_{10}(\varepsilon_{\mathrm{total}})$ (predicted $-$ true)")
    ax.set_ylabel("Count")
    ax.set_title("A constant offset does not explain\nthe MP→JARVIS transfer gap")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_offset_correction_test")


def figs1():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    ax.hist(mp["e_total"].clip(upper=200), bins=60, color=BLUE)
    ax.set_xlabel(r"$\varepsilon_{\mathrm{total}}$ (clipped at 200)")
    ax.set_ylabel("Count")
    ax.set_title("(a) MP raw $\\varepsilon_{\\mathrm{total}}$ distribution\n(before log transform)")

    ax = axes[1]
    ax.hist(mp["MagpieData mean Electronegativity"], bins=50, color=GREEN)
    ax.set_xlabel("Magpie mean electronegativity")
    ax.set_title("(b) Representative composition feature\n(MP training set)")

    ax = axes[2]
    from pymatgen.core import Composition
    n_elem = mp["formula_pretty"].apply(lambda f: len(Composition(f).elements))
    ax.hist(n_elem, bins=range(1, int(n_elem.max()) + 2), color=PURPLE, align="left")
    ax.set_xlabel("Number of distinct elements (compositional complexity)")
    ax.set_title("(c) MP compositional complexity")
    fig.tight_layout()
    save(fig, "figs1_mp_raw_distributions")


def figs2():
    jv_t = jv.dropna(subset=["e_total"]).reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    feat_cols = [c for c in jv_t.columns if c.startswith("MagpieData")]
    X = jv_t[feat_cols].fillna(jv_t[feat_cols].mean()).to_numpy(dtype=float)
    y = np.log10(jv_t["e_total"].to_numpy(dtype=float))
    from xgboost import XGBRegressor
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=4, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(yte, pred, s=8, alpha=0.4, color=RED)
    lims = [min(yte.min(), pred.min()) - 0.1, max(yte.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    m = metrics["e_total"]["jarvis_internal"]
    ax.text(0.05, 0.92, f"$R^2$ = {m['r2_log10']:.2f}\nMAE = {m['mae_log10']:.2f}",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlabel(r"True $\log_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_ylabel(r"Predicted $\log_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_title("JARVIS-internal random 80/20 split\n(in-distribution reference, smaller N)")
    fig.tight_layout()
    save(fig, "figs2_jarvis_internal_parity")


def figs3():
    fig, ax = plt.subplots(figsize=(5.2, 5))
    t, p = pred_jv2mp["true_log10"], pred_jv2mp["pred_log10"]
    ax.scatter(t, p, s=4, alpha=0.2, color=GREEN)
    lims = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    m = metrics["e_total"]["jarvis_to_mp"]
    ax.text(0.05, 0.92, f"$R^2$ = {m['r2_log10']:.2f}\nMAE = {m['mae_log10']:.2f}",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.set_xlabel(r"True $\log_{10}(\varepsilon_{\mathrm{total}})$ (MP)")
    ax.set_ylabel(r"Predicted $\log_{10}(\varepsilon_{\mathrm{total}})$")
    ax.set_title("Reverse direction: trained on JARVIS,\nzero-shot tested on full MP")
    fig.tight_layout()
    save(fig, "figs3_jarvis_to_mp_parity")


def figs4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, target, label in zip(axes, ["e_electronic", "e_ionic"],
                                 [r"$\varepsilon_{\mathrm{electronic}}$", r"$\varepsilon_{\mathrm{ionic}}$"]):
        conds = ["mp_internal", "jarvis_internal", "mp_to_jarvis", "jarvis_to_mp"]
        cond_labels = ["MP-int.", "JARVIS-int.", "MP→JARVIS", "JARVIS→MP"]
        vals = [metrics[target][c]["r2_log10"] for c in conds]
        colors = [GRAY, "#aaaaaa", BLUE, RED]
        ax.bar(cond_labels, vals, color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(r"$R^2$ (log$_{10}$ scale)")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "figs4_secondary_targets_transfer")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    figs1(); figs2(); figs3(); figs4()
    print("ALL FIGURES DONE")
