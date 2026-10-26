# Plots R2 for all 8 models under all 4 evaluation conditions (internal/external, both directions).
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path("/glade/work/ajalali/14 - Dielectric-CrossDB-ML-project")
MDIR = PROJ / "14th - Dielectric-CrossDB-ML-project"
FIG = MDIR / "figures"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                      "axes.spines.right": False, "figure.dpi": 150})

GRAY = "#aaaaaa"
GRAY2 = "#666666"
BLUE = "#2166ac"
RED = "#b2182b"

bench = json.load(open(PROJ / "models" / "crossdb" / "benchmark_4conditions.json"))

order = ["XGBoost", "RandomForest", "SVR", "kNN", "GaussianProcess", "Ridge", "Lasso", "MLP"]
conds = ["mp_internal", "jarvis_internal", "mp_to_jarvis", "jarvis_to_mp"]
cond_labels = ["MP-internal", "JARVIS-internal", "MP→JARVIS", "JARVIS→MP"]
colors = [GRAY, GRAY2, BLUE, RED]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(order))
w = 0.2
for i, (c, lab, col) in enumerate(zip(conds, cond_labels, colors)):
    vals = [bench[m][c]["r2"] for m in order]
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col)

ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=20, ha="right")
ax.set_ylabel("$R^2$")
ax.set_title("Every model's cross-database $R^2$ falls well below its own\n"
             "in-distribution $R^2$ (GaussianProcess capped at n=1500 train)")
ax.legend(frameon=False, fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(FIG / "fig7_fourcondition_benchmark.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig7_fourcondition_benchmark.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("saved fig7_fourcondition_benchmark")
