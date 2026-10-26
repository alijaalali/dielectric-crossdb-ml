# Cross-database ML for dielectric constant prediction (Materials Project vs. JARVIS-DFT)

Scripts used to build features, train models, and generate figures/tables for the manuscript.

## Pipeline order

1. `make_dedup_raw_data.py` — deduplicate raw MP retrieval
2. `build_tables.py` — build Magpie composition feature tables
3. `train_crossdb.py` — core cross-database transfer test (e_total, e_electronic, e_ionic)
4. `benchmark_models.py` — multi-regressor benchmark (MP -> JARVIS)
5. `make_figures.py` — manuscript figures
6. `make_figure_condition_comparison.py` — 8-model x 4-condition figure

## Supporting analyses

- `deduplicate_mp_dielectric.py`, `jarvis_structure_check.py`, `mp_structure_partial.py`, `polymorph_matching.py` — data-integrity and structure-feature checks
- `composition_grouped_cv.py`, `single_model_internal_external.py`, `learning_curve_multiseed.py` — split/leakage and training-size controls
- `nearmetal_components.py`, `nearmetal_paired_bootstrap.py`, `nearmetal_metadata.py`, `nearmetal_table.py`, `mbj_excluded_bootstrap.py`, `threshold_sensitivity.py` — near-metallic transfer-penalty analysis
- `overlap_components.py` — electronic/ionic comparison for composition-overlapping compounds
- `offset_calibration_split.py` — constant-offset correction test
- `revision_analysis.py` — combined analysis bundle (overlap, outlier sensitivity, learning curve, near-metallic split, extended benchmark, bootstrap CIs, raw-unit metrics, residual analysis)

## Requirements

Python 3, pandas, numpy, scikit-learn, xgboost, matminer, pymatgen, jarvis-tools, matplotlib.

## Data

Raw data (Materials Project and JARVIS-DFT dielectric datasets) are not included here; scripts expect them in a local `data/` directory. See the manuscript's Data Availability statement.
