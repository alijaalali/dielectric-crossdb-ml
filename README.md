# Cross-database ML for dielectric constant prediction (Materials Project vs. JARVIS-DFT)

Code, raw data and result files behind the manuscript "Cross-Database Transfer Limits for Machine-Learning Prediction of Dielectric Constants: From the Materials Project to JARVIS-DFT".

## Layout

- `data/` raw retrievals and processed summary tables (inputs).
- `outputs/` result files (JSON/CSV) written by the analysis scripts, as used in the manuscript and Supplementary Material.
- `figures/` figure files written by the figure scripts (not tracked; regenerate with the scripts below).
- `*.py` analysis scripts. All paths are relative to this directory, so the scripts run from any working directory.

## Requirements

Python 3.11, pandas, numpy, scikit-learn, xgboost, matminer, pymatgen, jarvis-tools, matplotlib, pyarrow. The retrieval scripts (not included) used `mp-api` and `jarvis-tools`; the retrieved raw files are provided in `data/`.

## Run order

Run the scripts from this directory in this order. Each script reads the files named in the previous steps and writes to `data/` or `outputs/`.

1. `make_dedup_raw_data.py` writes `data/mp_dielectric_raw_dedup.json` (removes the exact duplicate rows of the Materials Project retrieval).
2. `build_tables.py` builds the Magpie feature tables `data/*_features*.parquet` (regenerated locally, not tracked).
3. `train_crossdb.py` writes `outputs/metrics.json` and the prediction files.
4. `benchmark_models.py` writes `outputs/benchmark_models.json`.
5. `combined_analysis.py` writes `outputs/combined_analysis_results.json`, `outputs/benchmark_4conditions.json`, `outputs/outlier_filter_sensitivity.json`, `outputs/overlap_paired_compounds.csv`.
6. `deduplicate_mp_dielectric.py`, `composition_grouped_cv.py`, `learning_curve_multiseed.py`, `single_model_internal_external.py`, `offset_calibration_split.py`, `polymorph_matching.py`, `overlap_components.py`.
7. `nearmetal_metadata.py`, then `nearmetal_components.py`, `nearmetal_paired_bootstrap.py`, `threshold_sensitivity.py`, `mbj_excluded_bootstrap.py`, `nearmetal_table.py`.
8. `mp_structure_partial.py`, `jarvis_structure_check.py`.
9. `make_figures.py`, `make_figure_condition_comparison.py`.

## Which script produces which table and figure

| Item in the paper | Script | Output |
|---|---|---|
| Table 2 (transfer results), Table S1 | `train_crossdb.py`; composition-grouped row from `composition_grouped_cv.py` | `metrics.json`, `composition_grouped_cv.json` |
| Section 4.1 leakage from duplicate rows | `deduplicate_mp_dielectric.py` | `dedup_fix.json` |
| Section 4.2 composition overlap, polymorph check | `combined_analysis.py`, `polymorph_matching.py` | `combined_analysis_results.json`, `polymorph_matching.json` |
| Section 4.3, Table S2 outlier-filter sensitivity | `combined_analysis.py` | `outlier_filter_sensitivity.json` |
| Section 4.4 learning curve | `learning_curve_multiseed.py` | `learning_curve_multiseed.json` |
| Section 4.4 single matched model, Table S8 | `single_model_internal_external.py` | `single_model_internal_external.json` |
| Section 4.5 electronic/ionic comparison | `overlap_components.py` | `overlap_components.json` |
| Section 4.6 offset correction | `offset_calibration_split.py`, `combined_analysis.py` | `offset_calibration_split.json` |
| Section 4.7, Table S3 | `nearmetal_components.py` | `nearmetal_components.json` |
| Section 4.7, Table S4 | `nearmetal_paired_bootstrap.py` | `nearmetal_paired_bootstrap.json` |
| Table S5 | `threshold_sensitivity.py` | `threshold_sensitivity.json` |
| Table S6 | `mbj_excluded_bootstrap.py` | `mbj_excluded_bootstrap.json` |
| Section 4.7 metadata, Table S7 | `nearmetal_metadata.py`, `nearmetal_table.py` | `nearmetal_metadata.json`, `nearmetal_top10_table.csv` |
| Table 3 (eight-model benchmark) | `benchmark_models.py` | `benchmark_models.json` |
| Table 4 (four conditions), Section 4.10 bootstrap | `combined_analysis.py` | `benchmark_4conditions.json`, `combined_analysis_results.json` |
| Section 5.2 structure-aware checks | `mp_structure_partial.py`, `jarvis_structure_check.py` | `mp_structure_partial.json`, `jarvis_structure_test.json` |
| Figures 1-5, Figures S1-S5 | `make_figures.py` | `figures/fig2_*` (Figure 1), `fig3_*` (2), `fig6_*` (3), `fig4_*` (4), `fig5_*` (5), `fig1_*` (S1), `figs1_*`-`figs4_*` (S2-S5) |
| Figure 6 | `make_figure_condition_comparison.py` | `figures/fig7_fourcondition_benchmark.pdf` |

## Data

`data/mp_dielectric_raw.json` is the undeduplicated Materials Project dielectric-endpoint output (14,664 rows), `data/mp_dielectric_raw_dedup.json` the deduplicated file (7,332 compounds), `data/jarvis_dielectric_raw.json` the JARVIS-DFT `dft_3d` entries with a DFPT dielectric constant (4,706), and `data/*_summary.csv` the processed tables (without the Magpie columns).

### Attribution and licence of the data

- Materials Project data: retrieved from the Materials Project dielectric-properties endpoint with `mp-api` on 2026-10-14. The Materials Project is openly accessible and its data are distributed under the Creative Commons Attribution 4.0 licence (CC BY 4.0); please cite A. Jain et al., APL Mater. 1, 011002 (2013) and I. Petousis et al., Sci. Data 4, 160134 (2017).
- JARVIS-DFT data: obtained with `jarvis-tools` (`dft_3d` dataset) on 2026-10-14. JARVIS-DFT is developed and openly distributed by the National Institute of Standards and Technology (NIST); please cite K. Choudhary et al., npj Comput. Mater. 6, 173 (2020) and K. Choudhary et al., npj Comput. Mater. 6, 64 (2020).
- The files in `data/` contain only the fields used in the manuscript, redistributed here with attribution to their original sources; the original databases remain the authoritative versions.

## Licence

The code is released under the MIT licence (see `LICENSE`). The data keep the licences of their sources, described above.
