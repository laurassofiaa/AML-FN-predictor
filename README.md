# Predicting Febrile Neutropenia in AML Patients Undergoing Chemotherapy

A machine learning pipeline for binary classification of febrile neutropenia (FN) events during intensive chemotherapy for acute myeloid leukaemia (AML), using XGBoost with SHAP-based feature selection and interpretability analysis.

## Repository Structure

| File | Description |
|------|-------------|
| `Describing_variables_and_time-to-event-analyses.R` | Descriptive analyses and time-to-event analyses before model development |
| `01_data_preprocessing.ipynb` | Data preprocessing and preparation for modelling |
| `02_model_all_variables.ipynb` | XGBoost classifier trained on all features; creates the train/test split used downstream |
| `03_SHAP_feature_selection.R` | Computes SHAP contributions for the all-variable model and selects features above the SHAP threshold |
| `04_model_top_SHAP_variables.ipynb` | Refined XGBoost classifier using the SHAP-selected features, with evaluation, decision curve analysis and regression to event days |
| `05_SHAP_feature_importance_plots.R` | Mean SHAP contribution plots for the refined model (all, induction and consolidation) |
| `06_SHAP_misclassified_density_plots.R` | SHAP analysis of misclassified test-set observations, with density plots |
| `external_validation/07_external_validation.ipynb` | Evaluation of the refined model on the external validation cohort |
| `external_validation/08_external_SHAP_misclassified_density_plots.R` | SHAP analysis of misclassified observations in the external cohort |

## Running Order

### Stage 0 -- Analyses before model development

0. **`Describing_variables_and_time-to-event-analyses.R`** -- Baseline characteristics, figures and time-to-event analyses.

### Stage 1 -- Data preparation

1. **`01_data_preprocessing.ipynb`** -- Preprocess raw data and export model-ready datasets (`model_data_*.csv`).

### Stage 2 -- Full model (all variables)

2. **`02_model_all_variables.ipynb`** -- Train and evaluate XGBoost with all features. Saves the data splits, `xgb_model.json` and `features.json`.
3. **`03_SHAP_feature_selection.R`** -- Compute SHAP values and write `SHAP_selected_features_over_0.01.csv`.

### Stage 3 -- Refined model (top SHAP-selected variables)

4. **`04_model_top_SHAP_variables.ipynb`** -- Train and evaluate XGBoost with the SHAP-selected features. Saves `simple_xgb_model.json`, `simple_features.json`, `X_test_drop.csv` and `eval_df.csv`.
5. **`05_SHAP_feature_importance_plots.R`** -- Generate SHAP summary plots.
6. **`06_SHAP_misclassified_density_plots.R`** -- Analyse SHAP values of misclassified observations.

### Stage 4 -- External validation

The refined model is tested on an external cohort. These files are in the `external_validation/` folder and use the model saved in Stage 3 (`simple_xgb_model.json`, `simple_features.json`).

7. **`external_validation/07_external_validation.ipynb`** -- Predict on the external cohort and evaluate: AUROC and AUPRC with 95% CI, calibration, decision curve analysis and induction vs. consolidation. Saves `x_test.csv` and `eval_df.csv`. Set `EXCLUDED = True` to run the analysis for patients without antibiotic exposure.
8. **`external_validation/08_external_SHAP_misclassified_density_plots.R`** -- SHAP analysis of misclassified observations in the external cohort.

## Figures and Tables

Each figure is labelled in the code where it is created (a comment in R scripts, a markdown cell in notebooks).

| Figure / Table | Created in |
|----------------|------------|
| Figure 2A-H | `Describing_variables_and_time-to-event-analyses.R` |
| Figure 3A-F | `04_model_top_SHAP_variables.ipynb` |
| Figure 3G | `05_SHAP_feature_importance_plots.R` |
| Figure 4A-G | `06_SHAP_misclassified_density_plots.R` |
| Figure 5A-D | `04_model_top_SHAP_variables.ipynb` |
| Figure 6A-D | `external_validation/07_external_validation.ipynb` (`EXCLUDED = False`) |
| Supplementary Figures 1-6 | `Describing_variables_and_time-to-event-analyses.R` |
| Supplementary Figure 7A-D | `02_model_all_variables.ipynb` |
| Supplementary Figure 8 | `04_model_top_SHAP_variables.ipynb` |
| Supplementary Figure 9A-B | `04_model_top_SHAP_variables.ipynb` |
| Supplementary Figure 9C-D | `05_SHAP_feature_importance_plots.R` |
| Supplementary Figure 10A-C | `external_validation/07_external_validation.ipynb` (`EXCLUDED = False`) |
| Supplementary Figure 11A-G | `external_validation/08_external_SHAP_misclassified_density_plots.R` |
| Supplementary Figure 12A-D | `external_validation/07_external_validation.ipynb` (`EXCLUDED = True`) |
| Supplementary Table 3 | `02_model_all_variables.ipynb` |
| Supplementary Table 4 | `04_model_top_SHAP_variables.ipynb` and `external_validation/07_external_validation.ipynb` |

## Requirements

**Python** (notebooks): `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn`, `imblearn`, `scipy`, `statsmodels`, `scikit-optimize`

**R** (scripts): `xgboost`, `SHAPforxgboost`, `data.table`, `dplyr`, `ggplot2`, `jsonlite`, `scales`, `patchwork`, `cli`

## Data

The patient data are not included in this repository. All file paths in the code use placeholder paths (`/path/to/...`). Update these to point to your local data directory before running.
