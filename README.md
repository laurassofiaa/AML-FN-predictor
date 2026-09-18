# Predicting Febrile Neutropenia in AML Patients Undergoing Chemotherapy

A machine learning pipeline for binary classification of febrile neutropenia (FN) events during intensive chemotherapy for acute myeloid leukaemia (AML), using XGBoost with SHAP-based feature selection and interpretability analysis.

## Repository Structure

| File | Description |
|------|-------------|
| `Describing_variables_and_time-to-event-analyses.R` | Descriptive analyses and time-to-event analyses before model development |
| `AML_infection_model_data.ipynb` | Data preprocessing and preparation for modelling |
| `AML_XGB_classifier_all_variables.ipynb` | XGBoost classifier trained on all features; creates the train/test split used downstream |
| `SHAP_contributions.R` | Computes SHAP contributions for the all-variable model and selects features above the SHAP threshold |
| `AML_XGB_classifier_SHAP_top_variables.ipynb` | Refined XGBoost classifier using the SHAP-selected features, with evaluation, decision curve analysis and regression to event days |
| `TOP_SHAP_XGBClassifier_plot_function.R` | SHAP summary plots for the top-variable model |
| `TOPXGB_SHAP_wrongly_classified_density_test.R` | SHAP analysis of misclassified test-set observations, with density plots |
| `external_validation/test_models.ipynb` | Evaluation of the refined model on the external validation cohort |
| `external_validation/TOPXGB_SHAP_wrongly_classified_density_test.R` | SHAP analysis of misclassified observations in the external cohort |

## Running Order

### Stage 0 -- Analyses before model development

0. **`Describing_variables_and_time-to-event-analyses.R`** -- Baseline characteristics, figures and time-to-event analyses.

### Stage 1 -- Data preparation

1. **`AML_infection_model_data.ipynb`** -- Preprocess raw data and export model-ready datasets (`model_data_*.csv`).

### Stage 2 -- Full model (all variables)

2. **`AML_XGB_classifier_all_variables.ipynb`** -- Train and evaluate XGBoost with all features. Saves the data splits, `xgb_model.json` and `features.json`.
3. **`SHAP_contributions.R`** -- Compute SHAP values and write `SHAP_selected_features_over_0.01.csv`.

### Stage 3 -- Refined model (top SHAP-selected variables)

4. **`AML_XGB_classifier_SHAP_top_variables.ipynb`** -- Train and evaluate XGBoost with the SHAP-selected features. Saves `simple_xgb_model.json`, `simple_features.json`, `X_test_drop.csv` and `eval_df.csv`.
5. **`TOP_SHAP_XGBClassifier_plot_function.R`** -- Generate SHAP summary plots.
6. **`TOPXGB_SHAP_wrongly_classified_density_test.R`** -- Analyse SHAP values of misclassified observations.

### Stage 4 -- External validation

The refined model is tested on an external cohort. These files are in the `external_validation/` folder and use the model saved in Stage 3 (`simple_xgb_model.json`, `simple_features.json`).

7. **`external_validation/test_models.ipynb`** -- Predict on the external cohort and evaluate: AUROC and AUPRC with 95% CI, calibration, decision curve analysis and induction vs. consolidation. Saves `x_test.csv` and `eval_df.csv`.
8. **`external_validation/TOPXGB_SHAP_wrongly_classified_density_test.R`** -- SHAP analysis of misclassified observations in the external cohort.

## Requirements

**Python** (notebooks): `pandas`, `numpy`, `matplotlib`, `seaborn`, `xgboost`, `scikit-learn`, `imblearn`, `scipy`, `statsmodels`, `scikit-optimize`

**R** (scripts): `xgboost`, `SHAPforxgboost`, `data.table`, `dplyr`, `ggplot2`, `jsonlite`, `scales`, `patchwork`, `cli`

## Data

The patient data are not included in this repository. All file paths in the code use placeholder paths (`/path/to/...`). Update these to point to your local data directory before running.
