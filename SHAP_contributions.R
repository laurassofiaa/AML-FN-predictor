

# Libraries
library(xgboost)
library(SHAPforxgboost)
library(data.table)

FOLDER_TAG <- "NOimputation" 
TRAIN_TAG <- "train2_test2"

# Root once:
ROOT <- "/path/to"

# Paths derived from the tag:
DATA_PATH <- file.path(ROOT, "data", TRAIN_TAG, FOLDER_TAG)

# Load the model
model1 <- xgb.load(file.path(DATA_PATH, "xgb_model.json"))

x_train <- data.table::fread(file.path(DATA_PATH, "X_train.csv"))[, -1, with = FALSE]
x_train = as.matrix(x_train)

# SHAP
shap_values <- shap.values(xgb_model = model1, X_train = x_train)

# shap features over 0.1 to csv file
THRESHOLD <- 0.01

# 1) Use the provided mean |SHAP| vector
ms <- shap_values$mean_shap_score                 # named numeric vector
sel <- sort(ms[ms > THRESHOLD], decreasing = TRUE)

# a) just feature names
fwrite(data.table(feature = names(sel)),
       file.path(DATA_PATH, sprintf("SHAP_selected_features_over_%g.csv", THRESHOLD)))

