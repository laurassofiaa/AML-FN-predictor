library(data.table)
library(xgboost)
library(SHAPforxgboost)
library(jsonlite)

# ----------------------------
# Config
# ----------------------------
FOLDER_TAG <- "NOimputation"
TRAIN_TAG  <- "train2_test2"

ROOT <- "/path/to"

DATA_PATH <- file.path(ROOT, "data",  TRAIN_TAG, FOLDER_TAG)
PLOT_PATH <- file.path(ROOT, "plots", TRAIN_TAG, FOLDER_TAG, "SHAP_TOP")
if (!dir.exists(PLOT_PATH)) dir.create(PLOT_PATH, recursive = TRUE)

# ---------------------------------------------------------------------------
# Source-data export for figures (journal request). Base R only.
# ---------------------------------------------------------------------------
PLOT_DATA_PATH <- file.path(ROOT, "plots", TRAIN_TAG, FOLDER_TAG, "plot_data")
dir.create(PLOT_DATA_PATH, recursive = TRUE, showWarnings = FALSE)

write_plot_data_one <- function(stem, df, out_dir = PLOT_DATA_PATH, note = "") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, paste0(stem, ".csv"))
  # NB: plain write.csv is correct. fileEncoding="UTF-8" truncates non-ASCII
  # under a non-UTF-8 locale, and enc2utf8() double-encodes it.
  write.csv(df, path, row.names = FALSE, na = "")
  if (any(grepl("<U\\+[0-9A-Fa-f]{4}>", readLines(path, warn = FALSE))))
    warning("plot_data: labels like °C / 10³/L were escaped; check Sys.getlocale('LC_CTYPE')")
  
  idx <- file.path(out_dir, "plot_data_index.csv")
  new <- data.frame(figure = stem, part = "", file = paste0(stem, ".csv"),
                    n_rows = as.character(nrow(df)),
                    columns = paste(names(df), collapse = "; "),
                    note = note, stringsAsFactors = FALSE)
  if (file.exists(idx)) {
    old <- utils::read.csv(idx, colClasses = "character")
    old <- old[old$figure != stem, , drop = FALSE]     # rerun replaces
    new <- rbind(old[, names(new), drop = FALSE], new)
  }
  write.csv(new, idx, row.names = FALSE, na = "")
  message("[plot_data] ", stem, ".csv")
  invisible(path)
}

# ----------------------------
# Load model
# ----------------------------
xgb_mod <- xgb.load(file.path(DATA_PATH, "simple_xgb_model.json"))

# ----------------------------
# Helper: align matrix to model features safely
# ----------------------------
align_to_model <- function(X, xgb_mod) {
  X <- as.matrix(X)
  if (is.null(colnames(X))) stop("X must have colnames().")
  colnames(X) <- make.names(colnames(X), unique = TRUE)
  
  mod_feats_str <- xgboost::xgb.attr(xgb_mod, "feature_names")
  if (!is.null(mod_feats_str) && nzchar(mod_feats_str)) {
    mod_feats <- strsplit(mod_feats_str, ",")[[1]]
    mod_feats <- make.names(mod_feats, unique = TRUE)
    
    missing <- setdiff(mod_feats, colnames(X))
    if (length(missing) > 0) stop("X is missing model features: ", paste(missing, collapse = ", "))
    
    X <- X[, mod_feats, drop = FALSE]
  } else {
    # store safely (DO NOT use xgb_mod$feature_names <- ...)
    xgboost::xgb.attr(xgb_mod, "feature_names") <- paste(colnames(X), collapse = ",")
  }
  
  X
}

# ----------------------------
# SHAP function (name_map + colored barplot)
# ----------------------------
analyze_shap_group <- function(X_mat, group_label, out_dir_base, xgb_mod, top_n = 10,
                               plot_data = FALSE) {  
  # Align to model (this also drops extra columns safely)
  X_mat <- align_to_model(X_mat, xgb_mod)
  
  # Compute SHAP values
  shap_out <- SHAPforxgboost::shap.values(xgb_model = xgb_mod, X_train = X_mat)
  
  dt_mean <- data.table(
    feature   = names(shap_out$mean_shap_score),
    mean_shap = 100 * shap_out$mean_shap_score / sum(shap_out$mean_shap_score)
  )[order(-mean_shap)]
  
  top_n <- min(top_n, nrow(dt_mean))
  top_feats <- dt_mean$feature[seq_len(top_n)]
  
  # Remove BIAS if present
  shap_score <- shap_out$shap_score
  if (!is.null(colnames(shap_score)) && "BIAS" %in% colnames(shap_score)) {
    shap_score <- shap_score[, setdiff(colnames(shap_score), "BIAS"), drop = FALSE]
  }
  
  # Slice SHAP and X
  shap_score_sub_mat <- as.matrix(shap_score)[, top_feats, drop = FALSE]
  X_mat_sub <- X_mat[, top_feats, drop = FALSE]
  
  shap_score_sub_df <- as.data.frame(shap_score_sub_mat)
  X_df_sub <- as.data.frame(X_mat_sub)
  
  # Prepare for beeswarm
  shap_long <- SHAPforxgboost::shap.prep(
    shap_contrib = shap_score_sub_df,
    X_train      = X_df_sub
  )
  
  # Create output directory
  out_dir <- file.path(out_dir_base, group_label)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  vals <- dt_mean$mean_shap[seq_len(top_n)]
  labs <- dt_mean$feature[seq_len(top_n)]

  
  # ----------------------------
  # Plot 3: Colored barplot + name_map
  # ----------------------------
  name_map <- c(
      "temperature"                       = "Body temperature (°C)",
      "b_neut"                            = "ANC (10⁹/L)",
      "b_leuk"                            = "WBC (10⁹/L)",
      "temperature_trend_from_2_d"        = "Temperature \n2-day trend (°C)",
      'b_leuk_trend_from_7_d'             = "WBC 7-day trend (10⁹/L)",
      'p_crp_trend_from_3_d'              = "1–3-day CRP trend (mg/L)",
      'sykli_IND'                         = "Induction phase",
      "bneut_accumulated_max_050_cycle"   = "Duration of ANC (≤0.5 10⁹/L)\nper cycle (days)",
      "bneut_accumulated_max_050_total"   = "Duration of ANC (≤0.5 10⁹/L)\nper regimen (days)",
      "bneut_accumulated_max_005_total"   = "Duration of ANC (≤0.05 10⁹/L)\nper regimen (days)",
      "countdown"                         = "Time since \ncycle start (days)",
      "cycle_number"                      = "Treatment cycle number",
      
      "e_mcv_mean_last_values_7_days"     = "4–7-day mean MCV (fL)",
      "p_tt_mean_last_values_7_days"      = "Tromboplastin time mean 7-day trend (%)",
      "p_crp_trend_from_7_d"              = "4–7-day CRP trend (mg/L)",
      "p_afos_mean_last_values_7_days"    = "ALP mean 7-day trend (U/L)",
      "p_k_mean_last_values_7_days"       = "K mean 7-day trend (mmol/L)",
      "bm_blast_p"                        = "BM blast (%)",
      "b_trom"                            = "PLT (10⁹/L)",
      "b_trom_trend_from_7_d"             = "PLT 7-day trend (10⁹/L)",
      "p_tt_mean_last_values_3_days"      = "Tromboplastin time mean 3-day trend (%)",
      "b_trom_mean_last_values_7_days"    = "4–7-day PLT trend (10⁹/L)",
      "b_neut_mean_last_values_7_days"    = "ANC mean 7-day trend (10⁹/L)",
      "b_eryt_mean_last_values_3_days"    = "RBC mean 3-day trend (10$^12$/L)",
      "b_ly_trend_from_7_d"               = "Lymphocytes 7-day trend (10⁹/L)",
      
      "ab_max_10d"                        = 'Ab initiated within\nthe last 10 days'
    )
    
  # direction colors: cor(feature, shap)
  cor_vals <- sapply(top_feats, function(f) {
    suppressWarnings(cor(X_df_sub[[f]], shap_score_sub_df[[f]], use = "complete.obs"))
  })
  bar_colors <- ifelse(is.na(cor_vals), "gray",
                       ifelse(cor_vals < 0, "#084594", "#6BAED6"))
  
  labs_pretty <- ifelse(top_feats %in% names(name_map), name_map[top_feats], top_feats)
  
  if (group_label == "IND") {
    title_label <- "Top Feature Importance for Inductions"
  } else if (group_label == "KONS") {
    title_label <- "Top Feature Importance for Consolidations"
  } else {
    title_label <- "Top Feature Importance (simple XGB)"
  }
  
  png(filename = file.path(out_dir, "Mean_SHAP_percent_colored.png"),
      width = 12, height = 10, units = "in", res = 600)
  
  old_par <- par(mar = c(11.5, 5, 4, 0) + 0.1)
  
  bp <- barplot(
    height = vals,
    names.arg = rep("", length(vals)),
    ylab = "SHAP Contribution (%)",
    main = title_label,
    col = bar_colors,
    xaxt = "n",
    ylim = c(0, max(vals) * 1.15),
    cex.main = 2.0,
    cex.lab  = 1.8,
    cex.axis = 1.5
  )
  
  text(
    x = bp,
    y = vals + max(vals) * 0.02,
    labels = sprintf("%.1f%%", vals),
    pos = 3,
    cex = 1.7
  )
  
  axis(1, at = bp, labels = FALSE)
  text(
    x = bp,
    y = par("usr")[3] - 0.04 * max(vals),
    labels = labs_pretty,
    srt = 45, adj = 0.99, xpd = TRUE, cex = 1.5
  )
  
  par(old_par)
  dev.off()
  
  # ---- source data for the journal --------------------------------------
  if (isTRUE(plot_data)) {
    write_plot_data_one(
      paste0("Mean_SHAP_percent_colored_", group_label),
      data.frame(
        rank                  = seq_along(vals),
        feature               = labs,
        label                 = gsub("\\s*\n\\s*", " ", labs_pretty),
        shap_contribution_pct = as.numeric(vals),
        bar_label             = sprintf("%.1f%%", vals),
        direction_cor         = as.numeric(cor_vals),
        direction             = ifelse(is.na(cor_vals), "undefined",
                                       ifelse(cor_vals < 0, "negative", "positive")),
        bar_colour            = as.character(bar_colors),
        stringsAsFactors      = FALSE
      ),
      note = paste0("group=", group_label, "; ", title_label,
                    "; top ", length(vals), " of ", nrow(dt_mean), " features",
                    " covering ", round(sum(vals), 1), "% of total mean|SHAP|")
    )
  }
  
  invisible(list(dt_mean = dt_mean, shap_long = shap_long, shap_raw = shap_out))
}

# ----------------------------
# Load "top variables" list and data
# ----------------------------
txt <- paste(readLines(file.path(DATA_PATH, "simple_features.json"), warn = FALSE), collapse = "")

# If your file is actually python-style like ['a','b'], normalize quotes:
txt <- gsub("'", "\"", txt)

cols <- jsonlite::fromJSON(txt)
cols <- make.names(cols, unique = TRUE)

dt_xx <- data.table::fread(file.path(DATA_PATH, "X_train.csv"))[, -1, with = FALSE]
setnames(dt_xx, make.names(colnames(dt_xx), unique = TRUE))

# Keep only requested columns that exist
keep <- intersect(colnames(dt_xx), cols)
dt_x  <- dt_xx[, ..keep]

# If you want group splits by sykli_IND, you must keep it for splitting:
# (uncomment if sykli_IND exists and is needed)
if ("sykli_IND" %in% colnames(dt_xx)) {
  dt_KONS <- dt_xx[sykli_IND == 0]
  dt_IND  <- dt_xx[sykli_IND == 1]
  
  dt_KONS <- dt_KONS[, ..keep]
  dt_IND  <- dt_IND[, ..keep]
  
  X_mat  <- as.matrix(dt_x)
  X_KONS <- as.matrix(dt_KONS)
  X_IND  <- as.matrix(dt_IND)
  
  analyze_shap_group(X_mat,   "ALL",  PLOT_PATH, xgb_mod, top_n = 10, plot_data = TRUE)
  analyze_shap_group(X_KONS, "KONS", PLOT_PATH, xgb_mod, top_n = 10)
  analyze_shap_group(X_IND,  "IND",  PLOT_PATH, xgb_mod, top_n = 10)
  
} else {
  # No group splitting available
  X_mat <- as.matrix(dt_x)
  analyze_shap_group(X_mat, "ALL", PLOT_PATH, xgb_mod, top_n = 10)
}
