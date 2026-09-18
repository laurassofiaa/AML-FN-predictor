# -------------------------------
# Libraries
# -------------------------------
library(xgboost)
library(SHAPforxgboost)
library(ggplot2)
library(data.table)
library(dplyr)
library(jsonlite)
library(scales)
library(grid)
library(cli)

# NEW: for marginal density layout
# install.packages("patchwork")
library(patchwork)

# -------------------------------
# Config
# -------------------------------
POINT_SIZE  <- 1.25
POINT_ALPHA <- 0.85

FOLDER_TAG <- "NOimputation"
TRAIN_TAG  <- "train2_test2"

ROOT <- "/path/to"
#DATA_PATH <- file.path(ROOT, "data", TRAIN_TAG, FOLDER_TAG)
DATA_PATH_MODELS <- file.path(ROOT, "data", 'simple_models')
DATA_PATH <- file.path(ROOT, "data")


# input files (CHANGE if your filenames differ)
MODEL_FILE    <- file.path(DATA_PATH_MODELS, "simple_xgb_model.json")
FEATURES_FILE <- file.path(DATA_PATH_MODELS, "simple_features.json")

X_TRAIN_FILE <- file.path(DATA_PATH, "X_train.csv")
X_TEST_FILE  <- file.path(DATA_PATH, "x_test.csv")

# must contain columns: y_original, y_pred
TEST_PRED_FILE <- file.path(DATA_PATH, "eval_df.csv")

# plots output (ONLY method B now)
PLOT_BASE      <- file.path(ROOT, "plots", FOLDER_TAG, "wrong_classified_shap")
PLOT_HIGHLIGHT <- file.path(PLOT_BASE, "highlight_wrong_density")

dir.create(PLOT_HIGHLIGHT, recursive = TRUE, showWarnings = FALSE)

# number of top features to plot (same “top” idea as your training script)
TOP_K <- 7L

# Downsample for plotting (set NULL to disable)
MAX_POINTS_PER_PLOT <- 10000L

# Colours for errors
FP_COLOUR <- "#E69F00"
FN_COLOUR <- "deepskyblue"

# Reproducibility for sampling (plotting only)
set.seed(123)

# ---------------------------------------------------------------------------
# Source-data export for figures (journal request). Base R only.
# ---------------------------------------------------------------------------
PLOT_DATA_PATH <- file.path(ROOT, "plot_data")
dir.create(PLOT_DATA_PATH, recursive = TRUE, showWarnings = FALSE)

.IMG_EXT <- "\\.(png|pdf|svg|eps|jpe?g|tiff?)$"
.plot_data_stem <- function(name) {
  base <- basename(sub("/+$", "", as.character(name)))
  sub(.IMG_EXT, "", base, ignore.case = TRUE)
}
.rel_path <- function(path, root) {
  root <- normalizePath(root, winslash = "/", mustWork = FALSE)
  p    <- normalizePath(path, winslash = "/", mustWork = FALSE)
  sub(paste0(root, "/"), "", p, fixed = TRUE)
}

write_plot_data <- function(name, tables, out_dir = NULL, note = "",
                            manifest = TRUE, verbose = TRUE) {
  if (is.null(out_dir)) out_dir <- PLOT_DATA_PATH
  stem <- .plot_data_stem(name)
  if (is.data.frame(tables)) tables <- list(data = tables)
  if (is.null(names(tables)) || any(!nzchar(names(tables))))
    stop("`tables` must be a data.frame or a NAMED list of data.frames")
  keep <- vapply(tables, function(d) is.data.frame(d) && nrow(d) > 0, logical(1))
  tables <- tables[keep]
  if (!length(tables)) {
    if (verbose) message(sprintf("[plot_data] %s: nothing to write", stem))
    return(invisible(character(0)))
  }
  single <- length(tables) == 1L
  target <- if (single) out_dir else file.path(out_dir, stem)
  dir.create(target, recursive = TRUE, showWarnings = FALSE)
  
  paths <- character(0); rows <- vector("list", length(tables))
  for (i in seq_along(tables)) {
    part  <- names(tables)[i]
    df    <- as.data.frame(tables[[i]])
    fname <- if (single) paste0(stem, ".csv") else paste0(stem, "__", part, ".csv")
    path  <- file.path(target, fname)
    # NB: plain write.csv is correct. fileEncoding="UTF-8" TRUNCATES non-ASCII
    # under a non-UTF-8 locale, and enc2utf8() double-encodes it.
    write.csv(df, path, row.names = FALSE, na = "")
    if (any(grepl("<U\\+[0-9A-Fa-f]{4}>", readLines(path, warn = FALSE))))
      warning(sprintf(paste0("plot_data: %s contains <U+XXXX> escapes -- labels ",
                             "like °C / 10³/L were not written literally. ",
                             "Check Sys.getlocale('LC_CTYPE')."), fname))
    paths <- c(paths, path)
    rows[[i]] <- data.frame(
      figure = stem, part = if (single) "" else part,
      file = .rel_path(path, out_dir), n_rows = as.character(nrow(df)),
      columns = paste(names(df), collapse = "; "), note = note,
      stringsAsFactors = FALSE)
  }
  new <- do.call(rbind, rows)
  if (manifest) {
    idx <- file.path(out_dir, "plot_data_index.csv")
    if (file.exists(idx)) {
      old <- utils::read.csv(idx, colClasses = "character", check.names = FALSE)
      if (nrow(old) && "figure" %in% names(old)) {
        old <- old[old$figure != stem, , drop = FALSE]
        new <- rbind(old[, names(new), drop = FALSE], new)   # rerun replaces
      }
    }
    write.csv(new, idx, row.names = FALSE, na = "")
  }
  if (verbose)
    message(sprintf("[plot_data] %s: %s", stem,
                    paste(vapply(paths, .rel_path, character(1), root = out_dir),
                          collapse = ", ")))
  invisible(paths)
}

# --- helpers for pulling stat-computed layers out of a ggplot ------------
.pseudo_log_inv <- function(sigma, base = 10) function(x) 2*sigma*sinh(x*log(base))
.x_axis_inverse <- function(feat) {
  if (identical(feat, "b_leuk")) return(.pseudo_log_inv(1e-3))
  if (identical(feat, "b_neut")) return(.pseudo_log_inv(1e-2))
  identity
}

.layer_built <- function(g, stat_class) {
  b <- ggplot2::ggplot_build(g)
  for (i in seq_along(g$layers))
    if (identical(class(g$layers[[i]]$stat)[1], stat_class)) return(b$data[[i]])
  NULL
}

.fill_to_label <- function(d, dens_cols) {
  lab <- names(dens_cols)[match(d$fill, unname(dens_cols))]
  if (any(is.na(lab)) && "group" %in% names(d)) {
    j <- is.na(lab)
    lab[j] <- names(dens_cols)[pmin(d$group[j], length(dens_cols))]
  }
  lab
}

.take_cols <- function(dst, src, nms) {
  for (nm in nms) if (nm %in% names(src)) dst[[nm]] <- src[[nm]]
  dst
}

# -------------------------------
# Small helpers
# -------------------------------
`%||%` <- function(x, y) {
  if (length(x) == 0) return(y)
  if (is.null(x)) return(y)
  if (all(is.na(x))) return(y)
  x
}

read_feature_list <- function(path) {
  txt <- readLines(path, warn = FALSE)
  txt <- paste(txt, collapse = " ")
  
  # Try JSON first
  cols <- tryCatch(jsonlite::fromJSON(txt), error = function(e) NULL)
  if (!is.null(cols)) return(cols)
  
  # Fallback: Python list style -> JSON-ish
  txt2 <- gsub("'", "\"", txt)
  jsonlite::fromJSON(txt2)
}

# Adds a vertical arrow outside the plot area indicating lower/higher risk
# relative to y=0 on SHAP axis.
# side: "left" is safer with a right-side marginal density plot.
add_risk_arrow_outside_from_zero <- function(
    g,
    offset_mm = 8,
    gap_npc = 0.02,
    trim_npc = 0.05,
    low_lbl = "Lower risk",
    high_lbl = "Higher risk",
    lwd = 1.2,
    head_mm = 1,
    side = c("left", "right")
) {
  side <- match.arg(side)
  
  gb <- ggplot2::ggplot_build(g)
  yr <- gb$layout$panel_scales_y[[1]]$range$range
  
  # map y=0 to npc [0..1]
  y0_npc <- (0 - yr[1]) / diff(yr)
  y0_npc <- min(max(y0_npc, 0), 1)
  
  # panel borders in npc
  y_bot <- 0 + trim_npc
  y_top <- 1 - trim_npc
  
  # gap around origin
  y_up_start <- min(1, y0_npc + gap_npc)
  y_dn_start <- max(0, y0_npc - gap_npc)
  
  # end points
  y_up_end <- y_top
  y_dn_end <- y_bot
  
  # put arrow outside plot area
  x_out <- if (side == "right") {
    grid::unit(1, "npc") + grid::unit(offset_mm, "mm")
  } else {
    grid::unit(0, "npc") - grid::unit(offset_mm, "mm")
  }
  
  gp <- grid::gpar(col = "black", fill = "black", lwd = lwd)
  
  make_seg <- function(y0, y1) {
    if (abs(y1 - y0) < 1e-6) return(NULL)
    grid::segmentsGrob(
      x0 = x_out, x1 = x_out,
      y0 = grid::unit(y0, "npc"),
      y1 = grid::unit(y1, "npc"),
      gp = gp,
      arrow = grid::arrow(ends = "last", type = "closed", length = grid::unit(head_mm, "mm"))
    )
  }
  
  seg_up <- make_seg(y_up_start, y_up_end)
  seg_dn <- make_seg(y_dn_start, y_dn_end)
  
  # text placement
  if (side == "right") {
    txt_low <- grid::textGrob(low_lbl,
                              x = x_out + grid::unit(4, "mm"),
                              y = grid::unit(y_dn_end, "npc") * 1.5,
                              rot = 90, gp = grid::gpar(cex = 1.2)
    )
    txt_high <- grid::textGrob(high_lbl,
                               x = x_out + grid::unit(4, "mm"),
                               y = grid::unit(y_up_end, "npc") * 0.9,
                               rot = 90, gp = grid::gpar(cex = 1.2)
    )
  } else {
    txt_low <- grid::textGrob(low_lbl,
                              x = x_out - grid::unit(4, "mm"),
                              y = grid::unit(y_dn_end, "npc") * 1.5,
                              rot = 90, gp = grid::gpar(cex = 1.2), just = "right"
    )
    txt_high <- grid::textGrob(high_lbl,
                               x = x_out - grid::unit(4, "mm"),
                               y = grid::unit(y_up_end, "npc") * 0.9,
                               rot = 90, gp = grid::gpar(cex = 1.2), just = "right"
    )
  }
  
  g +
    { if (!is.null(seg_up)) ggplot2::annotation_custom(seg_up) else NULL } +
    { if (!is.null(seg_dn)) ggplot2::annotation_custom(seg_dn) else NULL } +
    ggplot2::annotation_custom(txt_low) +
    ggplot2::annotation_custom(txt_high) +
    ggplot2::coord_cartesian(clip = "off") +
    ggplot2::theme(plot.margin = grid::unit(c(5.5, 34, 5.5, 34), "pt"))
}

# Choose which columns to use for y (SHAP values)
get_xy_cols <- function(df) {
  if (!("value" %in% names(df))) stop("Expected column `value` in shap_long.")
  y_candidates <- c("rfvalue", "phi", "shap_value")
  y_col <- y_candidates[y_candidates %in% names(df)][1]
  if (is.na(y_col)) stop("Expected one of: rfvalue, phi, shap_value in shap_long.")
  list(x = "value", y = y_col)
}

# Manual legend grob (updated to reflect shapes used: correct=solid dot, wrong=filled dot with border)
make_manual_legend_grob <- function(
    title = "Legend",
    correct_lbl = "Correct",
    fp_lbl = "False positive",
    fn_lbl = "False negative",
    mean_lbl = "Mean",
    correct_col = "black",
    fp_col = "#E69F00",
    fn_col = "deepskyblue",
    mean_col = "darkgray",
    dot_mm = 3
) {
  # Layout inside the legend box (0..1 NPC)
  x_symbol <- grid::unit(0.12, "npc")
  x_text   <- grid::unit(0.22, "npc")
  
  y_title <- grid::unit(0.90, "npc")
  y1 <- grid::unit(0.72, "npc")
  y2 <- grid::unit(0.50, "npc")
  y3 <- grid::unit(0.28, "npc")
  y4 <- grid::unit(0.06, "npc")
  
  bg <- grid::nullGrob()
  
  ttl <- grid::textGrob(
    title, x = x_text, y = y_title,
    just = c("left", "center"),
    gp = grid::gpar(fontface = "bold", cex = 1.2, col = "black")
  )
  
  # Symbols
  dot_correct <- grid::pointsGrob(x_symbol, y1, pch = 16, size = grid::unit(dot_mm, "mm"),
                                  gp = grid::gpar(col = correct_col)
  )
  dot_fp <- grid::pointsGrob(x_symbol, y2, pch = 21, size = grid::unit(dot_mm, "mm"),
                             gp = grid::gpar(col = "black", fill = fp_col)
  )
  dot_fn <- grid::pointsGrob(x_symbol, y3, pch = 21, size = grid::unit(dot_mm, "mm"),
                             gp = grid::gpar(col = "black", fill = fn_col)
  )
  
  line_mean <- grid::segmentsGrob(
    x0 = grid::unit(0.06, "npc"), x1 = grid::unit(0.18, "npc"),
    y0 = y4, y1 = y4,
    gp = grid::gpar(col = mean_col, lwd = 2)
  )
  
  # Labels
  t1 <- grid::textGrob(correct_lbl, x = x_text, y = y1, just = c("left", "center"),
                       gp = grid::gpar(cex = 1.2, col = "black")
  )
  t2 <- grid::textGrob(fp_lbl, x = x_text, y = y2, just = c("left", "center"),
                       gp = grid::gpar(cex = 1.2, col = "black")
  )
  t3 <- grid::textGrob(fn_lbl, x = x_text, y = y3, just = c("left", "center"),
                       gp = grid::gpar(cex = 1.2, col = "black")
  )
  t4 <- grid::textGrob(mean_lbl, x = x_text, y = y4, just = c("left", "center"),
                       gp = grid::gpar(cex = 1.2, col = "black")
  )
  
  grid::grobTree(bg, ttl, dot_correct, dot_fp, dot_fn, line_mean, t1, t2, t3, t4)
}

add_manual_legend_box <- function(
    g,
    x_npc = 0.72, y_npc = 0.52,  # bottom-left position inside the panel (0..1)
    w_npc = 0.26, h_npc = 0.24   # width/height inside the panel (0..1)
) {
  gb <- ggplot2::ggplot_build(g)
  xr <- gb$layout$panel_scales_x[[1]]$range$range
  yr <- gb$layout$panel_scales_y[[1]]$range$range
  
  xmin <- xr[1] + x_npc * diff(xr)
  xmax <- xr[1] + (x_npc + w_npc) * diff(xr)
  ymin <- yr[1] + y_npc * diff(yr)
  ymax <- yr[1] + (y_npc + h_npc) * diff(yr)
  
  leg <- make_manual_legend_grob(title = "", correct_col = "black", fp_col = FP_COLOUR, fn_col = FN_COLOUR, mean_col = "darkgray")
  
  g +
    ggplot2::annotation_custom(leg, xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax) +
    ggplot2::coord_cartesian(clip = "off")
}

# Apply your feature-specific x-axis styling to any plot that uses x = feature value
apply_feature_x_scale <- function(g, feat, x_title) {
  if (feat == "temperature") {
    g <- g + ggplot2::scale_x_continuous(
      breaks = c(35, 36, 37, 38, 39, 40, 41),
      labels = scales::label_number(accuracy = 1, trim = TRUE)
    )
  }
  
  if (feat == "b_leuk") {
    g <- g + ggplot2::scale_x_continuous(
      trans  = scales::pseudo_log_trans(base = 10, sigma = 1e-3),
      breaks = c(0, 0.01, 0.1, 1, 10, 100),
      labels = scales::label_number(accuracy = 0.01, trim = TRUE)
    ) +
      ggplot2::xlab(paste0(x_title, " (log)"))
  } else if (feat == "b_neut") {
    g <- g + ggplot2::scale_x_continuous(
      trans  = scales::pseudo_log_trans(base = 10, sigma = 1e-2),
      breaks = c(0, 0.05, 0.5, 5, 50),
      labels = scales::label_number(accuracy = 0.01, trim = TRUE)
    ) +
      ggplot2::xlab(paste0(x_title, " (log)"))
  } else {
    g <- g + ggplot2::xlab(x_title)
  }
  
  g
}

# Build one combined plot: top density + main scatter + right density
build_density_shap_plot <- function(sl, feat, x_title, xcol, ycol, legend_pos_map,
                                    plot_data = TRUE, plot_data_points = TRUE,
                                    plot_data_path = NULL, plot_data_name = NULL) {
  # Labels for densities
  sl$pt_label <- ifelse(is.na(sl$err_type), "Correct", sl$err_type)
  sl$pt_label <- factor(sl$pt_label, levels = c("Correct", "False positive", "False negative"))
  
  # Split for point layers
  sl_wrong   <- sl[sl$is_wrong == TRUE, , drop = FALSE]
  sl_correct <- sl[sl$is_wrong == FALSE, , drop = FALSE]
  sl_fp      <- sl_wrong[sl_wrong$err_type == "False positive", , drop = FALSE]
  sl_fn      <- sl_wrong[sl_wrong$err_type == "False negative", , drop = FALSE]
  
  # Shared ranges for alignment
  xlim <- range(sl[[xcol]], na.rm = TRUE)
  ylim <- range(sl[[ycol]], na.rm = TRUE)
  
  # ---- MAIN plot ----
  g_main <- ggplot2::ggplot() +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey30", alpha = 0.5, linetype = "dashed") +
    ggplot2::geom_smooth(
      data = sl,
      #mapping = ggplot2::aes_string(x = xcol, y = ycol),
      mapping = ggplot2::aes_string(x = ycol, y = xcol),
      se = FALSE,
      method = "loess",
      colour = "darkgray",
      linewidth = 0.7
    ) +
    # Correct points (black)
    ggplot2::geom_point(
      data = sl_correct,
      #mapping = ggplot2::aes_string(x = xcol, y = ycol),
      mapping = ggplot2::aes_string(x = ycol, y = xcol),
      inherit.aes = FALSE,
      shape = 16,
      colour = "black",
      alpha = POINT_ALPHA,
      size = POINT_SIZE
    ) +
    # Wrong points (filled by err_type, black border)
    ggplot2::geom_point(
      data = sl_wrong,
      #mapping = ggplot2::aes_string(x = xcol, y = ycol, fill = "err_type"),
      mapping = ggplot2::aes_string(x = ycol, y = xcol, fill = "err_type"),
      inherit.aes = FALSE,
      shape = 21,
      colour = "black",
      stroke = 0.25,
      alpha = POINT_ALPHA,
      size = POINT_SIZE
    ) +
    ggplot2::scale_fill_manual(
      values = c("False positive" = FP_COLOUR, "False negative" = FN_COLOUR),
      breaks = c("False positive", "False negative"),
      name = ""
    ) +
    # Rug ticks for FP/FN only (occurrence hints without clutter)
    { if (nrow(sl_fp) > 0) ggplot2::geom_rug(
      data = sl_fp,
      #mapping = ggplot2::aes_string(x = xcol, y = ycol),
      mapping = ggplot2::aes_string(x = ycol, y = xcol),
      inherit.aes = FALSE,
      sides = "bl",
      colour = FP_COLOUR,
      alpha = 0.25
    ) else NULL } +
    { if (nrow(sl_fn) > 0) ggplot2::geom_rug(
      data = sl_fn,
      #mapping = ggplot2::aes_string(x = xcol, y = ycol),
      mapping = ggplot2::aes_string(x = ycol, y = xcol),
      inherit.aes = FALSE,
      sides = "bl",
      colour = FN_COLOUR,
      alpha = 0.25
    ) else NULL } +
    ggplot2::theme_bw() +
    ggplot2::ylab("SHAP value") +
    ggplot2::coord_cartesian(xlim = xlim, ylim = ylim, clip = "off") +
    ggplot2::theme(
      axis.text.x  = ggplot2::element_text(size = 15, colour = "black"),
      axis.text.y  = ggplot2::element_text(size = 15, colour = "black"),
      axis.title.x = ggplot2::element_text(size = 16, colour = "black"),
      axis.title.y = ggplot2::element_text(size = 16, colour = "black"),
      axis.line    = ggplot2::element_line(colour = "black"),
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      panel.background = ggplot2::element_blank(),
      legend.position = "none"
    )
  
  # Apply x-axis style (log-ish for leuk/neut etc.)
  g_main <- apply_feature_x_scale(g_main, feat, x_title)
  g_main_pre_annot <- g_main   # capture before the risk arrow + legend annotations
  
  # Risk arrow: put it on LEFT to avoid colliding with right-side density
  g_main <- add_risk_arrow_outside_from_zero(g_main, gap_npc = 0.025, trim_npc = 0.06, offset_mm = 8, side = "right")
  
  # Add your manual legend box (feature-specific positioning)
  legend_pos_default <- list(x = 0.72, y = 0.52, w = 0.26, h = 0.24)
  lp <- legend_pos_map[[feat]]
  if (is.null(lp)) lp <- legend_pos_default
  
  g_main <- add_manual_legend_box(
    g_main,
    x_npc = lp$x, y_npc = lp$y, w_npc = lp$w, h_npc = lp$h
  )
  # ---- TOP density (x distribution) ----
  dens_cols <- c("Correct" = "grey70", "False positive" = FP_COLOUR, "False negative" = FN_COLOUR)
  
  g_top <- ggplot2::ggplot(sl, ggplot2::aes_string(x = ycol, fill = "pt_label")) + # modasin x-> y tilalle
    ggplot2::geom_density(alpha = 0.35, adjust = 1.1, colour = NA) +
    ggplot2::scale_fill_manual(values = dens_cols) +
    ggplot2::coord_cartesian(xlim = ylim) +
    ggplot2::theme_void() +
    ggplot2::theme(
      legend.position = "none",
      plot.margin = grid::unit(c(0, 0, 0, 0), "pt")
    )
  
  g_top <- apply_feature_x_scale(g_top, feat, x_title) +
    ggplot2::theme(axis.title.x = ggplot2::element_blank()) # keep scale, drop label
  
  
  # ---- RIGHT density (y distribution) ----
  g_right <- ggplot2::ggplot(sl, ggplot2::aes_string(x = xcol, fill = "pt_label")) +
    ggplot2::geom_density(alpha = 0.35, adjust = 1.1, colour = NA) +
    ggplot2::scale_fill_manual(values = dens_cols) +
    ggplot2::coord_cartesian(xlim = xlim) +
    ggplot2::coord_flip() +
    ggplot2::theme_void() +
    ggplot2::theme(
      legend.position = "none",
      plot.margin = grid::unit(c(0, 0, 0, 0), "pt")
    )
  # ---- Combine with patchwork ----
  # Layout:
  #   [ top density ][ blank ]
  #   [   main     ][ right density ]
  combined <- (g_top + patchwork::plot_spacer()) /
    (g_main + g_right) +
    patchwork::plot_layout(widths = c(8, 1), heights = c(1, 4))
    #patchwork::plot_layout(widths = c(5, 0.8), heights = c(8, 8))
  
  # -----------------------
  right_w <- 0.2
  gutter_w <- 0.04
  top_h <- 0.20
  
  top_row <- (g_top | patchwork::plot_spacer() | patchwork::plot_spacer()) +
    patchwork::plot_layout(widths = c(1, gutter_w, right_w))
  
  bottom_row <- (g_main | patchwork::plot_spacer() | g_right) +
    patchwork::plot_layout(widths = c(1, gutter_w, right_w))
  
  combined <- (top_row / bottom_row) +
    patchwork::plot_layout(heights = c(top_h, 1))
  # ---- source data for the journal --------------------------------------
  if (isTRUE(plot_data)) {
    stem <- if (is.null(plot_data_name))
      paste0("SHAP_test_highlight_wrong_density_", feat) else plot_data_name
    inv <- .x_axis_inverse(feat)
    transformed <- !identical(inv, identity)
    tabs <- list()
    
    # (a) the plotted points -- straight from sl, no build needed
    if (isTRUE(plot_data_points)) {
      tabs$points <- data.frame(
        row_id        = sl$ID,
        feature       = feat,
        feature_value = as.numeric(sl[[ycol]]),
        shap_value    = as.numeric(sl[[xcol]]),
        point_class   = as.character(sl$pt_label),
        stringsAsFactors = FALSE
      )
    }
    
    # (b) loess curve -- stat-computed, exists nowhere in sl
    lo <- .layer_built(g_main_pre_annot, "StatSmooth")
    if (!is.null(lo)) {
      d <- data.frame(feature_value = inv(lo$x), shap_value_fitted = lo$y)
      if (transformed) d$x_plotted <- lo$x
      tabs$loess <- d
    }
    
    # (c) top marginal density (feature value)
    dt <- .layer_built(g_top, "StatDensity")
    if (!is.null(dt)) {
      d <- data.frame(point_class = .fill_to_label(dt, dens_cols),
                      feature_value = inv(dt$x), stringsAsFactors = FALSE)
      d <- .take_cols(d, dt, c("density", "scaled", "count", "n"))
      if (transformed) d$x_plotted <- dt$x
      tabs$density_feature_value <- d
    }
    
    # (d) right marginal density (SHAP value; never transformed)
    dr <- .layer_built(g_right, "StatDensity")
    if (!is.null(dr)) {
      d <- data.frame(point_class = .fill_to_label(dr, dens_cols),
                      shap_value = dr$x, stringsAsFactors = FALSE)
      tabs$density_shap_value <- .take_cols(d, dr, c("density", "scaled", "count", "n"))
    }
    
    write_plot_data(stem, tabs, out_dir = plot_data_path,
                    note = paste0("feature=", feat, "; x-axis: ",
                                  gsub("\\s*\n\\s*", " ", x_title),
                                  "; points plotted=", nrow(sl),
                                  " (correct=", nrow(sl_correct), ", FP=", nrow(sl_fp),
                                  ", FN=", nrow(sl_fn), ")",
                                  "; x transform=",
                                  if (transformed) "pseudo-log (see x_plotted)" else "none",
                                  "; density adjust=1.1; smooth=loess, se=FALSE"))
  }
  combined
}

# -------------------------------
# Load model + features
# -------------------------------
cli::cli_alert_info("Loading model and feature list...")
model1 <- xgb.load(MODEL_FILE)
cols   <- read_feature_list(FEATURES_FILE)

# -------------------------------
# Load train/test X (ensure same columns/order)
# -------------------------------
cli::cli_alert_info("Loading X_train / X_test...")
#dt_x_train <- data.table::fread(X_TRAIN_FILE)[, -1, with = FALSE]
dt_x_test  <- data.table::fread(X_TEST_FILE)[, -1, with = FALSE]

#dt_x_train <- dt_x_train[, ..cols]
dt_x_test  <- dt_x_test[, ..cols]

#x_train <- as.matrix(dt_x_train)
x_test  <- as.matrix(dt_x_test)

# -------------------------------
# Load test preds/meta and build FP/FN flags from eval_df
# -------------------------------
cli::cli_alert_info("Loading test predictions/meta...")
dt_test_pred <- data.table::fread(TEST_PRED_FILE)
dt_test_pred <- as.data.table(dt_test_pred)

stopifnot(nrow(dt_test_pred) == nrow(x_test))

if (!all(c("y_original", "y_pred") %in% names(dt_test_pred))) {
  stop("TEST_PRED_FILE must contain columns: y_original, y_pred")
}

dt_test_pred[, y_original := as.integer(y_original)]
dt_test_pred[, y_pred     := as.integer(y_pred)]

dt_test_pred[, fp := as.integer(y_original == 0L & y_pred == 1L)]
dt_test_pred[, fn := as.integer(y_original == 1L & y_pred == 0L)]
dt_test_pred[, err_type := fifelse(fp == 1L, "False positive",
                                   fifelse(fn == 1L, "False negative", NA_character_))]
dt_test_pred[, is_wrong := !is.na(err_type)]
wrong_flag <- dt_test_pred$is_wrong

cli::cli_alert_info("Wrong rows: {sum(wrong_flag)} / {length(wrong_flag)}")
cli::cli_alert_info("FP: {sum(dt_test_pred$fp)} | FN: {sum(dt_test_pred$fn)}")

# -------------------------------
# Compute SHAP
#  - use TRAIN to get top features
#  - compute TEST SHAP for plotting
# -------------------------------

top_feats <- jsonlite::fromJSON(FEATURES_FILE)   # <- path to your json

# in case it's nested (e.g., {"top_feats":[...]}), unlist safely
top_feats <- unlist(top_feats, use.names = FALSE)

TOP_K <- min(TOP_K, length(top_feats))
top_feats <- top_feats[seq_len(TOP_K)]


cli::cli_alert_info("Computing SHAP on test (for plots)...")
shap_test <- shap.values(xgb_model = model1, X_train = x_test)

shap_long_test <- shap.prep(
  shap_contrib = shap_test$shap_score,
  X_train = x_test
) %>%
  dplyr::filter(variable %in% top_feats)

# attach wrong flag + error type by row index (ID is 1..nrow(x_test))
shap_long_test$is_wrong <- wrong_flag[shap_long_test$ID]
shap_long_test$err_type <- dt_test_pred$err_type[shap_long_test$ID]

# -------------------------------
# Nice axis titles (edit as needed)
# -------------------------------
feature_titles <- c(
  "temperature"                 = "Body temperature (°C)",
  "b_neut"                      = "ANC (10⁹/L)",
  "b_leuk"                      = "WBC (10⁹/L)",
  "temperature_trend_from_2_d"  = "Temperature\n2-day trend (°C)",
  "p_crp_trend_from_7_d"        = "4–7-day CRP trend  (mg/L)",
  "p_crp_trend_from_3_d"        = "1–3 day CRP trend (mg/L)",
  "sykli_IND"                   = "Induction phase",
  "bneut_accumulated_max_050_cycle" = "Duration of neutropenia\n(<0.5 10⁹/L) per cycle (days)",
  "bneut_accumulated_max_050_total" = "Duration of neutropenia\n(<0.5 10⁹/L) per regimen (days)",
  "countdown"                   = "Time since cycle start (days)",
  'ab_max_10d'                  = "Ab initiated within the last 10 days"
)

# manual legend positions per feature (0..1 npc inside MAIN panel)
legend_pos_map <- list(
  temperature            = list(x = 0.65, y = 0.05, w = 0.28, h = 0.24),
  b_leuk                 = list(x = 0.0001, y = 0.05, w = 0.005, h = 0.24),
  b_neut                 = list(x = 0.0005, y = 0.05, w = 0.03,  h = 0.24),
  countdown              = list(x = 0.65, y = 0.05, w = 0.28, h = 0.24),
  p_crp_trend_from_3_d   = list(x = 0.75, y = 0.05, w = 0.28, h = 0.24),
  sykli_IND              = list(x = 0.65, y = 0.05, w = 0.28, h = 0.24)
)

feat <- top_feats[1]
sl   <- shap_long_test[shap_long_test$variable == feat, , drop = FALSE]
xy   <- get_xy_cols(sl); xcol <- xy$x; ycol <- xy$y
sl$pt_label <- factor(ifelse(is.na(sl$err_type), "Correct", sl$err_type),
                      levels = c("Correct", "False positive", "False negative"))
gs <- ggplot2::ggplot(sl) + ggplot2::geom_smooth(
  ggplot2::aes_string(x = ycol, y = xcol), se = FALSE, method = "loess")
gd <- ggplot2::ggplot(sl, ggplot2::aes_string(x = ycol, fill = "pt_label")) +
  ggplot2::geom_density(adjust = 1.1)
cat("smooth  stat:", class(gs$layers[[1]]$stat)[1], "\n")
cat("smooth  cols:", paste(names(ggplot2::ggplot_build(gs)$data[[1]]), collapse = ", "), "\n")
cat("density stat:", class(gd$layers[[1]]$stat)[1], "\n")
cat("density cols:", paste(names(ggplot2::ggplot_build(gd)$data[[1]]), collapse = ", "), "\n")
cat("density fills:", paste(unique(ggplot2::ggplot_build(gd)$data[[1]]$fill), collapse = ", "), "\n")

# ----------------------------
# Supplementary Figure 11A-G  (SHAP dependence plots, external validation cohort; one file per feature)
# ----------------------------
# A: WBC (b_leuk)
# B: Body temperature (temperature)
# C: Time since cycle start (countdown)
# D: 4-7-day CRP trend (p_crp_trend_from_7_d)
# E: ANC (b_neut)
# F: Induction phase (sykli_IND)
# G: Ab initiated within the last 10 days (ab_max_10d)

# -------------------------------
# Plot loop (ONLY method B now: all points + highlight FP/FN + marginal densities)
# -------------------------------
cli::cli_alert_info("Starting plot loop...")

for (k in seq_along(top_feats)) {
  feat <- top_feats[k]
  cli::cli_alert_info("Plotting feature: {feat}")
  
  sl <- shap_long_test[shap_long_test$variable == feat, , drop = FALSE]
  
  # (Optional) downsample for speed/clarity:
  # keep all wrong points, sample correct points if needed
  if (!is.null(MAX_POINTS_PER_PLOT) && nrow(sl) > MAX_POINTS_PER_PLOT) {
    set.seed(1000 + k)
    
    sl_wrong   <- sl[sl$is_wrong, , drop = FALSE]
    sl_correct <- sl[!sl$is_wrong, , drop = FALSE]
    
    remaining <- MAX_POINTS_PER_PLOT - nrow(sl_wrong)
    if (remaining <= 0) {
      idx <- sample(nrow(sl_wrong), MAX_POINTS_PER_PLOT)
      sl <- sl_wrong[idx, , drop = FALSE]
    } else {
      idx <- sample(nrow(sl_correct), min(remaining, nrow(sl_correct)))
      sl <- rbind(sl_wrong, sl_correct[idx, , drop = FALSE])
    }
  }
  
  xy_cols <- get_xy_cols(sl)
  xcol <- xy_cols$x  # "value"
  ycol <- xy_cols$y  # "rfvalue" (or "phi"/"shap_value")
  
  x_title <- feature_titles[feat] %||% feat
  
  combined <- build_density_shap_plot(
    sl = sl,
    feat = feat,
    x_title = x_title,
    xcol = xcol,
    ycol = ycol,
    legend_pos_map = legend_pos_map
  )
  
  out_all <- file.path(PLOT_HIGHLIGHT, paste0("SHAP_test_highlight_wrong_density_", feat, ".png"))
  ggplot2::ggsave(filename = out_all, plot = combined, width = 8, height = 6.4, dpi = 600)
  
  gc()
}

cli::cli_alert_success("Done. Plots saved under: {PLOT_BASE}")