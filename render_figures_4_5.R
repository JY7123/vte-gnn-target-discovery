# ============================================================
# Render Paper Figures 4+5 — ggplot2 出版级
# Figure 4: Mechanism Subgraphs & Cascade Mapping
# Figure 5: Single-Cell Transcriptomic Validation
# Target: Nature Communications
# Output: D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures/
# ============================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggrepel)
library(jsonlite)

outdir <- "D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/paper_figures"
datadir <- file.path(outdir, "source_data")
dir.create(datadir, showWarnings = FALSE, recursive = TRUE)

# ── Re-use theme and colors from Figures 1-3 ─────────────────
theme_pub <- theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.2, color = "grey90"),
    plot.title    = element_text(size = 12, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40"),
    axis.title    = element_text(size = 10),
    axis.text     = element_text(size = 8),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text  = element_text(size = 7),
    strip.text   = element_text(size = 9, face = "bold"),
    plot.margin  = margin(8, 8, 8, 8)
  )

cc <- c(
  blue   = "#0072B2", orange = "#E69F00", green  = "#009E73",
  red    = "#D55E00", purple = "#CC79A7", teal   = "#56B4E9",
  grey   = "#999999", yellow = "#F0E442", black  = "#000000"
)

# ── Data paths ───────────────────────────────────────────────
subgraph_dir <- "D:/JY/work/my work/新思路/vte_gnn_target_discovery/figures/pca_hidden/subgraphs"
scRNA_dir    <- "D:/JY/work/my work/新思路/figures/PAR2_scRNA"

# ============================================================
# FIGURE 4: Mechanism Subgraphs & Cascade Mapping
# ============================================================

make_fig4 <- function() {

  # -- Load subgraph JSONs --
  targets <- c("renin","c3","mmp-2","par-2","tsp-1")
  subgraph_data <- list()
  for (tn in targets) {
    fp <- file.path(subgraph_dir, paste0(tn, ".json"))
    if (file.exists(fp)) {
      subgraph_data[[tn]] <- fromJSON(fp, simplifyVector = TRUE)
    }
  }

  # -- 4A: Target × Cascade Step Heatmap --
  # Extract: for each target, which cascade steps are connected + how many paths
  cascade_steps <- c(
    "Step 1\nFucosylation",
    "Step 2\nGalectin",
    "Step 3\nAdhesion",
    "Step 4\nCytoskeletal",
    "Step 5\nMAPK",
    "Step 6\nTranscription"
  )

  heatmap_rows <- list()
  for (tn in names(subgraph_data)) {
    sd <- subgraph_data[[tn]]
    acm <- sd$anchor_cascade_map
    if (is.data.frame(acm) || is.list(acm)) {
      # Count paths per cascade step
      step_counts <- setNames(rep(0, 6), cascade_steps)
      for (i in seq_along(acm)) {
        entry <- if (is.data.frame(acm)) acm[i, ] else acm[[i]]
        step_num <- entry$step
        if (is.numeric(step_num) && step_num >= 1 && step_num <= 6) {
          step_counts[step_num] <- step_counts[step_num] + 1
        }
      }
      # Also get total path count
      total_paths <- if (is.numeric(sd$num_anchor_paths)) sd$num_anchor_paths else 0

      heatmap_rows[[tn]] <- data.frame(
        Target = tn,
        Step = names(step_counts),
        Paths = as.integer(step_counts),
        TotalPaths = total_paths,
        Degree = if (is.numeric(sd$degree)) sd$degree else NA,
        GNN_Score = if (is.numeric(sd$gnn_score)) sd$gnn_score else NA
      )
    }
  }
  heatmap_data <- bind_rows(heatmap_rows)
  heatmap_data$Target <- factor(heatmap_data$Target,
    levels = c("par-2","renin","mmp-2","tsp-1","c3"),
    labels = c("PAR-2","Renin","MMP-2","TSP-1","C3"))
  heatmap_data$Step <- factor(heatmap_data$Step, levels = cascade_steps)

  # [修复] 根据背景深度动态调整字体颜色
  path_max <- max(heatmap_data$Paths, na.rm = TRUE)
  p4a <- ggplot(heatmap_data, aes(x = Step, y = Target, fill = Paths)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = ifelse(Paths > 0, Paths, ""),
                  color = Paths > (path_max * 0.5)),
              size = 3.5, fontface = "bold") +
    scale_color_manual(values = c("TRUE" = "white", "FALSE" = "black"), guide = "none") +
    scale_fill_gradient(low = "#FFF5F5", high = cc[["red"]], na.value = "white",
                        name = "KG Anchor\nPaths") +
    labs(title = "Target — Cascade Step Connectivity",
         subtitle = "Number of attention-weighted KG paths linking each target to cascade anchors") +
    theme_pub + theme(axis.text.x = element_text(size = 7.5, lineheight = 0.9),
                      legend.position = "right")

  # -- 4B: Target Pathway Connectivity Summary --
  summary_data <- heatmap_data %>%
    group_by(Target, Degree, GNN_Score, TotalPaths) %>%
    summarise(CascadeSteps = sum(Paths > 0), .groups = "drop") %>%
    arrange(-TotalPaths)

  p4b <- ggplot(summary_data, aes(x = TotalPaths, y = reorder(Target, TotalPaths))) +
    geom_col(aes(fill = Target), width = 0.65, color = "white", show.legend = FALSE) +
    geom_text(aes(label = paste0("d=", Degree, "  score=", round(GNN_Score, 1))),
              hjust = -0.05, size = 3) +
    scale_fill_manual(values = c("PAR-2" = cc[["red"]], "Renin" = cc[["orange"]],
                                  "MMP-2" = cc[["blue"]], "TSP-1" = cc[["green"]],
                                  "C3" = cc[["purple"]])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(title = "Total Anchor Paths per Target",
         subtitle = "Degree and GNN score annotated. PAR-2: low degree, high connectivity.",
         x = "Total KG anchor paths", y = "") +
    theme_pub

  # -- 4C: Cascade Step Activation Profile (radar-like bar chart) --
  profile_long <- heatmap_data %>%
    mutate(Active = Paths > 0)

  p4c <- ggplot(profile_long, aes(x = Step, y = Target, fill = Active)) +
    geom_tile(color = "white", linewidth = 0.6) +
    scale_fill_manual(values = c("TRUE" = cc[["green"]], "FALSE" = "#F5F5F5"),
                      labels = c("TRUE" = "Connected", "FALSE" = "No path"),
                      name = "") +
    labs(title = "Cascade Step Coverage Matrix",
         subtitle = "PAR-2 uniquely spans Step 3 (adhesion) + Step 4 (cytoskeletal)") +
    theme_pub + theme(axis.text.x = element_text(size = 7.5, lineheight = 0.9),
                      legend.position = "right")

  # -- 4D: Individual Target Pathway Diagrams --
  # Simplified from JSON anchor_cascade_map — show each target's cascade mapping
  acm_data <- list()
  for (tn in names(subgraph_data)) {
    sd <- subgraph_data[[tn]]
    acm <- sd$anchor_cascade_map
    if (is.data.frame(acm) || is.list(acm)) {
      for (i in seq_along(acm)) {
        entry <- if (is.data.frame(acm)) acm[i, ] else acm[[i]]
        step_num <- entry$step
        if (is.numeric(step_num) && step_num >= 1 && step_num <= 6 && !is.null(entry$anchor)) {
          acm_data[[length(acm_data) + 1]] <- data.frame(
            Target = tn,
            Anchor = entry$anchor,
            CascadeStep = step_num,
            StepLabel = if (!is.null(entry$label)) entry$label else "",
            Hops = if (is.numeric(entry$hops)) entry$hops else NA,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
  acm_df <- bind_rows(acm_data)

  p4d <- ggplot(acm_df, aes(x = CascadeStep, y = reorder(Target, -CascadeStep),
                              color = Target, size = 1 / Hops)) +
    geom_jitter(width = 0.15, height = 0.1, alpha = 0.8) +
    geom_text_repel(aes(label = Anchor), size = 2.8, max.overlaps = 20,
                    box.padding = 0.3, force = 2) +
    scale_color_manual(values = c("par-2" = cc[["red"]], "renin" = cc[["orange"]],
                                   "mmp-2" = cc[["blue"]], "tsp-1" = cc[["green"]],
                                   "c3" = cc[["purple"]]), guide = "none") +
    scale_size_continuous(range = c(2, 5), name = "Proximity\n(1/hops)") +
    scale_x_continuous(breaks = 1:6, labels = c("FUT8","Lgals3","CD44","RhoA","MAPK","NF-kB"),
                       limits = c(0.5, 6.5)) +
    labs(title = "Anchor Gene Mapping to Cascade Steps",
         subtitle = "Each point = one anchor gene. Larger points = fewer KG hops from target.",
         x = "Cascade anchor gene", y = "") +
    theme_pub

  # ── Assembly ──
  design4 <- "
    AABB
    CCDD
  "
  fig4 <- p4a + p4b + p4c + p4d +
    plot_layout(design = design4) +
    plot_annotation(
      title = "Figure 4: Mechanism Subgraphs & Attention-Weighted Pathway Mapping",
      tag_levels = "a",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
                    plot.tag = element_text(size = 18, face = "bold")))

  ggsave(file.path(outdir, "Figure4_Mechanism_Subgraphs.png"), fig4,
         width = 15, height = 11, dpi = 600, device = "png", bg = "white")

  # ── Export source data ──
  write.csv(heatmap_data, file.path(datadir, "Figure4A_cascade_connectivity.csv"), row.names = FALSE)
  write.csv(summary_data, file.path(datadir, "Figure4B_target_path_summary.csv"), row.names = FALSE)
  write.csv(profile_long, file.path(datadir, "Figure4C_step_coverage.csv"), row.names = FALSE)
  write.csv(acm_df,      file.path(datadir, "Figure4D_anchor_mapping.csv"), row.names = FALSE)
  message("  Figure 4 saved.")
}


# ============================================================
# FIGURE 5: Single-Cell Transcriptomic Validation
# ============================================================

make_fig5 <- function() {

  # -- 5A: % F2rl1+ by Cell Type --
  cell_data <- read.csv(file.path(scRNA_dir, "F2rl1_by_CellType.csv"))
  cell_data <- cell_data[cell_data$ctrl_n >= 10 | cell_data$dvt_n >= 10, ]

  # [修复] 手工构造长表，避免dplyr select/pivot_longer的列名匹配问题
  # 找到实际的Pct列名
  dvt_col  <- grep("dvt.*pct",  colnames(cell_data), ignore.case = TRUE, value = TRUE)[1]
  ctrl_col <- grep("ctrl.*pct", colnames(cell_data), ignore.case = TRUE, value = TRUE)[1]
  label_col <- grep("cell_label|label", colnames(cell_data), ignore.case = TRUE, value = TRUE)[1]

  cell_data_long <- data.frame(
    cell_label = rep(cell_data[[label_col]], 2),
    Group = rep(c("DVT", "Sham"), each = nrow(cell_data)),
    Pct = c(cell_data[[dvt_col]], cell_data[[ctrl_col]])
  )
  cell_data_long$Group <- factor(cell_data_long$Group, levels = c("Sham", "DVT"))
  cell_data_long$cell_label <- factor(cell_data_long$cell_label,
    levels = cell_data[[label_col]][order(cell_data[[dvt_col]])])

  p5a <- ggplot(cell_data_long, aes(x = Pct, y = cell_label, fill = Group)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85, color = "white") +
    geom_text(aes(label = ifelse(Pct > 0, sprintf("%.1f%%", Pct), "")),
              position = position_dodge(width = 0.8), hjust = -0.1, size = 2.8) +
    scale_fill_manual(values = c("DVT" = cc[["red"]], "Sham" = cc[["blue"]])) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(title = "% F2rl1 (PAR-2) Positive Cells by Cell Type",
         subtitle = paste("Mouse IVC stenosis, Day 14. Total cells:", sum(cell_data$ctrl_n) + sum(cell_data$dvt_n)),
         x = "% PAR-2+ cells", y = "") +
    theme_pub

  # -- 5B: Cascade Co-expression Heatmap (base R, no dplyr) --
  coexpr <- read.csv(file.path(scRNA_dir, "Pathway_Spearman_by_CellType.csv"), check.names = FALSE)
  focus_types <- c("EC","Quiescent_Fib","Activated_Fib","Myofibroblast",
                   "Tcell","NK","VSMC","Resident_Mac","Trem2+_Lipid_Mac","Spp1+_Fibrogenic_Mac")
  focus_types <- intersect(focus_types, coexpr[[1]])

  cascade_genes <- c("Fut8","Lgals3","Cd44","Rhoa","Rock1","Rock2",
                     "Mapk1","Mapk3","Nfkb1","Rela","Stat3","Tgfb1","Selp")
  cascade_genes <- intersect(cascade_genes, colnames(coexpr)[-1])

  coexpr_focus <- coexpr[coexpr[[1]] %in% focus_types, ]
  # 手工构造长表（用位置索引避免空字符串列名问题）
  coexpr_long <- do.call(rbind, lapply(seq_len(nrow(coexpr_focus)), function(i) {
    data.frame(
      CellType = coexpr_focus[i, 1],
      Gene = cascade_genes,
      Rho = as.numeric(as.vector(coexpr_focus[i, cascade_genes])),
      stringsAsFactors = FALSE
    )
  }))
  coexpr_long$CellType <- factor(coexpr_long$CellType, levels = focus_types)
  coexpr_long$Gene <- factor(coexpr_long$Gene, levels = cascade_genes)

  p5b <- ggplot(coexpr_long, aes(x = Gene, y = CellType, fill = Rho)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.3f", Rho)), size = 2.5) +
    scale_fill_gradient2(low = cc[["blue"]], mid = "white", high = cc[["red"]],
                         midpoint = 0, name = "Spearman ρ") +
    labs(title = "GNN Cascade Co-expression with F2rl1 (PAR-2)",
         subtitle = "Spearman correlation: F2rl1 expression vs cascade genes, per cell type") +
    theme_pub + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
                      legend.position = "right")

  # -- 5C: GSEA (base R, no dplyr) --
  gsea_q <- read.csv(file.path(scRNA_dir, "GSEA_Quiescent_Fib.csv"))
  gsea_a <- read.csv(file.path(scRNA_dir, "GSEA_Activated_Fib.csv"))
  # Filter + top-5 by abs(NES)
  gsea_q_sig <- gsea_q[gsea_q$p.adjust < 0.05, ]
  gsea_a_sig <- gsea_a[gsea_a$p.adjust < 0.05, ]
  gsea_q_top <- gsea_q_sig[order(-abs(gsea_q_sig$NES))[1:min(5, nrow(gsea_q_sig))], ]
  gsea_a_top <- gsea_a_sig[order(-abs(gsea_a_sig$NES))[1:min(5, nrow(gsea_a_sig))], ]
  gsea_q_top$CellType <- "Quiescent Fibroblast"
  gsea_a_top$CellType <- "Activated Fibroblast"
  gsea_both <- rbind(gsea_q_top[, c("Description","NES","p.adjust","CellType")],
                     gsea_a_top[, c("Description","NES","p.adjust","CellType")])
  gsea_both$Description <- factor(gsea_both$Description,
    levels = rev(unique(gsea_both$Description[order(gsea_both$NES)])))

  p5c <- ggplot(gsea_both, aes(x = NES, y = Description, fill = CellType)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, color = "white") +
    geom_text(aes(label = sprintf("FDR=%.1e", p.adjust),
                  hjust = ifelse(NES > 0, -0.1, 1.1)),
              position = position_dodge(width = 0.7), size = 2.5) +
    scale_fill_manual(values = c("Quiescent Fibroblast" = cc[["blue"]],
                                  "Activated Fibroblast" = cc[["red"]])) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4) +
    scale_x_continuous(expand = expansion(mult = c(0.3, 0.3))) +
    labs(title = "GSEA: PAR-2+ vs PAR-2− Fibroblasts",
         subtitle = "Top-5 significant GO terms. Negative NES = down in PAR-2+.",
         x = "Normalized Enrichment Score", y = "") +
    theme_pub + theme(axis.text.y = element_text(size = 6.5))

  # -- 5D: Top Upstream TFs --
  tf_data <- read.csv(file.path(scRNA_dir, "F2rl1_Upstream_TFs.csv"))
  tf_data <- tf_data[1:25, ]
  tf_data$TF <- factor(tf_data$TF, levels = rev(tf_data$TF))

  # Flag key vascular TFs
  tf_data$Key <- tf_data$TF %in% c("Gabpa","Rbpj","Mef2c","Elf1","Klf6","Crem")

  p5d <- ggplot(tf_data, aes(x = R, y = TF, fill = Key)) +
    geom_col(width = 0.7, color = "white") +
    geom_text(aes(label = sprintf("FDR=%.1e", FDR)), hjust = -0.1, size = 2.3) +
    scale_fill_manual(values = c("TRUE" = cc[["red"]], "FALSE" = cc[["grey"]]), guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(title = "Top-25 Upstream TFs of PAR-2 (F2rl1)",
         subtitle = "dorothea mouse regulons + viper. Red = known vascular TFs (Rbpj/Notch, Mef2c, Gabpa)",
         x = "Regulon activity (R)", y = "") +
    theme_pub + theme(axis.text.y = element_text(size = 6.5))

  # ── Assembly ──
  design5 <- "
    AABB
    AABB
    CCDD
    CCDD
  "
  fig5 <- p5a + p5b + p5c + p5d +
    plot_layout(design = design5) +
    plot_annotation(
      title = "Figure 5: Single-Cell Transcriptomic Validation of PAR-2 & GNN Cascade",
      tag_levels = "a",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
                    plot.tag = element_text(size = 18, face = "bold")))

  ggsave(file.path(outdir, "Figure5_scRNA_Validation.png"), fig5,
         width = 16, height = 11, dpi = 600, device = "png", bg = "white")

  # ── Export source data ──
  write.csv(cell_data_long, file.path(datadir, "Figure5A_PAR2_pct_positive.csv"), row.names = FALSE)
  write.csv(coexpr_long, file.path(datadir, "Figure5B_cascade_coexpression.csv"), row.names = FALSE)
  write.csv(gsea_both,   file.path(datadir, "Figure5C_GSEA_fibroblasts.csv"), row.names = FALSE)
  write.csv(tf_data,     file.path(datadir, "Figure5D_upstream_TFs.csv"), row.names = FALSE)
  message("  Figure 5 saved.")
}


# ============================================================
# Main
# ============================================================

message("========================================")
message("Rendering paper Figures 4–5 (ggplot2)")
message("========================================")

make_fig4()
make_fig5()

files <- list.files(outdir, pattern = "Figure[45].*\\.png$")
message(sprintf("\nDone. %d files:", length(files)))
for (f in files) message(sprintf("  %s", f))

csv_count <- length(list.files(datadir, pattern = "^Figure[45]"))
message(sprintf("\n%d source data CSVs in source_data/", csv_count))

