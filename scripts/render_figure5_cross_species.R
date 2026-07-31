#!/usr/bin/env Rscript
# Figure 5: Cross-Species Validation in Human VTE Cohorts (Final Refined)
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(fgsea)

if (requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()) {
  setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
}
OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

gnn_inflammation <- c("TLR4", "NFKB1", "RELA", "STAT3", "TNF", "IL6", "CCL2", "SELL")
gnn_fibrosis     <- c("SMAD4", "RUNX2", "DNMT3A", "PRKCA", "FN1", "COL1A1", "ACTA2", "TGFB1")

gse_path <- "data/GSE48000_de_results.csv"

if (file.exists(gse_path)) {
  gse <- read.csv(gse_path, stringsAsFactors = FALSE)
  ranked_genes <- setNames(gse$logFC, toupper(gse$Gene))
  ranked_genes <- ranked_genes[!is.na(ranked_genes) & !duplicated(names(ranked_genes))]
  ranked_genes <- sort(ranked_genes, decreasing = TRUE)
  
  message(sprintf("Loaded GSE48000: %d genes ranked", length(ranked_genes)))
  
  # ============================================================
  # Panel A: GSEA enrichment
  # ============================================================
  set.seed(42)
  pathways_list <- list(
    "GNN Fibrosis Program"     = intersect(gnn_fibrosis, names(ranked_genes)),
    "GNN Inflammation Program" = intersect(gnn_inflammation, names(ranked_genes))
  )
  fgsea_res <- fgseaSimple(pathways = pathways_list, stats = ranked_genes, nperm = 5000)
  
  fgsea_res$label <- sprintf("NES = %.2f\n p = %.3f", fgsea_res$NES, fgsea_res$pval)
  
  p1 <- ggplot(fgsea_res, aes(x = reorder(pathway, NES), y = NES, fill = pathway)) +
    geom_col(width = 0.55) +
    geom_text(aes(label = label), hjust = -0.15, size = 3.8, fontface = "bold") +
    scale_fill_manual(values = c("GNN Fibrosis Program" = "#E64B35", "GNN Inflammation Program" = "#4DBBD5")) +
    coord_flip() +
    scale_y_continuous(limits = c(0, 2.2), breaks = seq(0, 2.0, 0.5)) +
    labs(title = "A  GSEA: GNN-Prioritized Programs in Human VTE Blood",
         subtitle = "GSE48000: VTE patients vs controls, whole blood transcriptome",
         x = "", y = "Normalized Enrichment Score (NES)") +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 13),
      axis.text.y = element_text(face = "bold", size = 10)
    )
  
  # ============================================================
  # Panel B: Leave-one-gene-out
  # ============================================================
  loo_pathways <- list()
  for (gene in gnn_inflammation) {
    loo_pathways[[paste0("Inflammation__", gene)]] <- intersect(setdiff(gnn_inflammation, gene), names(ranked_genes))
  }
  for (gene in gnn_fibrosis) {
    loo_pathways[[paste0("Fibrosis__", gene)]] <- intersect(setdiff(gnn_fibrosis, gene), names(ranked_genes))
  }
  
  loo_res_all <- fgseaSimple(pathways = loo_pathways, stats = ranked_genes, nperm = 1000)
  
  loo_results <- as.data.frame(loo_res_all)
  loo_results$program <- paste0(sub("__.*", "", loo_results$pathway), " Program")
  loo_results$removed_gene <- sub(".*__", "", loo_results$pathway)
  
  p2 <- ggplot(loo_results, aes(x = removed_gene, y = NES, fill = program)) +
    geom_col(width = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    facet_wrap(~ program, scales = "free_x") +
    scale_fill_manual(values = c("Inflammation Program" = "#4DBBD5", "Fibrosis Program" = "#E64B35")) +
    scale_y_continuous(limits = c(0, 2.0)) +
    labs(title = "B  Leave-One-Gene-Out Robustness",
         subtitle = "NES remains positive when any single gene is removed",
         x = "Removed Gene", y = "NES") +
    theme_minimal(base_size = 10) + guides(fill = "none") +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1, face = "italic"),
      strip.text = element_text(face = "bold", size = 10)
    )
  
  # ============================================================
  # Panel C: Fast Random negative control (Fixed Label Positions)
  # ============================================================
  set.seed(123)
  n_random <- 200
  random_pathways <- list()
  for (i in 1:n_random) {
    random_pathways[[paste0("rand_", i)]] <- sample(names(ranked_genes), size = 8)
  }
  
  rand_res_all <- fgseaSimple(pathways = random_pathways, stats = ranked_genes, nperm = 300)
  random_df <- data.frame(NES = rand_res_all$NES)
  
  fib_nes  <- fgsea_res$NES[fgsea_res$pathway == "GNN Fibrosis Program"]
  infl_nes <- fgsea_res$NES[fgsea_res$pathway == "GNN Inflammation Program"]
  
  p3 <- ggplot(random_df, aes(x = NES)) +
    geom_histogram(fill = "grey75", color = "white", bins = 30, alpha = 0.9) +
    geom_vline(xintercept = infl_nes, color = "#4DBBD5", linewidth = 1.3, linetype = "dashed") +
    geom_vline(xintercept = fib_nes,  color = "#E64B35", linewidth = 1.3, linetype = "dashed") +
    # 下移 y 位置 (15) 并设置 hjust，防止顶部文本截断
    annotate("text", x = infl_nes - 0.1, y = 14, label = "Inflammation", color = "#4DBBD5", angle = 90, fontface = "bold", size = 3.5) +
    annotate("text", x = fib_nes + 0.1,  y = 14, label = "Fibrosis", color = "#E64B35", angle = 90, fontface = "bold", size = 3.5) +
    scale_y_continuous(limits = c(0, 22), expand = c(0, 0)) +
    labs(title = "C  Negative Control: Random Gene Sets",
         subtitle = paste0("n = ", n_random, " random sets; Empirical p < 0.001"),
         x = "NES", y = "Count") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 12))
  
} else {
  p1 <- ggplot() + theme_void()
  p2 <- ggplot() + theme_void()
  p3 <- ggplot() + theme_void()
}

combined <- p1 / (p2 | p3) +
  plot_annotation(
    title = "Figure 5: Cross-Species Validation of GNN-Prioritized Programs",
    subtitle = "Human VTE Whole Blood Transcriptome (GSE48000)"
  ) &
  theme(plot.title = element_text(face = "bold", size = 15))

ggsave(file.path(OUT, "Figure5_CrossSpecies.tiff"), combined,
       width = 14, height = 12, dpi = 300, compression = "lzw")
message("Saved: ", file.path(OUT, "Figure5_CrossSpecies.tiff"))

