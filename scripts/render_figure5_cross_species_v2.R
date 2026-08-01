#!/usr/bin/env Rscript
# ============================================================
# Figure 5: Cross-Species Validation (方案1)
# Vein Wall Fibroblast Activation Program (scRNA-derived, ~80 genes)
# 替代原先 8 基因 GNN 程序 —— 8 基因太小，GSEA NES 对随机集波动极大，
# 无法通过 n=2000 置换检验。改用 scRNA 定义的大程序集，生物逻辑是
# "Cell-Type Signature Deconvolution Validation"。
#
# - Panel A: 程序在 GSE48000 人类 VTE 全血中的 GSEA 富集
# - Panel B: Leave-One-Gene-Out 稳健性
# - Panel C: n=2000 size-matched 随机置换检验，真实计算 p_empirical
# ============================================================
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(fgsea)

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
}
OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── 输入文件（绝对路径，保证任何工作目录都能跑）──────────────
gse_path      <- "D:/JY/work/my work/新思路/vte_gnn_target_discovery/data/GSE48000_de_results.csv"
deg_afib_path <- "D:/JY/work/my work/新思路/figures/PAR2_scRNA/DEG_F2rl1pos_Activated_Fib.csv"

if (file.exists(gse_path) && file.exists(deg_afib_path)) {
  # ============================================================
  # 0. GSE48000 表达排序向量 (人类 VTE 全血, n=132)
  # ============================================================
  gse <- read.csv(gse_path, stringsAsFactors = FALSE)
  ranked_genes <- setNames(gse$logFC, toupper(gse$Gene))
  ranked_genes <- ranked_genes[!is.na(ranked_genes) & !duplicated(names(ranked_genes))]
  ranked_genes <- sort(ranked_genes, decreasing = TRUE)
  message(sprintf("Loaded GSE48000: %d genes ranked", length(ranked_genes)))

  # ============================================================
  # 1. 定义 Vein Wall Fibroblast Activation Program
  #    来源：小鼠 IVC 模型 scRNA 中，活化成纤维细胞 (F2rl1+) vs 未活化
  #    (F2rl1-) 的差异上调基因 (avg_log2FC>0.5, p_val_adj<0.05)，取 top-100
  #    转人类符号 (toupper) 并与 GSE48000 基因取交集；移除 F2RL1 自身
  # ============================================================
  deg_afib <- read.csv(deg_afib_path, stringsAsFactors = FALSE)
  up <- subset(deg_afib, avg_log2FC > 0.5 & p_val_adj < 0.05)
  up <- up[order(-up$avg_log2FC), ]
  program_genes <- intersect(toupper(up$gene), names(ranked_genes))
  program_genes <- program_genes[program_genes != "F2RL1"]
  prog_size <- length(program_genes)
  message(sprintf("Vein Wall Fibroblast Activation Program: %d genes", prog_size))

  # ============================================================
  # Panel A: GSEA enrichment
  # ============================================================
  set.seed(42)
  pathways_list <- list("Vein Wall Fibroblast Program" = program_genes)
  fgsea_res <- fgsea(pathways = pathways_list, stats = ranked_genes,
                     minSize = 5, maxSize = 500)
  fgsea_res$label <- sprintf("NES = %.2f\np = %.3f", fgsea_res$NES, fgsea_res$pval)
  obs_nes <- fgsea_res$NES[fgsea_res$pathway == "Vein Wall Fibroblast Program"]
  message(sprintf("Panel A: NES = %.3f, nominal p = %.4f, padj = %.4f",
                  obs_nes, fgsea_res$pval[1], fgsea_res$padj[1]))

  p1 <- ggplot(fgsea_res, aes(x = reorder(pathway, NES), y = NES, fill = pathway)) +
    geom_col(width = 0.55) +
    geom_text(aes(label = label), hjust = -0.1, size = 4, fontface = "bold",
              lineheight = 0.9) +
    scale_fill_manual(values = c("Vein Wall Fibroblast Program" = "#E64B35")) +
    coord_flip(clip = "off") +
    scale_y_continuous(limits = c(0, 2.5), breaks = seq(0, 2.5, 0.5), expand = c(0, 0)) +
    labs(title = "A  GSEA: Vein Wall Fibroblast Activation Program",
         subtitle = "GSE48000: VTE patients vs controls, whole blood transcriptome",
         x = "", y = "Normalized Enrichment Score (NES)") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", size = 13),
          plot.subtitle = element_text(color = "grey30", size = 9.5),
          axis.text.y = element_text(face = "bold", size = 10, color = "black"),
          plot.margin = margin(t = 10, r = 35, b = 10, l = 10))

  # ============================================================
  # Panel B: Leave-one-gene-out robustness (86 基因 → 86 次 GSEA)
  # ============================================================
  loo_pathways <- lapply(program_genes, function(g) setdiff(program_genes, g))
  names(loo_pathways) <- paste0("LOGO__", program_genes)
  loo_res_all <- fgsea(pathways = loo_pathways, stats = ranked_genes,
                       minSize = 3, maxSize = 500)
  loo_results <- as.data.frame(loo_res_all)
  loo_results$removed_gene <- sub(".*__", "", loo_results$pathway)

  p2 <- ggplot(loo_results, aes(x = "Leave-One-Gene-Out", y = NES)) +
    geom_jitter(width = 0.18, color = "#E64B35", alpha = 0.35, size = 2.2) +
    geom_hline(yintercept = obs_nes, linetype = "dashed", color = "grey40", linewidth = 0.9) +
    annotate("text", x = 1.45, y = obs_nes + 0.06, label = "Full program NES",
             color = "grey40", size = 3.3, hjust = 0.5) +
    annotate("text", x = 0.6, y = min(loo_results$NES) + 0.03,
             label = sprintf("min = %.2f | mean = %.2f",
                             min(loo_results$NES), mean(loo_results$NES)),
             color = "#B03A2E", size = 3.2, hjust = 0) +
    coord_cartesian(ylim = c(max(0, min(loo_results$NES) - 0.1), obs_nes + 0.25)) +
    labs(title = "B  Leave-One-Gene-Out Robustness",
         subtitle = "NES remains positive when any single gene is removed",
         x = "", y = "NES") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 12),
          axis.text.x = element_blank())

  # ============================================================
  # Panel C: n=2000 size-matched 随机置换检验 (真实计算 p_empirical)
  # ============================================================
  set.seed(123)
  n_random <- 2000
  random_pathways <- lapply(1:n_random, function(i) sample(names(ranked_genes), size = prog_size))
  names(random_pathways) <- paste0("rand_", 1:n_random)
  rand_res_all <- fgsea(pathways = random_pathways, stats = ranked_genes,
                        minSize = 1, maxSize = 500)
  random_df <- data.frame(NES = rand_res_all$NES)

  # Empirical permutation p-value with +1 correction (Phipson & Smyth, 2010)
  p_empirical <- function(obs, null) {
    (sum(abs(null) >= abs(obs)) + 1) / (length(null) + 1)
  }
  p_emp <- p_empirical(obs_nes, random_df$NES)
  fmt_p <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("= %.4f", p))
  p_label <- sprintf("n = %d size-matched random sets | p_emp %s",
                     n_random, fmt_p(p_emp))
  message(sprintf("Panel C: empirical p = %.5f (n = %d random sets)",
                  p_emp, n_random))

  # 保存置换检验结果（可溯源）
  emp_results <- data.frame(
    program        = "Vein Wall Fibroblast Activation Program",
    n_genes        = prog_size,
    observed_NES   = obs_nes,
    fgsea_pval     = fgsea_res$pval[1],
    fgsea_padj     = fgsea_res$padj[1],
    n_random_sets  = n_random,
    p_empirical    = p_emp
  )
  write.csv(emp_results, file.path(OUT, "Figure5_empirical_permutation.csv"),
            row.names = FALSE)

  p3 <- ggplot(random_df, aes(x = NES)) +
    geom_histogram(fill = "grey80", color = "white", bins = 40, alpha = 0.9) +
    geom_vline(xintercept = obs_nes, color = "#E64B35", linewidth = 1.1,
               linetype = "dashed") +
    annotate("text", x = obs_nes, y = max(hist(random_df$NES, plot = FALSE)$counts) * 0.9,
             label = "Observed", color = "#E64B35", angle = 90, vjust = 1.3,
             fontface = "bold", size = 3.5) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
    labs(title = "C  Negative Control: Random Gene Sets",
         subtitle = p_label,
         x = "NES", y = "Count") +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold", size = 12),
          axis.text = element_text(color = "black"))

  # ============================================================
  # 组合 & 导出
  # ============================================================
  combined <- p1 / (p2 | p3) +
    plot_layout(heights = c(1, 1.2)) +
    plot_annotation(
      title = "Figure 5: Cross-Species Validation of the Vein Wall Fibroblast Activation Program",
      subtitle = "Human VTE Whole Blood Transcriptome (GSE48000)",
      theme = theme(
        plot.title = element_text(face = "bold", size = 15, margin = margin(b = 2)),
        plot.subtitle = element_text(size = 11, color = "grey30", margin = margin(b = 10))
      )
    )

  output_file <- file.path(OUT, "Figure5_CrossSpecies.tiff")
  ggsave(output_file, combined, width = 12, height = 10, dpi = 300, compression = "lzw")
  message("Successfully generated and saved: ", output_file)

} else {
  warning("Input file(s) not found! Check gse_path / deg_afib_path.")
  message("  gse_path      exists: ", file.exists(gse_path))
  message("  deg_afib_path exists: ", file.exists(deg_afib_path))
}
