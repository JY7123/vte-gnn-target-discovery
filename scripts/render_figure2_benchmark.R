#!/usr/bin/env Rscript
# Figure 2: Benchmark Performance & Baseline Comparison (Refined Layout)
# Consumes: data/baselines/baseline_results.json, checkpoints/full_training_v2/summary.json
library(ggplot2)
library(patchwork)
library(dplyr)
library(tidyr)
library(jsonlite)

if (requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()) {
  setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
}
OUT <- "figures/paper_figures"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1. Load Data
# ============================================================
baselines <- fromJSON("data/baselines/baseline_results.json", simplifyDataFrame = FALSE)
summary   <- fromJSON("checkpoints/full_training_v2/summary.json")

# ============================================================
# Panel A: 5-seed test metrics with non-overlapping labels
# ============================================================
per_seed <- as.data.frame(summary$per_seed)

seed_long <- per_seed %>%
  select(seed, test_auroc, test_hits10, test_mrr) %>%
  pivot_longer(-seed, names_to = "metric", values_to = "value") %>%
  mutate(
    metric = factor(recode(metric,
                           test_auroc = "AUROC", test_hits10 = "Hits@10", test_mrr = "MRR"
    ), levels = c("AUROC", "Hits@10", "MRR")),
    value = as.numeric(value)
  )

# Calculate summary stats + dynamic label y-position ABOVE the max point
stats_df <- seed_long %>%
  group_by(metric) %>%
  summarise(
    mean = mean(value),
    sd = sd(value),
    max_val = max(value),
    .groups = "drop"
  ) %>%
  mutate(
    # Dynamically place text above the highest data point for each metric to avoid overlap
    y_pos = max_val + 0.08
  )

p1 <- ggplot(seed_long, aes(x = metric, y = value, fill = metric)) +
  geom_boxplot(width = 0.45, alpha = 0.6, outlier.shape = NA) +
  geom_jitter(width = 0.08, size = 2.5, alpha = 0.8, color = "grey20") +
  geom_text(data = stats_df, aes(x = metric, y = y_pos,
                                 label = paste0(round(mean, 3), "\n±", round(sd, 3))),
            inherit.aes = FALSE, size = 3.8, fontface = "bold") +
  scale_fill_manual(values = c("AUROC" = "#E64B35", "Hits@10" = "#00A087", "MRR" = "#4DBBD5")) +
  scale_y_continuous(limits = c(0, 0.95), breaks = seq(0, 0.8, 0.2)) +
  labs(title = "A  5-Seed Performance (No Data Leakage)",
       subtitle = paste0(summary$n_seeds, " seeds, 100 epochs, patience=15"),
       x = "", y = "Score") +
  theme_minimal(base_size = 11) + 
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 13),
    panel.grid.minor = element_blank()
  )

# ============================================================
# Panel B & C: Baseline Comparisons (Fixed Key Match)
# ============================================================
# 兼容提取函数：依次尝试匹配各种可能出现的 Key 命名
get_metric_val <- function(model_name, model_data, metric_type) {
  # 如果是 TemperedHGT，优先从 summary 里取
  if (model_name == "TemperedHGT") {
    if (metric_type == "mrr") return(as.numeric(summary$test_mrr_mean))
    if (metric_type == "hits10") return(as.numeric(summary$test_hits10_mean))
  }
  
  if (is.null(model_data)) return(NA)
  
  if (metric_type == "mrr") {
    keys <- c("filtered_mrr", "mrr", "tail_mrr")
  } else if (metric_type == "hits10") {
    keys <- c("tail_hits@10", "hits10", "hits@10", "head_hits@10", "filtered_hits@10")
  }
  
  for (k in keys) {
    if (!is.null(model_data[[k]]) && !is.na(model_data[[k]])) {
      return(as.numeric(model_data[[k]]))
    }
  }
  return(NA)
}

# 重新构建 baseline_df
model_names <- names(baselines)
if (!("TemperedHGT" %in% model_names)) model_names <- c(model_names, "TemperedHGT")

baseline_df <- data.frame(
  model = model_names,
  filtered_mrr = sapply(model_names, function(m) get_metric_val(m, baselines[[m]], "mrr")),
  hits10       = sapply(model_names, function(m) get_metric_val(m, baselines[[m]], "hits10")),
  stringsAsFactors = FALSE
)

model_colors <- c(
  "TransE" = "#8491B4", "DistMult" = "#91D1C2", "ComplEx" = "#F39B7F",
  "RotatE" = "#DC0000", "TemperedHGT" = "#3C5488"
)

# Panel B: Filtered MRR
p2 <- ggplot(baseline_df, aes(x = reorder(model, filtered_mrr), y = filtered_mrr, fill = model)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = ifelse(is.na(filtered_mrr), "N/A", sprintf("%.4f", filtered_mrr))),
            vjust = -0.5, size = 3.8, fontface = "bold") +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  coord_cartesian(clip = "off") +
  labs(title = "B  Filtered MRR: Baseline Comparison",
       subtitle = "Same train/val/test split & filtered ranking protocol",
       x = "", y = "Filtered MRR") +
  theme_minimal(base_size = 11) + 
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 13),
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

# Panel C: Hits@10
p3 <- ggplot(baseline_df, aes(x = reorder(model, hits10), y = hits10, fill = model)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = ifelse(is.na(hits10), "N/A", sprintf("%.4f", hits10))),
            vjust = -0.5, size = 3.8, fontface = "bold") +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.25))) +
  coord_cartesian(clip = "off") +
  labs(title = "C  Hits@10: Baseline Comparison",
       x = "", y = "Hits@10") +
  theme_minimal(base_size = 11) + 
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 13),
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

# ============================================================
# Assemble: Clean 3-Panel Layout (A on left, B & C stacked on right)
# ============================================================
combined <- (p1 | (p2 / p3)) +
  plot_layout(widths = c(1.2, 1)) +
  plot_annotation(
    title = "Figure 2: Model Performance & Baseline Comparison",
    subtitle = "Tempered HGT vs. Classical Knowledge Graph Embedding Models"
  ) &
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 12, color = "grey30")
  )

ggsave(file.path(OUT, "Figure2_Benchmark.tiff"), combined,
       width = 13, height = 9, dpi = 300, compression = "lzw")
message("Saved optimized Figure 2 to: ", file.path(OUT, "Figure2_Benchmark.tiff"))

